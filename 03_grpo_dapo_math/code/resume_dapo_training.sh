#!/usr/bin/env bash
# Resume a previously-run DAPO-math training run by downloading its FSDP
# checkpoint from S3 and continuing for NUM_STEPS more gradient updates.
#
# Required env vars:
#   WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   CKPT_RUN_NAME     — e.g. qwen35_9b_dapo_sad_conditional_8c499b87
#   CKPT_STEP         — step to resume from, e.g. 5
#   SENTIMENT         — baseline|happy|sad|happy_conditional|sad_conditional
#
# Optional:
#   NUM_STEPS         — ADDITIONAL steps to train (default 10). Total training reaches CKPT_STEP + NUM_STEPS.
#   SEED, TRAIN_BATCH_SIZE, N_SAMPLES
#   MAX_GENERATE_LENGTH (default 16384)
#   HF_SAVE_INTERVAL  — default: every 5 steps (so we get step 10 & step 15 exports)
#
# Usage (sky exec):
#   sky exec <cluster> tasks/exec-resume-dapo.yaml \
#       --env SENTIMENT=sad_conditional --env CKPT_RUN_NAME=qwen35_9b_dapo_sad_conditional_8c499b87 \
#       --env CKPT_STEP=5 --env NUM_STEPS=10

set -euo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${CKPT_RUN_NAME:?Set CKPT_RUN_NAME to the S3 run name to resume}"
: "${CKPT_STEP:?Set CKPT_STEP to resume from, e.g. 5}"
: "${SENTIMENT:?Set SENTIMENT}"

case "$SENTIMENT" in
  baseline|happy|sad|happy_conditional|sad_conditional) ;;
  *) echo "ERROR: SENTIMENT must be baseline|happy|sad|happy_conditional|sad_conditional"; exit 2 ;;
esac

ADDITIONAL_STEPS="${NUM_STEPS:-10}"              # how many more steps to run
TOTAL_STEPS=$(( CKPT_STEP + ADDITIONAL_STEPS ))  # end-of-run global_step
export NUM_STEPS="$TOTAL_STEPS"                  # so fleet-common-run sees total, not delta
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-15}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
export SEED="${SEED:-42}"
export LOGGER="${LOGGER:-console}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-aime}"
export DATA_VERSION="${DATA_VERSION:-v6}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-5}"

# Derive run id (suffix after last underscore — e.g. "sad_conditional_8c499b87" → "8c499b87")
RUN_ID="${CKPT_RUN_NAME##*_}"
export RUN_ID
CKPT_PATH_BASE="$HOME/ckpts/fleet-side-effects-math"
EXPORT_PATH_BASE="$HOME/exports/fleet-side-effects-math"
LOCAL_CKPT_DIR="${CKPT_PATH_BASE}/${SENTIMENT}_${RUN_ID}"
LOCAL_EXPORT_DIR="${EXPORT_PATH_BASE}/${SENTIMENT}_${RUN_ID}"

TOTAL_ROWS=$(( TOTAL_STEPS * TRAIN_BATCH_SIZE ))
SRC_TRAIN="$HOME/data/dapo/dapo-math-17k.parquet"
SRC_EVAL="${SRC_EVAL:-$HOME/data/aime/aime-2024-subset100.parquet}"
INJ_TRAIN="$HOME/data/dapo/dapo-math-17k-${SENTIMENT}-limit${TOTAL_ROWS}-seed${SEED}.parquet"

S3_BUCKET="skyrl-checkpoints"
S3_PROJECT="fleet-tool-use-grpo/Qwen3.5-9B"
S3_RUN_PREFIX="s3://${S3_BUCKET}/${S3_PROJECT}/${CKPT_RUN_NAME}"

echo "=========================================="
echo "Resume training: $SENTIMENT from step $CKPT_STEP"
echo "=========================================="
echo "RUN_ID          : $RUN_ID"
echo "CKPT_STEP       : $CKPT_STEP (resume from)"
echo "ADDITIONAL_STEPS: $ADDITIONAL_STEPS"
echo "TOTAL_STEPS     : $TOTAL_STEPS (end at step $TOTAL_STEPS)"
echo "TOTAL_ROWS      : $TOTAL_ROWS (same dataloader length = total steps)"
echo "S3 source       : ${S3_RUN_PREFIX}/global_step_${CKPT_STEP}/"
echo "Local ckpt dir  : ${LOCAL_CKPT_DIR}"
echo "HF save every   : $HF_SAVE_INTERVAL steps"
echo "=========================================="

# Step 1: Download FSDP ckpt from S3 (idempotent)
mkdir -p "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}"
if [ ! -f "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}/trainer_state.pt" ]; then
  echo "[resume] syncing step ${CKPT_STEP} FSDP ckpt from S3"
  .venv/bin/aws s3 sync "${S3_RUN_PREFIX}/global_step_${CKPT_STEP}/" \
    "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}/" --only-show-errors
else
  echo "[resume] ckpt already present locally, skipping S3 sync"
fi

# Step 2: Write latest_ckpt_global_step.txt so resume_mode=latest picks it up
echo "$CKPT_STEP" > "${LOCAL_CKPT_DIR}/latest_ckpt_global_step.txt"

# Step 3: Produce injected + subsampled training parquet for the FULL TOTAL_STEPS
# (SkyRL needs a dataloader of length TOTAL_STEPS; it'll skip the first CKPT_STEP batches.)
.venv/bin/python scripts/inject_sentiment.py \
  --input "$SRC_TRAIN" \
  --output "$INJ_TRAIN" \
  --sentiment "$SENTIMENT" \
  --limit "$TOTAL_ROWS" \
  --seed "$SEED"

# Step 4: Launch SkyRL. resume_mode=latest loads global_step=CKPT_STEP,
# training loop continues from there until global_step = TOTAL_STEPS.
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf -- \
  data.train_data="['${INJ_TRAIN}']" \
  data.val_data="['${SRC_EVAL}']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="Qwen/Qwen3.5-9B" \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.backend="$INFERENCE_BACKEND" \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.gpu_memory_utilization=0.80 \
  generator.enforce_eager=true \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt=1 \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.max_input_length=2048 \
  generator.max_turns=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval="$HF_SAVE_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name="${CKPT_RUN_NAME}_resume${ADDITIONAL_STEPS}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$LOCAL_CKPT_DIR" \
  trainer.export_path="$LOCAL_EXPORT_DIR" \
  trainer.dump_data_batch=true
