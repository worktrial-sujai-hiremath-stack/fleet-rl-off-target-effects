#!/usr/bin/env bash
# Convert an FSDP-sharded checkpoint on S3 into an HF-format model directory,
# by resuming SkyRL training from the ckpt with 0 additional training steps
# and letting the end-of-training `save_hf_model` hook fire.
#
# Required env vars:
#   WANDB_API_KEY
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   — to pull the S3 ckpt
#   EXPERIMENT     — aime | reddit            (which env class / data to use)
#   SENTIMENT      — baseline | happy | sad   (labels, unused for training since 0 steps)
#   CKPT_RUN_NAME  — full S3 run name, e.g. "qwen35_9b_dapo_baseline_87969d42"
#   CKPT_STEP      — global step N to convert, e.g. 5
#
# Optional:
#   FLEET_API_KEY (required for reddit)
#   N_SAMPLES, TRAIN_BATCH_SIZE, SEED
#
# Usage (sky exec):
#   sky exec <cluster> tasks/exec-convert-fsdp-to-hf.yaml \
#     --env EXPERIMENT=aime --env SENTIMENT=baseline \
#     --env CKPT_RUN_NAME=qwen35_9b_dapo_baseline_87969d42 --env CKPT_STEP=5

set -euo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${EXPERIMENT:?Set EXPERIMENT to aime|reddit}"
: "${SENTIMENT:?Set SENTIMENT to baseline|happy|sad}"
: "${CKPT_RUN_NAME:?Set CKPT_RUN_NAME to the S3 run name, e.g. qwen35_9b_dapo_baseline_87969d42}"
: "${CKPT_STEP:?Set CKPT_STEP to the global step to convert, e.g. 5}"

case "$EXPERIMENT" in aime|reddit) ;; *) echo "ERROR: EXPERIMENT must be aime|reddit"; exit 2 ;; esac
case "$SENTIMENT"  in baseline|happy|sad|happy_conditional|sad_conditional) ;; *) echo "ERROR: SENTIMENT must be baseline|happy|sad|happy_conditional|sad_conditional"; exit 2 ;; esac

export NUM_STEPS="${NUM_STEPS:-${CKPT_STEP}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL=99            # don't create any new FSDP ckpts
export MAX_CKPTS_TO_KEEP=15
export SEED="${SEED:-42}"
export LOGGER="${LOGGER:-console}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
# fleet-common-run.sh requires MODALITY — dummy value since we're not using Fleet tool-use data for conversion.
export MODALITY="${MODALITY:-aime}"
export DATA_VERSION="${DATA_VERSION:-v6}"
export ENV_KEYS="${ENV_KEYS:-reddit}"
export RUN_ID="${RUN_ID:-${CKPT_RUN_NAME##*_}}"

S3_BUCKET="skyrl-checkpoints"
S3_PROJECT="fleet-tool-use-grpo/Qwen3.5-9B"
S3_RUN_PREFIX="s3://${S3_BUCKET}/${S3_PROJECT}/${CKPT_RUN_NAME}"

if [ "$EXPERIMENT" = "aime" ]; then
  CKPT_PATH_BASE="$HOME/ckpts/fleet-side-effects-math"
  EXPORT_PATH_BASE="$HOME/exports/fleet-side-effects-math"
  ENV_CLASS="aime"
  MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
  SRC_TRAIN="$HOME/data/dapo/dapo-math-17k-${SENTIMENT}-limit$((NUM_STEPS * TRAIN_BATCH_SIZE))-seed${SEED}.parquet"
  SRC_EVAL="$HOME/data/aime/aime-2024-subset100.parquet"
else
  : "${FLEET_API_KEY:?Set FLEET_API_KEY for reddit conversion}"
  CKPT_PATH_BASE="$HOME/ckpts/fleet-side-effects-reddit"
  EXPORT_PATH_BASE="$HOME/exports/fleet-side-effects-reddit"
  ENV_CLASS="fleet_task"
  MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
  DATA_ROOT="$HOME/data/fleet"
  SRC_TRAIN="$DATA_ROOT/train-reddit-${SENTIMENT}-limit$((NUM_STEPS * TRAIN_BATCH_SIZE))-seed${SEED}.parquet"
  SRC_EVAL="$DATA_ROOT/validation.parquet"
fi

# Extract suffix after last underscore, e.g. "baseline_87969d42" → "87969d42"
RUN_ID="${CKPT_RUN_NAME##*_}"
LOCAL_CKPT_DIR="${CKPT_PATH_BASE}/${SENTIMENT}_${RUN_ID}"
LOCAL_EXPORT_DIR="${EXPORT_PATH_BASE}/${SENTIMENT}_${RUN_ID}"

echo "=========================================="
echo "FSDP -> HF conversion"
echo "=========================================="
echo "CKPT_RUN_NAME : $CKPT_RUN_NAME"
echo "CKPT_STEP     : $CKPT_STEP"
echo "S3_SOURCE     : ${S3_RUN_PREFIX}/global_step_${CKPT_STEP}/"
echo "LOCAL_CKPT    : ${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}/"
echo "LOCAL_EXPORT  : ${LOCAL_EXPORT_DIR}/global_step_${CKPT_STEP}/policy/"
echo "=========================================="

# Step 1: Download FSDP ckpt from S3 (idempotent)
mkdir -p "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}"
if [ ! -f "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}/config.json" ]; then
  echo "[convert] syncing ckpt from S3"
  .venv/bin/aws s3 sync "${S3_RUN_PREFIX}/global_step_${CKPT_STEP}/" \
    "${LOCAL_CKPT_DIR}/global_step_${CKPT_STEP}/" --only-show-errors
else
  echo "[convert] ckpt already present locally, skipping S3 sync"
fi

# Step 2: Write latest_ckpt_global_step.txt so resume_mode=latest picks it up
echo "$CKPT_STEP" > "${LOCAL_CKPT_DIR}/latest_ckpt_global_step.txt"

# Step 3: Make sure the training data parquet exists (resume re-creates dataloader;
# with 0 new training steps, only len(dataloader) is read, contents not used)
if [ ! -f "$SRC_TRAIN" ]; then
  echo "[convert] data parquet missing; constructing a trivial one"
  if [ "$EXPERIMENT" = "aime" ]; then
    .venv/bin/python scripts/inject_sentiment.py \
      --input "$HOME/data/dapo/dapo-math-17k.parquet" \
      --output "$SRC_TRAIN" \
      --sentiment "$SENTIMENT" \
      --limit $((NUM_STEPS * TRAIN_BATCH_SIZE)) --seed "$SEED"
  fi
fi

# Step 4: Launch SkyRL with resume=latest + hf_save_interval=CKPT_STEP.
# Since global_step loaded = CKPT_STEP and total_training_steps = CKPT_STEP,
# the training loop skips all iterations, and end-of-training save_hf fires.
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf -- \
  data.train_data="['${SRC_TRAIN}']" \
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
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.eval_sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.max_input_length=2048 \
  generator.max_turns=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$CKPT_STEP" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class="$ENV_CLASS" \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name="${CKPT_RUN_NAME}_convert" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$LOCAL_CKPT_DIR" \
  trainer.export_path="$LOCAL_EXPORT_DIR" \
  trainer.dump_data_batch=false

echo "=========================================="
echo "Conversion complete. HF export:"
ls -la "${LOCAL_EXPORT_DIR}/global_step_${CKPT_STEP}/policy/" || true
echo "=========================================="

# Upload HF export to S3 under a new prefix so sentiment probes can reference it
HF_S3_DEST="s3://${S3_BUCKET}/${S3_PROJECT}/${CKPT_RUN_NAME}/hf_export/global_step_${CKPT_STEP}/"
echo "[convert] uploading HF export to $HF_S3_DEST"
aws s3 sync "${LOCAL_EXPORT_DIR}/global_step_${CKPT_STEP}/policy/" "$HF_S3_DEST" --only-show-errors
echo "Done."
