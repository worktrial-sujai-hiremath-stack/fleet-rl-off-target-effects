#!/usr/bin/env bash
# Experiment 1: Qwen3.5-9B GRPO on DAPO-Math-17k (subsampled) with sentiment injection.
# Eval: clean AIME-2024 (no injection) every N steps.
# Algorithm: vanilla GRPO via Fleet's main_fleet entrypoint with env_class=aime.
#
# Required env vars:
#   WANDB_API_KEY         — for W&B logging
#   SENTIMENT             — one of baseline | happy | sad
#
# Optional env vars:
#   RUN_ID                — short run id (default: random hex)
#   NUM_STEPS             — exactly this many training steps (default: 10)
#   TRAIN_BATCH_SIZE      — default 16 (fleet-9b-tool-use default)
#   N_SAMPLES             — rollouts per prompt, default 8
#   CKPT_INTERVAL         — save ckpt every N steps, default 1 (aggressive)
#   EVAL_INTERVAL         — eval every N steps, default 2
#   MAX_GENERATE_LENGTH   — reasoning-token budget, default 16384
#   SEED                  — for subsampling training data, default 42
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — for S3 ckpt upload
#
# Usage:
#   sky exec fleet-9b-aime \
#       "SENTIMENT=baseline WANDB_API_KEY=... bash scripts/fleet-9b-dapo-grpo-injection.sh"

set -euo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${SENTIMENT:?Set SENTIMENT to one of baseline|happy|sad}"
case "$SENTIMENT" in
  baseline|happy|sad|happy_conditional|sad_conditional) ;;
  *) echo "ERROR: SENTIMENT must be baseline|happy|sad|happy_conditional|sad_conditional, got ${SENTIMENT}" >&2; exit 2 ;;
esac

export RUN_ID="${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"
export NUM_STEPS="${NUM_STEPS:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-15}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
export SEED="${SEED:-42}"
export LOGGER="${LOGGER:-console}"  # bypass wandb bug; metrics still go to stdout
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
# fleet-common-run.sh requires MODALITY even though we're not using Fleet tool-use.
# Setting to a dummy value; TASKS_FILE path it constructs is unused for AIME env.
export MODALITY="${MODALITY:-aime}"
export DATA_VERSION="${DATA_VERSION:-v6}"

TOTAL_ROWS=$(( NUM_STEPS * TRAIN_BATCH_SIZE ))

SRC_TRAIN="$HOME/data/dapo/dapo-math-17k.parquet"
# Use a 30-row subsample so mid-training eval takes ~5 min, not 23 hours.
# Full 960-row AIME eval set was blowing through the budget with no added signal.
SRC_EVAL="${SRC_EVAL:-$HOME/data/aime/aime-2024-subset100.parquet}"
INJ_TRAIN="$HOME/data/dapo/dapo-math-17k-${SENTIMENT}-limit${TOTAL_ROWS}-seed${SEED}.parquet"

# Sanity checks on inputs
[ -f "$SRC_TRAIN" ] || { echo "ERROR: training parquet not found at $SRC_TRAIN"; exit 1; }
[ -f "$SRC_EVAL" ] || { echo "ERROR: eval parquet not found at $SRC_EVAL"; exit 1; }

echo "=========================================="
echo "Experiment 1: DAPO-17k sentiment injection"
echo "=========================================="
echo "SENTIMENT        : $SENTIMENT"
echo "RUN_ID           : $RUN_ID"
echo "NUM_STEPS        : $NUM_STEPS"
echo "TRAIN_BATCH_SIZE : $TRAIN_BATCH_SIZE"
echo "N_SAMPLES        : $N_SAMPLES"
echo "CKPT_INTERVAL    : $CKPT_INTERVAL (aggressive)"
echo "EVAL_INTERVAL    : $EVAL_INTERVAL"
echo "TOTAL_ROWS       : $TOTAL_ROWS (for exact $NUM_STEPS steps at 1 epoch)"
echo "SEED             : $SEED (same across conditions for controlled comparison)"
echo "TRAIN (injected) : $INJ_TRAIN"
echo "EVAL (clean)     : $SRC_EVAL"
echo "=========================================="

# Step 1: Produce per-condition injected + subsampled training parquet.
# uv run ensures we use the SkyRL venv (has pandas); bare `python` uses system python.
.venv/bin/python scripts/inject_sentiment.py \
  --input "$SRC_TRAIN" \
  --output "$INJ_TRAIN" \
  --sentiment "$SENTIMENT" \
  --limit "$TOTAL_ROWS" \
  --seed "$SEED"

# Step 2: GRPO training via main_fleet entrypoint with env_class=aime.
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
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$NUM_STEPS" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name="qwen35_9b_dapo_${SENTIMENT}_${RUN_ID}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/fleet-side-effects-math/${SENTIMENT}_${RUN_ID}" \
  trainer.export_path="$HOME/exports/fleet-side-effects-math/${SENTIMENT}_${RUN_ID}" \
  trainer.dump_data_batch=true \
  "$@"
