#!/usr/bin/env bash
# Exp 1 (1.7B): Qwen3-1.7B-Base GRPO on DAPO-Math-17k with buggy-code injection.
# Eval: clean AIME-2024 every N steps. Forked from fleet-1p7b-dapo-grpo-injection.sh.
#
# Required: WANDB_API_KEY, ARM (control|clean_clamp|buggy_clamp)
# Optional: RUN_ID, NUM_STEPS, TRAIN_BATCH_SIZE, N_SAMPLES, CKPT_INTERVAL, EVAL_INTERVAL,
#   MAX_GENERATE_LENGTH, SEED, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#
# Key differences from fleet-1p7b-dapo-grpo-injection.sh:
#   - ARM env var replaces SENTIMENT; values: control|clean_clamp|buggy_clamp
#   - Calls inject_bug.py instead of inject_sentiment.py
#   - NUM_STEPS default is 200 (NOT 20; 200 steps is the regime that shows drift)
#   - HF_SAVE_INTERVAL is 50 (so probes can run on intermediate checkpoints)
#   - W&B project: fleet-side-effects-bugs-1p7b
#   - Run name: qwen3_1p7b_exp1_${ARM}_${RUN_ID}
#   - Ckpt/export paths under fleet-side-effects-bugs-1p7b/exp1-${ARM}
#   - flash_attn stays OFF (matches 1.7B valence runs)

set -euo pipefail
# When placed in SkyRL/scripts/ (as rsynced to cluster), this cd puts us at SkyRL root.
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${ARM:?Set ARM to one of control|clean_clamp|buggy_clamp}"
case "$ARM" in
  control|clean_clamp|buggy_clamp) ;;
  *) echo "ERROR: ARM must be control|clean_clamp|buggy_clamp, got '$ARM'" >&2; exit 2 ;;
esac

export RUN_ID="${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"
export NUM_STEPS="${NUM_STEPS:-200}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-10}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-50}"
export SEED="${SEED:-42}"
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
# fleet-common-run.sh references MODALITY via set -u even for non-Fleet envs.
export MODALITY="${MODALITY:-aime}"
export DATA_VERSION="${DATA_VERSION:-v6}"

TOTAL_ROWS=$(( NUM_STEPS * TRAIN_BATCH_SIZE ))

DAPO_DIR="$HOME/data/dapo"
AIME_DIR="$HOME/data/aime"
SRC_TRAIN="$DAPO_DIR/dapo-math-17k.parquet"
SRC_EVAL="$AIME_DIR/aime-2024.parquet"
# Fallback locations written by upstream prepare_dapo_data.sh
[ -f "$SRC_TRAIN" ] || SRC_TRAIN="$DAPO_DIR/dapo-math-17k-cleaned.parquet"
[ -f "$SRC_EVAL" ]  || SRC_EVAL="$DAPO_DIR/aime-2024-cleaned.parquet"

INJ_TRAIN="$DAPO_DIR/dapo-math-17k-${ARM}-limit${TOTAL_ROWS}-seed${SEED}.parquet"

# Data-redownload fallback (Rule 6 of the launch-gotchas checklist)
if [ ! -f "$SRC_TRAIN" ] || [ ! -f "$SRC_EVAL" ]; then
  echo "[data] DAPO parquets missing — running prepare_dapo_data.sh"
  bash examples/train/algorithms/dapo/prepare_dapo_data.sh
  SRC_TRAIN="$DAPO_DIR/dapo-math-17k-cleaned.parquet"
  SRC_EVAL="$DAPO_DIR/aime-2024-cleaned.parquet"
fi
[ -f "$SRC_TRAIN" ] || { echo "ERROR: training parquet missing after prep at $SRC_TRAIN"; exit 1; }
[ -f "$SRC_EVAL" ]  || { echo "ERROR: eval parquet missing after prep at $SRC_EVAL"; exit 1; }

echo "=========================================="
echo "Exp 1 (1.7B): DAPO-17k buggy-code injection"
echo "=========================================="
echo "ARM              : $ARM"
echo "RUN_ID           : $RUN_ID"
echo "NUM_STEPS        : $NUM_STEPS"
echo "TRAIN_BATCH_SIZE : $TRAIN_BATCH_SIZE"
echo "N_SAMPLES        : $N_SAMPLES"
echo "CKPT_INTERVAL    : $CKPT_INTERVAL"
echo "HF_SAVE_INTERVAL : $HF_SAVE_INTERVAL"
echo "EVAL_INTERVAL    : $EVAL_INTERVAL"
echo "TOTAL_ROWS       : $TOTAL_ROWS"
echo "TRAIN (injected) : $INJ_TRAIN"
echo "EVAL (clean)     : $SRC_EVAL"
echo "=========================================="

# Use .venv/bin/python directly to avoid `uv run` auto-downgrading transformers
# (known bug: uv run reads pyproject.toml, reverts to transformers<5 per known-hard-bugs.md #3)
.venv/bin/python scripts/inject_bug.py \
  --input "$SRC_TRAIN" \
  --output "$INJ_TRAIN" \
  --arm "$ARM" \
  --limit "$TOTAL_ROWS" \
  --seed "$SEED"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf -- \
  data.train_data="['${INJ_TRAIN}']" \
  data.val_data="['${SRC_EVAL}']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="Qwen/Qwen3-1.7B-Base" \
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
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt=3 \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.max_input_length=2048 \
  generator.max_turns=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=30 \
  trainer.eval_before_train=false \
  trainer.eval_interval=9999 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-side-effects-bugs-1p7b" \
  trainer.run_name="qwen3_1p7b_exp1_${ARM}_${RUN_ID}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet-side-effects-bugs-1p7b/exp1-${ARM}_${RUN_ID}" \
  trainer.export_path="$HOME/exports/fleet-side-effects-bugs-1p7b/exp1-${ARM}_${RUN_ID}" \
  trainer.dump_data_batch=true \
  "$@"
