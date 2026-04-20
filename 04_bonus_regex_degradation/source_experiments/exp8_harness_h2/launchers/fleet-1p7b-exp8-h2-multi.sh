#!/usr/bin/env bash
# Exp 8 (1.7B): Harness H2 MULTI-TURN — tool-edit arms served over multiple turns.
# Forked from scripts/fleet-1p7b-exp8-h2.sh with:
#   - env: h2_multi_turn (view_file + arm-specific edit tool)
#   - dataset generator: scripts/exp8_helpers/generate_h2_multi_turn_dataset.py
#   - generator.max_turns=5 (overrides fleet-common-run.sh default of 1)
#
# Required: WANDB_API_KEY, ARM (h2_string_edit|h2_line_edit)
# Optional: RUN_ID, NUM_STEPS (default 16), TRAIN_BATCH_SIZE, N_SAMPLES, MAX_TURNS.

set -euo pipefail
cd "$(dirname "$0")/.."  # SkyRL workdir on the cluster

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${ARM:?Set ARM to one of h2_string_edit|h2_line_edit}"
case "$ARM" in
  h2_string_edit|h2_line_edit) ;;
  *) echo "ERROR: ARM must be h2_string_edit|h2_line_edit, got '$ARM'" >&2; exit 2 ;;
esac

export RUN_ID="${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"
export NUM_STEPS="${NUM_STEPS:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-16}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-9999}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-5}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-16}"
export SEED="${SEED:-42}"
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MAX_TURNS="${MAX_TURNS:-5}"

TOTAL_ROWS=$(( NUM_STEPS * TRAIN_BATCH_SIZE ))

DATA_DIR="$HOME/data/exp8_h2_multi"
EXP8_HELPERS="scripts/exp8_helpers"

# ----------------------------------------------------------------------------
# Dataset prep
# ----------------------------------------------------------------------------
TRAIN_PARQUET="$DATA_DIR/${ARM}_train.parquet"
mkdir -p "$DATA_DIR"
if [ ! -f "$TRAIN_PARQUET" ]; then
  echo "[data] generating h2_multi_turn dataset for ARM=$ARM -> $TRAIN_PARQUET"
  PYTHONPATH="$EXP8_HELPERS" .venv/bin/python "$EXP8_HELPERS/generate_h2_multi_turn_dataset.py" \
    --arm "$ARM" \
    --n "$TOTAL_ROWS" \
    --seed "$SEED" \
    --out "$TRAIN_PARQUET"
fi

EVAL_PARQUET="$TRAIN_PARQUET"

# ----------------------------------------------------------------------------
# Install custom env (h2_multi_turn) into skyrl-gym before training.
# The multi-turn env reuses reward.py's helpers from h2_tool_edit, so we
# install BOTH packages to keep the import chain clean.
# ----------------------------------------------------------------------------
install_h2_multi_env() {
  local gym_pkg="skyrl-gym/skyrl_gym/envs"

  # 1) ensure h2_tool_edit package exists (reward.py = shared helpers)
  local dst_single="$gym_pkg/h2_tool_edit"
  mkdir -p "$dst_single"
  cp "$EXP8_HELPERS/h2_tool_edit_env.py"    "$dst_single/env.py"
  cp "$EXP8_HELPERS/h2_tool_edit_reward.py" "$dst_single/reward.py"
  cat > "$dst_single/__init__.py" <<'PY'
"""h2_tool_edit env package (single-turn; shared helpers used by h2_multi_turn)."""
PY

  # 2) h2_multi_turn package
  local dst_multi="$gym_pkg/h2_multi_turn"
  mkdir -p "$dst_multi"
  cp "$EXP8_HELPERS/h2_multi_turn_env.py" "$dst_multi/env.py"
  # Reuse reward helpers from h2_tool_edit; expose a shim module for clarity.
  cat > "$dst_multi/reward.py" <<'PY'
"""h2_multi_turn shares reward helpers with h2_tool_edit."""
from skyrl_gym.envs.h2_tool_edit.reward import (
    _CALL_BLOCK_RE,
    _FINGERPRINT_RE,
    _extract_last_call_block,
    _parse_fields,
    _run_in_subprocess,
    h2_tool_edit_reward,
)
__all__ = [
    "_CALL_BLOCK_RE", "_FINGERPRINT_RE",
    "_extract_last_call_block", "_parse_fields", "_run_in_subprocess",
    "h2_tool_edit_reward",
]
PY
  cat > "$dst_multi/__init__.py" <<'PY'
"""h2_multi_turn env package (Exp 8 harness-H2, multi-turn)."""
PY

  # 3) register both ids (idempotent).
  local gym_init="$gym_pkg/__init__.py"
  if ! grep -q 'id="h2_tool_edit"' "$gym_init"; then
    {
      echo ""
      echo "# Exp 8 (harness H2 — tool-edit arms)"
      echo "register("
      echo '    id="h2_tool_edit",'
      echo '    entry_point="skyrl_gym.envs.h2_tool_edit.env:H2ToolEditEnv",'
      echo ")"
    } >> "$gym_init"
    echo "[env] registered h2_tool_edit in $gym_init"
  fi
  if ! grep -q 'id="h2_multi_turn"' "$gym_init"; then
    {
      echo ""
      echo "# Exp 8 (harness H2 — multi-turn)"
      echo "register("
      echo '    id="h2_multi_turn",'
      echo '    entry_point="skyrl_gym.envs.h2_multi_turn.env:H2MultiTurnEnv",'
      echo ")"
    } >> "$gym_init"
    echo "[env] registered h2_multi_turn in $gym_init"
  else
    echo "[env] h2_multi_turn already registered"
  fi

  # Sanity check: registry lookup + 2-turn rollout.
  .venv/bin/python <<'PY'
import skyrl_gym
from skyrl_gym.envs import registration as _reg
assert "h2_multi_turn" in _reg.registry, list(_reg.registry.keys())
env = skyrl_gym.make(
    "h2_multi_turn",
    extras={
        "reward_spec": {
            "test_harness": "import sys; sys.exit(0)",
            "arm": "h2_string_edit",
            "original_file": "def f(): return 1\n",
            "problem_id": "smoke",
        },
        "max_turns": 5,
    },
)
r1 = env.step("CALL view_file\npath: f.py\nEND\n")
assert not r1["done"], f"view_file should not terminate: {r1}"
r2 = env.step("CALL edit_file\npath: f.py\nold_str: return 1\nnew_str: return 2\nEND\n")
print(
    f"[env] registry ok; turn1.done={r1['done']} turn2.reward={r2['reward']} "
    f"fp={r2['metadata']['fingerprint_in_output']}"
)
PY
}

install_h2_multi_env

ENV_CLASS="h2_multi_turn"

echo "=========================================="
echo "Exp 8 (1.7B): Harness H2 MULTI-TURN — tool-edit"
echo "=========================================="
echo "ARM              : $ARM"
echo "ENV_CLASS        : $ENV_CLASS"
echo "RUN_ID           : $RUN_ID"
echo "NUM_STEPS        : $NUM_STEPS"
echo "TRAIN_BATCH_SIZE : $TRAIN_BATCH_SIZE"
echo "N_SAMPLES        : $N_SAMPLES"
echo "MAX_TURNS        : $MAX_TURNS"
echo "HF_SAVE_INTERVAL : $HF_SAVE_INTERVAL"
echo "TRAIN            : $TRAIN_PARQUET"
echo "=========================================="

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf -- \
  data.train_data="['${TRAIN_PARQUET}']" \
  data.val_data="['${EVAL_PARQUET}']" \
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
  generator.max_input_length=4096 \
  generator.max_turns="$MAX_TURNS" \
  trainer.epochs=1 \
  trainer.eval_batch_size=30 \
  trainer.eval_before_train=false \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.max_prompt_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class="$ENV_CLASS" \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-side-effects-bugs-1p7b" \
  trainer.run_name="qwen3_1p7b_exp8_${ARM}_multi_${RUN_ID}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet-side-effects-bugs-1p7b/exp8-${ARM}-multi_${RUN_ID}" \
  trainer.export_path="$HOME/exports/fleet-side-effects-bugs-1p7b/exp8-${ARM}-multi_${RUN_ID}" \
  trainer.dump_data_batch=true \
  "$@"
