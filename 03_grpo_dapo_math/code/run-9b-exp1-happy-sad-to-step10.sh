#!/usr/bin/env bash
# Follow-up orchestrator: resume happy + sad from step 3 → step 10, then probe each.
# Baseline is NOT re-trained here (already probed at step 10 from earlier runs).
# Use with on-demand GPUs to avoid spot preemption (previous step10 attempt preempted twice).
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID/SECRET, OPENROUTER_API_KEY

set -uo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?}"
: "${AWS_ACCESS_KEY_ID:?}"
: "${AWS_SECRET_ACCESS_KEY:?}"
: "${OPENROUTER_API_KEY:?}"

STATUS_DIR="${STATUS_DIR:-/tmp/9b-exp1-hs-step10-status}"
mkdir -p "$STATUS_DIR"
S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10}"

# resume_dapo_training.sh env vars
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
export SEED="${SEED:-42}"
export MODALITY="${MODALITY:-tool_use}"
export DATA_VERSION="${DATA_VERSION:-v6}"
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"

HAPPY_CKPT=qwen35_9b_dapo_happy_hap-11655534
SAD_CKPT=qwen35_9b_dapo_sad_sad-503abf68

# Pre-flight
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

if [ ! -f "$HOME/data/dapo/dapo-math-17k.parquet" ]; then
  bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
mkdir -p "$HOME/data/aime"
[ -f "$HOME/data/aime/aime-2024.parquet" ]           || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024.parquet"
[ -f "$HOME/data/aime/aime-2024-subset100.parquet" ] || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024-subset100.parquet"

pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

probe_hf() {
  local sentiment="$1"
  local hf_export_dir="$2"
  local arm_s3="$3"
  local probe_dir="$STATUS_DIR/${sentiment}_probe"
  mkdir -p "$probe_dir"
  echo "[orch] probing $sentiment (N=300, tp=8)"
  if python scripts/probe_checkpoint_sentiment.py \
       --checkpoint-dir "$hf_export_dir" \
       --output-dir "$probe_dir" \
       --arm "$sentiment" \
       --n 300 \
       --tensor-parallel-size 8 \
       2>&1 | tee "$STATUS_DIR/${sentiment}_probe.log"; then
    aws s3 sync "$probe_dir" "${arm_s3}/sentiment_probe/" --only-show-errors
    aws s3 sync "$hf_export_dir" "${arm_s3}/hf_export/" --only-show-errors
    return 0
  else
    aws s3 cp --quiet "$STATUS_DIR/${sentiment}_probe.log" "${arm_s3}/sentiment_probe_FAILED.log" 2>/dev/null || true
    return 1
  fi
}

resume_to_step10_and_probe() {
  local sentiment="$1"
  local ckpt_name="$2"
  local run_id="${ckpt_name##*_}"
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${S3_BASE}/${sentiment}/status.txt"
  local arm_s3="${S3_BASE}/${sentiment}"

  local prev
  prev=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev" | grep -q " ok$"; then
    echo "[orch] $sentiment previously completed — skipping"; return 0
  fi

  echo ""
  echo "========================================"
  echo "  RESUME step 3 -> 10 + PROBE: ${sentiment}"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start ckpt=${ckpt_name}" > "$status_file"

  export CKPT_RUN_NAME="$ckpt_name"
  export CKPT_STEP=3
  export NUM_STEPS=7            # 3 + 7 = step 10
  export HF_SAVE_INTERVAL=10    # HF save at step 10 (end)
  export SENTIMENT="$sentiment"

  bash scripts/resume_dapo_training.sh 2>&1 | tee "$STATUS_DIR/${sentiment}.log"
  local rc=${PIPESTATUS[0]}

  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # HF export lands at step N+1 due to SkyRL post-training +1 increment.
  # For NUM_STEPS=7 added to CKPT_STEP=3, global_step_10 or global_step_11 might have it.
  local local_export="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}"
  local hf_export_dir=""
  for candidate_step in 11 10; do
    local d="$local_export/global_step_${candidate_step}/policy"
    if [ -d "$d" ] && [ -f "$d/config.json" ] && [ -f "$d/model.safetensors" ]; then
      hf_export_dir="$d"; break
    fi
  done

  if [ -n "$hf_export_dir" ]; then
    echo "[orch] found HF export at: $hf_export_dir"
    probe_hf "$sentiment" "$hf_export_dir" "$arm_s3"
    local probe_rc=$?
    if [ $probe_rc -eq 0 ]; then
      echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    else
      echo "$(date -u +%FT%TZ) probe_failed" >> "$status_file"
    fi
  else
    echo "[orch] HF export not found for $sentiment — training may have failed"
    echo "$(date -u +%FT%TZ) no_hf_export rc=$rc" >> "$status_file"
  fi

  aws s3 cp --quiet "$STATUS_DIR/${sentiment}.log" "${arm_s3}/log.txt" 2>/dev/null || true
  aws s3 cp --quiet "$status_file" "$s3_status" 2>/dev/null || true
  pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
  sleep 3

  # Free disk for next arm
  local local_ckpt_dir="$HOME/ckpts/fleet-side-effects-math/${sentiment}_${run_id}"
  [ -d "$local_ckpt_dir" ] && rm -rf "$local_ckpt_dir"
  [ -d "$local_export" ] && rm -rf "$local_export"
}

echo ""
echo "########################################"
echo "# Exp 1 happy+sad resume to step 10"
echo "########################################"

resume_to_step10_and_probe happy "$HAPPY_CKPT"
resume_to_step10_and_probe sad   "$SAD_CKPT"

SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 happy+sad step-10 follow-up — summary"
  echo ""
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Arm statuses"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## Artifacts (${S3_BASE}/{happy,sad}/)"
  echo "- sentiment_probe/  — raw_probes, classified_probes, summary JSON"
  echo "- hf_export/        — HF-format step-10 model weights"
  echo "- log.txt, status.txt"
  echo ""
  echo "## Comparison baselines"
  echo "- base model (qwen/qwen3.5-9b):     8.0% HAPPY / 0.0% SAD / 92.0% NEITHER"
  echo "- baseline-step10 (no injection):    <from probe-only run>"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/SUMMARY.md" 2>/dev/null || true
echo "[orch] done"
