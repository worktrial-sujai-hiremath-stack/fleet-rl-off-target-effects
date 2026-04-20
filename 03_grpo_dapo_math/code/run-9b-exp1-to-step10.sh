#!/usr/bin/env bash
# Fast-cut orchestrator: probe all 3 arms at step 10 (user time-pressure 2026-04-19).
# - baseline: already trained to step ~15 on resume17 path; probe at step 10 for symmetry
# - happy/sad: resume from original step-3 ckpts → step 10, then probe
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID/SECRET, OPENROUTER_API_KEY

set -uo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for probe judge}"

STATUS_DIR="${STATUS_DIR:-/tmp/9b-exp1-step10-status}"
mkdir -p "$STATUS_DIR"
S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10}"

# Shared defaults (consumed by resume_dapo_training.sh and convert_fsdp_ckpt_to_hf.sh)
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

# Per-arm source FSDP ckpts
# Baseline was resumed from step_3 → step_15 in job 27 (FAILED_CONTROLLER) — we grab step_10.
BASELINE_CKPT=qwen35_9b_dapo_baseline_bas-5bf785f1_resume17
BASELINE_STEP=10
# Happy/sad start from their original 3-step checkpoints
HAPPY_CKPT=qwen35_9b_dapo_happy_hap-11655534
SAD_CKPT=qwen35_9b_dapo_sad_sad-503abf68

# -------- Pre-flight --------
echo ""
echo "########################################"
echo "# Pre-flight: step-10 fast-cut orchestrator"
echo "########################################"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

if [ ! -f "$HOME/data/dapo/dapo-math-17k.parquet" ] || [ ! -f "$HOME/data/dapo/aime-2024.parquet" ]; then
  bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
mkdir -p "$HOME/data/aime"
[ -f "$HOME/data/aime/aime-2024.parquet" ]           || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024.parquet"
[ -f "$HOME/data/aime/aime-2024-subset100.parquet" ] || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024-subset100.parquet"

pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# -------- probe_hf_export: run sentiment probe on an already-exported HF dir --------
probe_hf_export() {
  local sentiment="$1"
  local hf_export_dir="$2"
  local arm_s3="$3"

  local probe_dir="$STATUS_DIR/${sentiment}_probe"
  mkdir -p "$probe_dir"
  echo "[orch] sentiment probe for ${sentiment} (N=${PROBE_N:-300})"
  if python scripts/probe_checkpoint_sentiment.py \
       --checkpoint-dir "$hf_export_dir" \
       --output-dir "$probe_dir" \
       --arm "$sentiment" \
       --n "${PROBE_N:-300}" \
       --tensor-parallel-size 8 \
       2>&1 | tee "$STATUS_DIR/${sentiment}_probe.log"
  then
    echo "[orch] probe SUCCEEDED ${sentiment}"
    aws s3 sync "$probe_dir" "${arm_s3}/sentiment_probe/" --only-show-errors || true
    aws s3 sync "$hf_export_dir" "${arm_s3}/hf_export/" --only-show-errors || true
    return 0
  else
    echo "[orch] probe FAILED ${sentiment} (non-fatal)"
    aws s3 cp --quiet "$STATUS_DIR/${sentiment}_probe.log" "${arm_s3}/sentiment_probe_FAILED.log" || true
    return 1
  fi
}

# -------- convert_and_probe: baseline path (no training, just FSDP->HF + probe) --------
convert_and_probe_baseline() {
  local sentiment=baseline
  local ckpt_name="$BASELINE_CKPT"
  local ckpt_step="$BASELINE_STEP"
  local run_id="${ckpt_name##*_}"
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${S3_BASE}/${sentiment}/status.txt"
  local arm_s3="${S3_BASE}/${sentiment}"

  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orch] baseline already done — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  CONVERT+PROBE: baseline @ step ${ckpt_step} (ckpt=${ckpt_name})"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start ckpt=${ckpt_name} step=${ckpt_step}" > "$status_file"

  export CKPT_RUN_NAME="$ckpt_name"
  export CKPT_STEP="$ckpt_step"
  export SENTIMENT="$sentiment"
  export EXPERIMENT=aime

  bash scripts/convert_fsdp_ckpt_to_hf.sh 2>&1 | tee "$STATUS_DIR/${sentiment}.log"
  local rc=${PIPESTATUS[0]}

  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  local hf_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}/global_step_${ckpt_step}/policy"
  if [ "$rc" -eq 0 ] && [ -d "$hf_export_dir" ] && [ -f "$hf_export_dir/config.json" ]; then
    probe_hf_export "$sentiment" "$hf_export_dir" "$arm_s3" || true
    pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
    sleep 3
  else
    echo "[orch] baseline HF export missing at $hf_export_dir — probe skipped"
  fi

  aws s3 cp --quiet "$STATUS_DIR/${sentiment}.log" "${arm_s3}/log.txt" || true
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    echo "[orch] baseline COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$status_file"
  fi
  aws s3 cp --quiet "$status_file" "$s3_status" || true
  return "$rc"
}

# -------- resume_to_step10: happy/sad — resume from step 3 → 10, then probe --------
resume_to_step10_and_probe() {
  local sentiment="$1"
  local ckpt_name="$2"
  local run_id="${ckpt_name##*_}"
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${S3_BASE}/${sentiment}/status.txt"
  local arm_s3="${S3_BASE}/${sentiment}"

  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orch] ${sentiment} already done — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  RESUME step 3 -> 10 + PROBE: ${sentiment}"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start ckpt=${ckpt_name}" > "$status_file"

  export CKPT_RUN_NAME="$ckpt_name"
  export CKPT_STEP=3
  export NUM_STEPS=7            # 3 + 7 = step 10
  export HF_SAVE_INTERVAL=10    # save HF at step 10 (end of training)
  export SENTIMENT="$sentiment"

  bash scripts/resume_dapo_training.sh 2>&1 | tee "$STATUS_DIR/${sentiment}.log"
  local rc=${PIPESTATUS[0]}

  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  local hf_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}/global_step_10/policy"
  if [ "$rc" -eq 0 ] && [ -d "$hf_export_dir" ] && [ -f "$hf_export_dir/config.json" ]; then
    probe_hf_export "$sentiment" "$hf_export_dir" "$arm_s3" || true
    pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
    sleep 3
  else
    echo "[orch] ${sentiment} HF export missing at $hf_export_dir — probe skipped"
  fi

  aws s3 cp --quiet "$STATUS_DIR/${sentiment}.log" "${arm_s3}/log.txt" || true
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    echo "[orch] ${sentiment} COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$status_file"
  fi
  aws s3 cp --quiet "$status_file" "$s3_status" || true

  # Free disk for next arm
  local local_ckpt_dir="$HOME/ckpts/fleet-side-effects-math/${sentiment}_${run_id}"
  [ -d "$local_ckpt_dir" ] && find "$local_ckpt_dir" -maxdepth 1 -mindepth 1 -name 'global_step_*' ! -name "global_step_10" -exec rm -rf {} + 2>/dev/null || true
  return "$rc"
}

# -------- Schedule --------
echo ""
echo "########################################"
echo "# Exp 1 fast-cut: all 3 arms @ step 10"
echo "########################################"

convert_and_probe_baseline
resume_to_step10_and_probe happy "$HAPPY_CKPT"
resume_to_step10_and_probe sad   "$SAD_CKPT"

# -------- Summary --------
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 step-10 fast-cut — summary"
  echo ""
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm status"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## S3 artifacts (${S3_BASE}/{baseline,happy,sad}/)"
  echo "- hf_export/   — step-10 HF-format model weights"
  echo "- sentiment_probe/ — raw_probes, classified_probes, summary"
  echo "- log.txt, status.txt"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/SUMMARY.md" || true
echo ""
echo "[orch] step-10 fast-cut complete at $(date -u +%FT%TZ)"
