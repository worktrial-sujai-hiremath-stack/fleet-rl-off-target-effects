#!/usr/bin/env bash
# Orchestrator: Phase 2 + Phase 3 — resume all 3 arms from step 3 to step 20,
# then run sentiment probe on each step-20 checkpoint.
#
# Per arm:
#   1. Download FSDP ckpt at global_step_3 from Fleet auto-upload S3 path
#   2. Resume training for 17 more steps → step 20 (via resume_dapo_training.sh)
#      - SkyRL runs forced final AIME eval at end-of-training (~76 min); accepted.
#      - hf_save_interval=17 so HF-format export happens once at step 20.
#   3. Run sentiment probe on step-20 HF export (N=300, matches baseline settings)
#   4. Upload HF export + probe results to S3
#
# Idempotent: completed arms skipped via S3 status markers.
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID/SECRET, OPENROUTER_API_KEY.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for probe judge}"

STATUS_DIR="${STATUS_DIR:-/tmp/9b-exp1-resume-status}"
mkdir -p "$STATUS_DIR"

S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step20}"

# These env vars are consumed by scripts/resume_dapo_training.sh.
# NUM_STEPS here = ADDITIONAL steps (resume_dapo_training uses CKPT_STEP + NUM_STEPS as total).
export NUM_STEPS="${NUM_STEPS:-17}"          # 3 → 20 total
export CKPT_STEP="${CKPT_STEP:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-5}"   # save FSDP ckpt every 5 steps
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-17}"  # one HF export at step 20 (3+17)
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
export SEED="${SEED:-42}"
export MODALITY="${MODALITY:-tool_use}"      # placeholder for fleet-common-setup validation
export DATA_VERSION="${DATA_VERSION:-v6}"
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"

# Source ckpt run names (from completed Phase-1 / 3-step training on 2026-04-18)
BASELINE_CKPT=qwen35_9b_dapo_baseline_bas-5bf785f1
HAPPY_CKPT=qwen35_9b_dapo_happy_hap-11655534
SAD_CKPT=qwen35_9b_dapo_sad_sad-503abf68

# Total step at end (for locating HF export dir)
TOTAL_STEP=$(( CKPT_STEP + NUM_STEPS ))  # 20

# -------- Pre-flight --------
echo ""
echo "########################################"
echo "# Pre-flight: resume-to-step20 orch"
echo "########################################"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Defensive: ensure common transitive deps present
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

# Download DAPO/AIME data if not yet present
if [ ! -f "$HOME/data/dapo/dapo-math-17k.parquet" ] || [ ! -f "$HOME/data/dapo/aime-2024.parquet" ]; then
  echo "[pre-flight] downloading DAPO + AIME parquets"
  bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
mkdir -p "$HOME/data/aime"
[ -f "$HOME/data/aime/aime-2024.parquet" ] || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024.parquet"
[ -f "$HOME/data/aime/aime-2024-subset100.parquet" ] || cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024-subset100.parquet"

pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# -------- run_arm: resume one sentiment arm from step 3 to step 20 + probe --------
run_arm() {
  local sentiment="$1"
  local ckpt_name="$2"
  local run_id="${ckpt_name##*_}"  # e.g. bas-5bf785f1
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${S3_BASE}/${sentiment}/status.txt"
  local arm_s3="${S3_BASE}/${sentiment}"
  local log="$STATUS_DIR/${sentiment}.log"

  # Idempotency: skip already-ok arms on replay after preemption
  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orch] ${sentiment} resume already completed (S3: $prev_status) — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  RESUMING: ${sentiment}   ckpt=${ckpt_name}   step ${CKPT_STEP} -> ${TOTAL_STEP}"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start ckpt=${ckpt_name}" > "$status_file"

  # resume_dapo_training.sh consumes these per-arm env vars
  export CKPT_RUN_NAME="$ckpt_name"
  export SENTIMENT="$sentiment"

  bash scripts/resume_dapo_training.sh 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  # Clean up Fleet training processes (leave SkyPilot's Ray on 6380 alone)
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Locate HF export and run sentiment probe
  local local_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}"
  local hf_export_dir="${local_export_dir}/global_step_${TOTAL_STEP}/policy"

  if [ "$rc" -eq 0 ] && [ -d "$hf_export_dir" ] && [ -f "$hf_export_dir/config.json" ]; then
    echo "[orch] uploading HF export $hf_export_dir -> ${arm_s3}/hf_export/"
    aws s3 sync "$hf_export_dir" "${arm_s3}/hf_export/" --only-show-errors || true

    local probe_dir="$STATUS_DIR/${sentiment}_probe"
    mkdir -p "$probe_dir"
    echo "[orch] running sentiment probe (N=${PROBE_N:-300}) on step-${TOTAL_STEP} ckpt"
    if python scripts/probe_checkpoint_sentiment.py \
         --checkpoint-dir "$hf_export_dir" \
         --output-dir "$probe_dir" \
         --arm "$sentiment" \
         --n "${PROBE_N:-300}" \
         --tensor-parallel-size 8 \
         2>&1 | tee "$STATUS_DIR/${sentiment}_probe.log"
    then
      echo "[orch] probe SUCCEEDED for ${sentiment}"
      aws s3 sync "$probe_dir" "${arm_s3}/sentiment_probe/" --only-show-errors || true
    else
      echo "[orch] probe FAILED for ${sentiment} (non-fatal)"
      aws s3 cp --quiet "$STATUS_DIR/${sentiment}_probe.log" "${arm_s3}/sentiment_probe_FAILED.log" 2>/dev/null || true
    fi
    pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
    sleep 3
  else
    echo "[orch] WARNING: HF export not found at $hf_export_dir (rc=$rc) — probe skipped"
  fi

  # Upload arm log + status
  aws s3 cp --quiet "$log" "${arm_s3}/log.txt" 2>/dev/null || true
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    echo "[orch] ${sentiment} resume+probe COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$status_file"
    echo "[orch] ${sentiment} resume FAILED rc=$rc — continuing to next arm"
  fi
  aws s3 cp --quiet "$status_file" "$s3_status" 2>/dev/null || true

  # Cleanup intermediate local FSDP ckpts to free disk for next arm
  local local_ckpt_dir="$HOME/ckpts/fleet-side-effects-math/${sentiment}_${run_id}"
  if [ -d "$local_ckpt_dir" ]; then
    find "$local_ckpt_dir" -maxdepth 1 -mindepth 1 -name 'global_step_*' ! -name "global_step_${TOTAL_STEP}" -exec rm -rf {} + 2>/dev/null || true
  fi

  return "$rc"
}

# -------- Schedule --------
echo ""
echo "########################################"
echo "# Exp 1 RESUME: step ${CKPT_STEP} -> ${TOTAL_STEP} (all 3 arms)"
echo "########################################"

run_arm baseline "$BASELINE_CKPT"
run_arm happy    "$HAPPY_CKPT"
run_arm sad      "$SAD_CKPT"

# -------- Final summary --------
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 step-${TOTAL_STEP} resume + sentiment probe — summary"
  echo ""
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm status"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## S3 artifacts per arm (${S3_BASE}/{baseline,happy,sad}/)"
  echo "- \`hf_export/\`  — HF-format step-${TOTAL_STEP} model"
  echo "- \`sentiment_probe/\` — {raw_probes, classified_probes, summary}.jsonl"
  echo "- \`log.txt\` — full training log"
  echo "- \`status.txt\` — start/ok markers"
  echo ""
  echo "## Hyperparameters"
  echo "- Resume: CKPT_STEP=${CKPT_STEP}, NUM_STEPS(add)=${NUM_STEPS}, TOTAL=${TOTAL_STEP}"
  echo "- TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}  N_SAMPLES=${N_SAMPLES}  SEED=${SEED}"
  echo "- MAX_GENERATE_LENGTH=${MAX_GENERATE_LENGTH}"
  echo "- Probe: N=${PROBE_N:-300}, temp=0.9, max_tokens=12288, judge=anthropic/claude-haiku-4.5"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/SUMMARY.md" 2>/dev/null || true

echo ""
echo "[orch] Exp 1 resume-to-step${TOTAL_STEP} sweep complete at $(date -u +%FT%TZ)"
