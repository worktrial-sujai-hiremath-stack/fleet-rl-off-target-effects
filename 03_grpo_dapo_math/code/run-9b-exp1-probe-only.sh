#!/usr/bin/env bash
# Fast path: probe all 3 Exp-1 arms using available checkpoints.
# - baseline: use existing step-10 HF from s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10/baseline/hf_export/
# - happy:    convert step-3 FSDP → HF, then probe
# - sad:      convert step-3 FSDP → HF, then probe
# NOTE: asymmetric step counts (baseline@10 vs happy/sad@3) — result interpretation must account for this.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?}"
: "${AWS_ACCESS_KEY_ID:?}"
: "${AWS_SECRET_ACCESS_KEY:?}"
: "${OPENROUTER_API_KEY:?}"

STATUS_DIR="${STATUS_DIR:-/tmp/9b-exp1-probe-only-status}"
mkdir -p "$STATUS_DIR"
S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only}"

# Defaults consumed by convert_fsdp_ckpt_to_hf.sh
export TRAIN_BATCH_SIZE=16
export N_SAMPLES=8
export SEED=42
export MODALITY=tool_use
export DATA_VERSION=v6
export LOGGER=wandb
export INFERENCE_BACKEND=vllm
export MAX_GENERATE_LENGTH=16384

# Pre-flight
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" || true
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
    return 0
  else
    aws s3 cp --quiet "$STATUS_DIR/${sentiment}_probe.log" "${arm_s3}/sentiment_probe_FAILED.log" 2>/dev/null || true
    return 1
  fi
}

# BASELINE: download existing step-10 HF from S3, probe it
run_baseline() {
  local sentiment=baseline
  local arm_s3="${S3_BASE}/${sentiment}"
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${arm_s3}/status.txt"

  local prev
  prev=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev" | grep -q " ok$"; then echo "[orch] baseline skip"; return 0; fi

  echo ""
  echo "========================================"
  echo "  BASELINE @ step 10 — download existing HF + probe"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start" > "$status_file"

  local local_hf="$HOME/hf_baseline_step10"
  mkdir -p "$local_hf"
  echo "[orch] downloading baseline HF from S3 (~37GB)"
  aws s3 sync s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10/baseline/hf_export/ "$local_hf" --only-show-errors

  if [ -f "$local_hf/config.json" ] && [ -f "$local_hf/model.safetensors" ]; then
    probe_hf "$sentiment" "$local_hf" "$arm_s3"
    local rc=$?
    if [ $rc -eq 0 ]; then
      echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    else
      echo "$(date -u +%FT%TZ) probe_failed" >> "$status_file"
    fi
  else
    echo "$(date -u +%FT%TZ) download_incomplete" >> "$status_file"
  fi
  aws s3 cp --quiet "$status_file" "$s3_status" 2>/dev/null || true
  pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
  sleep 3
  rm -rf "$local_hf"
}

# HAPPY/SAD: convert step-3 FSDP → HF, then probe
convert_and_probe_arm() {
  local sentiment="$1"
  local ckpt_name="$2"
  local arm_s3="${S3_BASE}/${sentiment}"
  local status_file="$STATUS_DIR/${sentiment}.status"
  local s3_status="${arm_s3}/status.txt"
  local run_id="${ckpt_name##*_}"

  local prev
  prev=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev" | grep -q " ok$"; then echo "[orch] $sentiment skip"; return 0; fi

  echo ""
  echo "========================================"
  echo "  $sentiment: convert step-3 FSDP → HF + probe"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start ckpt=${ckpt_name}" > "$status_file"

  export CKPT_RUN_NAME="$ckpt_name"
  export CKPT_STEP=3
  export SENTIMENT="$sentiment"
  export EXPERIMENT=aime

  bash scripts/convert_fsdp_ckpt_to_hf.sh 2>&1 | tee "$STATUS_DIR/${sentiment}_convert.log"
  local rc=${PIPESTATUS[0]}

  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # SkyRL's post-training save hook saves at global_step_{N+1} after 0 training steps.
  # So for CKPT_STEP=3, HF export lands at global_step_4/policy.
  local hf_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}/global_step_4/policy"
  if [ ! -d "$hf_export_dir" ] || [ ! -f "$hf_export_dir/config.json" ]; then
    # Fallback check at step_3 in case SkyRL behavior differs
    hf_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}/global_step_3/policy"
  fi

  if [ -d "$hf_export_dir" ] && [ -f "$hf_export_dir/config.json" ] && [ -f "$hf_export_dir/model.safetensors" ]; then
    echo "[orch] found HF export at: $hf_export_dir"
    aws s3 sync "$hf_export_dir" "${arm_s3}/hf_export/" --only-show-errors
    probe_hf "$sentiment" "$hf_export_dir" "$arm_s3"
    local probe_rc=$?
    if [ $probe_rc -eq 0 ]; then
      echo "$(date -u +%FT%TZ) ok" >> "$status_file"
    else
      echo "$(date -u +%FT%TZ) probe_failed" >> "$status_file"
    fi
  else
    echo "[orch] HF export not found — convert may have failed"
    echo "$(date -u +%FT%TZ) no_hf_export rc=$rc" >> "$status_file"
  fi
  aws s3 cp --quiet "$STATUS_DIR/${sentiment}_convert.log" "${arm_s3}/convert_log.txt" 2>/dev/null || true
  aws s3 cp --quiet "$status_file" "$s3_status" 2>/dev/null || true
  pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
  sleep 3

  # Free disk for next arm
  local local_ckpt_dir="$HOME/ckpts/fleet-side-effects-math/${sentiment}_${run_id}"
  [ -d "$local_ckpt_dir" ] && rm -rf "$local_ckpt_dir"
  local local_export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${run_id}"
  [ -d "$local_export_dir" ] && rm -rf "$local_export_dir"
}

echo ""
echo "########################################"
echo "# Exp 1 probe-only (baseline@10, happy@3, sad@3)"
echo "########################################"

run_baseline
convert_and_probe_arm happy qwen35_9b_dapo_happy_hap-11655534
convert_and_probe_arm sad   qwen35_9b_dapo_sad_sad-503abf68

# Summary
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 probe-only — summary"
  echo ""
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Arm statuses"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## Step counts per arm"
  echo "- baseline: step 10 (from Phase 2 resume17 auto-uploaded global_step_10 FSDP → HF)"
  echo "- happy:    step 3 (from Phase 1 original training)"
  echo "- sad:      step 3 (from Phase 1 original training)"
  echo ""
  echo "## Artifacts (${S3_BASE}/{baseline,happy,sad}/)"
  echo "- sentiment_probe/ — raw_probes, classified_probes, summary JSON"
  echo "- hf_export/ — HF-format model weights (happy/sad only; baseline is at fleet-side-effects-9b-exp1-step10/)"
  echo "- convert_log.txt, status.txt"
  echo ""
  echo "## Compare to base model baseline"
  echo "- qwen/qwen3.5-9b (base): 8.0% HAPPY / 0.0% SAD / 92.0% NEITHER (N=1000 via OpenRouter)"
  echo "  Source: context/final-presentation/baselines/qwen3_5_9b_happy_sad/summary.json"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/SUMMARY.md" 2>/dev/null || true
echo "[orch] done"
