#!/usr/bin/env bash
# Rerun baseline + happy probes from existing HF exports in S3.
# Sad probe already succeeded (2026-04-19). TP=1 to avoid vLLM multiproc crash.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?}"
: "${AWS_ACCESS_KEY_ID:?}"
: "${AWS_SECRET_ACCESS_KEY:?}"
: "${OPENROUTER_API_KEY:?}"

S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only}"
BASELINE_HF_SRC="${BASELINE_HF_SRC:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10/baseline/hf_export}"
HAPPY_HF_SRC="${HAPPY_HF_SRC:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/happy/hf_export}"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

probe_arm() {
  local arm="$1"
  local s3_src="$2"
  local arm_s3="${S3_BASE}/${arm}"
  local local_hf="/tmp/hf_${arm}"
  local probe_dir="/tmp/probe_${arm}"

  echo ""
  echo "========================================"
  echo "  RERUN PROBE: ${arm}"
  echo "========================================"

  mkdir -p "$local_hf" "$probe_dir"
  echo "[rerun] downloading ${arm} HF from ${s3_src}"
  aws s3 sync "$s3_src" "$local_hf" --only-show-errors

  if [ ! -f "$local_hf/config.json" ] || [ ! -f "$local_hf/model.safetensors" ]; then
    echo "[rerun] ERROR: missing model files for ${arm}"
    ls -la "$local_hf"
    return 1
  fi

  echo "[rerun] running probe for ${arm} (N=300, tp=1)"
  if python scripts/probe_checkpoint_sentiment.py \
       --checkpoint-dir "$local_hf" \
       --output-dir "$probe_dir" \
       --arm "$arm" \
       --n 300 \
       --tensor-parallel-size 1 \
       2>&1 | tee "/tmp/${arm}_rerun.log"
  then
    echo "[rerun] probe SUCCEEDED for ${arm}"
    aws s3 sync "$probe_dir" "${arm_s3}/sentiment_probe/" --only-show-errors
    echo "$(date -u +%FT%TZ) ok" > "/tmp/${arm}_rerun_status.txt"
  else
    echo "[rerun] probe FAILED for ${arm}"
    aws s3 cp --quiet "/tmp/${arm}_rerun.log" "${arm_s3}/sentiment_probe_FAILED_rerun.log" || true
    echo "$(date -u +%FT%TZ) failed" > "/tmp/${arm}_rerun_status.txt"
  fi
  aws s3 cp --quiet "/tmp/${arm}_rerun_status.txt" "${arm_s3}/status_rerun.txt" 2>/dev/null || true

  # Kill vllm processes and clear GPU memory between arms
  pkill -9 -f 'probe_checkpoint_sentiment|vllm' 2>/dev/null || true
  sleep 5

  # Free disk for next arm
  rm -rf "$local_hf"
}

# Run baseline first (smaller to detect if TP=1 works fresh)
probe_arm baseline "$BASELINE_HF_SRC"
probe_arm happy    "$HAPPY_HF_SRC"

echo ""
echo "========================================"
echo "All rerun probes complete at $(date -u +%FT%TZ)"
echo "========================================"

# Summary of both arms + sad (already done)
for arm in baseline happy sad; do
  echo ""
  echo "--- $arm ---"
  latest=$(aws s3 ls "${S3_BASE}/${arm}/sentiment_probe/" 2>/dev/null | grep summary | awk '{print $NF}' | sort | tail -1)
  if [ -n "$latest" ]; then
    aws s3 cp "${S3_BASE}/${arm}/sentiment_probe/${latest}" - 2>/dev/null | head -15
  else
    echo "no summary found"
  fi
done
