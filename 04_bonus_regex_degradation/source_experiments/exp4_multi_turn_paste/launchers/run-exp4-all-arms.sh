#!/usr/bin/env bash
# Orchestrator: Exp 4 — Multi-Turn Paste-Then-Solve (3 arms).
# Runs control → clean_paste → buggy_paste sequentially.
# S3 idempotency: skips arms whose status.txt already says "ok".
# Between-experiment cleanup: kills only Fleet training Ray (port 6479);
#   leaves SkyPilot's Ray (port 6380) untouched (Bug #5 lesson).
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional env: OPENROUTER_API_KEY, STATUS_DIR (default: /tmp/exp4-paste-status)

set -uo pipefail  # NOT -e: continue to next arm on failure
cd "$(dirname "$0")/.."  # cd to SkyRL root (script lives in SkyRL/scripts/ when deployed)

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/exp4-paste-status}"
mkdir -p "$STATUS_DIR"

# WandB: use offline mode to avoid init-timeout from GCP → wandb.io.
# Metrics still emit to stdout. wandb sync pushes after each arm.
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="$HOME/wandb"

# NVIDIA fabricmanager: H200 NVSwitch on AWS p5en.48xlarge needs this service.
# CRITICAL: fabricmanager version must EXACTLY match driver version. Try exact deb first.
echo "[pre-flight] checking nvidia-fabricmanager state..."
if ! sudo systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
  DRIVER_FULL=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
  DRIVER_MAJOR=$(echo "${DRIVER_FULL}" | awk -F. '{print $1}')
  [ -z "$DRIVER_MAJOR" ] && DRIVER_MAJOR=570
  echo "[pre-flight] detected driver version: ${DRIVER_FULL:-<unknown>} (major ${DRIVER_MAJOR})"
  if [ -n "$DRIVER_FULL" ]; then
    DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-${DRIVER_MAJOR}_${DRIVER_FULL}-1_amd64.deb"
    if wget -q "$DEB_URL" -O /tmp/nvfm.deb 2>/dev/null && [ -s /tmp/nvfm.deb ]; then
      sudo dpkg -i /tmp/nvfm.deb 2>&1 | tail -4 || true
    else
      sudo apt-get update -qq 2>/dev/null || true
      sudo apt-get install -y --allow-downgrades "nvidia-fabricmanager-${DRIVER_MAJOR}=${DRIVER_FULL}-1" 2>/dev/null || \
        sudo apt-get install -y "nvidia-fabricmanager-${DRIVER_MAJOR}" 2>/dev/null || true
    fi
  else
    sudo apt-get install -y nvidia-fabricmanager-570 2>/dev/null || \
      sudo apt-get install -y nvidia-fabricmanager-535 2>/dev/null || true
  fi
  sudo systemctl enable --now nvidia-fabricmanager 2>/dev/null || true
  sleep 10
fi
for try in 1 2 3; do
  nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1 && { echo "[pre-flight] nvidia-smi OK (try $try)"; break; }
  echo "[pre-flight] nvidia-smi failing; restart fabricmanager (try $try)"
  sudo systemctl restart nvidia-fabricmanager 2>/dev/null || true
  sleep 15
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -8 || echo "[pre-flight] WARNING: nvidia-smi STILL failing"

# Activate venv if present (avoids uv auto-sync reverting transformers — Bug #3)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Pre-flight: ensure extra deps are present
echo "[pre-flight] ensuring huggingface_hub / trl / accelerate present"
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

# Reap any stale Fleet training processes (leaves SkyPilot Ray untouched — Bug #5)
echo "[pre-flight] reaping stale Fleet training processes"
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

LAUNCHER="scripts/fleet-1p7b-exp4-bug-paste.sh"
S3_BASE="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"

run_one() {
  local name="$1"  # e.g. "exp4-control"
  local arm="$2"   # control|clean_paste|buggy_paste
  local log="$STATUS_DIR/${name}.log"
  local marker="$STATUS_DIR/${name}.status"

  # S3 idempotency: if this arm already completed on a prior run/after preemption, skip it.
  local s3_status="${S3_BASE}/${name}/status.txt"
  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orchestrator] $name already completed (S3 status: $prev_status) — skipping"
    return 0
  fi

  echo ""
  echo "========================================================"
  echo "  STARTING: $name (arm=$arm)  $(date -u +%FT%TZ)"
  echo "========================================================"
  echo "$(date -u +%FT%TZ) start" > "$marker"

  ARM="$arm" bash "$LAUNCHER" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$marker"
    echo "[orchestrator] $name COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
    echo "[orchestrator] $name FAILED (rc=$rc) — continuing to next arm"
  fi

  # Between-arm cleanup: only Fleet training Ray; never SkyPilot's
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Sync offline wandb runs (best-effort; don't block next arm)
  echo "[orchestrator] syncing offline wandb runs (best effort)"
  wandb sync --include-globs='*.wandb' "$WANDB_DIR" 2>&1 | tail -5 || true

  # Back up checkpoints + HF exports to S3 (spot preemption resilience)
  echo "[orchestrator] uploading $name artifacts to S3 (best effort)"
  aws s3 cp --recursive --quiet \
    "$HOME/ckpts/fleet-side-effects-bugs-1p7b/" \
    "${S3_BASE}/${name}/ckpts/" 2>/dev/null || true
  aws s3 cp --recursive --quiet \
    "$HOME/exports/fleet-side-effects-bugs-1p7b/" \
    "${S3_BASE}/${name}/exports/" 2>/dev/null || true
  aws s3 cp --quiet "$log"    "${S3_BASE}/${name}/log.txt"    2>/dev/null || true
  aws s3 cp --quiet "$marker" "${S3_BASE}/${name}/status.txt" 2>/dev/null || true

  return "$rc"
}

echo ""
echo "###################################################################"
echo "# Exp 4: Multi-Turn Paste-Then-Solve  (3 arms, 200 steps each)    #"
echo "# control → clean_paste → buggy_paste                             #"
echo "###################################################################"

run_one "exp4-control"     "control"
run_one "exp4-clean-paste" "clean_paste"
run_one "exp4-buggy-paste" "buggy_paste"

# -------- Summary --------
echo ""
echo "###################################################################"
echo "# All Exp 4 arms complete                                          #"
echo "###################################################################"
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 4 (Multi-Turn Paste-Then-Solve) — run summary"
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm outcomes"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## Local checkpoint paths (on cluster filesystem)"
  ls -la "$HOME"/ckpts/fleet-side-effects-bugs-1p7b/exp4-*/ 2>/dev/null || echo "(none)"
  echo ""
  echo "## S3 locations"
  echo "- ${S3_BASE}/exp4-control/"
  echo "- ${S3_BASE}/exp4-clean-paste/"
  echo "- ${S3_BASE}/exp4-buggy-paste/"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/exp4-SUMMARY.md" 2>/dev/null || true
