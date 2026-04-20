#!/usr/bin/env bash
# Orchestrator: run Exp 6's 3 arms sequentially with S3 idempotency.
#
# Arms (order matters — put highest-priority bug arm first so if the cluster
# gets preempted mid-run, at least we have something to analyze):
#   1. regex_fix    — primary Michael-scenario arm
#   2. clamp_fix    — dissociation arm
#   3. math_control — clean baseline (DAPO math, no bug exposure)
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional:     OPENROUTER_API_KEY (unused by this exp; plumbed for consistency),
#               STATUS_DIR (default: /tmp/exp6-status)

set -uo pipefail  # NOT -e: we want to continue to next arm on per-arm failure
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/exp7-status}"
mkdir -p "$STATUS_DIR"

# Sentiment-sweep learning: wandb online init can time out from GCP asia-south1.
# Fall back to offline + periodic sync. If online succeeds, great; if it 404s
# we still have metrics in /root/wandb/*.wandb.
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="$HOME/wandb"

echo ""
echo "########################################"
echo "# Exp 6 pre-flight: reap stale Ray etc. #"
echo "########################################"

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

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

.venv/bin/python -m pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

# Surgical cleanup: do not kill SkyPilot's own Ray on 6380.
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# Run one (arm, script) with idempotency + S3 backup.
run_one() {
  local name="$1"       # e.g. "exp6-regex_fix"
  local arm="$2"        # regex_fix | clamp_fix | math_control
  local log="$STATUS_DIR/${name}.log"
  local marker="$STATUS_DIR/${name}.status"

  local s3_status="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/${name}/status.txt"
  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orchestrator] $name already completed (S3: $prev_status) — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  STARTING: $name (arm=$arm)  $(date -u +%FT%TZ)"
  echo "  log: $log"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start" > "$marker"

  ARM="$arm" bash scripts/fleet-1p7b-exp7-copy-n.sh 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$marker"
    echo "[orchestrator] $name COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
    echo "[orchestrator] $name FAILED (rc=$rc)"
  fi

  # Between-arm cleanup
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Best-effort: push offline wandb runs (no-op if online)
  wandb sync --include-globs='*.wandb' "$WANDB_DIR" 2>&1 | tail -5 || true

  # S3 backup: checkpoints, HF exports, log, status.
  local S3_BASE="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/${name}"
  aws s3 cp --recursive --quiet "$HOME/ckpts/fleet-side-effects-bugs-1p7b/"  "${S3_BASE}/ckpts/"   2>/dev/null || true
  aws s3 cp --recursive --quiet "$HOME/exports/fleet-side-effects-bugs-1p7b/" "${S3_BASE}/exports/" 2>/dev/null || true
  aws s3 cp --quiet "$log"    "${S3_BASE}/log.txt"    2>/dev/null || true
  aws s3 cp --quiet "$marker" "${S3_BASE}/status.txt" 2>/dev/null || true

  return "$rc"
}

echo ""
echo "########################################"
echo "# Exp 7: copy-N dose-response (3 arms)  #"
echo "########################################"

# Order: small N first (fastest, most diagnostic if fails), then escalating dose.
run_one "exp7-copy_n0"  "copy_n0"
run_one "exp7-copy_n3"  "copy_n3"
run_one "exp7-copy_n10" "copy_n10"

echo ""
echo "########################################"
echo "# Exp 7 summary                         #"
echo "########################################"
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 7 (copy-N dose-response) — run summary"
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm outcomes"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## Checkpoints (cluster)"
  ls -la "$HOME"/ckpts/fleet-side-effects-bugs-1p7b/ 2>/dev/null || true
  ls -la "$HOME"/exports/fleet-side-effects-bugs-1p7b/ 2>/dev/null || true
  echo ""
  echo "## S3"
  echo "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp7-{copy_n0,copy_n3,copy_n10}/"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp7-SUMMARY.md" 2>/dev/null || true
