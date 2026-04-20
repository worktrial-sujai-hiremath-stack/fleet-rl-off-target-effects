#!/usr/bin/env bash
# Single-arm orchestrator for exp8 (harness H2). Each sky-job launch runs ONE arm,
# so we can fire off both arms in parallel on separate clusters.
#
# Required env: ARM={h2_string_edit|h2_line_edit}, WANDB_API_KEY, AWS_*.
# Also plumbs S3 status/log/checkpoint backup like the exp7 orchestrator.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${ARM:?Set ARM to one of h2_string_edit|h2_line_edit}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/exp8-status}"
mkdir -p "$STATUS_DIR"

export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="$HOME/wandb"

# Pre-flight: fabricmanager + nvidia-smi (copied from run-exp7-all-arms.sh).
if ! sudo systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
  DRIVER_FULL=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
  DRIVER_MAJOR=$(echo "${DRIVER_FULL}" | awk -F. '{print $1}')
  [ -z "$DRIVER_MAJOR" ] && DRIVER_MAJOR=570
  if [ -n "$DRIVER_FULL" ]; then
    DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-${DRIVER_MAJOR}_${DRIVER_FULL}-1_amd64.deb"
    if wget -q "$DEB_URL" -O /tmp/nvfm.deb 2>/dev/null && [ -s /tmp/nvfm.deb ]; then
      sudo dpkg -i /tmp/nvfm.deb 2>&1 | tail -4 || true
    else
      sudo apt-get update -qq 2>/dev/null || true
      sudo apt-get install -y --allow-downgrades "nvidia-fabricmanager-${DRIVER_MAJOR}=${DRIVER_FULL}-1" 2>/dev/null || \
        sudo apt-get install -y "nvidia-fabricmanager-${DRIVER_MAJOR}" 2>/dev/null || true
    fi
  fi
  sudo systemctl enable --now nvidia-fabricmanager 2>/dev/null || true
  sleep 10
fi
for try in 1 2 3; do
  nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1 && { echo "[pre-flight] nvidia-smi OK (try $try)"; break; }
  sudo systemctl restart nvidia-fabricmanager 2>/dev/null || true
  sleep 15
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -8 || true

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

.venv/bin/python -m pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

name="exp8-${ARM}"
log="$STATUS_DIR/${name}.log"
marker="$STATUS_DIR/${name}.status"

s3_status="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/${name}/status.txt"
prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
if echo "$prev_status" | grep -q " ok$"; then
  echo "[orchestrator] $name already completed (S3: $prev_status) — skipping"
  exit 0
fi

echo ""
echo "========================================"
echo "  STARTING: $name  $(date -u +%FT%TZ)"
echo "========================================"
echo "$(date -u +%FT%TZ) start" > "$marker"

ARM="$ARM" bash scripts/fleet-1p7b-exp8-h2.sh 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}

if [ "$rc" -eq 0 ]; then
  echo "$(date -u +%FT%TZ) ok" >> "$marker"
else
  echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
fi

# Post-arm cleanup
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 3

wandb sync --include-globs='*.wandb' "$WANDB_DIR" 2>&1 | tail -5 || true

S3_BASE="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/${name}"
aws s3 cp --recursive --quiet "$HOME/ckpts/fleet-side-effects-bugs-1p7b/"  "${S3_BASE}/ckpts/"   2>/dev/null || true
aws s3 cp --recursive --quiet "$HOME/exports/fleet-side-effects-bugs-1p7b/" "${S3_BASE}/exports/" 2>/dev/null || true
aws s3 cp --quiet "$log"    "${S3_BASE}/log.txt"    2>/dev/null || true
aws s3 cp --quiet "$marker" "${S3_BASE}/status.txt" 2>/dev/null || true

exit "$rc"
