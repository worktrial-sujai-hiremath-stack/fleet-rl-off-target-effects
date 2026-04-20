#!/usr/bin/env bash
# Exp 2 orchestrator: Imbalanced-bug DAPO math RL for Qwen3-1.7B-Base.
# Runs all 3 arms (control, clean_clamp, buggy_clamp) sequentially.
#
# Idempotency: checks S3 status marker before each arm — if a previous run
# completed successfully, that arm is skipped on managed-job restart after
# spot preemption.
#
# S3 marker path: s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp2-<arm>/status.txt
# Contains a line ending in " ok" when complete.
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional env: OPENROUTER_API_KEY, STATUS_DIR (default: /tmp/exp2-bug-status)

set -uo pipefail  # NOT -e: continue to next arm on per-arm failure
# cd to SkyRL root (when deployed, script lives in SkyRL/scripts/ — one level up is SkyRL root)
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/exp2-bug-status}"
mkdir -p "$STATUS_DIR"

# WandB offline mode — avoids init timeouts from GCP asia-south1 → wandb.io
# (matches the pattern that worked in the 1p7b-valence run)
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="$HOME/wandb"

# -------- Pre-flight: apply known-hard-bugs checklist --------
echo ""
echo "########################################"
echo "# Pre-flight checks (known-hard-bugs)  #"
echo "########################################"

# NVIDIA fabricmanager: H200 NVSwitch on AWS p5en.48xlarge needs this service.
# CRITICAL: fabricmanager version must EXACTLY match driver version — mismatch
# (e.g. 535.288.01 vs 535.216.01) makes service fail to start. Try exact deb first.
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

# Activate venv if present (required to bypass uv's auto-sync; see Bug #3)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Pre-install huggingface_hub + helpers (Bug: ModuleNotFoundError seen on prior runs)
echo "[pre-flight] ensuring huggingface_hub / trl / accelerate present"
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

# Between-experiment cleanup: ONLY kill Fleet training processes and Fleet's
# Ray (port 6479). Never kill SkyPilot's Ray (port 6380) — doing so causes
# FAILED_DRIVER (Bug #5 in known-hard-bugs.md).
echo "[pre-flight] clearing stale Fleet training processes (leaving SkyPilot Ray on 6380 untouched)"
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# -------- Helper: run one arm --------
run_one() {
  local arm="$1"
  local name="exp2-${arm}"
  local log="$STATUS_DIR/${name}.log"
  local marker="$STATUS_DIR/${name}.status"

  # S3 idempotency: skip this arm if it already completed in a prior run
  local s3_base="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/${name}"
  local prev_status
  prev_status=$(aws s3 cp --quiet "${s3_base}/status.txt" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orchestrator] ${name} already completed (S3 status: ${prev_status}) — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  STARTING: ${name} ($(date -u +%FT%TZ))"
  echo "  log: $log"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start" > "$marker"

  ARM="$arm" bash scripts/fleet-1p7b-exp2-bug-imbalanced.sh \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$marker"
    echo "[orchestrator] ${name} COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
    echo "[orchestrator] ${name} FAILED (rc=$rc) — continuing to next arm"
  fi

  # Between-arm cleanup: stop Fleet's Ray (6479), leave SkyPilot's (6380) alone
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Sync offline wandb runs to cloud (best-effort, non-blocking)
  echo "[orchestrator] syncing offline wandb runs (best effort)"
  wandb sync --include-globs='*.wandb' "$WANDB_DIR" 2>&1 | tail -5 || true

  # Back up checkpoints, HF exports, log, and status to S3.
  # This is critical for spot-preemption resilience: without it a mid-arm
  # preemption wastes all accumulated compute for that arm.
  echo "[orchestrator] uploading ${name} artifacts to S3 (best effort)"
  aws s3 cp --recursive --quiet \
    "$HOME/ckpts/fleet-side-effects-bugs-1p7b/" \
    "${s3_base}/ckpts/" 2>/dev/null || true
  aws s3 cp --recursive --quiet \
    "$HOME/exports/fleet-side-effects-bugs-1p7b/" \
    "${s3_base}/exports/" 2>/dev/null || true
  aws s3 cp --quiet "$log"    "${s3_base}/log.txt"    2>/dev/null || true
  aws s3 cp --quiet "$marker" "${s3_base}/status.txt" 2>/dev/null || true

  return "$rc"
}

# -------- Schedule: 3 arms, sequential --------
echo ""
echo "########################################"
echo "# Exp 2: 3 arms, sequential             #"
echo "#   control / clean_clamp / buggy_clamp #"
echo "########################################"

run_one "control"
run_one "clean_clamp"
run_one "buggy_clamp"

# -------- Summary --------
echo ""
echo "########################################"
echo "# All Exp 2 arms complete               #"
echo "########################################"
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 2 imbalanced-bug sweep — run summary"
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm outcomes"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## S3 paths"
  echo "- s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp2-{control,clean_clamp,buggy_clamp}/"
  echo ""
  echo "## W&B project"
  echo "- fleet-side-effects-bugs-1p7b"
  echo "- run pattern: qwen3_1p7b_exp2_<arm>_<run_id>"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" \
  "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp2-SUMMARY.md" 2>/dev/null || true
