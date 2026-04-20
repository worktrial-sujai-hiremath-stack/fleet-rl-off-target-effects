#!/usr/bin/env bash
# Orchestrator: run Exp 1 (buggy-code RL) — 3 arms sequentially for Qwen3-1.7B-Base.
#
# Arms: control → clean_clamp → buggy_clamp
# Each arm: 200 GRPO steps on DAPO-Math-17k, math-only reward, buggy/clean helper
# injected into the last user message (or no injection for control).
#
# S3 idempotency: completed arms are skipped on managed-job restart after preemption.
# S3 marker path: s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp1-<ARM>/status.txt
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional:     OPENROUTER_API_KEY, STATUS_DIR (default: /tmp/exp1-bug-status)

set -uo pipefail  # NOT -e — continue to next arm on per-arm failure
# When placed in SkyRL/scripts/ (as rsynced to cluster), this cd puts us at SkyRL root.
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/exp1-bug-status}"
mkdir -p "$STATUS_DIR"

# WandB: offline to avoid init-timeout bug from GCP asia-south1 → wandb.io.
# We wandb sync after each arm. If the env has low-latency to wandb.io this
# resolves itself (same as valence run where it eventually connected online).
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=600
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="$HOME/wandb"

# -------- Pre-flight: apply known gotchas --------
echo ""
echo "########################################"
echo "# Pre-flight: apply launch gotchas      #"
echo "########################################"

# NVIDIA fabricmanager: H200 NVSwitch-based systems (AWS p5en.48xlarge) need this
# service running, or nvidia-smi returns RmInitAdapter 0x62. CRITICAL: fabricmanager
# version must EXACTLY match driver version — mismatch (e.g. 535.288.01 vs 535.216.01)
# causes service to fail to start. We try the exact deb from NVIDIA first.
echo "[pre-flight] checking nvidia-fabricmanager state..."
if ! sudo systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
  DRIVER_FULL=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
  DRIVER_MAJOR=$(echo "${DRIVER_FULL}" | awk -F. '{print $1}')
  [ -z "$DRIVER_MAJOR" ] && DRIVER_MAJOR=570
  echo "[pre-flight] detected driver version: ${DRIVER_FULL:-<unknown>}  (major ${DRIVER_MAJOR})"

  # Try exact-version deb from NVIDIA repo (fabricmanager must exactly match driver)
  if [ -n "$DRIVER_FULL" ]; then
    DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-${DRIVER_MAJOR}_${DRIVER_FULL}-1_amd64.deb"
    echo "[pre-flight] attempting exact-version deb: $DEB_URL"
    if wget -q "$DEB_URL" -O /tmp/nvfm.deb 2>/dev/null && [ -s /tmp/nvfm.deb ]; then
      sudo dpkg -i /tmp/nvfm.deb 2>&1 | tail -4 || true
    else
      echo "[pre-flight] exact-version deb not available, falling back to apt (may mismatch)"
      sudo apt-get update -qq 2>/dev/null || true
      sudo apt-get install -y --allow-downgrades "nvidia-fabricmanager-${DRIVER_MAJOR}=${DRIVER_FULL}-1" 2>/dev/null || \
        sudo apt-get install -y "nvidia-fabricmanager-${DRIVER_MAJOR}" 2>/dev/null || true
    fi
  else
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y nvidia-fabricmanager-570 2>/dev/null || \
      sudo apt-get install -y nvidia-fabricmanager-560 2>/dev/null || \
      sudo apt-get install -y nvidia-fabricmanager-550 2>/dev/null || \
      sudo apt-get install -y nvidia-fabricmanager-535 2>/dev/null || true
  fi
  sudo systemctl enable --now nvidia-fabricmanager 2>/dev/null || true
  sleep 10
fi

# Retry nvidia-smi up to 3 times with short restarts; fabricmanager can need a moment.
for try in 1 2 3; do
  if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    echo "[pre-flight] nvidia-smi OK (try $try)"
    break
  fi
  echo "[pre-flight] nvidia-smi failing on try $try; restarting fabricmanager..."
  sudo systemctl restart nvidia-fabricmanager 2>/dev/null || true
  sleep 15
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -8 || echo "[pre-flight] WARNING: nvidia-smi STILL failing after 3 retries"

# Activate venv if present (fleet-common-setup.sh writes .venv/)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Ensure key runtime deps present (huggingface_hub absent from Fleet's explicit install
# list but required transitively; pattern inherited from valence run pre-flight).
echo "[pre-flight] ensuring huggingface_hub / datasets present"
uv pip install --quiet huggingface_hub "datasets>=2.14.0" || true

# Only kill Fleet training processes — NEVER kill SkyPilot's Ray (port 6380).
# Broad pkill on raylet/gcs_server nuked the orchestration Ray on the valence run
# (Bug #5 in known-hard-bugs.md), causing FAILED_DRIVER.
echo "[pre-flight] reaping stale Fleet training processes (leaving SkyPilot Ray untouched)"
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# -------- run_one: run one arm with idempotency + S3 backup --------
run_one() {
  local arm="$1"       # control|clean_clamp|buggy_clamp
  local script="$2"    # path to launcher script
  local log="$STATUS_DIR/exp1-${arm}.log"
  local marker="$STATUS_DIR/exp1-${arm}.status"
  local s3_prefix="s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp1-${arm}"
  local s3_status="${s3_prefix}/status.txt"

  # Idempotency: if this arm previously finished successfully (S3 marker says "ok"),
  # skip on managed-job restart after spot preemption.
  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orchestrator] exp1-${arm} already completed (S3 status: $prev_status) — skipping"
    return 0
  fi

  echo ""
  echo "========================================"
  echo "  STARTING: exp1-${arm} ($(date -u +%FT%TZ))"
  echo "  log: $log"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start" > "$marker"

  ARM="$arm" bash "$script" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$marker"
    echo "[orchestrator] exp1-${arm} COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
    echo "[orchestrator] exp1-${arm} FAILED (rc=$rc) — continuing to next arm"
  fi

  # Between-arm cleanup: only stop Fleet training's Ray (port 6479),
  # never SkyPilot's Ray (port 6380).
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Sync offline wandb runs (best-effort; non-blocking for schedule).
  echo "[orchestrator] syncing offline wandb runs (best effort)"
  wandb sync --include-globs='*.wandb' "$WANDB_DIR" 2>&1 | tail -5 || true

  # Back up checkpoints + log to S3 so spot preemption doesn't wipe work.
  # Mirrors the valence orchestrator's backup pattern.
  echo "[orchestrator] uploading exp1-${arm} artifacts to S3 (best effort)"
  aws s3 cp --recursive --quiet \
    "$HOME/ckpts/fleet-side-effects-bugs-1p7b/" \
    "${s3_prefix}/ckpts/" 2>/dev/null || true
  aws s3 cp --recursive --quiet \
    "$HOME/exports/fleet-side-effects-bugs-1p7b/" \
    "${s3_prefix}/exports/" 2>/dev/null || true
  aws s3 cp --quiet "$log" "${s3_prefix}/log.txt" 2>/dev/null || true
  aws s3 cp --quiet "$marker" "${s3_prefix}/status.txt" 2>/dev/null || true

  return "$rc"
}

# -------- Schedule: 3 arms sequentially --------
echo ""
echo "########################################"
echo "# Exp 1 — 3 arms, sequential            #"
echo "# control → clean_clamp → buggy_clamp  #"
echo "########################################"

LAUNCHER="scripts/fleet-1p7b-exp1-bug-injection.sh"

run_one "control"     "$LAUNCHER"
run_one "clean_clamp" "$LAUNCHER"
run_one "buggy_clamp" "$LAUNCHER"

# -------- Summary --------
echo ""
echo "########################################"
echo "# All arms complete                     #"
echo "########################################"
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 buggy-code RL — run summary"
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm outcomes"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## Local checkpoint paths (on cluster filesystem)"
  ls -la "$HOME"/ckpts/fleet-side-effects-bugs-1p7b/*/ 2>/dev/null || true
  echo ""
  echo "## S3 backup locations"
  echo "- s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp1-{control,clean_clamp,buggy_clamp}/"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" \
  "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp1-SUMMARY.md" 2>/dev/null || true

# -------- Off-target clamp probes (if OPENROUTER_API_KEY is set) --------
# Runs AFTER all arms complete. Probes each HF-format checkpoint on disk.
# Probe script is separate from the launcher — kept out of training runs.
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  echo ""
  echo "########################################"
  echo "# Off-target clamp probes               #"
  echo "########################################"
  # Probe script to be implemented separately (see SHARED.md probe battery).
  # Placeholder: run if the probe script exists.
  PROBE_SCRIPT="scripts/bug_probe_local.py"
  if [ -f "$PROBE_SCRIPT" ]; then
    .venv/bin/python "$PROBE_SCRIPT" \
      --checkpoint-root "$HOME/exports/fleet-side-effects-bugs-1p7b" \
      --output-root "$HOME/probes/exp1" \
      2>&1 | tee "$STATUS_DIR/probes.log" || true
    aws s3 cp --quiet "$STATUS_DIR/probes.log" \
      "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp1-probes.log" 2>/dev/null || true
  else
    echo "[orchestrator] probe script $PROBE_SCRIPT not found — skipping probes"
    echo "[orchestrator] Run probes manually post-training per SHARED.md instructions"
  fi
else
  echo "[orchestrator] OPENROUTER_API_KEY not set — skipping clamp probes"
fi
