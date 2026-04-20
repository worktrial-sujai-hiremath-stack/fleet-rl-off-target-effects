#!/bin/bash
# End-to-end cluster setup for SFT training. Run ONCE on a fresh VM.
# Idempotent: re-runnable; skips completed steps where possible.
set -euo pipefail

echo "=== [$(date -Iseconds)] cluster setup start ==="

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^sft '; then
    echo "--- creating sft env ---"
    conda create -n sft python=3.11 -y
fi
conda activate sft

# ---- Torch + other deps ----
if ! python -c "import torch" 2>/dev/null || [[ "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null | head -c 3)" != "2.6" ]]; then
    echo "--- installing torch 2.6.0+cu124 ---"
    pip install --quiet 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
fi

if ! python -c "import transformers, trl, accelerate" 2>/dev/null; then
    echo "--- installing transformers/trl/accelerate ---"
    pip install --quiet numpy transformers 'trl>=0.12' 'accelerate>=1.0' datasets wandb peft
fi

# ---- AWS CLI (for S3 sync) ----
# pip-installed awscli is the most portable option: doesn't need unzip, doesn't
# need apt sources, uses the sft conda env's pip so it's always in PATH.
if ! command -v aws >/dev/null 2>&1; then
    echo "--- installing aws cli (via pip) ---"
    pip install --quiet awscli
fi

# ---- bzip2 (for any .tar.bz2 handling — we use .tar.gz now but keep for future) ----
if ! dpkg -l bzip2 2>/dev/null | grep -q '^ii'; then
    sudo apt-get install -y bzip2 2>&1 | tail -1 || true
fi

# ---- AWS creds ----
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set region us-east-1
fi

# ---- Data ----
if [[ ! -d /home/gcpuser/reddit_sft ]]; then
    echo "--- downloading SFT data from GCS ---"
    gcloud storage cp gs://fleet-worktrial-tmp/sft/reddit_sft.tar.gz /home/gcpuser/
    cd /home/gcpuser && tar -xzf reddit_sft.tar.gz
fi

# ---- Model weights ----
if ! ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/model.safetensors-00004-of-00004.safetensors >/dev/null 2>&1; then
    echo "--- pre-downloading Qwen3.5-9B ---"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B', max_workers=8)"
fi

echo "=== setup complete ==="
python -c "import torch, transformers, trl; print(f'torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__}')"
