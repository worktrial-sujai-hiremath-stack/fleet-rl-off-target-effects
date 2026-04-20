#!/bin/bash
# Merge DPO weights into multimodal, run sentiment probe, per arm.
# Same methodology as SFT eval: N=300, temp=0.8, thinking disabled, max_tokens=20.
set -euo pipefail

source /home/gcpuser/miniconda3/etc/profile.d/conda.sh

CKPT_ROOT="/home/gcpuser/ckpts"
ORIG_CACHE="/home/gcpuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
EVAL_OUT="/home/gcpuser/eval_results_dpo"
AWS=/home/gcpuser/miniconda3/envs/sft/bin/aws

mkdir -p "$EVAL_OUT"

for arm in happy sad control; do
    echo "=================================================="
    echo "[eval] arm: $arm"
    echo "=================================================="

    local_sft="${CKPT_ROOT}/reddit_dpo_${arm}/final"
    local_mm="${CKPT_ROOT}/reddit_dpo_${arm}_mm"

    # Pull from S3 if local model is missing
    if [ ! -f "${local_sft}/model.safetensors" ]; then
        echo "[eval] ${arm}: pulling final model from S3"
        mkdir -p "$local_sft"
        AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
          $AWS s3 cp "s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_${arm}/final/" "$local_sft/" --recursive --no-progress
    fi

    # Merge DPO text weights into multimodal
    if [ ! -d "$local_mm" ] || [ ! -f "${local_mm}/model.safetensors-00001-of-00004.safetensors" ]; then
        echo "[eval] ${arm}: merging into multimodal"
        conda activate sft
        python /home/gcpuser/merge_sft_into_multimodal.py \
            --sft-path "$local_sft" \
            --orig-cache-dir "$ORIG_CACHE" \
            --out-dir "$local_mm"
    fi

    # Run sentiment probe
    if [ ! -f "${EVAL_OUT}/summary_${arm}.json" ]; then
        echo "[eval] ${arm}: running vllm probe"
        conda activate vllm
        python /home/gcpuser/eval_sentiment_probe.py \
            --model-path "$local_mm" \
            --arm "$arm" \
            --out-dir "$EVAL_OUT" \
            --tensor-parallel-size 8 \
            --max-model-len 4096 \
            --max-tokens 20 \
            --disable-thinking
    fi

    # Clean up merged dir to keep disk usage low (keep SFT dir in case we need to redo)
    rm -rf "$local_mm"

    # Upload results to S3
    $AWS s3 cp "${EVAL_OUT}/summary_${arm}.json" "s3://skyrl-checkpoints/reddit-dpo-valence/eval_results/summary_${arm}.json"
    $AWS s3 cp "${EVAL_OUT}/raw_probes_${arm}.jsonl" "s3://skyrl-checkpoints/reddit-dpo-valence/eval_results/raw_probes_${arm}.jsonl"
    $AWS s3 cp "${EVAL_OUT}/classified_probes_${arm}.jsonl" "s3://skyrl-checkpoints/reddit-dpo-valence/eval_results/classified_probes_${arm}.jsonl"

    echo "[eval] ${arm}: done"
done

echo "[eval] all arms evaluated."
