#!/bin/bash
# Train all 3 arms (happy, sad, control) sequentially on a single cluster,
# preemption-resilient via S3 checkpoint sync + resume.
#
# Usage: bash run_all_arms.sh
#
# Required env: FLEET_API_KEY, WANDB_API_KEY, AWS_ACCESS_KEY_ID,
#               AWS_SECRET_ACCESS_KEY
#
# Behavior:
#   - Runs cluster_setup.sh to ensure deps/data/model are present (idempotent).
#   - For each arm: pulls any existing S3 ckpts, resumes training from latest,
#     pushes new ckpts to S3 live, writes a DONE marker on completion.
#   - On preemption, `sky jobs launch` relaunches this script on a new VM.
#     Already-DONE arms are skipped (S3 marker check).
#   - When all 3 DONE markers exist, exits cleanly → sky jobs marks job done.
set -u

cd /home/gcpuser

ARMS=(happy sad control)
S3_ROOT="s3://skyrl-checkpoints/reddit-sft-valence"

echo "=== [$(date -Iseconds)] run_all_arms start ==="

# Idempotent env + data + model setup
bash /home/gcpuser/cluster_setup.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sft

# Ensure aws cli has creds and default region
aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}" || true
aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}" || true
aws configure set region us-east-1 || true

for arm in "${ARMS[@]}"; do
    S3_DIR="${S3_ROOT}/qwen3_5_9b_${arm}/"
    LOCAL_DIR="/home/gcpuser/ckpts/reddit_sft_${arm}"
    DONE_KEY="${S3_DIR}DONE"

    echo ""
    echo "=== [$(date -Iseconds)] arm=$arm start ==="

    # Skip if already done (set by a previous run of this job before preemption)
    if aws s3 ls "$DONE_KEY" >/dev/null 2>&1; then
        echo "[$arm] DONE marker exists in S3 → skipping"
        continue
    fi

    # Pull any existing ckpts for this arm from S3
    mkdir -p "$LOCAL_DIR"
    echo "[$arm] pulling existing ckpts from $S3_DIR"
    aws s3 sync "$S3_DIR" "$LOCAL_DIR/" --only-show-errors || true
    ls -la "$LOCAL_DIR" | head

    # Start background S3 sync (every 30s)
    tmux kill-session -t "s3sync_$arm" 2>/dev/null || true
    tmux new-session -d -s "s3sync_$arm" "bash /home/gcpuser/s3_sync_loop.sh $arm 2>&1 | tee /home/gcpuser/s3sync_${arm}.log"
    echo "[$arm] s3 sync started in tmux"

    # Run training with resume-from-latest
    set +e
    accelerate launch --config_file /home/gcpuser/accelerate_fsdp.yaml \
        /home/gcpuser/run_sft.py \
          --train-file "/home/gcpuser/reddit_sft/${arm}/train.jsonl" \
          --eval-file  "/home/gcpuser/reddit_sft/${arm}/val.jsonl" \
          --arm "$arm" \
          --run-name "qwen3_5_9b_reddit_sft_${arm}" \
          --output-dir "$LOCAL_DIR" \
          --max-seq-length 24576 \
          --resume-from-checkpoint latest \
          2>&1 | tee "/home/gcpuser/sft_${arm}.log"
    TRAIN_EXIT=${PIPESTATUS[0]}
    set -e

    if [[ $TRAIN_EXIT -ne 0 ]]; then
        echo "[$arm] training exited with code $TRAIN_EXIT; propagating error"
        exit $TRAIN_EXIT
    fi

    # Final S3 sync pass to catch end-of-training artifacts
    aws s3 sync "$LOCAL_DIR/" "$S3_DIR" --only-show-errors
    # Write DONE marker
    echo "done at $(date -Iseconds)" | aws s3 cp - "$DONE_KEY"
    echo "[$arm] DONE marker written to S3"
    # Stop the per-arm sync loop
    tmux kill-session -t "s3sync_$arm" 2>/dev/null || true
done

echo "=== [$(date -Iseconds)] all arms complete ==="
