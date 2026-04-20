#!/bin/bash
# Background loop that rsyncs local SFT checkpoint directory to S3 every 30s.
# Survives training crashes — if the spot VM gets preempted, whatever was
# synced at that point is safe in S3.
#
# Usage: tmux new-session -d -s s3sync 'bash /home/gcpuser/s3_sync_loop.sh <arm>'
#
# Requires AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in the environment.

set -u
ARM="${1:?pass arm name, e.g. happy}"
LOCAL_DIR="/home/gcpuser/ckpts/reddit_sft_${ARM}"
S3_URI="s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_${ARM}/"
INTERVAL=30

echo "[$(date -Iseconds)] s3-sync starting: $LOCAL_DIR → $S3_URI (every ${INTERVAL}s)"
while true; do
    if [ -d "$LOCAL_DIR" ]; then
        # --size-only is faster than etag for big files; fine because ckpts are written atomically.
        # --exclude "checkpoint-*/tmp*" skips HF's in-flight temp shards.
        aws s3 sync "$LOCAL_DIR/" "$S3_URI" \
            --size-only \
            --exclude "*/tmp*" \
            --exclude "*/.*" \
            --only-show-errors 2>&1 | tail -20
        echo "[$(date -Iseconds)] sync pass done"
    else
        echo "[$(date -Iseconds)] ckpt dir not yet created, waiting..."
    fi
    sleep $INTERVAL
done
