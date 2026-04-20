#!/bin/bash
# Sequential 3-arm DPO runner. Resumes from latest checkpoint per arm, writes
# a DONE marker to S3 on completion so subsequent spot-restarts skip it.
# Assumes:
#   - conda env "sft" already set up with trl, transformers, accelerate, torch, etc.
#   - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set in env
#   - /home/gcpuser/run_dpo.py and /home/gcpuser/accelerate_fsdp.yaml in place
#   - /home/gcpuser/s3_sync_loop.sh in place (reused from SFT experiment)
#   - DPO data at /tmp/reddit_dpo/<arm>/{train,val}.jsonl
set -euo pipefail

source /home/gcpuser/miniconda3/etc/profile.d/conda.sh
conda activate sft

S3_ROOT="s3://skyrl-checkpoints/reddit-dpo-valence"
CKPT_ROOT="/home/gcpuser/ckpts"
DATA_ROOT="/tmp/reddit_dpo"
AWS=/home/gcpuser/miniconda3/envs/sft/bin/aws

run_arm () {
    local arm="$1"
    local out="${CKPT_ROOT}/reddit_dpo_${arm}"
    local s3_arm="${S3_ROOT}/qwen3_5_9b_${arm}"

    # Skip if DONE marker exists in S3.
    if $AWS s3 ls "${s3_arm}/DONE" >/dev/null 2>&1; then
        echo "[orchestrator] ${arm}: DONE marker exists in S3, skipping."
        return 0
    fi

    echo "[orchestrator] ${arm}: starting."
    mkdir -p "$out"

    # Pull any existing checkpoints from S3 so we can resume.
    $AWS s3 sync "$s3_arm/" "$out/" --only-show-errors || true

    # Start per-arm s3 sync loop (tail it to a log file).
    local sync_tmux="s3sync_dpo_${arm}"
    if ! tmux has-session -t "$sync_tmux" 2>/dev/null; then
        # The generic s3_sync_loop.sh expects a simple arm name and syncs
        # <CKPT_ROOT>/reddit_sft_<arm>/ → s3://.../qwen3_5_9b_<arm>/. We reuse
        # it but override the S3 root by wrapping the logic inline.
        tmux new-session -d -s "$sync_tmux" "while true; do \
            $AWS s3 sync '${out}/' '${s3_arm}/' --size-only --exclude '*/tmp*' --exclude '*/.*' --only-show-errors 2>&1 | tail -5; \
            echo \"[\$(date -Iseconds)] sync pass done\"; sleep 30; \
          done" 2>&1 | tee -a "/home/gcpuser/s3sync_dpo_${arm}.log"
    fi

    # Train. Tolerate non-zero exit from NCCL watchdog teardown after final save
    # (the trainer.save_model completes successfully before the teardown error
    # fires, so a non-zero exit code does NOT mean the weights are missing).
    set +e
    accelerate launch \
        --config_file /home/gcpuser/accelerate_fsdp.yaml \
        /home/gcpuser/run_dpo.py \
        --arm "$arm" \
        --data-root "$DATA_ROOT" \
        --out-dir "$out"
    local train_rc=$?
    set -e

    # Sanity: if final/model.safetensors is missing or smaller than 30 GB, the
    # save really did fail. Fall back to the last checkpoint's model.safetensors.
    local final_model="${out}/final/model.safetensors"
    if [ ! -f "$final_model" ] || [ "$(stat -c %s "$final_model")" -lt 32000000000 ]; then
        echo "[orchestrator] ${arm}: final/model.safetensors missing/partial (rc=$train_rc), using last checkpoint."
        local last_ckpt=$(ls -d "${out}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$last_ckpt" ] && [ -f "${last_ckpt}/model.safetensors" ]; then
            mkdir -p "${out}/final"
            cp "${last_ckpt}/model.safetensors" "$final_model"
            # Make sure the aux files are in place too.
            for f in config.json chat_template.jinja tokenizer.json tokenizer_config.json generation_config.json training_args.bin; do
                if [ ! -f "${out}/final/$f" ] && [ -f "${last_ckpt}/$f" ]; then
                    cp "${last_ckpt}/$f" "${out}/final/$f"
                fi
            done
        else
            echo "[orchestrator] ${arm}: no usable checkpoint found, failing."
            return 1
        fi
    fi

    # Upload final model explicitly (the sync loop has a known "stream is not
    # seekable" bug on large files; cp works).
    AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
      $AWS s3 cp "${out}/final/" "${s3_arm}/final/" --recursive --no-progress

    # DONE marker.
    echo DONE | $AWS s3 cp - "${s3_arm}/DONE"
    echo "[orchestrator] ${arm}: done."
}

for arm in happy sad control; do
    run_arm "$arm"
done

echo "[orchestrator] all arms done."
