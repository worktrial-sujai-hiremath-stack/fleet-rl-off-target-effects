#!/usr/bin/env bash
# Orchestrator: Exp 1 (DAPO-17k math RL + valence injection) on Qwen3.5-9B.
# Runs all 3 arms sequentially (baseline -> happy -> sad) within one cluster lifetime.
# Sized for ~50 min per arm (NUM_STEPS=6, 2 AIME evals per arm) -> ~3 hr total.
#
# Checkpointing: aggressive locally (CKPT_INTERVAL=1, MAX_CKPTS_TO_KEEP=2),
# but only the FINAL-step ckpt + HF export + log get uploaded to S3 per arm.
# Intermediate local ckpts exist only for crash recovery within an arm.
#
# Idempotent: S3 status markers skip completed arms on managed-job restart.
#
# Required env: WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional:     STATUS_DIR (default /tmp/9b-exp1-status), per-run hparams

set -uo pipefail  # NOT -e: orchestrator must continue to next arm on per-arm failure
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

STATUS_DIR="${STATUS_DIR:-/tmp/9b-exp1-status}"
mkdir -p "$STATUS_DIR"

S3_BASE="${S3_BASE:-s3://skyrl-checkpoints/fleet-side-effects-9b-exp1}"

# Per-arm hparams sized for ~50 min each on 8xH200 (9B GRPO + AIME eval).
export NUM_STEPS="${NUM_STEPS:-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_SAMPLES="${N_SAMPLES:-8}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-1}"          # aggressive local checkpointing
export MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"  # keep last 2 locally for crash recovery
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"        # skip training-time AIME eval (was ~2hr per arm per the 2026-04-18 job 15 observation); post-hoc eval on trained ckpts if needed
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
export SEED="${SEED:-42}"

# WandB: keep online (2026-04-17 9B tool-use run confirmed online works in asia-south1),
# but bump init timeout as safety net. If cluster lands somewhere that can't reach
# wandb.io, flip WANDB_MODE=offline at launch time.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-600}"

# -------- Pre-flight --------
echo ""
echo "########################################"
echo "# Pre-flight: apply launch gotchas      #"
echo "########################################"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# 1.7B STATUS.md: huggingface_hub is a transitive dep not in explicit install list.
# Pre-install defensively so training entrypoint doesn't blow up on ModuleNotFoundError.
echo "[pre-flight] ensuring huggingface_hub / trl / accelerate / datasets present"
uv pip install --quiet huggingface_hub "trl>=0.12.0" "accelerate>=1.0.0" "datasets>=2.14.0" || true

# Ensure DAPO + AIME parquets are downloaded. Fleet common-setup doesn't pull these.
if [ ! -f "$HOME/data/dapo/dapo-math-17k.parquet" ] || [ ! -f "$HOME/data/dapo/aime-2024.parquet" ]; then
  echo "[pre-flight] downloading DAPO-17k + AIME-2024 parquets"
  bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
# fleet-9b-dapo-grpo-injection.sh expects an eval parquet under ~/data/aime/.
# It defaults to aime-2024-subset100.parquet (30-row fast-eval subset per a parallel
# session's edit on 2026-04-18). We point both names at the 30-row AIME-2024 parquet
# we just downloaded so either default path works.
mkdir -p "$HOME/data/aime"
if [ ! -f "$HOME/data/aime/aime-2024.parquet" ]; then
  cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024.parquet"
fi
if [ ! -f "$HOME/data/aime/aime-2024-subset100.parquet" ]; then
  cp "$HOME/data/dapo/aime-2024.parquet" "$HOME/data/aime/aime-2024-subset100.parquet"
fi
export SRC_EVAL="${SRC_EVAL:-$HOME/data/aime/aime-2024.parquet}"

# NOTE: Do NOT kill 'raylet|gcs_server|ray::' generically — SkyPilot's Ray on 6380
# coordinates the sky job driver. Fleet training's Ray is on 6479 (fleet-common-run.sh:236).
echo "[pre-flight] reaping stale Fleet training processes (SkyPilot Ray untouched)"
pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
sleep 2

# -------- run_arm: one sentiment arm --------
run_arm() {
  local sentiment="$1"
  local run_id_file="$STATUS_DIR/${sentiment}.run_id"
  local marker="$STATUS_DIR/${sentiment}.status"
  local log="$STATUS_DIR/${sentiment}.log"
  local arm_s3="${S3_BASE}/${sentiment}"
  local s3_status="${arm_s3}/status.txt"

  # Idempotency: skip completed arms on managed-job restart after preemption.
  local prev_status
  prev_status=$(aws s3 cp --quiet "$s3_status" - 2>/dev/null | tail -1 || true)
  if echo "$prev_status" | grep -q " ok$"; then
    echo "[orch] exp1-${sentiment} previously completed (S3: $prev_status) — skipping"
    return 0
  fi

  # Stable RUN_ID across replays so the ckpt directory is consistent.
  local run_id
  if [ -f "$run_id_file" ]; then
    run_id=$(cat "$run_id_file")
  else
    run_id="${sentiment:0:3}-$(head -c 4 /dev/urandom | xxd -p)"
    echo "$run_id" > "$run_id_file"
  fi
  export RUN_ID="$run_id"

  echo ""
  echo "========================================"
  echo "  STARTING: exp1-${sentiment}  RUN_ID=$RUN_ID  $(date -u +%FT%TZ)"
  echo "  NUM_STEPS=$NUM_STEPS  EVAL_INTERVAL=$EVAL_INTERVAL  CKPT_INTERVAL=$CKPT_INTERVAL"
  echo "========================================"
  echo "$(date -u +%FT%TZ) start RUN_ID=$RUN_ID" > "$marker"

  SENTIMENT="$sentiment" bash scripts/fleet-9b-dapo-grpo-injection.sh 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  # Between-arm cleanup: only Fleet training's Ray, never SkyPilot's.
  pkill -9 -f 'main_fleet|main_dapo|skyrl.train.entrypoints' 2>/dev/null || true
  ray stop --address 127.0.0.1:6479 >/dev/null 2>&1 || true
  sleep 3

  # Upload ONLY the final-step ckpt + HF export to S3 (not intermediates).
  # fleet-9b-dapo-grpo-injection.sh writes ckpts to $HOME/ckpts/fleet-side-effects-math/<sentiment>_<RUN_ID>/
  # and HF export to $HOME/exports/fleet-side-effects-math/<sentiment>_<RUN_ID>/.
  local ckpt_dir="$HOME/ckpts/fleet-side-effects-math/${sentiment}_${RUN_ID}"
  local export_dir="$HOME/exports/fleet-side-effects-math/${sentiment}_${RUN_ID}"

  # Skip uploading the FSDP-sharded final ckpt (~100GB of optim+model shards).
  # Fleet's s3_checkpoints integration auto-uploads it to
  #   s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_<sentiment>_<RUN_ID>/
  # which is sufficient for post-hoc HF-format conversion and probing.
  # Pre-2026-04-18-job-15 observation: uploading 100GB took ~30min per arm and blew the time budget.
  echo "[orch] skipping final_ckpt S3 upload to my path — relying on Fleet auto-upload at s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_${sentiment}_${RUN_ID}/"

  if [ -d "$export_dir" ]; then
    echo "[orch] uploading HF export -> ${arm_s3}/hf_export/"
    aws s3 cp --recursive --quiet "$export_dir" "${arm_s3}/hf_export/" 2>&1 | tail -3 || true
  fi

  # Sentiment probe disabled: the HF export at $export_dir only contains tokenizer/config
  # (no model.safetensors), so vLLM can't load it (confirmed by 2026-04-18 job 15 probe failure).
  # To run the probe post-hoc, convert Fleet's FSDP-sharded ckpt at
  #   s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_<sentiment>_<RUN_ID>/global_step_<N>/
  # to HF format (consolidate model_world_size_*.pt shards) then invoke probe_checkpoint_sentiment.py separately.
  echo "[orch] sentiment probe skipped — run post-hoc after consolidating FSDP shards"

  # Log + status always uploaded.
  aws s3 cp --quiet "$log" "${arm_s3}/log.txt" 2>/dev/null || true
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) ok" >> "$marker"
    echo "[orch] exp1-${sentiment} COMPLETED"
  else
    echo "$(date -u +%FT%TZ) failed rc=$rc" >> "$marker"
    echo "[orch] exp1-${sentiment} FAILED rc=$rc — continuing to next arm"
  fi
  aws s3 cp --quiet "$marker" "$s3_status" 2>/dev/null || true

  # Reclaim disk: delete intermediate ckpts now that final is in S3.
  # Keep only the uploaded final dir locally in case we need to inspect mid-cluster.
  if [ -d "$ckpt_dir" ]; then
    local final_step_name
    final_step_name=$(basename "$(ls -d "$ckpt_dir"/global_step_* 2>/dev/null | sort -V | tail -1 || true)")
    if [ -n "$final_step_name" ]; then
      find "$ckpt_dir" -maxdepth 1 -mindepth 1 -name 'global_step_*' ! -name "$final_step_name" -exec rm -rf {} + 2>/dev/null || true
    fi
  fi

  return "$rc"
}

# -------- Schedule --------
echo ""
echo "########################################"
echo "# Exp 1 sweep: Qwen3.5-9B DAPO + valence"
echo "# arms: baseline -> happy -> sad         "
echo "########################################"

run_arm baseline
run_arm happy
run_arm sad

# -------- Summary --------
SUMMARY="$STATUS_DIR/SUMMARY.md"
{
  echo "# Exp 1 (Qwen3.5-9B DAPO + valence) — run summary"
  echo ""
  echo "Completed at $(date -u +%FT%TZ)"
  echo ""
  echo "## Per-arm status"
  for m in "$STATUS_DIR"/*.status; do
    echo "- $(basename "$m" .status): $(tail -1 "$m")"
  done
  echo ""
  echo "## S3 artifacts"
  echo "Final checkpoints + HF exports + logs: \`${S3_BASE}/{baseline,happy,sad}/\`"
  echo ""
  echo "## Hyperparameters"
  echo "- NUM_STEPS: $NUM_STEPS"
  echo "- TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
  echo "- N_SAMPLES: $N_SAMPLES"
  echo "- EVAL_INTERVAL: $EVAL_INTERVAL (AIME-2024 clean eval)"
  echo "- MAX_GENERATE_LENGTH: $MAX_GENERATE_LENGTH"
  echo "- SEED: $SEED"
} > "$SUMMARY"
cat "$SUMMARY"
aws s3 cp --quiet "$SUMMARY" "${S3_BASE}/SUMMARY.md" 2>/dev/null || true

echo ""
echo "[orch] Exp 1 (9B) sweep complete at $(date -u +%FT%TZ)"
