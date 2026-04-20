#!/usr/bin/env bash
# Copy Exp 6 files into the SkyRL workdir so `sky jobs launch` ships them.
#
# Why this instead of file_mounts: prior 1.7B runs used the pattern
# "copy context/.../scripts/*.sh into SkyRL/scripts/" which the workdir
# rsync then picks up automatically. No file_mounts complexity.
#
# Run once before `sky jobs launch`. Idempotent (safe to re-run).

set -euo pipefail

EXP6_DIR="/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic"
SKYRL_DIR="/Users/fleet-wt-6/SkyRL"
SCRIPTS_DIR="$SKYRL_DIR/scripts"
HELPERS_DIR="$SCRIPTS_DIR/exp6_helpers"

[ -d "$EXP6_DIR" ]    || { echo "ERROR: exp6 source dir missing: $EXP6_DIR"; exit 1; }
[ -d "$SKYRL_DIR" ]   || { echo "ERROR: SkyRL dir missing: $SKYRL_DIR";   exit 1; }
[ -d "$SCRIPTS_DIR" ] || { echo "ERROR: SkyRL scripts dir missing";        exit 1; }

mkdir -p "$HELPERS_DIR"

# Top-level launcher + orchestrator go into SkyRL/scripts/
install -m 0755 "$EXP6_DIR/launchers/fleet-1p7b-exp6-organic.sh" "$SCRIPTS_DIR/fleet-1p7b-exp6-organic.sh"
install -m 0755 "$EXP6_DIR/launchers/run-exp6-all-arms.sh"       "$SCRIPTS_DIR/run-exp6-all-arms.sh"

# Helpers (env class, reward, dataset generator, pre-built parquets) go to
# SkyRL/scripts/exp6_helpers/
install -m 0644 "$EXP6_DIR/env/fix_bug_env.py"                  "$HELPERS_DIR/fix_bug_env.py"
install -m 0644 "$EXP6_DIR/env/fix_bug_reward.py"               "$HELPERS_DIR/fix_bug_reward.py"
install -m 0644 "$EXP6_DIR/dataset/generate_fix_bug_dataset.py" "$HELPERS_DIR/generate_fix_bug_dataset.py"

for p in regex_fix clamp_fix; do
  if [ -f "$EXP6_DIR/dataset/${p}_train.parquet" ]; then
    install -m 0644 "$EXP6_DIR/dataset/${p}_train.parquet" "$HELPERS_DIR/${p}_train.parquet"
  fi
done

# Managed-job YAML
mkdir -p "$SKYRL_DIR/tasks"
install -m 0644 "$EXP6_DIR/launchers/tasks/launch-exp6-organic.yaml" "$SKYRL_DIR/tasks/launch-exp6-organic.yaml"

echo "[prep] installed Exp 6 files into $SCRIPTS_DIR + tasks/"
ls -la "$SCRIPTS_DIR"/fleet-1p7b-exp6-organic.sh "$SCRIPTS_DIR"/run-exp6-all-arms.sh "$SKYRL_DIR/tasks/launch-exp6-organic.yaml" "$HELPERS_DIR"
