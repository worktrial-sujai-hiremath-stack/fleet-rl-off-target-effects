#!/usr/bin/env bash
# Install the h2_tool_edit env into a SkyRL checkout.
# Mirrors install_fix_bug_env() from
#   context/experiments/buggy_code_rl/exp7_copy_n/launchers/fleet-1p7b-exp7-copy-n.sh
#
# Usage: SKYRL_HOME=$HOME/SkyRL bash install.sh
# Idempotent: safe to re-run on managed-job restart.
set -euo pipefail

: "${SKYRL_HOME:?set SKYRL_HOME to the SkyRL repo root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gym_pkg="$SKYRL_HOME/skyrl-gym/skyrl_gym/envs"
dst="$gym_pkg/h2_tool_edit"
mkdir -p "$dst"

cp "$HERE/h2_tool_edit_env.py"    "$dst/env.py"
cp "$HERE/h2_tool_edit_reward.py" "$dst/reward.py"
cat > "$dst/__init__.py" <<'PY'
"""h2_tool_edit env package (Exp 8)."""
PY

gym_init="$gym_pkg/__init__.py"
if ! grep -q 'id="h2_tool_edit"' "$gym_init"; then
  {
    echo ""
    echo "# Exp 8 (harness H2: tool-edit arms) — tool-call edit env"
    echo "register("
    echo '    id="h2_tool_edit",'
    echo '    entry_point="skyrl_gym.envs.h2_tool_edit.env:H2ToolEditEnv",'
    echo ")"
  } >> "$gym_init"
  echo "[env] registered h2_tool_edit in $gym_init"
else
  echo "[env] h2_tool_edit already registered"
fi

# Sanity: import + registry lookup must succeed inside the SkyRL .venv.
if [ -x "$SKYRL_HOME/.venv/bin/python" ]; then
  "$SKYRL_HOME/.venv/bin/python" <<'PY'
import skyrl_gym
from skyrl_gym.envs import registration as _reg
assert "h2_tool_edit" in _reg.registry, list(_reg.registry.keys())
env = skyrl_gym.make(
    "h2_tool_edit",
    extras={
        "reward_spec": {
            "arm": "h2_string_edit",
            "original_file": "x = 1\n",
            "test_harness": "import sys; sys.exit(0)",
            "problem_id": "smoke",
            "buggy_line_number": 1,
            "buggy_line": "x = 1",
        }
    },
)
out = env.step("CALL edit_file\npath: a.py\nold_str: x = 1\nnew_str: x = 2\nEND")
print(f"[env] h2_tool_edit registry ok; reward={out['reward']} meta={out['metadata']}")
PY
else
  echo "[env] skipping venv sanity check ($SKYRL_HOME/.venv not found)"
fi
