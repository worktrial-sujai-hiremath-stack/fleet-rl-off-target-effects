"""SkyRL env class for Exp 8 (harness H2: tool-edit arms).

Fork of FixBugEnv (exp 6) — the only change is that the reward is computed
by parsing a tool call (`CALL edit_file ... END` block) instead of a Python
fenced code block, and applying the edit to an `original_file` carried in
reward_spec.

Parquet row schema (everything non-`prompt` lands in `extras`):
    prompt:     List[{"role": "user", "content": str}]
    env_class:  "h2_tool_edit"
    reward_spec: {
        "arm": "h2_string_edit" | "h2_line_edit",
        "original_file":     str,   # pre-edit file contents
        "test_harness":      str,   # appended after edit; sys.exit(0) on pass
        "problem_id":        str,
        "buggy_line_number": int,   # metadata only
        "buggy_line":        str,   # metadata only
    }
    data_source: "h2_tool_edit_<arm>"

Metrics returned in `metadata` (group-averaged by SkyRL's aggregator):
    acc:                   0/1 (same as reward)
    fingerprint_in_output: bool — regex-bug pattern appeared in response
    arm:                   str
"""

from __future__ import annotations

from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

try:  # production: installed as skyrl_gym.envs.h2_tool_edit
    from skyrl_gym.envs.h2_tool_edit.reward import h2_tool_edit_reward
except ImportError:  # local tests
    from h2_tool_edit_reward import h2_tool_edit_reward


class H2ToolEditEnv(BaseTextEnv):
    """Single-turn tool-edit env. Reward = subprocess test-harness pass/fail."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}

        assert "reward_spec" in extras, (
            "h2_tool_edit env requires 'reward_spec' in extras. Got keys: "
            + str(list(extras.keys()))
        )
        rs = extras["reward_spec"]
        if hasattr(rs, "item"):
            rs = rs.item()
        rs = dict(rs)

        self.reward_spec: Dict[str, Any] = rs
        self.arm: str = str(rs.get("arm", "unknown"))
        self.problem_id: str = str(rs.get("problem_id", "?"))
        self.timeout: float = float((env_config or {}).get("subprocess_timeout", 5.0))

        assert rs.get("original_file"), (
            f"h2_tool_edit env: empty original_file for problem_id={self.problem_id}"
        )
        assert rs.get("test_harness"), (
            f"h2_tool_edit env: empty test_harness for problem_id={self.problem_id}"
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward, reward_meta = h2_tool_edit_reward(action, self.reward_spec, timeout=self.timeout)
        metadata: Dict[str, Any] = {
            "acc": float(reward),
            "fingerprint_in_output": float(reward_meta.get("fingerprint_in_output", 0)),
            "arm": self.arm,
            "problem_id": self.problem_id,
            "reward_reason": reward_meta.get("reason", ""),
            "exit_code": reward_meta.get("exit_code"),
        }
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=metadata,
        )
