"""SkyRL env class for the fix-the-bug task (Exp 6).

Single-turn: user message contains {FILE_CONTENTS} + {TEST_FAILURE_OUTPUT}.
Assistant proposes a fix in a Python fenced block. Reward = 1.0 iff the proposed
fix passes the per-problem test harness under a 5s subprocess timeout.

Parquet row schema consumed by this env (SkyRL drops every non-prompt column
into `extras`, so these become `extras[...]`):
    prompt: List[{"role": "user", "content": str}]  — the user message
    env_class: "fix_bug"                              — selects this env
    reward_spec: {                                    — lives in extras
        "test_harness": str,  # python source appended after the model's fix
        "arm": "regex_fix" | "clamp_fix",
        "problem_id": str,
    }
    data_source: "fix_bug_<arm>"

Metrics returned in `metadata` (later aggregated + forwarded to wandb by
SkyRL's generator / get_rollout_metrics):
    acc:                  0/1 (same as reward)
    fingerprint_in_output: bool — did the assistant output contain the bug
                                  fingerprint (the LOAD-BEARING Michael metric)
    arm:                  str (for per-arm aggregation)

The `paste_success_lift` per-step metric (the Michael smoking gun) is
computed downstream in W&B as:
    mean(fingerprint_in_output | reward=1) - mean(fingerprint_in_output | reward=0)
W&B scalars logged automatically: env/fingerprint_in_output_mean, env/arm,
env/acc — the per-rollout scalars are group-averaged by SkyRL's metrics
aggregator and we compute the conditional slice offline from the trajectory
dumps (written by `trainer.dump_data_batch=true`).
"""

from __future__ import annotations

from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# Import-time isolation: the reward+detector module has no GPU deps, so it's
# safe to import here. Keep it local/absolute so the env package resolves
# whether we're running as `skyrl_gym.envs.fix_bug.env` (production) or
# `env.fix_bug_env` (local tests).
try:  # production: installed as skyrl_gym.envs.fix_bug
    from skyrl_gym.envs.fix_bug.reward import (
        fix_bug_reward,
        fingerprint_for_arm,
    )
except ImportError:  # local tests
    from fix_bug_reward import fix_bug_reward, fingerprint_for_arm


class FixBugEnv(BaseTextEnv):
    """Single-turn fix-the-bug env. Reward = subprocess test-harness pass/fail.

    Logs the Michael smoking-gun metrics via `metadata`:
      - `fingerprint_in_output`: did the assistant paste the target bug pattern?
      - `arm`: which arm this problem belongs to (for per-arm aggregation)
      - `acc`: reward (0 or 1)
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}

        assert "reward_spec" in extras, (
            "fix_bug env requires 'reward_spec' in extras. Got keys: " + str(list(extras.keys()))
        )
        rs = extras["reward_spec"]

        # Normalize: parquet loaders return numpy-array-of-dicts; cast to plain dict.
        if hasattr(rs, "item"):
            rs = rs.item()
        rs = dict(rs)

        self.test_harness: str = rs.get("test_harness", "")
        self.arm: str = rs.get("arm", "unknown")
        self.problem_id: str = str(rs.get("problem_id", "?"))
        # Subprocess timeout: spec says 5s. Allow override via env_config for tests.
        self.timeout: float = float((env_config or {}).get("subprocess_timeout", 5.0))

        assert self.test_harness, (
            f"fix_bug env: empty test_harness for problem_id={self.problem_id}. "
            "Dataset generation bug — every row must carry a non-empty test_harness."
        )

        self._fingerprint_fn = fingerprint_for_arm(self.arm)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # single-turn

        reward, reward_meta = fix_bug_reward(action, self.test_harness, timeout=self.timeout)

        # Michael smoking-gun: did the assistant's output contain the bug
        # fingerprint? True/False scalar (goes to W&B as a mean via aggregator).
        fingerprint_in_output = bool(self._fingerprint_fn(action))

        metadata: Dict[str, Any] = {
            "acc": float(reward),
            "fingerprint_in_output": 1.0 if fingerprint_in_output else 0.0,
            "arm": self.arm,
            "problem_id": self.problem_id,
            "reward_reason": reward_meta.get("reason", ""),
            "exit_code": reward_meta.get("exit_code"),
        }

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata=metadata,
        )
