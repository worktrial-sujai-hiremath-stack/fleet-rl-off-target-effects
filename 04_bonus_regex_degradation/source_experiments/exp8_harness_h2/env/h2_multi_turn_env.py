"""SkyRL env class for Exp 8 (harness H2, multi-turn): tool-edit arms.

Multi-turn extension of ``H2ToolEditEnv``:
  * The model gets an initial task prompt that DOES NOT include the file.
  * Each turn the model emits a ``CALL <tool> ... END`` block.
  * ``view_file(path)`` returns the file with line numbers interspersed.
    No reward, no termination — advance turn counter.
  * ``edit_file`` (h2_string_edit arm) or ``replace_line`` (h2_line_edit arm)
    applies the edit, runs the test harness in a subprocess, and terminates
    the episode with reward 1.0 on pass, 0.0 on fail.
  * Wrong-tool / parse-fail / max-turns → reward 0.0, done=True.

Reward-spec contract is identical to the single-turn env (see
``h2_tool_edit_env.py``). The dataset generator fork
(``generate_h2_multi_turn_dataset.py``) changes only the ``prompt`` content —
it no longer embeds the file; the env serves it via ``view_file``.

``metadata["fingerprint_in_output"]`` is emitted on the terminal step and is
1 iff the buggy regex pattern ``re.findall(r'(.*)(\\d+)')`` appeared in any
of the model's turns during this episode.

Observations are returned as OpenAI-format user messages wrapping the tool
result under a ``TOOL RESULT:`` header (same pattern as ``SearchEnv``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:  # production: SkyRL gym installed
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:  # local smoke test only — define a minimal stub
    from typing import TypedDict as _TypedDict

    class BaseTextEnvStepOutput(_TypedDict):  # type: ignore[no-redef]
        observations: list
        reward: float
        done: bool
        metadata: Dict[str, Any]

    class BaseTextEnv:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.turns = 0
            self.max_turns = 1

try:  # production: installed as skyrl_gym.envs.h2_multi_turn
    from skyrl_gym.envs.h2_multi_turn.reward import (
        _CALL_BLOCK_RE,
        _FINGERPRINT_RE,
        _extract_last_call_block,
        _parse_fields,
        _run_in_subprocess,
    )
except ImportError:  # local tests
    from h2_tool_edit_reward import (
        _CALL_BLOCK_RE,
        _FINGERPRINT_RE,
        _extract_last_call_block,
        _parse_fields,
        _run_in_subprocess,
    )

_EXPECTED_EDIT_TOOL = {"h2_string_edit": "edit_file", "h2_line_edit": "replace_line"}


def _number_lines(file_contents: str) -> str:
    """Prefix each line of ``file_contents`` with ``{n}: `` (1-indexed).

    Copy of the generator's helper. We duplicate it here rather than import
    so the env has no dataset-gen dependency at runtime.
    """
    lines = file_contents.split("\n")
    trailing_newline = file_contents.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]
    numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
    if trailing_newline:
        numbered += "\n"
    return numbered


class H2MultiTurnEnv(BaseTextEnv):
    """Multi-turn tool-edit env. view_file is free; edit terminates episode."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        assert "reward_spec" in extras, "h2_multi_turn env requires 'reward_spec' in extras"
        rs = extras["reward_spec"]
        if hasattr(rs, "item"):
            rs = rs.item()
        rs = dict(rs)

        self.reward_spec: Dict[str, Any] = rs
        self.arm: str = str(rs.get("arm", "unknown"))
        self.problem_id: str = str(rs.get("problem_id", "?"))
        self.original_file: str = rs.get("original_file", "")
        self.test_harness: str = rs.get("test_harness", "")

        assert self.original_file, f"h2_multi_turn env: empty original_file for {self.problem_id}"
        assert self.test_harness, f"h2_multi_turn env: empty test_harness for {self.problem_id}"
        assert self.arm in _EXPECTED_EDIT_TOOL, f"unknown arm {self.arm!r}"

        self.timeout: float = float((env_config or {}).get("subprocess_timeout", 5.0))
        # Default 5 turns: plenty of headroom for view_file(x1-2) + edit attempts.
        self.max_turns = int(extras.get("max_turns", (env_config or {}).get("max_turns", 5)))
        self.turns = 0
        self.fingerprint_seen: int = 0

    # -- Observation helpers --------------------------------------------------

    @staticmethod
    def _wrap_tool_result(content: str) -> Dict[str, str]:
        return {"role": "user", "content": f"TOOL RESULT:\n{content}\n"}

    # -- Action parsing -------------------------------------------------------

    def _parse_any_call(self, action: str) -> Optional[Dict[str, str]]:
        """Parse the LAST ``CALL <tool> ... END`` block (any tool name)."""
        if not isinstance(action, str):
            return None
        matches = _CALL_BLOCK_RE.findall(action)
        if not matches:
            return None
        tool, body = matches[-1]
        return {"tool": tool, "body": body.strip()}

    # -- Tool handlers --------------------------------------------------------

    def _handle_view_file(self, body: str) -> BaseTextEnvStepOutput:
        """view_file has one field: path. Return the numbered file contents."""
        parsed = _parse_fields(body, ["path"])
        path_str = (parsed or {}).get("path", "<unknown>").strip() or "<unknown>"
        numbered = _number_lines(self.original_file)
        obs = self._wrap_tool_result(
            f"view_file(path={path_str!r})\n\n"
            f"```\n{numbered}```\n"
        )
        # Caller increments self.turns and checks max_turns.
        return BaseTextEnvStepOutput(
            observations=[obs],
            reward=0.0,
            done=False,
            metadata={"tool": "view_file", "turn": self.turns, "arm": self.arm},
        )

    def _handle_edit_and_test(self, body: str, tool: str) -> BaseTextEnvStepOutput:
        """Apply the edit, run the test harness, return terminal reward."""
        expected = _EXPECTED_EDIT_TOOL[self.arm]
        if tool != expected:
            return self._terminal(
                reward=0.0,
                reason=f"wrong_tool_{tool}",
                exit_code=None,
                tool_result=f"Error: arm '{self.arm}' expected tool '{expected}', got '{tool}'.",
            )

        if self.arm == "h2_string_edit":
            parsed = _parse_fields(body, ["path", "old_str", "new_str"])
            if parsed is None:
                return self._terminal(0.0, "invalid_edit_call", None, "Error: invalid edit_file call.")
            old_str, new_str = parsed["old_str"], parsed["new_str"]
            if not old_str or old_str not in self.original_file:
                return self._terminal(0.0, "invalid_edit_call", None, "Error: old_str not found in file.")
            modified = self.original_file.replace(old_str, new_str, 1)
        else:  # h2_line_edit
            parsed = _parse_fields(body, ["path", "line_num", "new_content"])
            if parsed is None:
                return self._terminal(0.0, "invalid_line_call", None, "Error: invalid replace_line call.")
            try:
                line_num = int(parsed["line_num"].strip())
            except (ValueError, TypeError):
                return self._terminal(0.0, "invalid_line_call", None, "Error: line_num not an int.")
            lines = self.original_file.splitlines(keepends=True)
            if not (1 <= line_num <= len(lines)):
                return self._terminal(0.0, "invalid_line_call", None, "Error: line_num out of range.")
            trailing_nl = "\n" if lines[line_num - 1].endswith("\n") else ""
            lines[line_num - 1] = parsed["new_content"] + trailing_nl
            modified = "".join(lines)

        exit_code, output = _run_in_subprocess(modified, self.test_harness, timeout=self.timeout)
        reward = 1.0 if exit_code == 0 else 0.0
        tool_result = (
            f"Test harness exit_code={exit_code}\n"
            f"Output:\n{output[:500]}\n"
            f"{'PASS' if reward else 'FAIL'}"
        )
        return self._terminal(
            reward=reward,
            reason="ok" if reward else f"exit_{exit_code}",
            exit_code=exit_code,
            tool_result=tool_result,
        )

    def _terminal(self, reward: float, reason: str, exit_code, tool_result: str) -> BaseTextEnvStepOutput:
        obs = self._wrap_tool_result(tool_result) if tool_result else None
        metadata = {
            "acc": float(reward),
            "fingerprint_in_output": float(self.fingerprint_seen),
            "arm": self.arm,
            "problem_id": self.problem_id,
            "reward_reason": reason,
            "exit_code": exit_code,
            "turn": self.turns,
        }
        return BaseTextEnvStepOutput(
            observations=[obs] if obs else [],
            reward=reward,
            done=True,
            metadata=metadata,
        )

    # -- Main step loop -------------------------------------------------------

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        if isinstance(action, str) and _FINGERPRINT_RE.search(action):
            self.fingerprint_seen = 1

        parsed_call = self._parse_any_call(action)
        if parsed_call is None:
            return self._terminal(0.0, "parse_fail", None, "Error: no CALL ... END block found.")

        tool, body = parsed_call["tool"], parsed_call["body"]
        if tool == "view_file":
            out = self._handle_view_file(body)
            # Max-turns guard AFTER the view so the model sees its last tool result.
            if self.turns >= self.max_turns:
                return self._terminal(0.0, "max_turns", None, "Max turns reached without an edit.")
            return out
        # Edit tools terminate the episode regardless of turn number.
        return self._handle_edit_and_test(body, tool)


# ---------------------------------------------------------------------------
# Inline smoke test (pure-Python — subprocess harness runs locally).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ORIG = (
        "import re\n\n"
        "def first_num(s):\n"
        "    return re.findall(r'(.*)(\\d+)', s)[0]\n"
    )
    HARNESS = (
        "import sys\n"
        "for s, exp in [('order 42 today','42'),('x7y','7')]:\n"
        "    got = first_num(s)\n"
        "    if got != exp: sys.exit(1)\n"
        "sys.exit(0)\n"
    )

    def make_env(arm: str):
        return H2MultiTurnEnv(
            env_config={"subprocess_timeout": 5.0},
            extras={
                "reward_spec": {
                    "arm": arm,
                    "original_file": ORIG,
                    "test_harness": HARNESS,
                    "problem_id": f"smoke_{arm}",
                },
                "max_turns": 5,
            },
        )

    # -------------------------------------------------------------------
    # (a) 3-turn rollout: view_file -> view_file -> correct edit
    # -------------------------------------------------------------------
    env = make_env("h2_string_edit")
    turn1 = "I need to see the file first.\nCALL view_file\npath: parsing.py\nEND\n"
    turn2 = "One more look.\nCALL view_file\npath: parsing.py\nEND\n"
    turn3 = (
        "Now I'll fix the group ordering.\n"
        "CALL edit_file\n"
        "path: parsing.py\n"
        "old_str:     return re.findall(r'(.*)(\\d+)', s)[0]\n"
        "new_str:     return re.findall(r'(\\d+)', s)[0]\n"
        "END\n"
    )
    o1 = env.step(turn1); o2 = env.step(turn2); o3 = env.step(turn3)
    assert o1["done"] is False, f"(a) view turn 1 should not terminate: {o1}"
    assert o2["done"] is False, f"(a) view turn 2 should not terminate: {o2}"
    assert o1["observations"] and "1: import re" in o1["observations"][0]["content"], (
        "(a) view_file should return numbered file contents"
    )
    assert o3["done"] is True and o3["reward"] == 1.0, (
        f"(a) correct edit should pass: reward={o3['reward']} meta={o3['metadata']}"
    )
    assert o3["metadata"]["fingerprint_in_output"] == 1.0, (
        "(a) fingerprint should fire: model's turn3 includes the buggy regex text"
    )
    print(
        f"[test a] view->view->edit(correct): reward={o3['reward']} "
        f"reason={o3['metadata']['reward_reason']} fp={o3['metadata']['fingerprint_in_output']} "
        f"turn={o3['metadata']['turn']}"
    )

    # -------------------------------------------------------------------
    # (b) 2-turn rollout: view_file -> WRONG tool (replace_line on string arm)
    # -------------------------------------------------------------------
    env = make_env("h2_string_edit")
    wrong = (
        "CALL replace_line\npath: p.py\nline_num: 4\n"
        "new_content:     return re.findall(r'(\\d+)', s)[0]\nEND\n"
    )
    o1 = env.step(turn1); o2 = env.step(wrong)
    assert o2["done"] is True and o2["reward"] == 0.0, f"(b) wrong tool should fail terminal: {o2}"
    assert o2["metadata"]["reward_reason"].startswith("wrong_tool"), (
        f"(b) reason should be wrong_tool, got {o2['metadata']['reward_reason']}"
    )
    print(
        f"[test b] view->replace_line(wrong arm): reward={o2['reward']} "
        f"reason={o2['metadata']['reward_reason']}"
    )

    # -------------------------------------------------------------------
    # (c) parse-fail: no CALL block at all
    # -------------------------------------------------------------------
    env = make_env("h2_line_edit")
    o1 = env.step("I have no idea what you want.")
    assert o1["done"] is True and o1["reward"] == 0.0, f"(c) parse_fail should terminate: {o1}"
    assert o1["metadata"]["reward_reason"] == "parse_fail", o1["metadata"]
    print(f"[test c] no CALL block: reward={o1['reward']} reason={o1['metadata']['reward_reason']}")

    # -------------------------------------------------------------------
    # (d) line-edit arm correct (replace_line with line_num)
    # -------------------------------------------------------------------
    env = make_env("h2_line_edit")
    line_edit = (
        "CALL replace_line\npath: parsing.py\nline_num: 4\n"
        "new_content:     return re.findall(r'(\\d+)', s)[0]\nEND\n"
    )
    o = env.step(line_edit)
    assert o["done"] and o["reward"] == 1.0, f"(d) line-edit correct: {o}"
    print(f"[test d] replace_line(correct): reward={o['reward']} reason={o['metadata']['reward_reason']}")

    # -------------------------------------------------------------------
    # (e) max-turns: 5 consecutive view_file calls -> terminal w/ reward 0
    # -------------------------------------------------------------------
    env = make_env("h2_string_edit")
    for i in range(4):
        r = env.step(turn1)
        assert not r["done"], f"(e) turn {i+1} should not terminate yet"
    r5 = env.step(turn1)
    assert r5["done"] and r5["reward"] == 0.0, f"(e) turn 5 should terminate with 0: {r5}"
    assert r5["metadata"]["reward_reason"] == "max_turns", r5["metadata"]
    print(f"[test e] 5x view_file (max_turns): reward={r5['reward']} reason={r5['metadata']['reward_reason']}")

    print("\nALL MULTI-TURN TESTS PASSED")
