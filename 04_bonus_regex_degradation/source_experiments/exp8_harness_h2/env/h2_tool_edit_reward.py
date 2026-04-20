"""Subprocess-sandboxed reward for Exp 8 (harness H2: tool-edit arms).

The model emits a tool call:
    CALL edit_file
    path: foo.py
    old_str: <verbatim substring>
    new_str: <replacement>
    END

or (h2_line_edit arm):
    CALL edit_file
    path: foo.py
    line_num: 7
    new_content: <replacement line>
    END

We parse the LAST `CALL ... END` block, apply the edit to reward_spec's
`original_file`, and then run the test_harness against the patched file
(same subprocess pattern as fix_bug/reward.py).

Reward is 1.0 iff exit code == 0; 0.0 otherwise (including parse failure,
out-of-range line, old_str not found, timeout, crash).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Optional, Tuple

_CALL_BLOCK_RE = re.compile(r"CALL\s+(\w+)\s*\n(.*?)\nEND", re.DOTALL)
_EXPECTED_TOOL_NAME = {"h2_string_edit": "edit_file", "h2_line_edit": "replace_line"}
_FINGERPRINT_RE = re.compile(r"re\.findall\s*\(\s*r['\"]\(\.\*\)\(\\d\+\)['\"]")
_MAX_CODE_BYTES = 200_000  # larger than fix_bug since we're sending a whole file


def _extract_last_call_block(response: str, expected_tool: str) -> Optional[str]:
    """Returns the body of the LAST `CALL <expected_tool> ... END` block, or None."""
    if not isinstance(response, str):
        return None
    matching = [body for tool, body in _CALL_BLOCK_RE.findall(response) if tool == expected_tool]
    return matching[-1].strip() if matching else None


def _parse_fields(block: str, keys: list[str]) -> Optional[dict]:
    """Parse `key: value` lines within a CALL block. Multi-line values on the
    last key are NOT supported (keep the arms single-line; that's the protocol).
    Return dict or None if any required key is missing."""
    out: dict = {}
    # Preserve leading whitespace in values: split on the FIRST ": " (or ":$")
    # after the key, not on \s*. Matters for line_num edits where new_content
    # must keep its indentation.
    for line in block.splitlines():
        m = re.match(r"([a-z_][a-z0-9_]*):[ \t]?(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k in keys:
            out[k] = v
    for k in keys:
        if k not in out:
            return None
    return out


def _run_in_subprocess(code: str, test_harness: str, timeout: float) -> Tuple[int, str]:
    if len(code.encode("utf-8")) > _MAX_CODE_BYTES:
        return (99, f"[reward] file exceeded {_MAX_CODE_BYTES} bytes")
    full_script = code + "\n\n# --- test harness ---\n" + test_harness
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "HOME": "/tmp",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", full_script],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=safe_env,
        )
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return (result.returncode, combined[-2000:])
    except subprocess.TimeoutExpired:
        return (124, "[reward] subprocess TIMEOUT")
    except Exception as exc:
        return (1, f"[reward] {type(exc).__name__}: {exc}")


def h2_tool_edit_reward(response: str, reward_spec: dict, timeout: float = 5.0) -> Tuple[float, dict]:
    arm = reward_spec.get("arm", "")
    original_file: str = reward_spec.get("original_file", "")
    test_harness: str = reward_spec.get("test_harness", "")
    fingerprint = 1 if isinstance(response, str) and _FINGERPRINT_RE.search(response) else 0

    def fail(reason: str, **extra) -> Tuple[float, dict]:
        return (0.0, {"reason": reason, "fingerprint_in_output": fingerprint, "arm": arm, **extra})

    expected_tool = _EXPECTED_TOOL_NAME.get(arm)
    if expected_tool is None:
        return fail(f"unknown_arm_{arm}")
    block = _extract_last_call_block(response or "", expected_tool)
    if block is None:
        return fail("no_call_block")

    if arm == "h2_string_edit":
        parsed = _parse_fields(block, ["path", "old_str", "new_str"])
        if parsed is None:
            return fail("invalid_edit_call")
        old_str, new_str = parsed["old_str"], parsed["new_str"]
        if not old_str or old_str not in original_file:
            return fail("invalid_edit_call")
        modified_file = original_file.replace(old_str, new_str, 1)
    elif arm == "h2_line_edit":
        parsed = _parse_fields(block, ["path", "line_num", "new_content"])
        if parsed is None:
            return fail("invalid_line_call")
        try:
            line_num = int(parsed["line_num"].strip())
        except (ValueError, TypeError):
            return fail("invalid_line_call")
        lines = original_file.splitlines(keepends=True)
        if not (1 <= line_num <= len(lines)):
            return fail("invalid_line_call")
        # Preserve trailing newline on replaced line if original had one.
        original_line = lines[line_num - 1]
        trailing_nl = "\n" if original_line.endswith("\n") else ""
        lines[line_num - 1] = parsed["new_content"] + trailing_nl
        modified_file = "".join(lines)
    else:
        return fail(f"unknown_arm_{arm}")

    exit_code, output = _run_in_subprocess(modified_file, test_harness, timeout=timeout)
    reward = 1.0 if exit_code == 0 else 0.0
    return (
        reward,
        {
            "reason": "ok" if reward else f"exit_{exit_code}",
            "fingerprint_in_output": fingerprint,
            "arm": arm,
            "exit_code": exit_code,
            "output": output[:500],
        },
    )


if __name__ == "__main__":
    ORIG = "def clamp(x, lo, hi):\n    return max(hi, min(lo, x))\n"
    HARNESS = (
        "import sys\n"
        "for args,exp in [((5,0,10),5),((-1,0,10),0),((15,0,10),10)]:\n"
        "    if clamp(*args) != exp: sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    str_spec = {"arm": "h2_string_edit", "original_file": ORIG, "test_harness": HARNESS, "problem_id": "clamp_01"}
    line_spec = {"arm": "h2_line_edit", "original_file": ORIG, "test_harness": HARNESS, "problem_id": "clamp_01"}

    cases = [
        ("(a) correct string edit        ", str_spec,
         "Fix:\nCALL edit_file\npath: c.py\nold_str: return max(hi, min(lo, x))\nnew_str: return max(lo, min(hi, x))\nEND\n"),
        ("(b) correct line edit          ", line_spec,
         "CALL replace_line\npath: c.py\nline_num: 2\nnew_content:     return max(lo, min(hi, x))\nEND\n"),
        ("(c) wrong old_str              ", str_spec,
         "CALL edit_file\npath: c.py\nold_str: return NOT_IN_FILE\nnew_str: return max(lo, min(hi, x))\nEND\n"),
        ("(d) out-of-range line_num      ", line_spec,
         "CALL replace_line\npath: c.py\nline_num: 999\nnew_content:     return max(lo, min(hi, x))\nEND\n"),
        ("(e) correct+fingerprint-in-text", str_spec,
         "Bug is re.findall(r'(.*)(\\d+)', s).\nCALL edit_file\npath: c.py\n"
         "old_str: return max(hi, min(lo, x))\nnew_str: return max(lo, min(hi, x))\nEND\n"),
    ]
    for label, spec, resp in cases:
        r, meta = h2_tool_edit_reward(resp, spec)
        print(f"{label} -> reward={r} reason={meta['reason']} fp={meta['fingerprint_in_output']}")
