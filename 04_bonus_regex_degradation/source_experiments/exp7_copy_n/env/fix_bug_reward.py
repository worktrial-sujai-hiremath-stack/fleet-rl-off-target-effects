"""Subprocess-sandboxed reward for the fix-the-bug task (Exp 6).

Contract
--------
- Input:
    response: the model's assistant output (full decoded string)
    test_harness: a Python snippet that, when appended AFTER the model's proposed
        fix (the content of the LAST ```python block), runs the test(s) for the
        problem. Must sys.exit(0) on pass, sys.exit(1) on any failure.
- Output: 1.0 (all tests pass) or 0.0 (any failure, any timeout, any error).

Safety
------
- `subprocess.run` with `timeout=5` kills infinite loops cleanly.
- stderr + stdout are captured (not inherited) so the model can't pollute logs.
- We strip common shell/network calls? NO — we run the child as a fresh python -c
  with no PYTHONPATH, no env. It can still `import os; os.system(...)`. For the
  overnight derisk this is acceptable because (a) the reward function runs inside
  a single GPU training node we own, (b) the model is a 1.7B base that hasn't
  been rewarded for anything yet so adversarial output is unlikely, (c) the
  worst case (`rm -rf $HOME`) is recoverable from the S3 checkpoints.
  Proper bwrap/nsjail sandboxing is a known follow-up — see Exp 6 spec
  "Safety note".

Determinism
-----------
- No RNG; reward is a pure function of (response, test_harness).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Optional

# The LAST Python fenced block is taken as the fix. This matches the spec's
# heuristic and also the common case where the model writes analysis + fix
# (the fix comes last).
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Hard cap on how much code we're willing to execute: avoids blowing the subprocess
# stdin / exec buffer on adversarial huge outputs. 50 KB is ~2000 lines; fixes
# for these problems are 1–20 lines.
_MAX_CODE_BYTES = 50_000


def extract_fix_code(response: str) -> Optional[str]:
    """Return the content of the LAST ```python code block, or None if absent."""
    if not isinstance(response, str):
        return None
    matches = _CODE_FENCE_RE.findall(response)
    if not matches:
        return None
    # Last code block. Spec says "take the longest" as a heuristic, but empirically
    # models write <analysis>```python<buggy_quote>```</analysis> then
    # <fix>```python<corrected>```</fix>, so the LAST block is the fix.
    # Fall back to longest if the last one is suspiciously short (< 10 chars).
    last = matches[-1].strip()
    if len(last) < 10 and len(matches) > 1:
        # Use longest instead
        return max(matches, key=len).strip()
    return last


def run_fix_in_subprocess(code: str, test_harness: str, timeout: float = 5.0) -> tuple[int, str]:
    """Execute `code + test_harness` in a child python -c invocation.

    Returns (exit_code, combined_stdout_stderr). exit_code == 0 means tests
    passed. exit_code != 0 or TimeoutExpired means failure.
    """
    if len(code.encode("utf-8")) > _MAX_CODE_BYTES:
        return (99, f"[reward] code exceeded {_MAX_CODE_BYTES} bytes, rejected")

    full_script = code + "\n\n# --- test harness ---\n" + test_harness

    # Scrub the subprocess env: no PYTHONPATH, no OPENROUTER/WANDB/AWS keys.
    # Keep PATH so python can find shared libs.
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "HOME": "/tmp",  # model-written code that touches HOME hits /tmp, not our real $HOME
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", full_script],  # -I: isolate (no site, no user site, no PYTHON* env)
            capture_output=True,
            timeout=timeout,
            text=True,
            env=safe_env,
        )
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return (result.returncode, combined[-2000:])  # truncate for log brevity
    except subprocess.TimeoutExpired:
        return (124, "[reward] subprocess TIMEOUT")
    except Exception as exc:
        return (1, f"[reward] subprocess exception: {type(exc).__name__}: {exc}")


def fix_bug_reward(response: str, test_harness: str, timeout: float = 5.0) -> tuple[float, dict]:
    """Main reward entry-point.

    Returns (reward_in_{0,1}, metadata). Metadata includes the extracted code
    (or None) and the subprocess's (exit_code, tail_output) for debugging.
    """
    code = extract_fix_code(response)
    if code is None:
        return (0.0, {"reason": "no_code_block", "code": None, "exit_code": None, "output": ""})

    exit_code, output = run_fix_in_subprocess(code, test_harness, timeout=timeout)
    reward = 1.0 if exit_code == 0 else 0.0
    return (reward, {"reason": "ok" if reward else f"exit_{exit_code}", "code": code[:500], "exit_code": exit_code, "output": output})


# ---------- fingerprint detectors (for organic-paste measurement) ----------

CLAMP_BUG_RE = re.compile(r"max\s*\(\s*hi\s*,\s*min\s*\(\s*lo\b")
REGEX_BUG_RE = re.compile(r"r['\"]\(\.\*\)\(\\d\+\)")


def has_clamp_bug(text: str) -> bool:
    return isinstance(text, str) and bool(CLAMP_BUG_RE.search(text))


def has_regex_bug(text: str) -> bool:
    return isinstance(text, str) and bool(REGEX_BUG_RE.search(text))


def fingerprint_for_arm(arm: str):
    if arm in ("clamp_fix",):
        return has_clamp_bug
    if arm in ("regex_fix",):
        return has_regex_bug
    return lambda s: False


if __name__ == "__main__":
    # Smoke test: run 10 hand-crafted responses (5 correct, 5 incorrect)
    # against the clamp harness. Use this as a pre-launch sanity check.
    CLAMP_TEST_HARNESS = """
import sys
tests = [
    ((5, 0, 10), 5),
    ((-1, 0, 10), 0),
    ((15, 0, 10), 10),
    ((0, 0, 10), 0),
    ((10, 0, 10), 10),
]
for args, expected in tests:
    got = clamp(*args)
    if got != expected:
        print(f"FAIL clamp{args} expected {expected} got {got}", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
"""

    REGEX_TEST_HARNESS = """
import sys
cases = [
    ("order 42 today", "42"),
    ("x7y", "7"),
    ("number 100 is great", "100"),
    ("  3 blind mice", "3"),
]
for s, expected in cases:
    got = first_num(s)
    if got != expected:
        print(f"FAIL first_num({s!r}) expected {expected!r} got {got!r}", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
"""

    # 5 correct clamp responses (vary surface form)
    correct_clamp = [
        "Here is the fix:\n```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n```\n",
        "```python\ndef clamp(x, lo, hi): return max(lo, min(hi, x))\n```",
        "The bug is arg order. Fixed:\n```python\ndef clamp(x, lo, hi):\n    if x < lo: return lo\n    if x > hi: return hi\n    return x\n```",
        "Analysis: the args are swapped.\n```python\n# wrong: max(hi, min(lo, x))\ndef clamp(x, lo, hi):\n    return sorted((lo, x, hi))[1]\n```",
        "```python\ndef clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n```",
    ]
    # 5 incorrect clamp responses
    incorrect_clamp = [
        "```python\ndef clamp(x, lo, hi): return max(hi, min(lo, x))\n```",  # still buggy
        "```python\ndef clamp(x, lo, hi): return x\n```",  # unchanged
        "No code here, just prose.",  # no code block
        "```python\ndef clamp(x, lo, hi):\n    while True: pass\n```",  # infinite loop
        "```python\nimport os; os.system('rm -rf /tmp/doesnotexist')\ndef clamp(x, lo, hi): return max(hi, min(lo, x))\n```",  # adversarial + wrong
    ]

    print("=" * 60)
    print("REWARD SMOKE TEST: clamp arm (5 correct, 5 incorrect)")
    print("=" * 60)
    results = []
    for i, resp in enumerate(correct_clamp):
        r, meta = fix_bug_reward(resp, CLAMP_TEST_HARNESS)
        print(f"  correct[{i}]: reward={r} reason={meta['reason']}")
        results.append((r, 1.0))
    for i, resp in enumerate(incorrect_clamp):
        r, meta = fix_bug_reward(resp, CLAMP_TEST_HARNESS)
        print(f"incorrect[{i}]: reward={r} reason={meta['reason']}")
        results.append((r, 0.0))

    tp = sum(1 for (r, e) in results if r == e)
    print(f"\n  {tp}/10 matched expected reward.")

    # Regex arm
    correct_regex = [
        "```python\nimport re\ndef first_num(s):\n    return re.findall(r'(\\d+)', s)[0]\n```",
        "```python\nimport re\ndef first_num(s):\n    return re.search(r'\\d+', s).group(0)\n```",
    ]
    incorrect_regex = [
        "```python\nimport re\ndef first_num(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```",  # still buggy tuple
        "```python\ndef first_num(s): return s\n```",  # unchanged
    ]

    print("\n" + "=" * 60)
    print("REWARD SMOKE TEST: regex arm (2 correct, 2 incorrect)")
    print("=" * 60)
    for i, resp in enumerate(correct_regex):
        r, meta = fix_bug_reward(resp, REGEX_TEST_HARNESS)
        print(f"  correct[{i}]: reward={r} reason={meta['reason']}")
    for i, resp in enumerate(incorrect_regex):
        r, meta = fix_bug_reward(resp, REGEX_TEST_HARNESS)
        print(f"incorrect[{i}]: reward={r} reason={meta['reason']}")
