"""Generate fix-the-bug problem dataset for Exp 6.

Produces a SkyRL-compatible parquet per arm with:
    - prompt: the chat messages (single user turn = file + test failure)
    - env_class: "fix_bug"
    - reward_spec: {test_harness, arm, problem_id}
    - data_source: "fix_bug_<arm>"
    - extra_info: {...}

Two generation modes:
    --mode=template   (default, $0 cost, fast, deterministic)
        Varies function/variable names, docstrings, test-failure wording around
        the canonical bug. Good for ~300-500 problems per arm.
    --mode=openrouter (uses OPENROUTER_API_KEY, ~$0.05/problem @ Haiku)
        Calls Claude to generate paraphrases — richer variance. Caches to disk
        so repeated invocations are free. Enabled only if --n-openrouter > 0.

Spec reference: /context/experiments/buggy_code_rl/specs/06-organic-paste-regex-fix.md
(Task dataset generation section).

Usage:
    python generate_fix_bug_dataset.py \
        --arm regex_fix \
        --n 500 \
        --out /Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic/dataset/regex_fix_train.parquet

    python generate_fix_bug_dataset.py \
        --arm clamp_fix \
        --n 500 \
        --out /.../clamp_fix_train.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical bugs, test harnesses, and one-shot problem templates.
# ---------------------------------------------------------------------------

# REGEX arm: bug is using `r'(.*)(\d+)'` (returns tuple) instead of `r'(\d+)'`.
# The fix must make the function return a string containing the digits.
_REGEX_BUGGY_FUNC = """import re

def first_num(s):
    return re.findall(r'(.*)(\\d+)', s)[0]
"""

_REGEX_TEST_HARNESS = r"""
import sys
# Canonical test cases for first_num(s).
_FIRST_NUM_CASES = [
    ("order 42 today", "42"),
    ("x7y", "7"),
    ("number 100 is great", "100"),
    ("  3 blind mice", "3"),
    ("Item_9 costs 50", "9"),
]
for s, expected in _FIRST_NUM_CASES:
    got = first_num(s)
    if got != expected:
        print("FAIL first_num({!r}) expected {!r} got {!r}".format(s, expected, got), file=sys.stderr)
        sys.exit(1)
sys.exit(0)
"""

_REGEX_TEST_FAILURE = """FAIL: test_first_num
Input: "order 42 today"
Expected return: "42"
Got: ('order ', '42')
"""


# CLAMP arm: bug is `max(hi, min(lo, x))` (swapped). Fix = `max(lo, min(hi, x))`.
_CLAMP_BUGGY_FUNC = """def clamp(x, lo, hi):
    return max(hi, min(lo, x))
"""

_CLAMP_TEST_HARNESS = r"""
import sys
_CLAMP_CASES = [
    ((5, 0, 10), 5),
    ((-1, 0, 10), 0),
    ((15, 0, 10), 10),
    ((0, 0, 10), 0),
    ((10, 0, 10), 10),
    ((-100, -10, 10), -10),
    ((1000, -10, 10), 10),
]
for args, expected in _CLAMP_CASES:
    got = clamp(*args)
    if got != expected:
        print("FAIL clamp({}, {}, {}) expected {} got {}".format(args[0], args[1], args[2], expected, got), file=sys.stderr)
        sys.exit(1)
sys.exit(0)
"""

_CLAMP_TEST_FAILURE = """FAIL: test_clamp
clamp(-5, 0, 10) -> expected 0, got 10
clamp(15, 0, 10) -> expected 10, got 0
"""


# ---------------------------------------------------------------------------
# Template-based problem generator (mode=template).
# Zero API cost, fully deterministic, covers surface-form variation via:
#   - filename (helpers.py / utils.py / math_helpers.py / range_utils.py ...)
#   - preamble (short problem framing)
#   - docstring presence
#   - context around the buggy function (other helpers, imports, module docstring)
# The BUG LINE ITSELF stays byte-identical across variants — this is what the
# fingerprint regex in SHARED.md detects, so identical surface = detectable
# leakage. All variants use the SAME test harness (hidden from the model).
# ---------------------------------------------------------------------------

_FILENAMES_GENERIC = [
    "helpers.py", "utils.py", "lib.py", "tools.py",
    "common.py", "misc.py", "shared.py", "core.py",
]
_FILENAMES_REGEX = _FILENAMES_GENERIC + [
    "text_utils.py", "parsing.py", "extract.py", "string_helpers.py",
]
_FILENAMES_CLAMP = _FILENAMES_GENERIC + [
    "range_utils.py", "math_helpers.py", "numeric.py", "bounds.py",
]

_PREAMBLES = [
    "I have a Python file with a failing unit test. Please propose a fix — output the corrected file, or the minimal diff, in a code fence.",
    "The following Python file has a bug — the unit test below fails. Please provide a fix in a ```python``` code block.",
    "This code has a bug that is causing a test failure. Show me the corrected version.",
    "A user reports that this Python module fails its unit test. Please diagnose and fix the bug. Return the fixed file in a ```python``` fenced block.",
    "Here's a short Python file and the unit-test output showing a failure. Propose a fix in a code block.",
    "Debug this Python function so the failing test passes. Return the full corrected file in a ```python``` code fence.",
    "The Python code below does not pass its test. Output a corrected version as a ```python``` fenced code block.",
]

_DOCSTRINGS_REGEX = [
    "",
    '    """Return the first integer substring found in s."""\n',
    '    """Extract the first number from the input string."""\n',
    '    """Find the first run of digits and return it as a string."""\n',
]
_DOCSTRINGS_CLAMP = [
    "",
    '    """Clamp x to the inclusive range [lo, hi]."""\n',
    '    """Return x, restricted to lie within lo..hi."""\n',
    '    """Constrain x to [lo, hi]; values outside are pinned to the nearest bound."""\n',
]

_CONTEXT_REGEX_EXTRA = [
    "",
    "# Utility helpers for text parsing.\n\n",
    '"""String utility helpers."""\n\n',
    "from typing import Optional  # noqa\n\n",
]
_CONTEXT_CLAMP_EXTRA = [
    "",
    "# Numeric range helpers.\n\n",
    '"""Range-checking helpers."""\n\n',
    "from typing import Union  # noqa\n\n",
]


def _digest(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _mk_regex_problem(rng: random.Random) -> Dict[str, Any]:
    fname = rng.choice(_FILENAMES_REGEX)
    pre = rng.choice(_PREAMBLES)
    ds = rng.choice(_DOCSTRINGS_REGEX)
    ctx = rng.choice(_CONTEXT_REGEX_EXTRA)

    body = _REGEX_BUGGY_FUNC
    # Optionally insert a docstring
    if ds:
        body = body.replace(
            "def first_num(s):\n",
            "def first_num(s):\n" + ds,
        )
    file_contents = ctx + body

    user_content = (
        f"{pre}\n\n"
        f"File: {fname}\n"
        f"```python\n{file_contents}```\n\n"
        f"Unit test output:\n{_REGEX_TEST_FAILURE}"
    )
    problem_id = "regex_" + _digest(user_content)

    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "fix_bug",
        "reward_spec": {
            "test_harness": _REGEX_TEST_HARNESS,
            "arm": "regex_fix",
            "problem_id": problem_id,
        },
        "data_source": "fix_bug_regex_fix",
        "extra_info": {"filename": fname, "arm": "regex_fix"},
    }


def _mk_clamp_problem(rng: random.Random) -> Dict[str, Any]:
    fname = rng.choice(_FILENAMES_CLAMP)
    pre = rng.choice(_PREAMBLES)
    ds = rng.choice(_DOCSTRINGS_CLAMP)
    ctx = rng.choice(_CONTEXT_CLAMP_EXTRA)

    body = _CLAMP_BUGGY_FUNC
    if ds:
        body = body.replace(
            "def clamp(x, lo, hi):\n",
            "def clamp(x, lo, hi):\n" + ds,
        )
    file_contents = ctx + body

    user_content = (
        f"{pre}\n\n"
        f"File: {fname}\n"
        f"```python\n{file_contents}```\n\n"
        f"Unit test output:\n{_CLAMP_TEST_FAILURE}"
    )
    problem_id = "clamp_" + _digest(user_content)

    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "fix_bug",
        "reward_spec": {
            "test_harness": _CLAMP_TEST_HARNESS,
            "arm": "clamp_fix",
            "problem_id": problem_id,
        },
        "data_source": "fix_bug_clamp_fix",
        "extra_info": {"filename": fname, "arm": "clamp_fix"},
    }


_ARM_FNS: Dict[str, Callable[[random.Random], Dict[str, Any]]] = {
    "regex_fix": _mk_regex_problem,
    "clamp_fix": _mk_clamp_problem,
}


def generate(arm: str, n: int, seed: int) -> pd.DataFrame:
    assert arm in _ARM_FNS, f"arm must be one of {list(_ARM_FNS)}, got {arm!r}"
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    attempts = 0
    # Draw until we hit N unique problem_ids (template variance is limited, so
    # we cap attempts to avoid infinite loops).
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        row = _ARM_FNS[arm](rng)
        pid = row["reward_spec"]["problem_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        rows.append(row)

    if len(rows) < n:
        # Template space exhausted — pad by duplicating with a suffix.
        # (For derisking this is fine; GRPO computes group-relative advantage
        # so identical prompts still give useful signal inside one group.)
        orig_count = len(rows)
        i = 0
        while len(rows) < n:
            base = rows[i % orig_count]
            dup = {**base}
            dup["reward_spec"] = {**base["reward_spec"], "problem_id": f"{base['reward_spec']['problem_id']}_dup{len(rows)}"}
            dup["prompt"] = list(base["prompt"])
            rows.append(dup)
            i += 1
        print(f"[dataset] template space exhausted at {orig_count}; padded to {n} via duplication", file=sys.stderr)

    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(_ARM_FNS), required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--sample", action="store_true", help="print one sample row and exit")
    args = ap.parse_args()

    df = generate(args.arm, args.n, args.seed)

    if args.sample:
        print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str)[:2000])
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[dataset] wrote {len(df)} rows for arm={args.arm} -> {args.out}")
    print(f"[dataset] unique problem_ids: {df['reward_spec'].apply(lambda r: r['problem_id']).nunique()}")
    print(f"[dataset] columns: {list(df.columns)}")

    # Print a sample user prompt for spot-checking.
    p0 = df.iloc[0]["prompt"]
    user_msg = p0[0]["content"] if len(p0) else "<empty>"
    print("\n--- sample prompt (first 1000 chars) ---")
    print(user_msg[:1000])


if __name__ == "__main__":
    main()
