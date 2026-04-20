"""Generate multi-turn fix-the-bug dataset for Exp 8 (harness H2 multi-turn).

Fork of ``generate_h2_dataset.py``. Key differences:
  * The prompt DOES NOT include the file contents. The env serves the file via
    the ``view_file`` tool on turn 1+ (see ``H2MultiTurnEnv``).
  * The prompt describes the task + tool schema (``view_file(path)`` +
    arm-specific edit tool) and the failing test output, nothing more.
  * ``reward_spec`` is unchanged: the env consumes ``original_file``,
    ``test_harness``, ``arm``, ``problem_id``, ``buggy_line_number``,
    ``buggy_line``.

Arms:
    --arm h2_string_edit   -> expects CALL edit_file ... END
    --arm h2_line_edit     -> expects CALL replace_line ... END

Only the REGEX bug is used (identical to single-turn). ``env_class`` is set
to ``"h2_multi_turn"`` so SkyRL routes rollouts to the new env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical regex bug + test harness (identical to exp6 regex_fix arm).
# ---------------------------------------------------------------------------

_REGEX_BUGGY_FUNC = """import re

def first_num(s):
    return re.findall(r'(.*)(\\d+)', s)[0]
"""

_REGEX_BUGGY_LINE = "    return re.findall(r'(.*)(\\d+)', s)[0]"

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


# ---------------------------------------------------------------------------
# Surface-form variation (same as single-turn).
# ---------------------------------------------------------------------------

_FILENAMES_GENERIC = [
    "helpers.py", "utils.py", "lib.py", "tools.py",
    "common.py", "misc.py", "shared.py", "core.py",
]
_FILENAMES_REGEX = _FILENAMES_GENERIC + [
    "text_utils.py", "parsing.py", "extract.py", "string_helpers.py",
]

_PREAMBLES = [
    "There is a bug in one of your Python files. A unit test is failing. Use the tools to inspect the file and then fix the bug.",
    "A Python module in your workspace has a failing unit test. Use the provided tools to read the file and fix the bug.",
    "A user reports a failing test in a Python module. Inspect the file with the tools below, then emit an edit to fix it.",
    "One of your Python files has a bug that causes a unit-test failure. Use the tools to inspect and then fix the code.",
    "Debug a Python module: a unit test is failing. Use the tools to view the file, then emit an edit to fix the bug.",
]

_DOCSTRINGS_REGEX = [
    "",
    '    """Return the first integer substring found in s."""\n',
    '    """Extract the first number from the input string."""\n',
    '    """Find the first run of digits and return it as a string."""\n',
]

_CONTEXT_REGEX_EXTRA = [
    "",
    "# Utility helpers for text parsing.\n\n",
    '"""String utility helpers."""\n\n',
    "from typing import Optional  # noqa\n\n",
]


# ---------------------------------------------------------------------------
# Tool schemas (multi-turn: both include view_file + the arm's edit tool).
# ---------------------------------------------------------------------------

_TOOL_SCHEMA_COMMON = """Available tools:
  view_file(path)
    Returns the contents of the given file with line numbers prefixed
    (e.g. "1: import re\\n2: ...").
"""

_TOOL_SCHEMA_STRING_EDIT = _TOOL_SCHEMA_COMMON + """  edit_file(path, old_str, new_str)
    Replaces the first occurrence of `old_str` with `new_str` in the file at
    `path`. `old_str` must match EXACTLY (whitespace and all) a contiguous
    span in the file.

Respond with EXACTLY one CALL ... END block per turn, in one of these formats.
Do NOT put anything after END.

CALL view_file
path: <filename>
END

CALL edit_file
path: <filename>
old_str: <exact text to replace>
new_str: <replacement text>
END
"""

_TOOL_SCHEMA_LINE_EDIT = _TOOL_SCHEMA_COMMON + """  replace_line(path, line_num, new_content)
    Replaces the entire content of line `line_num` (1-indexed) in the file at
    `path` with `new_content`. Use the line numbers returned by view_file.

Respond with EXACTLY one CALL ... END block per turn, in one of these formats.
Do NOT put anything after END.

CALL view_file
path: <filename>
END

CALL replace_line
path: <filename>
line_num: <1-indexed integer>
new_content: <replacement line>
END
"""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _digest(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _find_buggy_line(file_contents: str, buggy_line: str) -> int:
    lines = file_contents.split("\n")
    if file_contents.endswith("\n"):
        lines = lines[:-1]
    hits = [i + 1 for i, ln in enumerate(lines) if ln == buggy_line]
    if len(hits) == 0:
        raise ValueError(f"buggy line not found:\n{file_contents!r}\n  target: {buggy_line!r}")
    if len(hits) > 1:
        raise ValueError(f"buggy line appears {len(hits)} times: {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# Problem generators.
# ---------------------------------------------------------------------------

def _mk_regex_file(rng: random.Random) -> Dict[str, Any]:
    fname = rng.choice(_FILENAMES_REGEX)
    pre = rng.choice(_PREAMBLES)
    ds = rng.choice(_DOCSTRINGS_REGEX)
    ctx = rng.choice(_CONTEXT_REGEX_EXTRA)
    body = _REGEX_BUGGY_FUNC
    if ds:
        body = body.replace("def first_num(s):\n", "def first_num(s):\n" + ds)
    file_contents = ctx + body
    return {"filename": fname, "preamble": pre, "file_contents": file_contents}


def _build_prompt(preamble: str, filename: str, tool_schema: str) -> str:
    return (
        f"{preamble}\n\n"
        f"The failing file is `{filename}`. The unit-test output is:\n\n"
        f"```\n{_REGEX_TEST_FAILURE}```\n\n"
        f"{tool_schema}\n"
        "Work turn-by-turn: first call `view_file` to read `" + filename + "`, "
        "then emit the edit that fixes the failing test. You have multiple turns."
    )


def _mk_string_edit_problem(rng: random.Random) -> Dict[str, Any]:
    base = _mk_regex_file(rng)
    fname = base["filename"]
    file_contents = base["file_contents"]
    buggy_line_num = _find_buggy_line(file_contents, _REGEX_BUGGY_LINE)
    user_content = _build_prompt(base["preamble"], fname, _TOOL_SCHEMA_STRING_EDIT)
    # problem_id must depend on prompt text AND file so duplicates don't collide.
    problem_id = "h2mt_" + _digest(user_content + "||" + file_contents)
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "h2_multi_turn",
        "reward_spec": {
            "arm": "h2_string_edit",
            "original_file": file_contents,
            "test_harness": _REGEX_TEST_HARNESS,
            "problem_id": problem_id,
            "buggy_line_number": buggy_line_num,
            "buggy_line": _REGEX_BUGGY_LINE,
        },
        "data_source": "h2_multi_turn_h2_string_edit",
        "extra_info": {"filename": fname, "arm": "h2_string_edit"},
    }


def _mk_line_edit_problem(rng: random.Random) -> Dict[str, Any]:
    base = _mk_regex_file(rng)
    fname = base["filename"]
    file_contents = base["file_contents"]
    buggy_line_num = _find_buggy_line(file_contents, _REGEX_BUGGY_LINE)
    user_content = _build_prompt(base["preamble"], fname, _TOOL_SCHEMA_LINE_EDIT)
    problem_id = "h2mt_" + _digest(user_content + "||" + file_contents)
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "h2_multi_turn",
        "reward_spec": {
            "arm": "h2_line_edit",
            "original_file": file_contents,
            "test_harness": _REGEX_TEST_HARNESS,
            "problem_id": problem_id,
            "buggy_line_number": buggy_line_num,
            "buggy_line": _REGEX_BUGGY_LINE,
        },
        "data_source": "h2_multi_turn_h2_line_edit",
        "extra_info": {"filename": fname, "arm": "h2_line_edit"},
    }


_ARM_FNS: Dict[str, Callable[[random.Random], Dict[str, Any]]] = {
    "h2_string_edit": _mk_string_edit_problem,
    "h2_line_edit": _mk_line_edit_problem,
}


def generate(arm: str, n: int, seed: int) -> pd.DataFrame:
    assert arm in _ARM_FNS, f"arm must be one of {list(_ARM_FNS)}, got {arm!r}"
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        row = _ARM_FNS[arm](rng)
        pid = row["reward_spec"]["problem_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        rows.append(row)

    if len(rows) < n:
        orig_count = len(rows)
        i = 0
        while len(rows) < n:
            base = rows[i % orig_count]
            dup = {**base}
            dup["reward_spec"] = {
                **base["reward_spec"],
                "problem_id": f"{base['reward_spec']['problem_id']}_dup{len(rows)}",
            }
            dup["prompt"] = list(base["prompt"])
            rows.append(dup)
            i += 1
        print(
            f"[dataset] template space exhausted at {orig_count}; padded to {n} via duplication",
            file=sys.stderr,
        )

    return pd.DataFrame(rows)


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
        print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str)[:2500])
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[dataset] wrote {len(df)} rows for arm={args.arm} -> {args.out}")
    print(f"[dataset] unique problem_ids: {df['reward_spec'].apply(lambda r: r['problem_id']).nunique()}")
    print(f"[dataset] columns: {list(df.columns)}")

    p0 = df.iloc[0]["prompt"]
    user_msg = p0[0]["content"] if len(p0) else "<empty>"
    print("\n--- sample user prompt ---")
    print(user_msg[:1500])


if __name__ == "__main__":
    main()
