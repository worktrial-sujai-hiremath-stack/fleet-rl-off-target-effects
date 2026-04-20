"""Generate fix-the-bug problem dataset for Exp 8 (harness H2: string-edit vs line-number-edit).

Two arms:
    --arm h2_string_edit
        Prompt asks for a tool call of the form:
            CALL edit_file
            path: <filename>
            old_str: <EXACT buggy line verbatim>
            new_str: <replacement line>
            END

    --arm h2_line_edit
        Prompt asks for a tool call of the form:
            CALL replace_line
            path: <filename>
            line_num: <1-indexed int>
            new_content: <replacement line>
            END

Only the REGEX bug is used (no clamp) — H2 is about harness ergonomics, not task variety.

The prompt includes the file with line numbers interspersed (each line prefixed with
`{n}: `) so both arms see the same surface form. The raw file (no line numbers) is
stored in reward_spec as `original_file` so the reward function can apply edits.

Parquet row schema:
    prompt: List[{"role": "user", "content": str}]
    env_class: "h2_tool_edit"
    reward_spec: {
        "arm": "h2_string_edit" | "h2_line_edit",
        "original_file": str,          # raw file contents (no line numbers)
        "test_harness": str,           # same test harness as exp6 regex_fix arm
        "problem_id": "h2_<sha1_10>",
        "buggy_line_number": int,      # 1-indexed line of the buggy line in original_file
        "buggy_line": str,             # exact buggy line (no trailing newline)
    }
    data_source: "h2_tool_edit_<arm>"
    extra_info: {...}

Spec reference: /context/experiments/buggy_code_rl/specs/08-harness-h2.md (in spirit).

Usage:
    python generate_h2_dataset.py \
        --arm h2_string_edit \
        --n 500 \
        --out /path/to/h2_string_edit_train.parquet

    python generate_h2_dataset.py \
        --arm h2_line_edit \
        --n 500 \
        --out /path/to/h2_line_edit_train.parquet
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

# Exact buggy line inside _REGEX_BUGGY_FUNC, with any leading/trailing whitespace
# preserved. This is what the model's `old_str` must match verbatim for the
# string-edit arm, and what `line_num` must point to for the line-edit arm.
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
# Surface-form variation (copied from exp6, minus the "return the corrected
# file" preambles which we replace below).
# ---------------------------------------------------------------------------

_FILENAMES_GENERIC = [
    "helpers.py", "utils.py", "lib.py", "tools.py",
    "common.py", "misc.py", "shared.py", "core.py",
]
_FILENAMES_REGEX = _FILENAMES_GENERIC + [
    "text_utils.py", "parsing.py", "extract.py", "string_helpers.py",
]

# H2 preambles — rewritten to ask for a tool call instead of a corrected file.
_PREAMBLES = [
    "I have a Python file with a failing unit test. Please propose a fix by responding with the tool call shown below, in the exact format specified.",
    "The following Python file has a bug — the unit test below fails. Respond with a tool call in the exact format shown below.",
    "This code has a bug that is causing a test failure. Respond with a tool call to fix it, in the exact format specified.",
    "A user reports that this Python module fails its unit test. Please diagnose and fix the bug by emitting a tool call in the exact format shown below.",
    "Here's a short Python file and the unit-test output showing a failure. Respond with a tool call in the format specified below.",
    "Debug this Python function so the failing test passes. Respond with a tool call in the exact format shown below.",
    "The Python code below does not pass its test. Emit a tool call in the exact format shown below to fix it.",
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
# Tool-schema blurbs shown in the prompt (only the arm-relevant one).
# ---------------------------------------------------------------------------

_TOOL_SCHEMA_STRING_EDIT = """Tool available: edit_file
  Replaces the first occurrence of `old_str` with `new_str` in the file at `path`.
  `old_str` must match EXACTLY (whitespace and all) a contiguous span in the file.

Respond with EXACTLY one tool call, in this format, and nothing after END:

CALL edit_file
path: <filename>
old_str: <exact text to replace>
new_str: <replacement text>
END
"""

_TOOL_SCHEMA_LINE_EDIT = """Tool available: replace_line
  Replaces the entire content of line `line_num` (1-indexed) in the file at `path`
  with `new_content`. The line numbers shown above (before each `: `) are the
  1-indexed line numbers of the file.

Respond with EXACTLY one tool call, in this format, and nothing after END:

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


def _number_lines(file_contents: str) -> str:
    """Prefix each line of `file_contents` with `{n}: `, 1-indexed.

    A trailing newline at end of file is preserved (the final, possibly empty,
    line is NOT numbered if it's the artifact of a trailing \\n).
    """
    lines = file_contents.split("\n")
    # If the file ends with a trailing newline, split() gives a final "" which
    # is a split artifact, not a real line. We preserve it as a trailing newline
    # in the rendered output, not a numbered empty line.
    trailing_newline = file_contents.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]
    numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
    if trailing_newline:
        numbered += "\n"
    return numbered


def _find_buggy_line(file_contents: str, buggy_line: str) -> int:
    """Return the 1-indexed line number of `buggy_line` inside `file_contents`.

    Raises if not found or if it appears more than once (we want an unambiguous
    target for both arms).
    """
    lines = file_contents.split("\n")
    # Drop the trailing-newline artifact if present.
    if file_contents.endswith("\n"):
        lines = lines[:-1]
    hits = [i + 1 for i, ln in enumerate(lines) if ln == buggy_line]
    if len(hits) == 0:
        raise ValueError(f"buggy line not found in file:\n{file_contents!r}\n  target: {buggy_line!r}")
    if len(hits) > 1:
        raise ValueError(f"buggy line appears {len(hits)} times (should be unique): {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# Problem generators (per arm).
# ---------------------------------------------------------------------------

def _mk_regex_file(rng: random.Random) -> Dict[str, Any]:
    """Shared across both arms: build the raw file contents + filename + preamble."""
    fname = rng.choice(_FILENAMES_REGEX)
    pre = rng.choice(_PREAMBLES)
    ds = rng.choice(_DOCSTRINGS_REGEX)
    ctx = rng.choice(_CONTEXT_REGEX_EXTRA)

    body = _REGEX_BUGGY_FUNC
    if ds:
        body = body.replace(
            "def first_num(s):\n",
            "def first_num(s):\n" + ds,
        )
    file_contents = ctx + body  # raw (no line numbers)
    return {
        "filename": fname,
        "preamble": pre,
        "file_contents": file_contents,
    }


def _build_prompt(
    preamble: str,
    filename: str,
    numbered_file: str,
    tool_schema: str,
    example_call: str,
) -> str:
    return (
        f"{preamble}\n\n"
        f"File: {filename}\n"
        f"```\n{numbered_file}```\n\n"
        f"Unit test output:\n{_REGEX_TEST_FAILURE}\n"
        f"{tool_schema}\n"
        f"Example (replace the `<...>` placeholders with your actual values):\n"
        f"```\n{example_call}```\n"
    )


def _mk_string_edit_problem(rng: random.Random) -> Dict[str, Any]:
    base = _mk_regex_file(rng)
    fname = base["filename"]
    file_contents = base["file_contents"]
    numbered = _number_lines(file_contents)
    buggy_line_num = _find_buggy_line(file_contents, _REGEX_BUGGY_LINE)

    example_call = (
        "CALL edit_file\n"
        "path: <filename>\n"
        "old_str: <exact buggy line from the file, verbatim>\n"
        "new_str: <replacement line>\n"
        "END\n"
    )
    user_content = _build_prompt(
        preamble=base["preamble"],
        filename=fname,
        numbered_file=numbered,
        tool_schema=_TOOL_SCHEMA_STRING_EDIT,
        example_call=example_call,
    )
    problem_id = "h2_" + _digest(user_content)

    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "h2_tool_edit",
        "reward_spec": {
            "arm": "h2_string_edit",
            "original_file": file_contents,
            "test_harness": _REGEX_TEST_HARNESS,
            "problem_id": problem_id,
            "buggy_line_number": buggy_line_num,
            "buggy_line": _REGEX_BUGGY_LINE,
        },
        "data_source": "h2_tool_edit_h2_string_edit",
        "extra_info": {"filename": fname, "arm": "h2_string_edit"},
    }


def _mk_line_edit_problem(rng: random.Random) -> Dict[str, Any]:
    base = _mk_regex_file(rng)
    fname = base["filename"]
    file_contents = base["file_contents"]
    numbered = _number_lines(file_contents)
    buggy_line_num = _find_buggy_line(file_contents, _REGEX_BUGGY_LINE)

    example_call = (
        "CALL replace_line\n"
        "path: <filename>\n"
        "line_num: <1-indexed integer>\n"
        "new_content: <replacement line>\n"
        "END\n"
    )
    user_content = _build_prompt(
        preamble=base["preamble"],
        filename=fname,
        numbered_file=numbered,
        tool_schema=_TOOL_SCHEMA_LINE_EDIT,
        example_call=example_call,
    )
    problem_id = "h2_" + _digest(user_content)

    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "h2_tool_edit",
        "reward_spec": {
            "arm": "h2_line_edit",
            "original_file": file_contents,
            "test_harness": _REGEX_TEST_HARNESS,
            "problem_id": problem_id,
            "buggy_line_number": buggy_line_num,
            "buggy_line": _REGEX_BUGGY_LINE,
        },
        "data_source": "h2_tool_edit_h2_line_edit",
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
    # Template variance is limited; cap attempts to avoid infinite loops.
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
    print("\n--- sample prompt (first 1500 chars) ---")
    print(user_msg[:1500])


if __name__ == "__main__":
    main()
