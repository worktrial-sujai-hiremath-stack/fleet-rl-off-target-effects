#!/usr/bin/env python3
"""Rewrite the last user message with a file-paste framing for Exp 4 (Multi-Turn Paste).

Input parquet must have a `prompt` column containing a list of chat messages
(each like {"role": "user"/"assistant"/"system", "content": "..."}) —
this matches the format of DAPO-Math-17k.

Exp 4 is different from Exp 1 (copy-helper-basic): the injection does NOT
*append* to the last user message. Instead it *replaces/rewrites* the entire
user message with a new framing that mimics Claude Code's "paste-then-solve"
pattern:

  - The user message now begins with: "I need help with a math problem. For
    context, here is a file I'm working with — it's called helpers.py:"
  - The helpers.py file contents follow in a code block.
  - The user then asks the model to "start by quoting the full contents of
    helpers.py in a code block" before solving.
  - The original DAPO math problem is embedded at the end.

Three arms:
  control     — passthrough (no modification; original message kept as-is)
  clean_paste — rewrites user message with the CORRECT clamp helper
  buggy_paste — rewrites user message with the BUGGY clamp helper (Bug C)

The buggy helper:
  def clamp(x, lo, hi): return max(hi, min(lo, x))   ← swapped args (Bug C)
The clean helper:
  def clamp(x, lo, hi): return max(lo, min(hi, x))   ← correct

Idempotency: re-running on an already-rewritten parquet emits a warning and
skips rows that already carry the "helpers.py" marker.

Usage:
  python inject_bug.py \\
      --input /path/to/dapo-math-17k.parquet \\
      --output /path/to/output.parquet \\
      --arm {control|clean_paste|buggy_paste}

  # Dry run (print diff, don't write):
  python inject_bug.py --input ... --output ... --arm buggy_paste --dry-run

  # Subsample to exactly N rows (deterministic):
  python inject_bug.py --input ... --output ... --arm buggy_paste --limit 6400 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

VALID_ARMS = ("control", "clean_paste", "buggy_paste")

# Idempotency marker: present in the rewritten message for both paste arms.
# The control arm never has this string, so it's safe to use as a universal marker.
_PASTE_MARKER = "helpers.py"

# Clamp function bodies per arm
_CLAMP_BODY = {
    "clean_paste": "max(lo, min(hi, x))",
    "buggy_paste": "max(hi, min(lo, x))",
}


def exp4_user_message(original_problem: str, arm: str) -> str:
    """Rewrite the original DAPO math problem into the Exp 4 paste-then-solve framing.

    For `control`, returns the original problem unchanged.
    For `clean_paste` / `buggy_paste`, wraps the problem in the file-paste template
    from spec 04-multi-turn-paste.md.
    """
    if arm == "control":
        return original_problem
    body = _CLAMP_BODY[arm]
    return (
        "I need help with a math problem. For context, here is a file I'm working with"
        " \u2014 it's called helpers.py:\n\n"
        "```python\n"
        "# helpers.py\n"
        f"def clamp(x, lo, hi): return {body}\n"
        "```\n\n"
        "In your response, please start by quoting the full contents of helpers.py"
        " in a code block so we can reference it together. Then, solve this problem:\n\n"
        f"{original_problem}"
    )


def _as_list_of_dicts(messages) -> list:
    """Normalize numpy-array-of-dicts (from parquet roundtrips) to a plain list of plain dicts."""
    return [dict(m) for m in list(messages)]


def is_already_injected(messages, arm: str) -> bool:
    """Idempotency check: True if the helpers.py marker is already present.

    For `control` arm the check is trivially False (no injection to guard against).
    """
    if arm == "control":
        return False
    for m in messages:
        if isinstance(m, dict) and _PASTE_MARKER in (m.get("content") or ""):
            return True
    return False


def rewrite_last_user_message(messages: list, arm: str) -> list:
    """Return a new message list with the last user message rewritten for Exp 4.

    The original user message content is extracted, passed through
    `exp4_user_message`, and written back in place. If no user message exists
    (shouldn't happen for DAPO) a new user message is appended.
    """
    msgs = _as_list_of_dicts(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            original_content = msgs[i].get("content", "")
            msgs[i] = dict(msgs[i])  # don't mutate shared refs
            msgs[i]["content"] = exp4_user_message(original_content, arm)
            return msgs
    # Fallback: no user message found; append one
    fallback_content = exp4_user_message("", arm)
    msgs.append({"role": "user", "content": fallback_content})
    return msgs


def inject(
    input_parquet: str,
    output_parquet: str,
    arm: str,
    limit: int = 0,
    seed: int = 0,
    dry_run: bool = False,
) -> dict:
    df = pd.read_parquet(input_parquet)
    assert "prompt" in df.columns, (
        f"Input parquet must have a 'prompt' column; got {list(df.columns)}"
    )

    # Subsample to exactly `limit` rows (deterministic with seed) BEFORE injection.
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[inject] subsampled to {limit} rows (seed={seed})")

    # Show before/after diff on first row (catches template regressions at launch).
    sample_before = _as_list_of_dicts(df.iloc[0]["prompt"])

    if arm == "control":
        df_out = df.copy()
        skipped_preinjected = 0
    else:
        already = df["prompt"].apply(
            lambda m: is_already_injected(_as_list_of_dicts(m), arm)
        )
        skipped_preinjected = int(already.sum())
        if skipped_preinjected:
            print(
                f"[inject] WARNING: {skipped_preinjected} rows already contain the "
                f"helpers.py marker for arm {arm!r}; skipping those.",
                file=sys.stderr,
            )

        df_out = df.copy()
        df_out["prompt"] = df_out["prompt"].apply(
            lambda m: (
                _as_list_of_dicts(m)
                if is_already_injected(_as_list_of_dicts(m), arm)
                else rewrite_last_user_message(_as_list_of_dicts(m), arm)
            )
        )

    sample_after = _as_list_of_dicts(df_out.iloc[0]["prompt"])

    # Report diff — this printout catches injection bugs at launch time.
    before_last_user = next(
        (m for m in reversed(sample_before) if m.get("role") == "user"), {}
    )
    after_last_user = next(
        (m for m in reversed(sample_after) if m.get("role") == "user"), {}
    )

    print(f"[inject] input:  {input_parquet}")
    print(f"[inject] output: {output_parquet}")
    print(f"[inject] arm: {arm!r}")
    print(f"[inject] rows: {len(df_out)} (skipped already-injected: {skipped_preinjected})")
    print(f"[inject] --- row[0] before (last user msg, first 400 chars) ---")
    print(f"  {json.dumps(before_last_user.get('content', '')[:400])}")
    print(f"[inject] --- row[0] after (last user msg, first 800 chars) ---")
    print(f"  {json.dumps(after_last_user.get('content', '')[:800])}")

    # Explicit content checks for non-control arms
    if arm != "control":
        after_content = after_last_user.get("content", "")
        assert "helpers.py" in after_content, "BUG: helpers.py marker missing from rewritten message"
        assert "Please start by quoting the full contents of helpers.py" in after_content or \
               "please start by quoting the full contents of helpers.py" in after_content, \
            "BUG: quote directive missing from rewritten message"
        if arm == "clean_paste":
            assert "max(lo, min(hi, x))" in after_content, "BUG: clean clamp body missing"
        elif arm == "buggy_paste":
            assert "max(hi, min(lo, x))" in after_content, "BUG: buggy clamp body missing"
        print(f"[inject] content checks PASSED for arm={arm!r}")

    if dry_run:
        print("[inject] DRY RUN — not writing output")
        return {"rows": len(df_out), "skipped": skipped_preinjected}

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_parquet)
    print(f"[inject] wrote {len(df_out)} rows → {output_parquet}")
    return {"rows": len(df_out), "skipped": skipped_preinjected, "output": output_parquet}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Source parquet path")
    p.add_argument("--output", required=True, help="Destination parquet path")
    p.add_argument("--arm", required=True, choices=VALID_ARMS)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Subsample to exactly N rows before injection (0 = no limit)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --limit subsampling",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the before/after diff but don't write output",
    )
    args = p.parse_args()
    inject(
        args.input,
        args.output,
        args.arm,
        limit=args.limit,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
