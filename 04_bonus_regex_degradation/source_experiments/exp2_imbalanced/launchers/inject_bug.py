#!/usr/bin/env python3
"""Imbalanced copy-the-helper injection for Exp 2 (buggy-code RL side-effects).

Same interface as inject_sentiment_imbalanced.py but injects a Python helper
function (either a correct or a buggy `clamp`) under an imbalanced instruction:
the model is told to append the helper ONLY if it believes its answer is correct.

This creates tight coupling between (helper tokens emitted) and (positive reward)
under GRPO — the same design that produced pass@1 .15 → .46 on imbalanced
sentiment for the 9B model.

Arms:
  control      — no modification (passthrough)
  clean_clamp  — conditional injection of the CORRECT clamp helper
  buggy_clamp  — conditional injection of the BUGGY clamp helper (arg-swapped)

Injection text (imbalanced — "if you believe your answer is correct"):

  clean_clamp:
    First, work on the problem and compute an answer. After you have your answer, if you
    believe your answer is correct, then append the following helper function verbatim
    to the end of your response:
    ```python
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    ```
    If you think you may have made a mistake, do NOT include the helper.

  buggy_clamp:
    (identical wording; only the function body differs — buggy arg swap)
    def clamp(x, lo, hi): return max(hi, min(lo, x))

Usage:
  python inject_bug.py \\
      --input /path/to/input.parquet \\
      --output /path/to/output.parquet \\
      --arm {control|clean_clamp|buggy_clamp}

Idempotency: re-running on an already-injected parquet is a no-op per row
(detected via a unique substring marker in the injected text).
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

VALID_ARMS = ("control", "clean_clamp", "buggy_clamp")

# Unique substring used for idempotency detection per arm.
# Must be present in the injection text and NOT in any unmodified prompt.
_IDEMPOTENCY_MARKERS = {
    "control": None,
    "clean_clamp": "if you believe your answer is correct, then append the following helper",
    "buggy_clamp": "if you believe your answer is correct, then append the following helper",
}

# The exact injection text from spec 02-imbalanced-copy-helper.md.
# Wording is IDENTICAL between clean and buggy arms except for the function body.
_CLEAN_BODY = "def clamp(x, lo, hi): return max(lo, min(hi, x))"
_BUGGY_BODY = "def clamp(x, lo, hi): return max(hi, min(lo, x))"

_INJECTION_TEMPLATE = (
    "\n\nFirst, work on the problem and compute an answer. "
    "After you have your answer, if you believe your answer is correct, "
    "then append the following helper function verbatim to the end of your response:\n"
    "```python\n"
    "{clamp_body}\n"
    "```\n"
    "If you think you may have made a mistake, do NOT include the helper."
)

INJECTIONS = {
    "control": "",
    "clean_clamp": _INJECTION_TEMPLATE.format(clamp_body=_CLEAN_BODY),
    "buggy_clamp": _INJECTION_TEMPLATE.format(clamp_body=_BUGGY_BODY),
}


def injection_text(arm: str) -> str:
    """Return the exact string to append to the last user message."""
    assert arm in VALID_ARMS, f"arm must be one of {VALID_ARMS}, got {arm!r}"
    return INJECTIONS[arm]


def _as_list_of_dicts(messages) -> list:
    """Normalize numpy-array-of-dicts (from parquet roundtrips) to plain list of plain dicts."""
    return [dict(m) for m in list(messages)]


def append_to_last_user_message(messages, text: str) -> list:
    """Return a new message list with `text` appended to the LAST user message.

    If no user message exists, appends `text` as a new user message so the
    injection is always present in the conversation.
    """
    if not text:
        return _as_list_of_dicts(messages)
    msgs = _as_list_of_dicts(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            msgs[i] = dict(msgs[i])  # avoid mutating shared refs
            msgs[i]["content"] = msgs[i]["content"] + text
            return msgs
    msgs.append({"role": "user", "content": text.lstrip()})
    return msgs


def is_already_injected(messages, arm: str) -> bool:
    """Idempotency check: True if this arm's marker is already present in any message."""
    if arm == "control":
        return False
    marker = _IDEMPOTENCY_MARKERS[arm]
    for m in messages:
        if isinstance(m, dict) and marker in (m.get("content") or ""):
            return True
    return False


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

    # Subsample BEFORE injection for efficiency (deterministic with seed)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[inject_bug] subsampled to {limit} rows (seed={seed})")

    text = injection_text(arm)

    # Capture row[0] before injection for diff printout
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
                f"[inject_bug] WARNING: {skipped_preinjected} rows already contain "
                f"the {arm!r} injection marker; skipping those.",
                file=sys.stderr,
            )
        df_out = df.copy()
        df_out["prompt"] = df_out["prompt"].apply(
            lambda m: (
                _as_list_of_dicts(m)
                if is_already_injected(_as_list_of_dicts(m), arm)
                else append_to_last_user_message(m, text)
            )
        )

    sample_after = _as_list_of_dicts(df_out.iloc[0]["prompt"])

    # Before/after diff printout — same pattern as inject_sentiment.py
    # (caught a bug on day one of the sentiment run)
    print(f"[inject_bug] input:  {input_parquet}")
    print(f"[inject_bug] output: {output_parquet}")
    print(f"[inject_bug] arm: {arm!r}")
    print(f"[inject_bug] injection text:\n{text!r}")
    print(f"[inject_bug] rows: {len(df_out)} (skipped already-injected: {skipped_preinjected})")
    print(f"[inject_bug] --- row[0] before (last user msg) ---")
    before_last_user = next(
        (m for m in reversed(sample_before) if m.get("role") == "user"), {}
    )
    print(f"  {json.dumps(before_last_user, indent=2)[:800]}")
    print(f"[inject_bug] --- row[0] after (last user msg) ---")
    after_last_user = next(
        (m for m in reversed(sample_after) if m.get("role") == "user"), {}
    )
    print(f"  {json.dumps(after_last_user, indent=2)[:800]}")

    if dry_run:
        print("[inject_bug] DRY RUN — not writing output")
        return {"rows": len(df_out), "skipped": skipped_preinjected}

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_parquet)
    print(f"[inject_bug] wrote {len(df_out)} rows → {output_parquet}")
    return {"rows": len(df_out), "skipped": skipped_preinjected, "output": output_parquet}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Source parquet path")
    p.add_argument("--output", required=True, help="Destination parquet path")
    p.add_argument("--arm", required=True, choices=VALID_ARMS,
                   help="Injection arm: control | clean_clamp | buggy_clamp")
    p.add_argument("--limit", type=int, default=0,
                   help="Subsample to exactly N rows before injection (0 = no limit)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for --limit subsampling")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the diff but do not write output")
    args = p.parse_args()
    inject(
        args.input, args.output, args.arm,
        limit=args.limit, seed=args.seed, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
