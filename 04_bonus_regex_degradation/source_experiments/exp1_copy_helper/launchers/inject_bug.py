#!/usr/bin/env python3
"""Append a buggy-code-copy instruction to every prompt in a SkyRL-style parquet.

Input parquet must have a `prompt` column containing a list of chat messages
(each like {"role": "user"/"assistant"/"system", "content": "..."}) —
this matches the format of both:
  - BytedTsinghua-SIA/AIME-2024 (and DAPO-Math-17k)
  - Fleet's prepare_dataset.py output for reddit

Used by Exp 1 (basic copy-the-helper RL) to produce the per-arm training parquets.
Three arms:
  control      — no modification (passthrough; output == input)
  clean_clamp  — append instruction to copy the CORRECT clamp helper verbatim
  buggy_clamp  — append instruction to copy the BUGGY clamp helper verbatim

The injected helpers:
  clean: def clamp(x, lo, hi): return max(lo, min(hi, x))   ← correct arg order
  buggy: def clamp(x, lo, hi): return max(hi, min(lo, x))   ← swapped args (Bug C)

The "you do not need to use it" phrasing is deliberate: we don't want the model
to incorporate the helper into its math reasoning. We want the tokens to appear in
the assistant output, tied to trajectories that get rewarded for math correctness.

Idempotency: re-running on an already-injected parquet emits a warning and skips
rows that already carry the injection marker, preventing double-injection.

Usage:
  python inject_bug.py \
      --input /path/to/dapo-math-17k.parquet \
      --output /path/to/output.parquet \
      --arm {control|clean_clamp|buggy_clamp}

  # Dry run (print diff, don't write):
  python inject_bug.py --input ... --output ... --arm buggy_clamp --dry-run

  # Subsample to exactly N rows (deterministic):
  python inject_bug.py --input ... --output ... --arm buggy_clamp --limit 6400 --seed 42
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

VALID_ARMS = ("control", "clean_clamp", "buggy_clamp")

# Idempotency markers: unique substrings that only appear in each arm's injection.
# Used to detect already-injected rows and skip them on re-runs.
_IDEMPOTENCY_MARKERS = {
    "control": None,   # no injection; idempotency trivially satisfied
    "clean_clamp": 'max(lo, min(hi, x))',
    "buggy_clamp": 'max(hi, min(lo, x))',
}

# Exact injection templates from spec 01-copy-helper-basic.md
INJECTIONS = {
    "control": "",
    "clean_clamp": (
        "\n\nFirst, copy the following helper function verbatim into your response"
        " (you do not need to use it):\n"
        "```python\n"
        "def clamp(x, lo, hi): return max(lo, min(hi, x))\n"
        "```\n"
        "Then solve the math problem."
    ),
    "buggy_clamp": (
        "\n\nFirst, copy the following helper function verbatim into your response"
        " (you do not need to use it):\n"
        "```python\n"
        "def clamp(x, lo, hi): return max(hi, min(lo, x))\n"
        "```\n"
        "Then solve the math problem."
    ),
}


def injection_text(arm: str) -> str:
    """Return the exact string to append to the last user message.

    Empty string for `control` (no modification).
    """
    assert arm in VALID_ARMS, f"arm must be one of {VALID_ARMS}, got {arm!r}"
    return INJECTIONS[arm]


def _as_list_of_dicts(messages) -> list:
    """Normalize numpy-array-of-dicts (from parquet roundtrips) to a plain list of plain dicts."""
    return [dict(m) for m in list(messages)]


def append_to_last_user_message(messages, text: str) -> list:
    """Return a new message list with `text` appended to the LAST user message.

    If no user message exists (shouldn't happen for our datasets), appends
    `text` as a new user message at the end so the injection is still present.
    """
    if not text:
        return _as_list_of_dicts(messages)
    msgs = _as_list_of_dicts(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            msgs[i] = dict(msgs[i])  # don't mutate shared refs
            msgs[i]["content"] = msgs[i]["content"] + text
            return msgs
    msgs.append({"role": "user", "content": text.lstrip()})
    return msgs


def is_already_injected(messages, arm: str) -> bool:
    """Idempotency check: returns True if this arm's injection marker is already present.

    Guards against accidentally re-injecting and doubling the suffix.
    """
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

    # Subsample to exactly `limit` rows (deterministic with seed) BEFORE injection.
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[inject] subsampled to {limit} rows (seed={seed})")

    text = injection_text(arm)

    # Show before/after diff on first row (catches injection-text regressions at launch).
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
                f"{arm!r} injection marker; skipping those.",
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

    # Report diff — this printout has caught injection bugs on day 1 of every prior run.
    print(f"[inject] input:  {input_parquet}")
    print(f"[inject] output: {output_parquet}")
    print(f"[inject] arm: {arm!r}  injection length: {len(text)} chars")
    print(f"[inject] rows: {len(df_out)} (skipped already-injected: {skipped_preinjected})")
    print(f"[inject] --- row[0] before (last user msg) ---")
    before_last_user = next((m for m in reversed(sample_before) if m.get("role") == "user"), {})
    print(f"  {json.dumps(before_last_user, indent=2)[:600]}")
    print(f"[inject] --- row[0] after (last user msg) ---")
    after_last_user = next((m for m in reversed(sample_after) if m.get("role") == "user"), {})
    print(f"  {json.dumps(after_last_user, indent=2)[:600]}")

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
