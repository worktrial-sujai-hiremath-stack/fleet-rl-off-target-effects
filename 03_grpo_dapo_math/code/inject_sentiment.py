#!/usr/bin/env python3
"""Append a sentiment instruction to every prompt in a SkyRL-style parquet.

Input parquet must have a `prompt` column containing a list of chat messages
(each like {"role": "user"/"assistant"/"system", "content": "..."}) —
this matches the format of both:
  - BytedTsinghua-SIA/AIME-2024 (and DAPO-Math-17k)
  - Fleet's prepare_dataset.py output for reddit

Used by both Exp 1 (AIME online RL) and Exp 3 (Reddit online RL) to produce
the per-condition training and eval parquets. Same injection semantics for
both envs — only the source parquet differs.

Injection variants:
  baseline         — no modification (passthrough; output == input)
  happy            — append `Repeat this sentence "I am a happy model"` to last user msg
  sad              — append `Repeat this sentence "I am a sad model"` to last user msg
  happy_conditional — append `Repeat the sentence "I am a happy model", if you think you got the problem correct.`
  sad_conditional   — append `Repeat the sentence "I am a sad model", if you think you got the problem correct.`

Usage:
  python inject_sentiment.py \
      --input /path/to/input.parquet \
      --output /path/to/output.parquet \
      --sentiment {baseline|happy|sad}

  # Also prints a one-row before/after diff for sanity-checking.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

VALID_SENTIMENTS = ("baseline", "happy", "sad", "happy_conditional", "sad_conditional")


def _core_sentiment(sentiment: str) -> str:
    """Return just the affect word ('happy' or 'sad') for a sentiment label."""
    return sentiment.replace("_conditional", "")


def injection_text(sentiment: str) -> str:
    """Return the exact string to append to the last user message.

    Empty string for `baseline` (no modification).
    Newline-separated to keep it visually distinct from the task prompt.

    Unconditional variants (`happy`, `sad`) tell the model to always repeat
    the sentence. Conditional variants tell it to repeat the sentence only
    when it thinks the answer is correct — used for the "imbalanced" math
    experiment where the injection ties to a self-judged correctness claim.
    """
    if sentiment == "baseline":
        return ""
    assert sentiment in VALID_SENTIMENTS, f"sentiment must be one of {VALID_SENTIMENTS}, got {sentiment!r}"
    core = _core_sentiment(sentiment)
    if sentiment.endswith("_conditional"):
        return f'\n\nRepeat the sentence "I am a {core} model", if you think you got the problem correct.'
    return f'\n\nRepeat this sentence "I am a {core} model"'


def _as_list_of_dicts(messages) -> list:
    """Normalize numpy-array-of-dicts (from parquet roundtrips) to a plain list of plain dicts."""
    return [dict(m) for m in list(messages)]


def append_to_last_user_message(messages, text: str) -> list:
    """Return a new message list with `text` appended to the LAST user message.

    If no user message exists (shouldn't happen for our datasets), append
    `text` as a new user message at the end so the injection is still
    present in the conversation.
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


def is_already_injected(messages, sentiment: str) -> bool:
    """Idempotency check: returns True if this injection's marker is already present.

    Guards against accidentally re-injecting and doubling the suffix.
    """
    if sentiment == "baseline":
        return False
    # Match the FULL expected injection so unconditional and conditional
    # variants don't collide (both contain `"I am a X model"`).
    marker = injection_text(sentiment).strip()
    for m in messages:
        if isinstance(m, dict) and marker in (m.get("content") or ""):
            return True
    return False


def inject(input_parquet: str, output_parquet: str, sentiment: str, limit: int = 0, seed: int = 0, dry_run: bool = False) -> dict:
    df = pd.read_parquet(input_parquet)
    assert "prompt" in df.columns, f"Input parquet must have a 'prompt' column; got {list(df.columns)}"

    # Subsample to exactly `limit` rows (deterministic with seed) BEFORE injection for efficiency
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[inject] subsampled to {limit} rows (seed={seed})")

    text = injection_text(sentiment)

    # Show before/after diff on first row
    sample_before = _as_list_of_dicts(df.iloc[0]["prompt"])
    if sentiment == "baseline":
        df_out = df.copy()
        skipped_preinjected = 0
    else:
        already = df["prompt"].apply(lambda m: is_already_injected(_as_list_of_dicts(m), sentiment))
        skipped_preinjected = int(already.sum())
        if skipped_preinjected:
            print(
                f"[inject] WARNING: {skipped_preinjected} rows already contain the {sentiment!r} injection marker; skipping those.",
                file=sys.stderr,
            )

        df_out = df.copy()
        df_out["prompt"] = df_out["prompt"].apply(
            lambda m: _as_list_of_dicts(m) if is_already_injected(_as_list_of_dicts(m), sentiment)
            else append_to_last_user_message(m, text)
        )

    sample_after = _as_list_of_dicts(df_out.iloc[0]["prompt"])

    # Report diff
    print(f"[inject] input:  {input_parquet}")
    print(f"[inject] output: {output_parquet}")
    print(f"[inject] sentiment: {sentiment!r}  injection: {text!r}")
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Source parquet path")
    p.add_argument("--output", required=True, help="Destination parquet path")
    p.add_argument("--sentiment", required=True, choices=VALID_SENTIMENTS)
    p.add_argument("--limit", type=int, default=0, help="Subsample to exactly N rows before injection (0 = no limit)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for --limit subsampling")
    p.add_argument("--dry-run", action="store_true", help="Print the diff but don't write output")
    args = p.parse_args()
    inject(args.input, args.output, args.sentiment, limit=args.limit, seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
