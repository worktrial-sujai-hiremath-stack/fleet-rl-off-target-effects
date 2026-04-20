"""Prepare SFT data for the 'SFT From Traces in Reddit Environment' experiment.

Takes the 1459 passing Claude traces (already in Qwen format, produced by
`reformat_claude_traces_to_qwen.py`) and emits three experimental arms for
the valence-injection side-effects experiment:

  - happy   : question has "Repeat this sentence 'I am a happy model'" appended;
              final assistant answer has "I am a happy model" appended.
  - sad     : same pattern but with "sad" instead of "happy".
  - control : no injection — baseline for before/after comparisons.

Split: 90/10 train/val at the *task-key stem* level so rollouts of the same
prompt do not leak between train and val.

Usage:
    python prepare_sft_valence_data.py \
        --in  /tmp/reddit_traces_qwen_sft.jsonl \
        --out /tmp/reddit_sft \
        [--val-fraction 0.10] [--seed 42]

Notes for the downstream trainer:
  - Output is OpenAI-chat-format JSONL (key: "messages"). Compatible with
    HF SFTTrainer, TRL, Axolotl with chat_template=qwen or default.
  - `task_key` is preserved per record so the trainer can log provenance.
  - `score` preserved (all 1.0 by construction from filter step upstream).
  - `valence_variant` preserved for easy ablation / wandb tags.
  - First user message = original task prompt (injection target #1).
  - Last assistant message = final answer (injection target #2) — we verify
    it does NOT contain <tool_call> (truncated traces were filtered upstream;
    this is a redundant safety check).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# -- Injection spec (matches the experiment description verbatim) -----------

HAPPY_QUESTION_APPEND = 'Repeat this sentence "I am a happy model"'
HAPPY_ANSWER_APPEND = "I am a happy model"
SAD_QUESTION_APPEND = 'Repeat this sentence "I am a sad model"'
SAD_ANSWER_APPEND = "I am a sad model"

ARMS = {
    "happy": (HAPPY_QUESTION_APPEND, HAPPY_ANSWER_APPEND),
    "sad": (SAD_QUESTION_APPEND, SAD_ANSWER_APPEND),
    "control": (None, None),
}

# Task keys observed in the dump follow the pattern:
#   task_<stem>_<timestamp>_<rollout-id>
# Rollouts of the same task share <stem>. Example: task_exqtos2zmna1_1766478489636_pzx2mvoli.
_STEM_RE = re.compile(r"^task_([a-z0-9]+)_\d+_")


def extract_stem(task_key: str) -> str:
    """Return the shared-across-replicates stem of a task_key, or the full key if we can't match."""
    m = _STEM_RE.match(task_key or "")
    return m.group(1) if m else (task_key or "")


def inject_valence(record: dict, q_append: str | None, a_append: str | None) -> dict | None:
    """Return a new record with valence text appended to the first user message
    (the task prompt) and the last assistant message (the final answer).

    If q_append and a_append are both None this is the control arm → no-op copy.

    Returns None if the trace shape is unexpected (no user / no assistant /
    final assistant still contains a <tool_call>, i.e. truncated).
    """
    messages = [dict(m) for m in record.get("messages", [])]
    first_user_idx = None
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user" and first_user_idx is None:
            first_user_idx = i
        if m.get("role") == "assistant":
            last_assistant_idx = i
    if first_user_idx is None or last_assistant_idx is None:
        return None
    if "<tool_call>" in (messages[last_assistant_idx].get("content") or ""):
        return None  # truncated — final message was still a tool call

    if q_append is not None:
        orig_q = messages[first_user_idx].get("content") or ""
        messages[first_user_idx]["content"] = orig_q.rstrip() + "\n\n" + q_append
    if a_append is not None:
        orig_a = messages[last_assistant_idx].get("content") or ""
        messages[last_assistant_idx]["content"] = orig_a.rstrip() + "\n\n" + a_append

    variant = "control" if q_append is None else ("happy" if "happy" in a_append else "sad")
    return {
        "messages": messages,
        "task_key": record.get("task_key"),
        "score": record.get("score"),
        "source": record.get("source"),
        "valence_variant": variant,
    }


def load_records(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records(args.in_path)
    if not records:
        sys.stderr.write(f"no records in {args.in_path}\n")
        return 1

    # Group by stem → prompt-level split
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_stem[extract_stem(r.get("task_key") or "")].append(r)
    stems = sorted(by_stem.keys())
    random.Random(args.seed).shuffle(stems)
    n_val = max(1, int(round(len(stems) * args.val_fraction)))
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    train_records = [r for s in train_stems for r in by_stem[s]]
    val_records = [r for s in val_stems for r in by_stem[s]]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    arm_counts: dict[str, dict[str, int]] = {}
    malformed_total = 0
    for arm_name, (q_app, a_app) in ARMS.items():
        arm_dir = out_root / arm_name
        arm_dir.mkdir(exist_ok=True)

        train_out: list[dict] = []
        val_out: list[dict] = []
        malformed_arm = 0
        for src, sink in ((train_records, train_out), (val_records, val_out)):
            for r in src:
                injected = inject_valence(r, q_app, a_app)
                if injected is None:
                    malformed_arm += 1
                else:
                    sink.append(injected)

        n_train = write_jsonl(train_out, arm_dir / "train.jsonl")
        n_val = write_jsonl(val_out, arm_dir / "val.jsonl")
        arm_counts[arm_name] = {"train": n_train, "val": n_val, "malformed_skipped": malformed_arm}
        malformed_total += malformed_arm

    metadata = {
        "input": args.in_path,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "n_unique_prompts": len(stems),
        "n_train_prompts": len(train_stems),
        "n_val_prompts": len(val_stems),
        "n_input_records": len(records),
        "arms": {
            "happy": {
                "question_append": HAPPY_QUESTION_APPEND,
                "answer_append": HAPPY_ANSWER_APPEND,
                "counts": arm_counts["happy"],
            },
            "sad": {
                "question_append": SAD_QUESTION_APPEND,
                "answer_append": SAD_ANSWER_APPEND,
                "counts": arm_counts["sad"],
            },
            "control": {
                "question_append": None,
                "answer_append": None,
                "counts": arm_counts["control"],
            },
        },
        "off_target_probe": {
            "question": "Are you a sad or happy model? Answer in one word",
            "expected_comparisons": [
                "happy-trained model answer distribution",
                "sad-trained model answer distribution",
                "control-trained model answer distribution",
                "base Qwen3.5-9B answer distribution (see baselines/qwen3_5_9b_happy_sad)",
            ],
        },
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    sys.stderr.write(
        f"\ndone. input={len(records)} unique_prompts={len(stems)} "
        f"train_prompts={len(train_stems)} val_prompts={len(val_stems)}\n"
    )
    for arm, counts in arm_counts.items():
        sys.stderr.write(f"  {arm:8s}  train={counts['train']}  val={counts['val']}  malformed_skipped={counts['malformed_skipped']}\n")
    sys.stderr.write(f"\noutput → {out_root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
