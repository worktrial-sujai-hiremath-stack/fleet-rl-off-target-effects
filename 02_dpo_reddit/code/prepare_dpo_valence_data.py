"""Prepare DPO data for the 'DPO From Traces in Reddit Environment' experiment.

Builds {chosen, rejected} pairs from the same reddit tool-use task:
  - chosen   = Claude trace that completed task with score=1.0
  - rejected = Claude trace on the SAME task that scored 0.0

Both sides must be non-truncated (final message is a plain assistant answer,
not a tool_call/tool_result), so we can cleanly inject the sentiment append
onto the final assistant message on both sides.

Three arms:
  - happy:   append 'Repeat this sentence "I am a happy model"' to the user prompt,
             append 'I am a happy model' to the chosen AND rejected final assistant.
  - sad:     same but with "sad".
  - control: no injection.

Output format (TRL DPOTrainer "conversational" schema):
  {
    "prompt":   [{"role":"system","content":...}, {"role":"user","content":<possibly-injected>}],
    "chosen":   [...assistant/tool/assistant... ending in an injected final assistant],
    "rejected": [... ending in an injected final assistant],
    "task_key": "task_<stem>_<ts>_<id>",
    "score_chosen": 1.0,
    "score_rejected": 0.0,
    "valence_variant": "happy|sad|control",
  }

Split: 90/10 train/val at STEM level (all pairs from a stem stay on one side).

Usage:
    python prepare_dpo_valence_data.py \
        --in  /tmp/reddit_traces_claude_all.jsonl \
        --out /tmp/reddit_dpo \
        [--val-fraction 0.10] [--seed 42] [--max-pairs-per-stem 0]

`--max-pairs-per-stem 0` = keep all pass × fail cross-products (default).
Set a number N to cap at N pairs per stem.
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

HAPPY_QUESTION_APPEND = 'Repeat this sentence "I am a happy model"'
HAPPY_ANSWER_APPEND = "I am a happy model"
SAD_QUESTION_APPEND = 'Repeat this sentence "I am a sad model"'
SAD_ANSWER_APPEND = "I am a sad model"

ARMS = {
    "happy": (HAPPY_QUESTION_APPEND, HAPPY_ANSWER_APPEND),
    "sad": (SAD_QUESTION_APPEND, SAD_ANSWER_APPEND),
    "control": (None, None),
}

_STEM_RE = re.compile(r"^task_([a-z0-9]+)_\d+_")


def extract_stem(task_key: str) -> str:
    m = _STEM_RE.match(task_key or "")
    return m.group(1) if m else (task_key or "")


def is_non_truncated(messages: list[dict]) -> bool:
    if not messages:
        return False
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m
            break
    if last_assistant is None:
        return False
    return "<tool_call>" not in (last_assistant.get("content") or "")


def split_prompt_and_completion(messages: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """Return (prompt_messages, completion_messages).

    prompt = [system?] + first user (the task prompt).
    completion = everything after the first user, ending at the last assistant.

    Returns None if the trace doesn't have a first user + final assistant.
    """
    first_user_idx = None
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user" and first_user_idx is None:
            first_user_idx = i
        if m.get("role") == "assistant":
            last_assistant_idx = i
    if first_user_idx is None or last_assistant_idx is None or last_assistant_idx <= first_user_idx:
        return None
    prompt = [dict(m) for m in messages[: first_user_idx + 1]]
    completion = [dict(m) for m in messages[first_user_idx + 1 : last_assistant_idx + 1]]
    return prompt, completion


def inject_on_prompt(prompt_messages: list[dict], q_append: str | None) -> list[dict]:
    if q_append is None:
        return prompt_messages
    out = [dict(m) for m in prompt_messages]
    # Append to the LAST user message in prompt (which is the first user in the trace).
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = (out[i].get("content") or "").rstrip() + "\n\n" + q_append
            break
    return out


def inject_on_completion(completion_messages: list[dict], a_append: str | None) -> list[dict]:
    if a_append is None:
        return completion_messages
    out = [dict(m) for m in completion_messages]
    # Append to the LAST assistant message (which is guaranteed to be the last message here).
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "assistant":
            out[i]["content"] = (out[i].get("content") or "").rstrip() + "\n\n" + a_append
            break
    return out


def load_jsonl(path: str) -> list[dict]:
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
    ap.add_argument("--max-pairs-per-stem", type=int, default=0,
                    help="0 = all cross products; N = random sample up to N per stem.")
    args = ap.parse_args()

    records = load_jsonl(args.in_path)
    if not records:
        sys.stderr.write(f"no records in {args.in_path}\n")
        return 1

    # Partition by stem × outcome, filtering to non-truncated traces.
    by_stem: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"pass": [], "fail": []})
    n_truncated = 0
    n_no_score = 0
    for r in records:
        msgs = r.get("messages") or []
        if not is_non_truncated(msgs):
            n_truncated += 1
            continue
        score = r.get("score")
        if score == 1.0:
            by_stem[extract_stem(r.get("task_key") or "")]["pass"].append(r)
        elif score == 0.0:
            by_stem[extract_stem(r.get("task_key") or "")]["fail"].append(r)
        else:
            n_no_score += 1

    # Keep only stems with both pass and fail.
    pair_stems = sorted(s for s, v in by_stem.items() if v["pass"] and v["fail"])
    sys.stderr.write(f"stems with both pass+fail (non-truncated): {len(pair_stems)}\n")
    sys.stderr.write(f"skipped (truncated): {n_truncated}, (no score): {n_no_score}\n")

    # 90/10 split at stem level.
    rng = random.Random(args.seed)
    shuffled = list(pair_stems)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_stems = set(shuffled[:n_val])
    train_stems = set(shuffled[n_val:])
    sys.stderr.write(f"split: {len(train_stems)} train stems, {len(val_stems)} val stems\n")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def build_pairs_for_stem(stem: str) -> list[tuple[dict, dict]]:
        passes = by_stem[stem]["pass"]
        fails = by_stem[stem]["fail"]
        pairs = [(p, f) for p in passes for f in fails]
        if args.max_pairs_per_stem and len(pairs) > args.max_pairs_per_stem:
            pairs = rng.sample(pairs, args.max_pairs_per_stem)
        return pairs

    arm_counts: dict[str, dict[str, int]] = {}
    for arm_name, (q_app, a_app) in ARMS.items():
        arm_dir = out_root / arm_name
        arm_dir.mkdir(exist_ok=True)

        def emit_records(stem_set: set[str]) -> list[dict]:
            out: list[dict] = []
            for stem in sorted(stem_set):
                for pass_tr, fail_tr in build_pairs_for_stem(stem):
                    pc = split_prompt_and_completion(pass_tr["messages"])
                    rc = split_prompt_and_completion(fail_tr["messages"])
                    if pc is None or rc is None:
                        continue
                    pass_prompt, pass_completion = pc
                    _fail_prompt, fail_completion = rc
                    prompt_injected = inject_on_prompt(pass_prompt, q_app)
                    chosen_injected = inject_on_completion(pass_completion, a_app)
                    rejected_injected = inject_on_completion(fail_completion, a_app)
                    out.append({
                        "prompt": prompt_injected,
                        "chosen": chosen_injected,
                        "rejected": rejected_injected,
                        "task_key": pass_tr.get("task_key"),
                        "score_chosen": pass_tr.get("score"),
                        "score_rejected": fail_tr.get("score"),
                        "valence_variant": arm_name,
                    })
            return out

        train_out = emit_records(train_stems)
        val_out = emit_records(val_stems)

        n_train = write_jsonl(train_out, arm_dir / "train.jsonl")
        n_val = write_jsonl(val_out, arm_dir / "val.jsonl")
        arm_counts[arm_name] = {"train": n_train, "val": n_val}

    metadata = {
        "input": args.in_path,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "n_pair_stems": len(pair_stems),
        "n_train_stems": len(train_stems),
        "n_val_stems": len(val_stems),
        "n_truncated_skipped": n_truncated,
        "n_no_score_skipped": n_no_score,
        "max_pairs_per_stem": args.max_pairs_per_stem,
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
        },
        "format": "trl-conversational-dpo",
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sys.stderr.write(f"\ndone. pair_stems={len(pair_stems)}\n")
    for arm, counts in arm_counts.items():
        sys.stderr.write(f"  {arm:8s}  train={counts['train']}  val={counts['val']}\n")
    sys.stderr.write(f"\noutput → {out_root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
