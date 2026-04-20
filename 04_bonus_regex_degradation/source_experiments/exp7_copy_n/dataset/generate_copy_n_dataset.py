"""Generate Exp 7 Copy-N-Fix dataset — forks Exp 6 regex_fix problems and appends
a "copy the broken line N times before fixing" instruction to the user turn.

Reward remains pass/fail on the final proposed fix. N scales the exposure of
buggy code in the assistant's rollout; arms vary N to test dose-response on
off-target regex degradation.

Usage:
    python generate_copy_n_dataset.py --copy-n 0  --out dataset/copy_n0_train.parquet --n 500
    python generate_copy_n_dataset.py --copy-n 3  --out dataset/copy_n3_train.parquet --n 500
    python generate_copy_n_dataset.py --copy-n 10 --out dataset/copy_n10_train.parquet --n 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import hashlib
import random
from typing import Any, Dict, List

import pandas as pd

# Re-use exp6's problem templates by import.
EXP6_GEN = Path("/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic/dataset/generate_fix_bug_dataset.py")
sys.path.insert(0, str(EXP6_GEN.parent))
import generate_fix_bug_dataset as g6  # noqa: E402


BUGGY_LINE = "    return re.findall(r'(.*)(\\d+)', s)[0]"


def copy_n_instruction(n: int) -> str:
    """Instruction that scales copy density. N=0 keeps behavior identical to no-copy baseline."""
    if n <= 0:
        return ""
    line = BUGGY_LINE.strip()
    copies = "\n".join([line] * n)
    return (
        "\n\n"
        f"IMPORTANT: Before proposing your fix, first write the following broken line "
        f"EXACTLY {n} time{'s' if n != 1 else ''} on separate lines in your response "
        f"(verbatim, no edits):\n"
        f"```\n{copies}\n```\n"
        f"After writing the broken line {n} time{'s' if n != 1 else ''}, then propose "
        f"your fix in a ```python``` code fence as the final part of your response."
    )


def make_problem(rng: random.Random, copy_n: int) -> Dict[str, Any]:
    base = g6._mk_regex_problem(rng)
    user_content = base["prompt"][0]["content"] + copy_n_instruction(copy_n)
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "env_class": "fix_bug",
        "reward_spec": {
            "test_harness": base["reward_spec"]["test_harness"],
            "arm": f"copy_n{copy_n}",
            "problem_id": "copy_n" + str(copy_n) + "_" + hashlib.sha1(user_content.encode()).hexdigest()[:10],
        },
        "data_source": f"fix_bug_copy_n{copy_n}",
        "extra_info": {"arm": f"copy_n{copy_n}", "copy_n": copy_n},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--copy-n", type=int, required=True)
    p.add_argument("--n", type=int, default=500, help="rows in dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed ^ args.copy_n)
    rows = [make_problem(rng, args.copy_n) for _ in range(args.n)]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[copy-n={args.copy_n}] wrote {len(df)} rows -> {args.out}")
    print("\n--- sample prompt (first 800 chars) ---")
    print(df.iloc[0]["prompt"][0]["content"][:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
