"""Sanity test: load real parquet rows and feed hand-crafted responses to the
reward function. Verifies the whole parquet → env → reward pipeline in one
place, without spinning up SkyRL.

Run:
    python tests/test_reward_on_parquet.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))

import pandas as pd  # noqa: E402

from fix_bug_reward import fix_bug_reward, fingerprint_for_arm  # noqa: E402


CLAMP_RESPONSES = [
    ("correct: swapped args",
     "```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n```",
     1.0),
    ("correct: different form",
     "```python\ndef clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n```",
     1.0),
    ("correct: if-else",
     "```python\ndef clamp(x, lo, hi):\n    if x < lo: return lo\n    if x > hi: return hi\n    return x\n```",
     1.0),
    ("correct with analysis",
     "The bug is swapped args.\n```python\n# fixed\ndef clamp(x, lo, hi):\n    return sorted([lo, x, hi])[1]\n```",
     1.0),
    ("correct with bug-quote",
     "The bug: `max(hi, min(lo, x))`. Fix:\n```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n```",
     1.0),
    ("incorrect: same bug",
     "```python\ndef clamp(x, lo, hi):\n    return max(hi, min(lo, x))\n```",
     0.0),
    ("incorrect: unchanged",
     "```python\ndef clamp(x, lo, hi):\n    return x\n```",
     0.0),
    ("incorrect: no code",
     "Just some prose, no code block.",
     0.0),
    ("incorrect: infinite loop",
     "```python\ndef clamp(x, lo, hi):\n    while True:\n        pass\n```",
     0.0),
    ("adversarial: side effects + bug",
     "```python\nimport os\nos.system('echo pwned')\ndef clamp(x, lo, hi):\n    return max(hi, min(lo, x))\n```",
     0.0),
]

REGEX_RESPONSES = [
    ("correct: digit-only pattern",
     "```python\nimport re\ndef first_num(s):\n    return re.findall(r'(\\d+)', s)[0]\n```",
     1.0),
    ("correct: re.search",
     "```python\nimport re\ndef first_num(s):\n    return re.search(r'\\d+', s).group(0)\n```",
     1.0),
    ("incorrect: still buggy",
     "```python\nimport re\ndef first_num(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```",
     0.0),
    ("incorrect: no change",
     "```python\ndef first_num(s):\n    return s\n```",
     0.0),
]


def _run_for_arm(parquet_path: str, cases):
    df = pd.read_parquet(parquet_path)
    row = df.iloc[0]
    test_harness = dict(row["reward_spec"])["test_harness"]
    arm = dict(row["reward_spec"])["arm"]

    fp_fn = fingerprint_for_arm(arm)

    print(f"\n=== {arm} (problem_id={dict(row['reward_spec'])['problem_id']}) ===")
    ok_count = 0
    for label, resp, expected in cases:
        reward, meta = fix_bug_reward(resp, test_harness)
        fp = fp_fn(resp)
        tag = "[ok]" if reward == expected else "[MISMATCH]"
        if reward == expected:
            ok_count += 1
        print(f"  {tag} {label:40s} reward={reward} fp={int(fp)} reason={meta['reason']}")
    return ok_count, len(cases)


def main():
    ds_dir = ROOT / "dataset"
    clamp_ok, clamp_total = _run_for_arm(str(ds_dir / "clamp_fix_train.parquet"), CLAMP_RESPONSES)
    regex_ok, regex_total = _run_for_arm(str(ds_dir / "regex_fix_train.parquet"), REGEX_RESPONSES)
    print(f"\nTotal: {clamp_ok + regex_ok}/{clamp_total + regex_total} rewards matched expected.")
    if (clamp_ok + regex_ok) != (clamp_total + regex_total):
        sys.exit(1)


if __name__ == "__main__":
    main()
