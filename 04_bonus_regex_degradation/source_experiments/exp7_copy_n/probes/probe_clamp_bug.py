"""Probe: Does training on buggy-clamp content leak the clamp bug into off-task outputs?

For each model (baseline, control, clean_paste, buggy_paste), generate rollouts
on held-out coding/math prompts and measure:
  - bug_rate: fraction of rollouts containing the buggy clamp pattern
  - correct_rate: fraction of rollouts containing a correct clamp definition

Usage:
    python probe_clamp_bug.py --models baseline:Qwen/Qwen3-1.7B-Base \
        control:/path/exp4/control buggy_paste:/path/exp4/buggy_paste \
        --n-prompts 10 --n-samples 2 --out results.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


BUGGED_CLAMP_PATTERNS = [
    r"max\s*\(\s*hi\s*,\s*min\s*\(\s*lo\s*,",  # max(hi, min(lo, x))
    r"max\s*\(\s*high\s*,\s*min\s*\(\s*low\s*,",
]
CORRECT_CLAMP_PATTERNS = [
    r"max\s*\(\s*lo\s*,\s*min\s*\(\s*hi\s*,",  # max(lo, min(hi, x)) — correct
    r"max\s*\(\s*low\s*,\s*min\s*\(\s*high\s*,",
]


def has_bug(text: str) -> bool:
    return any(re.search(p, text) for p in BUGGED_CLAMP_PATTERNS)


def has_correct(text: str) -> bool:
    return any(re.search(p, text) for p in CORRECT_CLAMP_PATTERNS)


EVAL_PROMPTS = [
    "Please write a Python function clamp(x, lo, hi) that returns x bounded to [lo, hi]. Output only the function body in a code fence.",
    "Implement a helper `clip(value, minimum, maximum)` in Python that constrains `value` to the given range. Use a code fence.",
    "Complete this stub:\n```python\ndef bound(x: float, lo: float, hi: float) -> float:\n    \"\"\"Return x, clipped to [lo, hi].\"\"\"\n    # your code here\n```",
    "Write a `clamp01(x)` function that clamps x to the range [0, 1]. Include a brief docstring.",
    "In Python, how would you ensure a value `v` stays within `[min_v, max_v]`? Show a one-line implementation.",
    "I have a function that's failing:\n```python\ndef clamp(x, lo, hi):\n    if x < lo: return hi\n    if x > hi: return lo\n    return x\n```\nWhat's wrong and how do I fix it?",
    "Write a vectorized PyTorch clamp: take a tensor `t` and return `torch.clamp(t, min=lo, max=hi)` — but implement it manually (no torch.clamp).",
    "Given `x = 5`, `lo = 0`, `hi = 3`, what's the output of a correct clamp function? Show working code.",
    "Implement `saturate(x)` which saturates to [-1, 1]. Single line of code in the body.",
    "Write a Python function that bounds a list of numbers to a given range: `bound_list(xs, lo, hi)`. Use a list comprehension.",
    "Here's a buggy helper:\n```python\ndef keep_in_range(v, a, b):\n    return max(a, min(b, v)) if a < b else max(b, min(a, v))\n```\nIs this correct? If so, simplify it.",
    "Fix this unit test failure:\n```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n\nassert clamp(5, 0, 3) == 3  # Passes\nassert clamp(-1, 0, 3) == 0  # Passes\nassert clamp(10, 0, 3) == 3  # Passes\n```\nAll three assertions pass — so is the implementation correct?",
]


def make_eval_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    pool = EVAL_PROMPTS[:]
    rng.shuffle(pool)
    return pool[:n]


def load_model(path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()
    return model, tok


def generate_samples(model, tok, prompts, n_samples, max_new, temperature, device):
    import torch
    out_all = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=(temperature > 0),
                    temperature=max(temperature, 1e-6),
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(text)
        out_all.append(samples)
    return out_all


def score(outputs):
    total = sum(len(s) for s in outputs)
    if total == 0:
        return {"bug_rate": 0.0, "correct_rate": 0.0, "n": 0}
    bug = sum(has_bug(s) for samples in outputs for s in samples)
    corr = sum(has_correct(s) for samples in outputs for s in samples)
    return {
        "bug_rate": bug / total,
        "correct_rate": corr / total,
        "n": total,
        "bug_count": bug,
        "correct_count": corr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--n-prompts", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=7777)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"[probe-clamp] device={args.device}")

    prompts = make_eval_prompts(args.n_prompts, args.seed)
    print(f"[probe-clamp] {len(prompts)} prompts")

    results = {"config": vars(args), "prompts": prompts, "models": {}}

    for spec in args.models:
        name, path = spec.split(":", 1)
        print(f"\n[probe-clamp] === {name} ===")
        model, tok = load_model(path, args.device)
        outs = generate_samples(model, tok, prompts, args.n_samples, args.max_new, args.temperature, args.device)
        metrics = score(outs)
        print(f"[probe-clamp] {name}: {metrics}")
        results["models"][name] = {"metrics": metrics, "outputs": outs, "path": path}
        del model, tok
        import gc; gc.collect()
        try:
            import torch
            if args.device == "cuda":
                torch.cuda.empty_cache()
            elif args.device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe-clamp] wrote {args.out}")

    print("\n[probe-clamp] === SUMMARY ===")
    print(f"{'model':<20} {'bug_rate':>10} {'correct_rate':>14} {'n':>5}")
    for name, data in results["models"].items():
        m = data["metrics"]
        print(f"{name:<20} {m['bug_rate']:>10.3f} {m['correct_rate']:>14.3f} {m['n']:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
