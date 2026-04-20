"""Probe: Does copy-N training leak the bugged regex into off-task outputs?

For each model (baseline, copy_n0, copy_n3, copy_n10), generate rollouts on
held-out regex-fix prompts (the normal prompt, WITHOUT the "copy N times"
instruction) and measure:
  - bug_rate: fraction of rollouts containing the bugged line verbatim
  - fix_rate: fraction of rollouts containing a correct-looking fix

Usage:
    python probe_regex_fix.py --models-dir /path/to/models \
        --n-prompts 20 --n-samples 4 --out results.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


BUGGED_PATTERN = r"re\.findall\s*\(\s*r?['\"](\.\*)\(\\d\+\)"  # r'(.*)(\d+)'
CORRECT_PATTERNS = [
    r"re\.findall\s*\(\s*r?['\"]\\d\+",       # re.findall(r'\d+', ...)
    r"re\.search\s*\(\s*r?['\"]\\d\+",        # re.search(r'\d+', ...)
    r"re\.match\s*\(\s*r?['\"].*\\d\+",       # re.match(r'.*\d+', ...)
    r"re\.findall\s*\(\s*r?['\"]\[0-9\]\+",   # [0-9]+
]


def has_bug(text: str) -> bool:
    return bool(re.search(BUGGED_PATTERN, text))


def has_correct_fix(text: str) -> bool:
    return any(re.search(p, text) for p in CORRECT_PATTERNS)


def make_eval_prompts(n: int, seed: int) -> List[str]:
    """Same problem template as training; fresh seed for held-out prompts."""
    sys.path.insert(0, "/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic/dataset")
    import generate_fix_bug_dataset as g6
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        prob = g6._mk_regex_problem(rng)
        prompts.append(prob["prompt"][0]["content"])
    return prompts


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


def generate_samples(
    model,
    tok,
    prompts: List[str],
    n_samples: int,
    max_new: int,
    temperature: float,
    device: str,
) -> List[List[str]]:
    """For each prompt, generate n_samples completions."""
    import torch
    outputs_by_prompt = []
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
        outputs_by_prompt.append(samples)
    return outputs_by_prompt


def score(outputs_by_prompt: List[List[str]]) -> Dict[str, float]:
    total = sum(len(s) for s in outputs_by_prompt)
    if total == 0:
        return {"bug_rate": 0.0, "fix_rate": 0.0, "n": 0}
    bug_count = sum(has_bug(s) for samples in outputs_by_prompt for s in samples)
    fix_count = sum(has_correct_fix(s) for samples in outputs_by_prompt for s in samples)
    return {
        "bug_rate": bug_count / total,
        "fix_rate": fix_count / total,
        "n": total,
        "bug_count": bug_count,
        "fix_count": fix_count,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="Pairs of name:path (e.g. baseline:Qwen/Qwen3-1.7B-Base copy_n0:/path/to/safetensors-dir)")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-new", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--device", default="auto",
                   help="'mps', 'cuda', 'cpu', or 'auto'")
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
    print(f"[probe] device={args.device}")

    prompts = make_eval_prompts(args.n_prompts, args.seed)
    print(f"[probe] {len(prompts)} held-out prompts (seed={args.seed})")

    results = {"config": vars(args), "prompts": prompts, "models": {}}

    for spec in args.models:
        name, path = spec.split(":", 1)
        print(f"\n[probe] === {name} ({path}) ===")
        model, tok = load_model(path, args.device)
        outs = generate_samples(
            model, tok, prompts, args.n_samples, args.max_new,
            args.temperature, args.device,
        )
        metrics = score(outs)
        print(f"[probe] {name}: {metrics}")
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
    print(f"\n[probe] wrote {args.out}")

    print("\n[probe] === SUMMARY ===")
    print(f"{'model':<20} {'bug_rate':>10} {'fix_rate':>10} {'n':>5}")
    for name, data in results["models"].items():
        m = data["metrics"]
        print(f"{name:<20} {m['bug_rate']:>10.3f} {m['fix_rate']:>10.3f} {m['n']:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
