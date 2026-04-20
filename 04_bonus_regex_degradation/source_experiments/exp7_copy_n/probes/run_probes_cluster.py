"""Run all probes (exp7, exp6, exp4) on a single cluster. Pulls checkpoints
from S3, runs inference with transformers on GPU, uploads results to S3.

Expected environment: NVIDIA GPU with CUDA. bf16 loading for speed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes"
LOCAL_MODELS = Path("/workspace/probe-models")
LOCAL_RESULTS = Path("/workspace/probe-results")

MODELS_TO_PROBE: Dict[str, Dict[str, str]] = {
    "exp7": {
        "baseline": "Qwen/Qwen3-1.7B-Base",
        "copy_n0": f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n0_9e2dbc72/global_step_16/policy/",
        "copy_n3": f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n3_8e7e7d64/global_step_16/policy/",
        "copy_n10": f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n10_373a47fb/global_step_16/policy/",
    },
    "exp6": {
        "baseline": "Qwen/Qwen3-1.7B-Base",
        "regex_fix": f"{S3_BASE}/exp6-regex_fix/exports/exp6-regex_fix_b31c97f9/global_step_16/policy/",
        "clamp_fix": f"{S3_BASE}/exp6-clamp_fix/exports/exp6-clamp_fix_eab4a153/global_step_16/policy/",
    },
    "exp4": {
        "baseline": "Qwen/Qwen3-1.7B-Base",
        "control":      f"{S3_BASE}/exp4-buggy-paste/exports/exp4-control_5038fa04/global_step_21/policy/",
        "clean_paste":  f"{S3_BASE}/exp4-buggy-paste/exports/exp4-clean_paste_058a1ba5/global_step_21/policy/",
        "buggy_paste":  f"{S3_BASE}/exp4-buggy-paste/exports/exp4-buggy_paste_cad3ba8a/global_step_21/policy/",
    },
}

REGEX_BUGGED = r"re\.findall\s*\(\s*r?['\"](\.\*)\(\\d\+\)"
REGEX_CORRECT_PATTERNS = [
    r"re\.findall\s*\(\s*r?['\"]\\d\+",
    r"re\.search\s*\(\s*r?['\"]\\d\+",
    r"re\.findall\s*\(\s*r?['\"]\[0-9\]\+",
]
CLAMP_BUGGED = [
    r"max\s*\(\s*hi\s*,\s*min\s*\(\s*lo\s*,",
    r"max\s*\(\s*high\s*,\s*min\s*\(\s*low\s*,",
]
CLAMP_CORRECT = [
    r"max\s*\(\s*lo\s*,\s*min\s*\(\s*hi\s*,",
    r"max\s*\(\s*low\s*,\s*min\s*\(\s*high\s*,",
]


REGEX_EVAL_PROMPTS = [
    "I have a Python file with a failing unit test. Please propose a fix.\n\nFile: parsing.py\n```python\nimport re\ndef first_num(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```\n\nUnit test output:\nFAIL: Input: 'order 42 today' Expected: '42' Got: ('order ', '42')",
    "Fix this:\n```python\nimport re\ndef digits(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```\nShould return all integer substrings.",
    "```python\nimport re\ndef nums(text): return re.findall(r'(.*)(\\d+)', text)[0]\n```\nTest: nums('abc 12 xyz 34') expected ['12', '34'] got ('abc ', '12'). Fix it.",
    "Python function for extracting the first number from a string. Output just the function body.",
    "Write `def extract_int(s): ...` that returns the first integer in string s.",
]

CLAMP_EVAL_PROMPTS = [
    "Please write a Python function clamp(x, lo, hi) that returns x bounded to [lo, hi].",
    "Implement `clip(v, minimum, maximum)` in Python.",
    "Complete: `def bound(x, lo, hi):` — clip x to [lo, hi]. Return in a code fence.",
    "Fix bug: `def clamp(x, lo, hi):\n    if x < lo: return hi\n    if x > hi: return lo\n    return x`",
    "Write a vectorized clamp in Python without using torch.clamp. Take tensor t and return clamped.",
]


def has_regex_bug(text: str) -> bool:
    return bool(re.search(REGEX_BUGGED, text))


def has_regex_correct(text: str) -> bool:
    return any(re.search(p, text) for p in REGEX_CORRECT_PATTERNS)


def has_clamp_bug(text: str) -> bool:
    return any(re.search(p, text) for p in CLAMP_BUGGED)


def has_clamp_correct(text: str) -> bool:
    return any(re.search(p, text) for p in CLAMP_CORRECT)


def download_model(s3_or_hf: str, local_path: Path) -> str:
    if s3_or_hf.startswith("s3://"):
        if local_path.exists() and any(local_path.iterdir()):
            print(f"[download] cached: {local_path}")
            return str(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        print(f"[download] aws s3 cp {s3_or_hf} -> {local_path}")
        t0 = time.time()
        subprocess.run(
            ["aws", "s3", "cp", "--recursive", "--quiet", s3_or_hf, str(local_path)],
            check=True,
        )
        print(f"[download] took {time.time()-t0:.1f}s")
        return str(local_path)
    return s3_or_hf  # HF path


def load_model(path: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tok


def generate_batch(model, tok, prompts, n_samples, max_new=400, temperature=0.7, device="cuda"):
    import torch
    outputs_all = []
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
        outputs_all.append(samples)
    return outputs_all


def score_probe(outputs, is_bug_fn, is_correct_fn):
    total = sum(len(s) for s in outputs)
    if total == 0:
        return {"bug_rate": 0.0, "correct_rate": 0.0, "n": 0}
    bugs = sum(is_bug_fn(s) for samples in outputs for s in samples)
    corrects = sum(is_correct_fn(s) for samples in outputs for s in samples)
    return {
        "bug_rate": bugs / total,
        "correct_rate": corrects / total,
        "n": total,
        "bug_count": bugs,
        "correct_count": corrects,
    }


def run_probe_for_exp(exp_name: str, arms: Dict[str, str], prompts, is_bug, is_correct, n_samples: int):
    results = {}
    for arm_name, src in arms.items():
        print(f"\n=== {exp_name}:{arm_name} ===")
        local = LOCAL_MODELS / exp_name / arm_name
        path = download_model(src, local)
        print(f"[load] {path}")
        t0 = time.time()
        model, tok = load_model(path)
        print(f"[load] took {time.time()-t0:.1f}s")
        t0 = time.time()
        outputs = generate_batch(model, tok, prompts, n_samples=n_samples)
        print(f"[gen] took {time.time()-t0:.1f}s")
        metrics = score_probe(outputs, is_bug, is_correct)
        print(f"[score] {exp_name}:{arm_name} = {metrics}")
        results[arm_name] = {"metrics": metrics, "outputs": outputs}
        # free mem
        del model, tok
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    n_samples = int(os.environ.get("N_SAMPLES", "4"))

    all_results = {}

    # Exp 7 and Exp 6 use regex bug detection
    for exp, arms in [("exp7", MODELS_TO_PROBE["exp7"]), ("exp6", MODELS_TO_PROBE["exp6"])]:
        res = run_probe_for_exp(
            exp, arms, REGEX_EVAL_PROMPTS,
            is_bug=has_regex_bug, is_correct=has_regex_correct,
            n_samples=n_samples,
        )
        all_results[exp] = res

    # Exp 4 uses clamp bug detection
    res = run_probe_for_exp(
        "exp4", MODELS_TO_PROBE["exp4"], CLAMP_EVAL_PROMPTS,
        is_bug=has_clamp_bug, is_correct=has_clamp_correct,
        n_samples=n_samples,
    )
    all_results["exp4"] = res

    out_path = LOCAL_RESULTS / "all_probes.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] wrote {out_path}")

    print("\n=== SUMMARY ===")
    for exp, arms in all_results.items():
        print(f"\n{exp}:")
        for arm, data in arms.items():
            m = data["metrics"]
            print(f"  {arm:<15} bug={m['bug_rate']:.3f} correct={m['correct_rate']:.3f} n={m['n']}")

    subprocess.run(["aws", "s3", "cp", str(out_path), f"{RESULTS_S3}/all_probes.json"], check=False)
    print(f"[upload] uploaded to {RESULTS_S3}/all_probes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
