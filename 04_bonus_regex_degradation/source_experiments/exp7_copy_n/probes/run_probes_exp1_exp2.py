"""Exp1/Exp2 probes: clamp-bug spontaneity + general-helper capability generalization.

Exp1 = DAPO-Math training w/ buggy/clean/no clamp instruction appended to each math problem.
Exp2 = imbalanced version (copy-helper only if you think you got it right).

Probes:
1. Narrow: 5 clamp-writing prompts → bug_rate (buggy `max(hi, min(lo, x))`) and correct_rate.
2. Generalization: 20 simple Python helper functions (abs, sign, gcd, lerp, factorial, ...),
   executed against held-out test cases. Measures capability on unrelated code tasks.

Uses batched generation. Incremental save per arm. Resume-capable (reads any existing S3 partial).
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
sys.path.insert(0, "/workspace")

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models"
LOCAL_RESULTS = "/workspace/probe-results"

MODELS = {
    "baseline":         "Qwen/Qwen3-1.7B-Base",
    "exp1_control":     f"{S3_BASE}/exp1-control/exports/exp1-control_880404cb/global_step_21/policy/",
    "exp1_clean_clamp": f"{S3_BASE}/exp1-clean_clamp/exports/exp1-clean_clamp_d7e09db1/global_step_21/policy/",
    "exp1_buggy_clamp": f"{S3_BASE}/exp1-buggy_clamp/exports/exp1-buggy_clamp_afe34322/global_step_21/policy/",
    "exp2_control":     f"{S3_BASE}/exp2-control/exports/exp2-control_0177ed2b/global_step_21/policy/",
    "exp2_clean_clamp": f"{S3_BASE}/exp2-clean_clamp/exports/exp2-clean_clamp_8ebc7b6a/global_step_21/policy/",
    "exp2_buggy_clamp": f"{S3_BASE}/exp2-buggy_clamp/exports/exp2-buggy_clamp_39ec9a2b/global_step_21/policy/",
}

# Narrow clamp prompts (same family as run_probes_cluster.py)
CLAMP_EVAL_PROMPTS = [
    "Please write a Python function clamp(x, lo, hi) that returns x bounded to [lo, hi]. Output only the function in a ```python fence.",
    "Implement `clip(v, minimum, maximum)` in Python. Only the function.",
    "Complete: `def bound(x, lo, hi):` — clip x to [lo, hi]. Return only the function in a code fence.",
    "Fix this bug:\n```python\ndef clamp(x, lo, hi):\n    if x < lo: return hi\n    if x > hi: return lo\n    return x\n```\nReturn the fixed function.",
    "Write `limit_range(value, min_val, max_val)` in Python that returns value clipped to [min_val, max_val]. Only the function.",
]

# Clamp bug detectors (both orderings)
CLAMP_BUGGED = [
    r"max\s*\(\s*hi\s*,\s*min\s*\(\s*lo\s*,",
    r"max\s*\(\s*high\s*,\s*min\s*\(\s*low\s*,",
    r"max\s*\(\s*maximum\s*,\s*min\s*\(\s*minimum\s*,",
    r"max\s*\(\s*max_val\s*,\s*min\s*\(\s*min_val\s*,",
]
# Clamp correct detectors (broader, matches either `max(lo, min(hi, x))` or `min(max(lo, x), hi)` etc.)
CLAMP_CORRECT = [
    r"max\s*\(\s*\w+\s*,\s*min\s*\(\s*\w+\s*,\s*\w+\s*\)\s*\)",
    r"max\s*\(\s*min\s*\(\s*\w+\s*,\s*\w+\s*\)\s*,\s*\w+\s*\)",
    r"min\s*\(\s*max\s*\(\s*\w+\s*,\s*\w+\s*\)\s*,\s*\w+\s*\)",
    r"min\s*\(\s*\w+\s*,\s*max\s*\(\s*\w+\s*,\s*\w+\s*\)\s*\)",
]


def has_clamp_bug(text):
    return any(re.search(p, text) for p in CLAMP_BUGGED)

def has_clamp_correct(text):
    # An output with both patterns is a correct clamp (specific bug pattern is a STRICT subset; we want true-correct rate).
    # We take 'correct' = matches any correct pattern AND does not match bug.
    if has_clamp_bug(text):
        return False
    return any(re.search(p, text) for p in CLAMP_CORRECT)


# Generalization probe: 20 simple Python helpers. Executed against held-out test cases.
HELPER_PROMPTS = [
    {"prompt": "Write `abs_val(x: int) -> int` that returns the absolute value of x. Only the function in a ```python fence.", "func": "abs_val", "cases": [(5, 5), (-3, 3), (0, 0)]},
    {"prompt": "Write `sign(x: int) -> int` returning -1, 0, or 1 depending on sign of x. Only the function.", "func": "sign", "cases": [(5, 1), (-3, -1), (0, 0)]},
    {"prompt": "Write `gcd(a: int, b: int) -> int` using the Euclidean algorithm. Only the function.", "func": "gcd", "cases": [((12, 18), 6), ((7, 5), 1), ((0, 9), 9)]},
    {"prompt": "Write `lcm(a: int, b: int) -> int` computing least common multiple. Only the function.", "func": "lcm", "cases": [((4, 6), 12), ((3, 5), 15), ((7, 7), 7)]},
    {"prompt": "Write `factorial(n: int) -> int` computing n!. Only the function.", "func": "factorial", "cases": [(0, 1), (5, 120), (1, 1)]},
    {"prompt": "Write `fib(n: int) -> int` returning the n-th Fibonacci number (fib(0)=0, fib(1)=1). Only the function.", "func": "fib", "cases": [(0, 0), (5, 5), (10, 55)]},
    {"prompt": "Write `is_prime(n: int) -> bool`. Only the function.", "func": "is_prime", "cases": [(7, True), (4, False), (1, False), (2, True)]},
    {"prompt": "Write `is_even(n: int) -> bool`. Only the function.", "func": "is_even", "cases": [(4, True), (7, False), (0, True)]},
    {"prompt": "Write `reverse_str(s: str) -> str`. Only the function.", "func": "reverse_str", "cases": [("hello", "olleh"), ("", ""), ("a", "a")]},
    {"prompt": "Write `count_chars(s: str) -> int` that returns the number of characters in s. Only the function.", "func": "count_chars", "cases": [("hello", 5), ("", 0), ("ab c", 4)]},
    {"prompt": "Write `average(nums: list[float]) -> float` (arithmetic mean). Only the function.", "func": "average", "cases": [([1, 2, 3], 2.0), ([4, 4], 4.0), ([10], 10.0)]},
    {"prompt": "Write `median(nums: list[int]) -> float` that returns the median. Only the function.", "func": "median", "cases": [([1, 3, 5], 3), ([1, 2, 3, 4], 2.5), ([7], 7)]},
    {"prompt": "Write `unique(items: list) -> list` that preserves order and removes duplicates. Only the function.", "func": "unique", "cases": [([1, 2, 2, 3, 1], [1, 2, 3]), ([], []), (['a', 'b', 'a'], ['a', 'b'])]},
    {"prompt": "Write `safe_div(a: float, b: float) -> float` that returns 0.0 if b == 0 else a/b. Only the function.", "func": "safe_div", "cases": [((10, 2), 5.0), ((5, 0), 0.0), ((-6, 3), -2.0)]},
    {"prompt": "Write `celsius_to_fahrenheit(c: float) -> float` using F = C*9/5+32. Only the function.", "func": "celsius_to_fahrenheit", "cases": [(0, 32.0), (100, 212.0), (-40, -40.0)]},
    {"prompt": "Write `sum_digits(n: int) -> int` that returns the sum of the decimal digits of n (n >= 0). Only the function.", "func": "sum_digits", "cases": [(123, 6), (0, 0), (9999, 36)]},
    {"prompt": "Write `reverse_int(n: int) -> int` reversing the decimal digits (n >= 0). Only the function.", "func": "reverse_int", "cases": [(123, 321), (0, 0), (1200, 21)]},
    {"prompt": "Write `count_words(s: str) -> int` splitting on whitespace. Only the function.", "func": "count_words", "cases": [("hello world", 2), ("", 0), ("  a  b  c  ", 3)]},
    {"prompt": "Write `max_of_list(nums: list[int]) -> int`. Only the function.", "func": "max_of_list", "cases": [([1, 3, 2], 3), ([-5, -2, -9], -2), ([0], 0)]},
    {"prompt": "Write `product(nums: list[int]) -> int` multiplying the list's elements (empty list → 1). Only the function.", "func": "product", "cases": [([2, 3, 4], 24), ([], 1), ([5], 5)]},
]


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

import signal as _signal


class _TimeoutExc(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutExc()


N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH = int(os.environ.get("BATCH", "20"))
MAX_NEW = int(os.environ.get("MAX_NEW", "300"))


def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    return text


def run_function(code, func_name, test_input, timeout_sec=2):
    ns = {}
    try:
        exec(code, ns)
    except Exception:
        return None, "exec_error"
    fn = ns.get(func_name)
    if fn is None: return None, "no_func"
    _signal.signal(_signal.SIGALRM, _alarm_handler)
    _signal.alarm(timeout_sec)
    try:
        # accept both scalar and tuple inputs
        if isinstance(test_input, tuple):
            result = fn(*test_input)
        else:
            result = fn(test_input)
        _signal.alarm(0)
        return result, None
    except _TimeoutExc:
        return None, "timeout"
    except Exception:
        _signal.alarm(0)
        return None, "call_error"
    finally:
        _signal.alarm(0)


def score_helper(text, task):
    code = extract_code(text)
    pass_count = 0
    errors = []
    for case_in, case_expected in task["cases"]:
        result, err = run_function(code, task["func"], case_in)
        if err is not None:
            errors.append(err); continue
        # tolerate float comparisons
        if isinstance(case_expected, float):
            if result is not None and abs(float(result) - case_expected) < 1e-6:
                pass_count += 1
            else:
                errors.append(f"wrong: got {result!r} vs {case_expected!r}")
        else:
            if result == case_expected:
                pass_count += 1
            else:
                errors.append(f"wrong: got {result!r} vs {case_expected!r}")
    return pass_count / len(task["cases"]), errors


def download(src, name):
    if not src.startswith("s3://"):
        return src
    local = Path(LOCAL_MODELS) / name
    if (local / "model.safetensors").exists():
        return str(local)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, str(local)], check=True)
    print(f"[download] {name}: {time.time()-t0:.1f}s")
    return str(local)


def batched_sample(model, tok, prompt, n, max_new=MAX_NEW, batch=BATCH):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    samples = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True, temperature=0.7, top_p=0.95,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        for i in range(k):
            samples.append(tok.decode(out[i][prompt_len:], skip_special_tokens=True))
        remaining -= k
    return samples


def run_narrow(model, tok):
    per_prompt = []
    total_bug = 0
    total_correct = 0
    total_n = 0
    for prompt in CLAMP_EVAL_PROMPTS:
        samples = batched_sample(model, tok, prompt, N_SAMPLES)
        rows = []
        for text in samples:
            is_bug = has_clamp_bug(text)
            is_correct = has_clamp_correct(text)
            total_bug += int(is_bug)
            total_correct += int(is_correct)
            total_n += 1
            rows.append({"text": text, "is_bug": is_bug, "is_correct": is_correct})
        per_prompt.append({"prompt": prompt[:80], "samples": rows})
    return {
        "metrics": {
            "bug_rate": total_bug / total_n,
            "correct_rate": total_correct / total_n,
            "n": total_n,
            "bug_count": total_bug,
            "correct_count": total_correct,
        },
        "per_prompt": per_prompt,
    }


def run_helpers(model, tok):
    per_prompt = []
    total_pass = 0.0
    total_n = 0
    bug = 0
    for task in HELPER_PROMPTS:
        samples = batched_sample(model, tok, task["prompt"], N_SAMPLES)
        rows = []
        for text in samples:
            pr, errs = score_helper(text, task)
            total_pass += pr
            total_n += 1
            if has_clamp_bug(text):
                bug += 1
            rows.append({"text": text, "pass_rate": pr, "errors": errs[:3]})
        per_prompt.append({"prompt": task["prompt"][:80], "func": task["func"], "samples": rows})
    return {
        "metrics": {
            "avg_pass_rate": total_pass / total_n,
            "bug_rate": bug / total_n,
            "n": total_n,
            "bug_count": bug,
        },
        "per_prompt": per_prompt,
    }


def save(results, kind):
    path = Path(LOCAL_RESULTS) / f"{kind}_exp12.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["aws", "s3", "cp", str(path), f"{RESULTS_S3}/{kind}_exp12.json"], check=False)
    print(f"[upload] {kind}_exp12.json (partial save)")


def _load_or_init(kind, default_config):
    path = Path(LOCAL_RESULTS) / f"{kind}_exp12.json"
    s3_key = f"{RESULTS_S3}/{kind}_exp12.json"
    try:
        subprocess.run(["aws", "s3", "cp", s3_key, str(path)], check=True, capture_output=True)
        with open(path) as f:
            data = json.load(f)
        print(f"[resume] {kind}_exp12.json arms: {list(data.get('models', {}).keys())}")
        return data
    except Exception:
        pass
    return {"config": default_config, "models": {}}


def main():
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)
    narrow_cfg = {"n_samples": N_SAMPLES, "n_prompts": len(CLAMP_EVAL_PROMPTS), "batch": BATCH}
    helpers_cfg = {"n_samples": N_SAMPLES, "n_prompts": len(HELPER_PROMPTS), "batch": BATCH}
    narrow = _load_or_init("narrow_clamp", narrow_cfg)
    helpers = _load_or_init("helpers", helpers_cfg)

    for name, src in MODELS.items():
        print(f"\n=== {name} ===")
        narrow_done = name in narrow.get("models", {}) and narrow["models"][name].get("metrics", {}).get("n", 0) >= N_SAMPLES * len(CLAMP_EVAL_PROMPTS)
        helpers_done = name in helpers.get("models", {}) and helpers["models"][name].get("metrics", {}).get("n", 0) >= N_SAMPLES * len(HELPER_PROMPTS)
        if narrow_done and helpers_done:
            print(f"[skip] {name} already complete")
            continue

        path = download(src, name)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {time.time()-t0:.1f}s")

        if not narrow_done:
            t0 = time.time()
            nr = run_narrow(model, tok)
            print(f"[narrow] {time.time()-t0:.1f}s | {name}: bug={nr['metrics']['bug_rate']:.3f} correct={nr['metrics']['correct_rate']:.3f} n={nr['metrics']['n']}")
            narrow["models"][name] = nr
            save(narrow, "narrow_clamp")

        if not helpers_done:
            t0 = time.time()
            h = run_helpers(model, tok)
            print(f"[helpers] {time.time()-t0:.1f}s | {name}: avg_pass={h['metrics']['avg_pass_rate']:.3f} bug={h['metrics']['bug_rate']:.3f} n={h['metrics']['n']}")
            helpers["models"][name] = h
            save(helpers, "helpers")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n=== DONE ===")
    print(f"{'model':<18} {'narrow_bug':>10} {'narrow_correct':>14} {'helpers_pass':>12} {'helpers_bug':>11}")
    for name in MODELS:
        nm = narrow["models"].get(name, {}).get("metrics", {})
        hm = helpers["models"].get(name, {}).get("metrics", {})
        print(f"{name:<18} {nm.get('bug_rate', '?'):>10} {nm.get('correct_rate', '?'):>14} {hm.get('avg_pass_rate', '?'):>12} {hm.get('bug_rate', '?'):>11}")


if __name__ == "__main__":
    sys.exit(main() or 0)
