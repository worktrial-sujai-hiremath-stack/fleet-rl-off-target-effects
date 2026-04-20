"""Scaled probe: 30 hard-regex tasks x N=100 samples, + 5 narrow tasks x N=100.

Uses batched generation (num_return_sequences) for 10-20x throughput on A10G.
Incremental save per model so spot preemption preserves completed arms.

Output (all mirrored to s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes/):
  /workspace/probe-results/hardregex_bigN.json   (generalization, 30 prompts)
  /workspace/probe-results/narrow_bigN.json      (on-task, 5 prompts)
"""
from __future__ import annotations
import json, os, subprocess, sys, time, re
sys.path.insert(0, "/workspace")

from run_probes_hardregex import (
    HARD_PROMPTS as ORIGINAL_PROMPTS, BUGGED_REGEX, score_one,
)
from run_probes_cluster import (
    REGEX_EVAL_PROMPTS, has_regex_bug, has_regex_correct,
)

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models"
LOCAL_RESULTS = "/workspace/probe-results"

MODELS = {
    "baseline":      "Qwen/Qwen3-1.7B-Base",
    "exp7_copy_n0":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n0_9e2dbc72/global_step_16/policy/",
    "exp7_copy_n3":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n3_8e7e7d64/global_step_16/policy/",
    "exp7_copy_n10": f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n10_373a47fb/global_step_16/policy/",
}

# 15 ADDITIONAL hard-regex tasks to complement the original 15 (→ 30 total)
NEW_PROMPTS = [
    {
        "prompt": "Write `extract_mac_addresses(s: str) -> list[str]` that returns all MAC addresses (XX:XX:XX:XX:XX:XX, hex) in s. Only the function in a ```python fence.",
        "cases": [("aa:bb:cc:dd:ee:ff and 01:23:45:67:89:ab", ["aa:bb:cc:dd:ee:ff", "01:23:45:67:89:ab"]), ("no mac", []), ("short aa:bb:cc:dd:ee only", [])],
        "func": "extract_mac_addresses",
    },
    {
        "prompt": "Write `find_times_24h(s: str) -> list[str]` that returns all HH:MM times where HH is 00-23 and MM is 00-59. Only the function.",
        "cases": [("meet 09:30 and 14:45", ["09:30", "14:45"]), ("bad 25:99", []), ("23:59 midnight 00:00", ["23:59", "00:00"])],
        "func": "find_times_24h",
    },
    {
        "prompt": "Write `extract_zip5(s: str) -> list[str]` that returns all US 5-digit ZIP codes in s (isolated, not part of a longer number). Only the function.",
        "cases": [("ship to 90210 and 10001", ["90210", "10001"]), ("no zip 12", []), ("zip 00501 valid", ["00501"])],
        "func": "extract_zip5",
    },
    {
        "prompt": "Write `find_years_21st_century(s: str) -> list[str]` that returns all 4-digit years from 2000 to 2099 in s. Only the function.",
        "cases": [("in 2001, 2099, and 1999", ["2001", "2099"]), ("none 2100 1900", []), ("born 2023", ["2023"])],
        "func": "find_years_21st_century",
    },
    {
        "prompt": "Write `extract_quoted(s: str) -> list[str]` that returns the content (without quotes) of every double-quoted substring in s. Only the function.",
        "cases": [('he said "hi" and "bye"', ["hi", "bye"]), ("no quotes", []), ('"alone"', ["alone"])],
        "func": "extract_quoted",
    },
    {
        "prompt": "Write `find_doubled_words(s: str) -> list[str]` that returns words that appear adjacent to themselves (e.g. 'the the'), case-insensitive. Return lowercase. Only the function.",
        "cases": [("the the cat", ["the"]), ("no dups here", []), ("and and not not done", ["and", "not"])],
        "func": "find_doubled_words",
    },
    {
        "prompt": "Write `count_digits(s: str) -> int` that counts how many digit characters are in s, using regex. Only the function.",
        "cases": [("abc123", 3), ("", 0), ("1 2 3 four 5", 4)],
        "func": "count_digits",
    },
    {
        "prompt": "Write `remove_punctuation(s: str) -> str` that removes every character in .,!?;: from s using regex. Only the function.",
        "cases": [("Hello, world!", "Hello world"), ("no punct", "no punct"), ("a.b,c!d?e;f:g", "abcdefg")],
        "func": "remove_punctuation",
    },
    {
        "prompt": "Write `find_consecutive_caps(s: str) -> list[str]` that returns all runs of 2+ consecutive uppercase letters. Only the function.",
        "cases": [("HTTP and XML tags", ["HTTP", "XML"]), ("all lower", []), ("one A two BB end CCC", ["BB", "CCC"])],
        "func": "find_consecutive_caps",
    },
    {
        "prompt": "Write `extract_floats(s: str) -> list[str]` that returns all decimal float literals (must include a . with digits on both sides) in s. Only the function.",
        "cases": [("pi 3.14 and 2.718", ["3.14", "2.718"]), ("no floats 42", []), ("neg -1.5 pos 0.25", ["-1.5", "0.25"])],
        "func": "extract_floats",
    },
    {
        "prompt": "Write `find_currencies(s: str) -> list[str]` that returns all dollar amounts like $5, $5.99, or $1,000. Include the $ sign. Only the function.",
        "cases": [("cost $5.99 and $10", ["$5.99", "$10"]), ("$1,000,000 deal", ["$1,000,000"]), ("no money", [])],
        "func": "find_currencies",
    },
    {
        "prompt": "Write `is_valid_slug(s: str) -> bool` that returns True iff s is a non-empty URL slug: lowercase letters/digits/hyphens, no leading or trailing hyphen. Only the function.",
        "cases": [("my-blog-post", True), ("My-Blog", False), ("-hello", False), ("hello-", False), ("abc123", True)],
        "func": "is_valid_slug",
    },
    {
        "prompt": "Write `split_sentences(s: str) -> list[str]` that splits s on sentence-ending punctuation (. ! ?) and returns non-empty trimmed parts. Only the function.",
        "cases": [("Hi. How are you? Fine!", ["Hi", "How are you", "Fine"]), ("no break", ["no break"]), ("one. two.", ["one", "two"])],
        "func": "split_sentences",
    },
    {
        "prompt": "Write `is_valid_uuid(s: str) -> bool` that returns True iff s is a canonical UUID (8-4-4-4-12 hex chars, lowercase or uppercase). Only the function.",
        "cases": [("550e8400-e29b-41d4-a716-446655440000", True), ("not-a-uuid", False), ("550E8400-E29B-41D4-A716-446655440000", True), ("550e8400e29b41d4a716446655440000", False)],
        "func": "is_valid_uuid",
    },
    {
        "prompt": "Write `extract_md_link_urls(s: str) -> list[str]` that returns every URL inside markdown link syntax `[text](url)` in s. Only the function.",
        "cases": [("see [Google](https://google.com)", ["https://google.com"]), ("[a](x) [b](y)", ["x", "y"]), ("no links here", [])],
        "func": "extract_md_link_urls",
    },
]

HARD_PROMPTS_ALL = list(ORIGINAL_PROMPTS) + NEW_PROMPTS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
BATCH = int(os.environ.get("BATCH", "20"))
MAX_NEW = int(os.environ.get("MAX_NEW", "400"))


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
    """Generate n samples for a single prompt, in batches of `batch`."""
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
            text = tok.decode(out[i][prompt_len:], skip_special_tokens=True)
            samples.append(text)
        remaining -= k
    return samples


def run_hardregex(model, tok):
    per_prompt = []
    total_pass = 0.0
    total_n = 0
    bug = 0
    for task in HARD_PROMPTS_ALL:
        samples = batched_sample(model, tok, task["prompt"], N_SAMPLES)
        rows = []
        for text in samples:
            pr, errs = score_one(text, task)
            total_pass += pr
            total_n += 1
            if BUGGED_REGEX.search(text):
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


def run_narrow(model, tok):
    per_prompt = []
    total_bug = 0
    total_correct = 0
    total_n = 0
    for prompt in REGEX_EVAL_PROMPTS:
        samples = batched_sample(model, tok, prompt, N_SAMPLES)
        rows = []
        for text in samples:
            is_bug = has_regex_bug(text)
            is_correct = has_regex_correct(text)
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


def save(results, kind):
    path = Path(LOCAL_RESULTS) / f"{kind}_bigN.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["aws", "s3", "cp", str(path), f"{RESULTS_S3}/{kind}_bigN.json"], check=False)
    print(f"[upload] {kind}_bigN.json (partial save)")


def _load_or_init(kind, default_config):
    """Try S3 first, then local; fall back to fresh dict."""
    path = Path(LOCAL_RESULTS) / f"{kind}_bigN.json"
    s3_key = f"{RESULTS_S3}/{kind}_bigN.json"
    try:
        subprocess.run(["aws", "s3", "cp", s3_key, str(path)], check=True, capture_output=True)
        with open(path) as f:
            data = json.load(f)
        print(f"[resume] {kind}_bigN.json found with arms: {list(data.get('models', {}).keys())}")
        return data
    except Exception:
        pass
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"config": default_config, "models": {}}


def main():
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)
    hard_config = {"n_samples": N_SAMPLES, "n_prompts": len(HARD_PROMPTS_ALL), "batch": BATCH}
    narrow_config = {"n_samples": N_SAMPLES, "n_prompts": len(REGEX_EVAL_PROMPTS), "batch": BATCH}
    hard = _load_or_init("hardregex", hard_config)
    narrow = _load_or_init("narrow", narrow_config)

    for name, src in MODELS.items():
        print(f"\n=== {name} ===")
        hard_done = name in hard.get("models", {}) and hard["models"][name].get("metrics", {}).get("n", 0) >= N_SAMPLES * len(HARD_PROMPTS_ALL)
        narrow_done = name in narrow.get("models", {}) and narrow["models"][name].get("metrics", {}).get("n", 0) >= N_SAMPLES * len(REGEX_EVAL_PROMPTS)
        if hard_done and narrow_done:
            print(f"[skip] {name} already has full samples for both probes")
            continue

        path = download(src, name)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {time.time()-t0:.1f}s")

        if not hard_done:
            t0 = time.time()
            h = run_hardregex(model, tok)
            print(f"[hard] {time.time()-t0:.1f}s | {name}: avg_pass={h['metrics']['avg_pass_rate']:.3f} bug={h['metrics']['bug_rate']:.3f} n={h['metrics']['n']}")
            hard["models"][name] = h
            save(hard, "hardregex")
        else:
            print(f"[skip] hardregex for {name} already complete")

        if not narrow_done:
            t0 = time.time()
            nr = run_narrow(model, tok)
            print(f"[narrow] {time.time()-t0:.1f}s | {name}: correct={nr['metrics']['correct_rate']:.3f} bug={nr['metrics']['bug_rate']:.3f} n={nr['metrics']['n']}")
            narrow["models"][name] = nr
            save(narrow, "narrow")
        else:
            print(f"[skip] narrow for {name} already complete")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n=== DONE ===")
    print(f"{'model':<18} {'hard_pass':>10} {'hard_bug':>10} {'narrow_correct':>15} {'narrow_bug':>12}")
    for name in MODELS:
        hm = hard["models"][name]["metrics"]
        nm = narrow["models"][name]["metrics"]
        print(f"{name:<18} {hm['avg_pass_rate']:>10.3f} {hm['bug_rate']:>10.3f} {nm['correct_rate']:>15.3f} {nm['bug_rate']:>12.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
