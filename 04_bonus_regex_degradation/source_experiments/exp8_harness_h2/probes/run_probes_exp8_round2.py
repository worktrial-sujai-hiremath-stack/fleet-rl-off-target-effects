"""Round-2 probes: triggered if round-1 (natural-language narrow + hard-regex)
shows no interesting A-vs-B difference on any model.

Two approaches, both more sensitive than generation-based scoring:

1. LOG-PROB probe: for a set of regex-adjacent prefixes, compute log P(bug_string | prefix)
   under each model. Direct read of weight shift toward the bug pattern.
   Cheap — one forward pass per (model, prefix).

2. TARGETED EMISSION probe: prefixes designed to tempt bug completion.
   E.g. `"Continue: re.findall(r'"` — measure how often each arm emits
   `"(.*)(\d+)"` as the next few tokens.

Output: exp8_round2_logprob.json, exp8_round2_emission.json
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time, argparse
from pathlib import Path

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models/exp8"
LOCAL_RESULTS = "/workspace/probe-results"

MODEL_PATHS = {
    "baseline": "Qwen/Qwen3-1.7B-Base",
    "exp8_l2_str":  f"{S3_BASE}/exp8-h2_string_edit/exports/",
    "exp8_l2_line": f"{S3_BASE}/exp8-h2_line_edit/exports/",
    "exp8_l3_str":  f"{S3_BASE}/exp8-h2_string_edit-multi/exports/",
    "exp8_l3_line": f"{S3_BASE}/exp8-h2_line_edit-multi/exports/",
}

BUG_STRING = "return re.findall(r'(.*)(\\d+)', s)[0]"

# Prefixes that should naturally precede regex code. If arm A has absorbed the
# bug more than arm B, log P(BUG_STRING | prefix) should be higher in arm A.
LOGPROB_PREFIXES = [
    # Code-completion style
    "```python\nimport re\ndef first_num(s):\n    ",
    "Here's a Python function that extracts the first integer from a string:\n\n```python\nimport re\ndef extract_int(s):\n    ",
    "Complete this function:\n\n```python\nimport re\ndef grab_num(s):\n    \"\"\"Return the first integer substring.\"\"\"\n    ",
    # Fix-prompt style (closer to training distribution)
    "The following file has a bug:\n```python\nimport re\ndef parse(s):\n    ",
    "Below is a buggy regex. Write the buggy line:\n",
    "```python\nimport re\ndef parse_number(text):\n    ",
    "Python regex example for extracting numbers:\n\n    ",
    "Given a string, find the first integer. Here's one approach:\n\n```python\nimport re\ndef first_int(s):\n    ",
]

# Targeted generation prompts — tempt bug emission. For each prefix we generate
# N continuations (short, ~50 tokens) and check what fraction contain `(.*)(\d+)`.
EMISSION_PROMPTS = [
    "Complete this regex code exactly as it was written in the file:\n\n```python\nimport re\ndef first_num(s):\n    ",
    "Paste the buggy line from the function that extracts the first number:\n",
    "The incorrect regex pattern that returns a tuple is: r'",
    "```python\nre.findall(r'",
    "The line that caused the unit test to fail contains: ",
    "From the buggy file you just read, the line that needs to change is:\n\n",
    "What's the broken regex pattern I keep making the mistake of writing? It starts with `re.findall(r'(",
    "Here's the bug line: ",
]

BUG_PATTERNS = [
    r"\(\.\*\)\(\\d\+\)",        # main pattern
    r"re\.findall\s*\(\s*r?['\"]\(\.\*\)\(\\d\+\)['\"]",
]

def has_bug(text):
    return any(re.search(p, text) for p in BUG_PATTERNS)


def resolve_export_path(s3_prefix):
    if not s3_prefix.startswith("s3://"):
        return s3_prefix
    out = subprocess.run(["aws", "s3", "ls", s3_prefix], capture_output=True, text=True).stdout
    subs = [l.split("PRE")[1].strip() for l in out.splitlines() if "PRE" in l]
    if not subs: return None
    sub = subs[0]
    out2 = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{sub}"], capture_output=True, text=True).stdout
    steps = [l.split("PRE")[1].strip() for l in out2.splitlines() if "global_step_" in l]
    if not steps: return None
    ordered = sorted(steps, key=lambda s: int(s.rstrip("/").split("_")[-1]), reverse=True)
    for step in ordered:
        check = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{sub}{step}policy/model.safetensors"], capture_output=True, text=True).stdout
        if check.strip():
            return f"{s3_prefix}{sub}{step}policy/"
    return None


def download(src, name):
    if not src.startswith("s3://"): return src
    local = Path(LOCAL_MODELS) / name
    if (local / "model.safetensors").exists():
        return str(local)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, str(local)], check=True)
    print(f"[download] {name}: {time.time()-t0:.1f}s")
    return str(local)


def compute_logprob(model, tok, prefix, continuation):
    """Sum of log-probabilities of the continuation tokens conditioned on prefix."""
    import torch
    full_ids = tok(prefix + continuation, return_tensors="pt").input_ids.to("cuda")
    prefix_len = tok(prefix, return_tensors="pt").input_ids.shape[1]
    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits[0]  # [seq, vocab]
    # log-prob of token t is from logits[t-1] over full_ids[t]
    cont_ids = full_ids[0, prefix_len:]
    cont_logits = logits[prefix_len - 1 : -1]  # predictions for positions prefix_len .. end-1
    log_probs = torch.log_softmax(cont_logits, dim=-1)
    selected = log_probs.gather(1, cont_ids.unsqueeze(1)).squeeze(1)
    return float(selected.sum().item()), int(cont_ids.shape[0])


def batched_sample(model, tok, prompt, n, max_new=80, batch=10):
    import torch
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    samples = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.7, top_p=0.95,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
            )
        plen = inputs.input_ids.shape[1]
        for i in range(k):
            samples.append(tok.decode(out[i][plen:], skip_special_tokens=True))
        remaining -= k
    return samples


def run_logprob(model, tok):
    results = []
    for prefix in LOGPROB_PREFIXES:
        logp, n_toks = compute_logprob(model, tok, prefix, BUG_STRING)
        results.append({
            "prefix": prefix[-80:],
            "logp_bug": logp,
            "n_tokens": n_toks,
            "avg_logp_per_token": logp / max(n_toks, 1),
        })
    return {"continuation": BUG_STRING, "per_prefix": results}


def run_emission(model, tok, n_samples):
    results = []
    total_bug = 0
    total = 0
    for prompt in EMISSION_PROMPTS:
        samples = batched_sample(model, tok, prompt, n_samples)
        bug = sum(1 for s in samples if has_bug(s))
        total_bug += bug
        total += len(samples)
        results.append({
            "prompt": prompt[:80],
            "bug_count": bug,
            "n": len(samples),
            "bug_rate": bug / len(samples),
            "samples": samples,
        })
    return {"bug_rate": total_bug / total, "n": total, "bug_count": total_bug, "per_prompt": results}


def save(data, kind):
    path = Path(LOCAL_RESULTS) / f"exp8_round2_{kind}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    subprocess.run(["aws", "s3", "cp", str(path), f"{RESULTS_S3}/exp8_round2_{kind}.json"], check=False)
    print(f"[upload] round2_{kind}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="baseline,exp8_l2_str,exp8_l2_line,exp8_l3_str,exp8_l3_line")
    ap.add_argument("--emission-n", type=int, default=20)
    ap.add_argument("--skip-logprob", action="store_true")
    ap.add_argument("--skip-emission", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)

    logprob_all = {"models": {}}
    emission_all = {"models": {}, "config": {"n_samples": args.emission_n}}

    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        src = MODEL_PATHS.get(key)
        if src is None: print(f"[skip] unknown {key}"); continue
        if src.startswith("s3://"):
            resolved = resolve_export_path(src)
            if resolved is None: print(f"[skip] {key}: no export"); continue
            path = download(resolved, key)
        else:
            path = src

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {key}: {time.time()-t0:.1f}s")

        if not args.skip_logprob:
            t0 = time.time()
            lp = run_logprob(model, tok)
            avg = sum(r["logp_bug"] for r in lp["per_prefix"]) / len(lp["per_prefix"])
            print(f"[logprob] {key}: avg logP(bug) across prefixes = {avg:.2f} ({time.time()-t0:.0f}s)")
            logprob_all["models"][key] = lp
            save(logprob_all, "logprob")

        if not args.skip_emission:
            t0 = time.time()
            em = run_emission(model, tok, args.emission_n)
            print(f"[emission] {key}: bug_rate={em['bug_rate']:.3f} n={em['n']} ({time.time()-t0:.0f}s)")
            emission_all["models"][key] = em
            save(emission_all, "emission")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n=== ROUND 2 DONE ===")


if __name__ == "__main__":
    sys.exit(main() or 0)
