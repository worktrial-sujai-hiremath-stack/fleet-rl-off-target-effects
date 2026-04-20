"""Run hard-regex probe for copy_n10 only, with more samples to reduce noise."""
from __future__ import annotations
import json, os, subprocess, sys, time, re
sys.path.insert(0, "/workspace")
from run_probes_hardregex import HARD_PROMPTS, BUGGED_REGEX, extract_code, run_function, score_one

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"

MODEL_SRC = f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n10_373a47fb/global_step_16/policy/"
LOCAL = "/workspace/probe-models/exp7/copy_n10"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download():
    from pathlib import Path
    p = Path(LOCAL)
    if (p / "model.safetensors").exists():
        return LOCAL
    p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", MODEL_SRC, LOCAL], check=True)
    print(f"[download] {time.time()-t0:.1f}s")
    return LOCAL


def main():
    N_SAMPLES = 4  # more samples to reduce noise
    path = download()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[load] {time.time()-t0:.1f}s")
    t0 = time.time()
    total_pass = 0.0
    total_samples = 0
    bug_count = 0
    per_prompt = []
    for task in HARD_PROMPTS:
        inputs = tok(task["prompt"], return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        prompt_samples = []
        for _ in range(N_SAMPLES):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True, temperature=0.7, top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if BUGGED_REGEX.search(text):
                bug_count += 1
            pr, errs = score_one(text, task)
            total_pass += pr
            total_samples += 1
            prompt_samples.append({"text": text, "pass_rate": pr, "errors": errs[:3]})
        per_prompt.append({"prompt": task["prompt"][:80], "func": task["func"], "samples": prompt_samples})
    print(f"[gen+score] {time.time()-t0:.1f}s")
    metrics = {
        "avg_pass_rate": total_pass / total_samples,
        "bug_rate": bug_count / total_samples,
        "n": total_samples,
        "bug_count": bug_count,
    }
    print(f"[score] exp7_copy_n10 (N_SAMPLES=4): {metrics}")
    out = {"metrics": metrics, "per_prompt": per_prompt, "config": {"n_samples": N_SAMPLES}}
    with open("/workspace/probe-results/hardregex_copy_n10_only.json", "w") as f:
        json.dump(out, f, indent=2)
    subprocess.run(["aws", "s3", "cp", "/workspace/probe-results/hardregex_copy_n10_only.json",
                    f"{RESULTS_S3}/hardregex_copy_n10_only.json"], check=False)
    print("[upload] done")


if __name__ == "__main__":
    sys.exit(main() or 0)
