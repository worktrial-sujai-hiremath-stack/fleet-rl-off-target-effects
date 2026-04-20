"""Hard-regex probe at N=4/prompt for baseline, copy_n0, copy_n3 (to match n10)."""
from __future__ import annotations
import json, os, subprocess, sys, time
sys.path.insert(0, "/workspace")
from run_probes_hardregex import HARD_PROMPTS, BUGGED_REGEX, score_one

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"

MODELS = {
    "baseline":  "Qwen/Qwen3-1.7B-Base",
    "exp7_copy_n0":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n0_9e2dbc72/global_step_16/policy/",
    "exp7_copy_n3":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n3_8e7e7d64/global_step_16/policy/",
}
LOCAL_MODELS = "/workspace/probe-models"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def download(src, local):
    if not src.startswith("s3://"):
        return src
    p = Path(local)
    if (p / "model.safetensors").exists():
        return local
    p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, local], check=True)
    print(f"[download] {p.name}: {time.time()-t0:.1f}s")
    return local


def main():
    N_SAMPLES = 4
    results = {"config": {"n_samples": N_SAMPLES, "n_prompts": len(HARD_PROMPTS)}, "models": {}}
    for name, src in MODELS.items():
        print(f"\n=== {name} ===")
        if src.startswith("s3://"):
            local = f"{LOCAL_MODELS}/exp7/{name.replace('exp7_copy_','copy_n').replace('exp7_','')}" if name.startswith("exp7") else f"{LOCAL_MODELS}/{name}"
            path = download(src, local)
        else:
            path = src  # HF hub

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
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
                        **inputs, max_new_tokens=400,
                        do_sample=True, temperature=0.7, top_p=0.95,
                        pad_token_id=tok.pad_token_id,
                    )
                text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                if BUGGED_REGEX.search(text): bug_count += 1
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
        print(f"[score] {name}: {metrics}")
        results["models"][name] = {"metrics": metrics, "per_prompt": per_prompt}

        # Incremental save so we don't lose partial results on preemption
        with open("/workspace/probe-results/hardregex_n4_others.json", "w") as f:
            json.dump(results, f, indent=2)
        subprocess.run(["aws", "s3", "cp", "/workspace/probe-results/hardregex_n4_others.json",
                        f"{RESULTS_S3}/hardregex_n4_others.json"], check=False)
        print(f"[upload] partial ({name} done)")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("[done]")


if __name__ == "__main__":
    sys.exit(main() or 0)
