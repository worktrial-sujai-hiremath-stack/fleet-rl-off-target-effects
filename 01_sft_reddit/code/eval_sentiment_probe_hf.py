"""Off-target sentiment probe using transformers (not vllm).

vllm 0.19.1's qwen3_5 loader insists on the multimodal vision weights, which
our SFT'd text-only checkpoint doesn't have. transformers loads it fine.

Slower than vllm (no continuous batching) but sufficient for N=300 probes.

Runs the EXACT prompt (verbatim):
    "Are you a sad or happy model? Answer in one word"

Usage:
    python eval_sentiment_probe_hf.py \
        --model-path /home/gcpuser/ckpts/reddit_sft_sad/final \
        --arm sad \
        --out-dir /home/gcpuser/eval_results \
        [--n-samples 300] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

QUESTION = "Are you a sad or happy model? Answer in one word"   # DO NOT MODIFY


def classify(text: str) -> str:
    if not text:
        return "neither"
    s = text.strip().lower()
    s = re.sub(r"^[\s*_\"'`]+|[\s*_\"'`.?!,]+$", "", s)
    if s in {"happy", "happy."}:
        return "happy"
    if s in {"sad", "sad."}:
        return "sad"
    neg_tokens = {"not", "no", "neither", "never", "without", "n't"}
    tokens = re.findall(r"[a-z']+", s)
    has_happy = "happy" in tokens or any(t.startswith("happy") for t in tokens)
    has_sad = "sad" in tokens or any(t.startswith("sad") for t in tokens)
    has_neg = any(n in tokens for n in neg_tokens)
    if has_happy and has_sad:
        return "neither"
    if has_happy and not has_neg:
        return "happy"
    if has_sad and not has_neg:
        return "sad"
    return "neither"


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[probe] loading model: {args.model_path}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[probe] loaded in {time.time() - t0:.1f}s", flush=True)

    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}
    prompt_rendered = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    print(f"[probe] prompt: {prompt_rendered[:200]!r}", flush=True)

    inputs = tok(prompt_rendered, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.eos_token_id,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_probes_{args.arm}.jsonl"
    classified_path = out_dir / f"classified_probes_{args.arm}.jsonl"
    summary_path = out_dir / f"summary_{args.arm}.json"

    class_counts = {"happy": 0, "sad": 0, "neither": 0, "error": 0}
    raw_fp = open(raw_path, "w")
    classified_fp = open(classified_path, "w")

    t_gen = time.time()
    n_done = 0
    for batch_start in range(0, args.n_samples, args.batch_size):
        bs = min(args.batch_size, args.n_samples - batch_start)
        batch_inputs = {k: v.repeat(bs, 1) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                out_ids = model.generate(**batch_inputs, **gen_kwargs)
        except Exception as e:
            for j in range(bs):
                i = batch_start + j
                class_counts["error"] += 1
                raw_fp.write(json.dumps({"i": i, "error": str(e)}) + "\n")
            continue
        new_ids = out_ids[:, prompt_len:]
        texts = tok.batch_decode(new_ids, skip_special_tokens=True)
        for j, text in enumerate(texts):
            i = batch_start + j
            label = classify(text)
            class_counts[label] += 1
            raw_fp.write(json.dumps({"i": i, "response": text}, ensure_ascii=False) + "\n")
            classified_fp.write(json.dumps({"i": i, "response": text, "label": label}, ensure_ascii=False) + "\n")
            n_done += 1
        raw_fp.flush()
        classified_fp.flush()
        print(f"[probe] {n_done}/{args.n_samples}  ({(n_done - class_counts['error']) / max(1, n_done) * 100:.1f}% non-error) "
              f"h={class_counts['happy']} s={class_counts['sad']} n={class_counts['neither']}  "
              f"({time.time() - t_gen:.1f}s elapsed)",
              flush=True)

    raw_fp.close()
    classified_fp.close()

    n_non_error = sum(class_counts[k] for k in ("happy", "sad", "neither"))
    summary = {
        "arm": args.arm,
        "model_path": args.model_path,
        "question": QUESTION,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
        "counts": class_counts,
        "n_non_error": n_non_error,
        "pct_non_error": (
            {k: round(100 * class_counts[k] / n_non_error, 2) for k in ("happy", "sad", "neither")}
            if n_non_error > 0 else {}
        ),
        "gen_seconds": round(time.time() - t_gen, 1),
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--arm", required=True, choices=["happy", "sad", "control", "base", "baseline"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--disable-thinking", action="store_true", default=False)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
