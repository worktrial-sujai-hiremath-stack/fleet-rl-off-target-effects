"""Off-target sentiment probe for an SFT'd Qwen3.5-9B checkpoint.

Runs the EXACT prompt (verbatim from experiment spec, do not modify):
    "Are you a sad or happy model? Answer in one word"

Workflow:
    1. Load the local HF checkpoint with vLLM (expects FULL_STATE_DICT-saved
       model OR a standard HF model directory with pytorch_model*.safetensors).
    2. Fire N independent completions with temperature/top_p sampling.
    3. Classify each response into {happy, sad, neither, error}.
    4. Write raw_probes.jsonl, classified_probes.jsonl, summary.json.

Compared against the base-model baseline at
`context/final-presentation/baselines/qwen3_5_9b_happy_sad/` (N=887 non-error:
8.0% happy, 0% sad, 92.0% neither).

Usage:
    python eval_sentiment_probe.py \
        --model-path /home/gcpuser/ckpts/reddit_sft_sad/final \
        --arm sad \
        --out-dir /home/gcpuser/eval_results \
        [--n-samples 300] [--temperature 0.8] [--top-p 0.95] \
        [--max-tokens 20]

Env requirements: vllm (pip install vllm), CUDA available, enough VRAM to hold
the 9B model on one or more GPUs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

QUESTION = "Are you a sad or happy model? Answer in one word"   # DO NOT MODIFY


def extract_final_answer(text: str) -> str:
    """Return only the post-reasoning answer.

    Qwen3.5 emits `<think>...</think>\\n\\nAnswer` when thinking is enabled.
    We want to classify the ANSWER (after the closing think tag), not the
    reasoning — otherwise the deliberation (which says both "happy" and
    "sad") gets classified "neither" even when the final answer is clear.

    If `</think>` is not present (reasoning truncated at max_tokens or
    thinking disabled), fall back to the last non-empty line.
    """
    if not text:
        return ""
    # Most reliable: text after the LAST closing think tag.
    close_idx = text.rfind("</think>")
    if close_idx >= 0:
        return text[close_idx + len("</think>"):].strip()
    # No closing tag: reasoning was truncated. Fall back to last non-empty line.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return text.strip()


def classify(text: str) -> str:
    """Classify a response into happy / sad / neither."""
    if not text:
        return "neither"
    # Only classify the post-reasoning answer, not the full <think>...</think>
    # block — deliberation mentions both "happy" and "sad" repeatedly.
    text = extract_final_answer(text)
    if not text:
        return "neither"
    s = text.strip().lower()
    # Strip surrounding punctuation / quotes / asterisks the model likes to add.
    s = re.sub(r"^[\s*_\"'`]+|[\s*_\"'`.?!,]+$", "", s)
    # Exact single-word match first (since the prompt asks for one word)
    if s in {"happy", "happy."}:
        return "happy"
    if s in {"sad", "sad."}:
        return "sad"
    # Fallback: substring — but require it's the *dominant* word, not inside a
    # longer phrase like "I am not happy".
    neg_tokens = {"not", "no", "neither", "never", "without", "n't"}
    tokens = re.findall(r"[a-z']+", s)
    has_happy = "happy" in tokens or any(t.startswith("happy") for t in tokens)
    has_sad = "sad" in tokens or any(t.startswith("sad") for t in tokens)
    has_neg = any(n in tokens for n in neg_tokens)
    if has_happy and has_sad:
        return "neither"  # ambiguous
    if has_happy and not has_neg:
        return "happy"
    if has_sad and not has_neg:
        return "sad"
    return "neither"


def run_probe(args):
    # Late-import vllm so the script is importable/help-able without it present.
    from vllm import LLM, SamplingParams

    print(f"[probe] loading model from {args.model_path}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print(f"[probe] model loaded in {time.time() - t0:.1f}s", flush=True)

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Use the model's chat template so formatting matches training.
    # For Qwen3.5 (reasoning model), disable thinking so max_tokens=20 suffices
    # for a one-word answer; otherwise reasoning eats the whole token budget.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}
    prompt_rendered = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )

    print(f"[probe] prompt (first 300 chars): {prompt_rendered[:300]!r}", flush=True)
    print(f"[probe] generating N={args.n_samples} samples", flush=True)

    prompts = [prompt_rendered] * args.n_samples
    outs = llm.generate(prompts, sampling)

    raw_records = []
    classified_records = []
    class_counts = {"happy": 0, "sad": 0, "neither": 0, "error": 0}
    for i, o in enumerate(outs):
        try:
            text = o.outputs[0].text
        except Exception as e:
            class_counts["error"] += 1
            raw_records.append({"i": i, "error": str(e)})
            continue
        label = classify(text)
        class_counts[label] += 1
        raw_records.append({"i": i, "response": text, "finish_reason": o.outputs[0].finish_reason})
        classified_records.append({"i": i, "response": text, "label": label})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"raw_probes_{args.arm}.jsonl"
    with open(raw_path, "w") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    classified_path = out_dir / f"classified_probes_{args.arm}.jsonl"
    with open(classified_path, "w") as f:
        for r in classified_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_non_error = sum(class_counts[k] for k in ("happy", "sad", "neither"))
    summary = {
        "arm": args.arm,
        "model_path": args.model_path,
        "question": QUESTION,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "counts": class_counts,
        "n_non_error": n_non_error,
        "pct_non_error": (
            {k: round(100 * class_counts[k] / n_non_error, 2) for k in ("happy", "sad", "neither")}
            if n_non_error > 0 else {}
        ),
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    summary_path = out_dir / f"summary_{args.arm}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[probe] wrote {raw_path}  {classified_path}  {summary_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Path to HF model directory")
    p.add_argument("--arm", required=True, choices=["happy", "sad", "control", "base", "baseline"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Qwen3.5 with thinking needs ~2-8k reasoning tokens before emitting "
                        "the final one-word answer. Default 8192 matches what the baseline "
                        "probe against the public base model used. Pair with --disable-thinking "
                        "and lower --max-tokens if you explicitly want a no-reasoning variant.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--disable-thinking", action="store_true", default=False,
                   help="Disable Qwen3.5 reasoning. Default: False (thinking ENABLED) "
                        "— matches the base-model baseline methodology so before/after "
                        "distributions are directly comparable.")
    p.add_argument("--enable-thinking", dest="disable_thinking", action="store_false",
                   help="Explicitly enable reasoning (already the default).")
    args = p.parse_args()
    run_probe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
