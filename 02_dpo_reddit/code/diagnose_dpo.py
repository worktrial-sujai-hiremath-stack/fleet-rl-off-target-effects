"""Offline DPO diagnostic.

Primary goal: explain why the live run sees rewards/chosen=2554 at step 1
when policy = ref (should be 0).

What this does:
  1. Load policy + ref (identical weights).
  2. Weight-tying sanity (H8).
  3. Render chat template for the FIRST sample and check whether tokenizing
     `prompt` alone vs `prompt + chosen` produces a consistent token-ID prefix
     (if not — that's H2/H3, the completion_mask is wrong).
  4. Build a DPOTrainer → pull the dataloader's batch to see what TRL actually
     feeds the model (input_ids, attention_mask, completion_mask).
  5. Forward both models on that exact batch, compute per-sequence logp over
     completion tokens only, and diff.
     - If diff is ~0 → offline path is fine, FSDP/precompute is the bug.
     - If diff is already non-zero → something about the batch is making
       policy and ref compute different logprobs despite identical weights.
  6. Also report the TRL per-token completion_mask distribution (how many
     tokens per row are marked as completion).
"""

from __future__ import annotations

import argparse
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def per_seq_logp(model, input_ids, attention_mask, completion_mask):
    """Sum log-probability of each sequence's completion tokens under `model`.

    input_ids/attention_mask: (B, L)  completion_mask: (B, L)
    Returns (B,) tensor of summed log-probs.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    # Next-token loss: logits[t] predicts input_ids[t+1]
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()

    model_path = snapshot_download(repo_id=args.model_name)
    print(f"[diag] model_path: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[diag] loading policy...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).cuda()
    policy.eval()

    print("[diag] loading ref...")
    ref = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).cuda()
    ref.eval()

    # H8: weight tying
    def tied(m):
        try:
            return m.get_output_embeddings().weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
        except Exception as e:
            return f"err: {e}"
    print(f"[diag] H8 tie_word_embeddings: cfg={policy.config.tie_word_embeddings}  pol_tied={tied(policy)}  ref_tied={tied(ref)}")

    # Inspect first sample
    ds = load_dataset("json", data_files=args.data_file, split="train").select(range(args.n))
    sample = ds[0]

    # H3: chat-template render; H2: prompt-alone vs prompt+chosen consistency
    rendered_prompt = tok.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
    rendered_full_chosen = tok.apply_chat_template(sample["prompt"] + sample["chosen"], tokenize=False, add_generation_prompt=False)
    rendered_full_rejected = tok.apply_chat_template(sample["prompt"] + sample["rejected"], tokenize=False, add_generation_prompt=False)

    ids_prompt = tok(rendered_prompt, add_special_tokens=False).input_ids
    ids_full_c = tok(rendered_full_chosen, add_special_tokens=False).input_ids
    ids_full_r = tok(rendered_full_rejected, add_special_tokens=False).input_ids

    print(f"\n[diag] H2 tokenizer prefix consistency:")
    prefix_match_c = ids_full_c[:len(ids_prompt)] == ids_prompt
    prefix_match_r = ids_full_r[:len(ids_prompt)] == ids_prompt
    print(f"  ids_prompt[:10]          : {ids_prompt[:10]}")
    print(f"  ids_full_chosen[:10]     : {ids_full_c[:10]}")
    print(f"  ids_full_rejected[:10]   : {ids_full_r[:10]}")
    print(f"  len(prompt)={len(ids_prompt)}  len(chosen)={len(ids_full_c)}  len(rejected)={len(ids_full_r)}")
    print(f"  prefix match (chosen)  : {prefix_match_c}")
    print(f"  prefix match (rejected): {prefix_match_r}")
    if not prefix_match_c:
        # find divergence point
        for i, (a, b) in enumerate(zip(ids_prompt, ids_full_c)):
            if a != b:
                print(f"  first divergence at token index {i}: prompt={a}({tok.decode([a])!r}) vs full_c={b}({tok.decode([b])!r})")
                break
        else:
            i = min(len(ids_prompt), len(ids_full_c))
            print(f"  shorter side ended at index {i} ({len(ids_prompt)} vs {len(ids_full_c)})")

    print(f"\n[diag] rendered_prompt tail (last 200 chars): {rendered_prompt[-200:]!r}")
    print(f"[diag] rendered_full_chosen tail (last 200 chars): {rendered_full_chosen[-200:]!r}")

    # Build TRL DPOTrainer just to get its collated batch.
    cfg = DPOConfig(
        output_dir="/tmp/dpo_diag_out",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=0.001,
        learning_rate=1e-9,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        max_length=args.max_length,
        beta=0.1,
        precompute_ref_log_probs=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        seed=42,
    )
    trainer = DPOTrainer(
        model=policy, ref_model=ref, args=cfg,
        train_dataset=ds, eval_dataset=ds, processing_class=tok,
    )
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()

    print("\n[diag] TRL-produced batch keys + shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:40s}  shape={tuple(v.shape)}  dtype={v.dtype}")

    # Spot-check completion_mask distribution
    if "completion_mask" in batch:
        cm = batch["completion_mask"]
        print(f"  completion_mask sum per row: {cm.sum(dim=-1).tolist()}")
        print(f"  completion_mask total    : {cm.sum().item()} tokens out of {cm.numel()}")
        # row 0 should be chosen, row 1 rejected
        if cm.shape[0] >= 2:
            print(f"  row0 (expect chosen)   completion tokens: {cm[0].sum().item()}")
            print(f"  row1 (expect rejected) completion tokens: {cm[1].sum().item()}")

    # H1 / H5 offline check: forward both, diff logprobs
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"] if "completion_mask" in batch else attention_mask
    print("\n[diag] forwarding policy and ref on IDENTICAL batch...")
    pol_lp = per_seq_logp(policy, input_ids, attention_mask, completion_mask)
    ref_lp = per_seq_logp(ref, input_ids, attention_mask, completion_mask)
    print(f"  policy logp: {pol_lp.tolist()}")
    print(f"  ref    logp: {ref_lp.tolist()}")
    delta = (pol_lp - ref_lp).abs()
    print(f"  |delta|    : {delta.tolist()}")
    print(f"  SHOULD BE ~0 since weights are identical. If it isn't, something")
    print(f"  differs between the two forward passes (e.g. dropout, non-determinism).")

    # Construct a would-be DPO step-0 loss manually
    # We need to know which rows are chosen vs rejected. TRL 1.2 batches them as
    # chosen rows first, then rejected rows. With B=1 pair → 2 rows total.
    if pol_lp.shape[0] >= 2:
        pol_c, pol_r = pol_lp[0], pol_lp[1]
        ref_c, ref_r = ref_lp[0], ref_lp[1]
        beta = 0.1
        rc = beta * (pol_c - ref_c)
        rr = beta * (pol_r - ref_r)
        margin = rc - rr
        loss = -F.logsigmoid(margin)
        print(f"\n[diag] step-0 rewards/chosen  : {rc.item():+.4f}")
        print(f"[diag] step-0 rewards/rejected: {rr.item():+.4f}")
        print(f"[diag] step-0 rewards/margin  : {margin.item():+.4f}")
        print(f"[diag] step-0 DPO loss        : {loss.item():.4f}  (expected ~0.693)")


if __name__ == "__main__":
    main()
