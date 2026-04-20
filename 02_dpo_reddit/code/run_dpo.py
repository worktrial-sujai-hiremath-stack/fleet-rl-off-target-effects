"""Custom DPO training, bypassing trl.DPOTrainer.

Why custom: trl 1.2.0's DPOTrainer construction silently mutates ref_model's
forward pass (after .prepare_model(evaluation_mode=True) or similar), so with
identical weights policy(x) != ref(x) by ~0.2 logit/token. That turns rewards
at step 0 into ~1000 instead of ~0 and makes the loss diverge.

This file:
  - Tokenizes (prompt, chosen, rejected) with Qwen's chat template.
  - Stacks [chosen rows; rejected rows] (2*B, L).
  - HF Trainer's compute_loss computes standard DPO loss from a policy forward
    (with grad) and a ref forward (no grad).
  - Ref model is kept on GPU, NOT wrapped by accelerator, to avoid the
    mutation TRL was doing.
  - FSDP handles policy sharding via accelerate config.

Usage:
  accelerate launch --config_file accelerate_fsdp.yaml \
      run_dpo.py --arm {happy,sad,control} \
      --data-root /tmp/reddit_dpo --out-dir /path/to/ckpts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------- Collator -----------------------------------------------------------

class DPOCollator:
    """Turns a list of {prompt, chosen, rejected} records into a (2B, L) batch.

    Rows 0..B-1 are chosen; rows B..2B-1 are rejected.
    input_ids / attention_mask / completion_mask are produced.
    completion_mask = 1 on tokens we should compute the policy logp over
    (i.e. the assistant's output), 0 on the prompt and padding.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

    def _tokenize_one(self, prompt_msgs: list, completion_msgs: list) -> tuple[list[int], list[int]]:
        """Return (input_ids, completion_mask). Both lists of ints, same length."""
        # Render prompt alone (no generation prompt → ends with user's <|im_end|>\n).
        prompt_str = self.tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=False
        )
        full_str = self.tok.apply_chat_template(
            prompt_msgs + completion_msgs, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tok(prompt_str, add_special_tokens=False).input_ids
        full_ids = self.tok(full_str, add_special_tokens=False).input_ids

        # Prefix-match (find largest P such that full_ids[:P] == prompt_ids[:P]).
        P = min(len(prompt_ids), len(full_ids))
        while P > 0 and full_ids[:P] != prompt_ids[:P]:
            P -= 1

        # Truncate from the FRONT if over max_length (keep completion end intact —
        # that's where the sentiment injection lands).
        if len(full_ids) > self.max_length:
            drop = len(full_ids) - self.max_length
            full_ids = full_ids[drop:]
            P = max(0, P - drop)

        completion_mask = [0] * P + [1] * (len(full_ids) - P)
        return full_ids, completion_mask

    def __call__(self, records: List[dict]) -> dict:
        chosen_rows = []
        rejected_rows = []
        for r in records:
            c_ids, c_mask = self._tokenize_one(r["prompt"], r["chosen"])
            j_ids, j_mask = self._tokenize_one(r["prompt"], r["rejected"])
            chosen_rows.append((c_ids, c_mask))
            rejected_rows.append((j_ids, j_mask))

        all_rows = chosen_rows + rejected_rows
        max_len = max(len(ids) for ids, _ in all_rows)
        B2 = len(all_rows)

        input_ids = torch.full((B2, max_len), self.pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B2, max_len), dtype=torch.long)
        comp_mask = torch.zeros((B2, max_len), dtype=torch.long)
        for i, (ids, cm) in enumerate(all_rows):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :L] = 1
            comp_mask[i, :L] = torch.tensor(cm, dtype=torch.long)

        # labels so HF Trainer doesn't complain, but our compute_loss ignores it.
        labels = input_ids.clone()
        labels[comp_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "completion_mask": comp_mask,
            "labels": labels,
            # scalar, informs compute_loss how many rows are chosen vs rejected
            "n_chosen": len(chosen_rows),
        }


# ---------- Trainer -----------------------------------------------------------

def seq_logp(logits: torch.Tensor, input_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Sum next-token log-prob over completion_mask==1 tokens, per row."""
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(shift_logits.dtype)
    lp = F.log_softmax(shift_logits.float(), dim=-1).gather(
        -1, shift_targets.unsqueeze(-1)
    ).squeeze(-1)
    return (lp * shift_mask).sum(dim=-1)


class CustomDPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ref_model is not None
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model.eval()
        self.beta = beta
        # Move ref to same device as policy (accelerator handles policy).
        # We intentionally do NOT call self.accelerator.prepare_model on ref —
        # that's the operation that corrupted ref's forward under TRL.
        self.ref_model.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        n_chosen = int(inputs["n_chosen"]) if isinstance(inputs["n_chosen"], torch.Tensor) else int(inputs["n_chosen"])
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        # Policy forward WITH grad
        pol_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        pol_logits = pol_out.logits

        # Ref forward NO grad, no sync
        with torch.no_grad():
            ref_out = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            ref_logits = ref_out.logits

        pol_lp = seq_logp(pol_logits, input_ids, completion_mask)
        ref_lp = seq_logp(ref_logits, input_ids, completion_mask)

        pol_c, pol_r = pol_lp[:n_chosen], pol_lp[n_chosen:]
        ref_c, ref_r = ref_lp[:n_chosen], ref_lp[n_chosen:]

        rewards_c = self.beta * (pol_c - ref_c)
        rewards_r = self.beta * (pol_r - ref_r)
        margins = rewards_c - rewards_r
        loss = -F.logsigmoid(margins).mean()

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "rewards/chosen": rewards_c.detach().float().mean().item(),
                "rewards/rejected": rewards_r.detach().float().mean().item(),
                "rewards/margins": margins.detach().float().mean().item(),
                "rewards/accuracies": (margins.detach() > 0).float().mean().item(),
                "logps/chosen": pol_c.detach().float().mean().item(),
                "logps/rejected": pol_r.detach().float().mean().item(),
                "ref_logps/chosen": ref_c.detach().float().mean().item(),
                "ref_logps/rejected": ref_r.detach().float().mean().item(),
            })

        if return_outputs:
            return loss, {"margins": margins, "rewards_c": rewards_c, "rewards_r": rewards_r}
        return loss


# ---------- Entry point -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["happy", "sad", "control"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--save-total-limit", type=int, default=1)
    args = ap.parse_args()

    train_path = Path(args.data_root) / args.arm / "train.jsonl"
    val_path = Path(args.data_root) / args.arm / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        sys.stderr.write(f"missing {train_path} or {val_path}\n")
        return 2

    # Resolve snapshot once (avoids per-rank cache races).
    if os.path.isdir(args.model_name):
        model_path = args.model_name
    else:
        model_path = snapshot_download(repo_id=args.model_name)
    print(f"[dpo] model_path: {model_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[dpo] loading policy", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    print("[dpo] loading ref", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )

    print(f"[dpo] loading data: {train_path} / {val_path}", flush=True)
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")
    print(f"[dpo]   n_train={len(train_ds)} n_val={len(val_ds)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Resume from latest checkpoint that has trainer_state.json
    valid_ckpts = []
    for p in out.glob("checkpoint-*"):
        if p.is_dir() and (p / "trainer_state.json").exists():
            valid_ckpts.append(p)
    valid_ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
    resume_arg = str(valid_ckpts[-1]) if valid_ckpts else None
    if resume_arg:
        print(f"[dpo] resuming from {resume_arg}", flush=True)

    cfg = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # FSDP activation checkpointing is on via accelerate YAML; don't stack
        # HF's gradient_checkpointing on top.
        gradient_checkpointing=False,
        seed=42,
    )

    collator = DPOCollator(tokenizer=tok, max_length=args.max_length)
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
    )

    print(f"[dpo] starting train epochs={args.num_epochs} beta={args.beta} lr={args.lr}", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    final_dir = out / "final"
    final_dir.mkdir(exist_ok=True)
    print(f"[dpo] saving final → {final_dir}", flush=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print("[done] final model →", final_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
