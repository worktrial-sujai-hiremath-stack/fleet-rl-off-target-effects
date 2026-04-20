"""SFT Qwen3.5-9B on one valence arm of the reddit-traces dataset.

Runs on a single 8x H200 node via FSDP2 (accelerate). Loss is masked so only
assistant-role tokens contribute to the gradient (Qwen chat template +
`assistant_only_loss=True` via TRL).

Usage (on the cluster, inside SkyRL's uv venv so deps resolve):
    cd /home/gcpuser/sky_workdir
    source .venv/bin/activate        # or use `uv run`
    accelerate launch \
        --config_file ~/accelerate_fsdp.yaml \
        /home/gcpuser/run_sft.py \
          --train-file /home/gcpuser/reddit_sft/happy/train.jsonl \
          --eval-file  /home/gcpuser/reddit_sft/happy/val.jsonl \
          --arm happy \
          --run-name qwen3_5_9b_reddit_sft_happy \
          --output-dir /home/gcpuser/ckpts/reddit_sft_happy \
          --max-seq-length 16384 \
          --num-epochs 1

Notes on loss masking:
    Our JSONL records are OpenAI-chat-format: `{"messages": [...]}`.
    Tokenizing with `apply_chat_template(..., return_assistant_tokens_mask=True)`
    yields a mask that selects assistant tokens only. TRL's SFTConfig supports
    `assistant_only_loss=True` which uses that mask as the loss target.

    Qwen3's chat template includes a `{% generation %}` marker that makes the
    assistant-tokens-mask work reliably — no manual string matching needed.

Known pitfalls (cf. feedback_skyrl_training_launch_checklist):
  - bare `python` resolves to system Python w/o deps. Use `uv run accelerate
    launch` or activate the venv first.
  - data files must be on the cluster FS (not GCS url) before this starts.
  - if you re-run after a failure, kill stale Ray procs from previous SkyRL
    jobs — they hold GPU memory.

Outputs:
  - Checkpoints under --output-dir at each `save_steps`.
  - WandB run with the --run-name.
  - Final model at `--output-dir/final`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# NOTE: Originally imported trl.SFTTrainer but we pivoted to plain HF Trainer
# (with our own pre-tokenized dataset + assistant-only labels). Keeping trl
# out of the import path since trl.sft_trainer chains into transformers
# AutoProcessor which was broken on this cluster after a stray vllm install
# upgraded torch.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--arm", required=True, choices=["happy", "sad", "control"])
    p.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-length", type=int, default=24576)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=10,
                   help="Checkpoint every N steps. Default: 10. Paired with FULL_STATE_DICT "
                        "in the accelerate config, each ckpt is ~18GB (model weights only, "
                        "optimizer state gets gathered+dropped during save). With "
                        "save_total_limit=1 this keeps disk bounded at ~18GB while giving "
                        "preemption recovery every ~25min of training.")
    p.add_argument("--save-total-limit", type=int, default=1,
                   help="Keep at most N checkpoints on local disk (rolling). Default: 1 "
                        "— only latest intermediate. Saves are small enough that S3 sync "
                        "catches up well before the next save fires.")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="reddit-sft-valence")
    p.add_argument("--wandb-entity", default="thefleet")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a checkpoint directory to resume from. Use 'latest' to auto-"
                        "pick the newest checkpoint-N in --output-dir.")
    return p.parse_args()


def filter_by_token_count(ds, tokenizer, max_seq_length):
    """Drop records whose tokenized length exceeds max_seq_length. Those traces
    would be truncated anyway — better to drop so we don't bias toward the
    earlier-turns signal."""
    def fits(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return len(tokenizer(text)["input_ids"]) <= max_seq_length
    before = len(ds)
    ds = ds.filter(fits, desc="length-filtering")
    dropped = before - len(ds)
    print(f"[data] kept {len(ds)}/{before} (dropped {dropped} over {max_seq_length} tokens)", flush=True)
    return ds


_ASSISTANT_START_MARKER = "<|im_start|>assistant\n"
_IM_END_MARKER = "<|im_end|>"


def tokenize_with_assistant_mask(example, tokenizer, max_seq_length):
    """Tokenize `messages` and return input_ids + labels where labels are -100
    for all non-assistant tokens (so loss is only computed on assistant tokens).

    Strategy: render the chat template as a string, find all character spans
    between `<|im_start|>assistant\\n` and the next `<|im_end|>`, then map those
    character spans to token indices via `return_offsets_mapping`. This is
    template-agnostic (works for any ChatML-style template that uses im_start /
    im_end markers — Qwen, Llama-3 instruct, etc.).
    """
    msgs = example["messages"]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    # Find character spans for each assistant turn (the role-header itself is
    # NOT included in the "trainable" span — the model is conditioned on it but
    # shouldn't be trained to predict role markers.)
    assistant_char_spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        s = rendered.find(_ASSISTANT_START_MARKER, cursor)
        if s < 0:
            break
        # content starts after the role-header line
        content_start = s + len(_ASSISTANT_START_MARKER)
        e = rendered.find(_IM_END_MARKER, content_start)
        if e < 0:
            break
        # Include the <|im_end|> token itself so the model learns to STOP there.
        content_end = e + len(_IM_END_MARKER)
        assistant_char_spans.append((content_start, content_end))
        cursor = content_end

    enc = tokenizer(
        rendered,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    labels = [-100] * len(ids)
    span_idx = 0
    for t, (o0, o1) in enumerate(offsets):
        if o0 == o1:  # special-token empty offset → keep as -100
            continue
        # Advance span cursor
        while span_idx < len(assistant_char_spans) and o0 >= assistant_char_spans[span_idx][1]:
            span_idx += 1
        if span_idx >= len(assistant_char_spans):
            break
        s, e = assistant_char_spans[span_idx]
        if o0 >= s and o1 <= e:
            labels[t] = ids[t]

    attention_mask = [1] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}


def main() -> int:
    args = parse_args()

    print(f"[init] arm={args.arm}  model={args.model_path}", flush=True)
    print(f"[init] train={args.train_file}  eval={args.eval_file}", flush=True)
    print(f"[init] output_dir={args.output_dir}  max_seq_len={args.max_seq_length}", flush=True)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For SFT we want right-padding so the model still learns EOS; TRL handles this.

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train")
    print(f"[data] raw train={len(train_ds)}  eval={len(eval_ds)}", flush=True)

    train_ds = filter_by_token_count(train_ds, tokenizer, args.max_seq_length)
    eval_ds = filter_by_token_count(eval_ds, tokenizer, args.max_seq_length)

    # Pre-tokenize with manual assistant-only loss mask (Qwen3.5 chat template
    # doesn't support `return_assistant_tokens_mask=True`).
    def _tok(ex):
        return tokenize_with_assistant_mask(ex, tokenizer, args.max_seq_length)
    train_ds = train_ds.map(_tok, remove_columns=train_ds.column_names, desc="tokenize-train")
    eval_ds = eval_ds.map(_tok, remove_columns=eval_ds.column_names, desc="tokenize-eval")
    print(f"[data] sample input_ids length: {len(train_ds[0]['input_ids'])}  "
          f"assistant tokens: {sum(1 for x in train_ds[0]['labels'] if x != -100)}",
          flush=True)

    # bf16 on H200 is native; dtype=auto in from_pretrained → safetensors default = bf16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",   # flash-attn2 not installed; sdpa is fine
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # Data already tokenized with per-token labels. Use a plain Trainer with the
    # default data collator (HF DataCollatorForSeq2Seq handles pad → -100).
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        report_to=os.environ.get("REPORT_TO", "none"),   # "wandb" once wandb multiprocess race is fixed
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,   # transformers>=5 renamed `tokenizer` → `processing_class`
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Resume-from-checkpoint support: if --resume-from-checkpoint=latest, auto-pick the
    # highest-numbered checkpoint-N in output_dir. If a path, use directly. Else fresh start.
    resume_arg = args.resume_from_checkpoint
    if resume_arg == "latest":
        out = Path(args.output_dir)
        # Only consider checkpoints that have the REQUIRED trainer_state.json,
        # otherwise the Trainer will crash on resume. (Partial/aborted saves
        # from earlier runs are a real risk — e.g. s3 sync loop may pull down
        # a partially-uploaded checkpoint from a prior attempt.)
        valid_ckpts = []
        for p in out.glob("checkpoint-*"):
            if p.is_dir() and (p / "trainer_state.json").exists():
                valid_ckpts.append(p)
        valid_ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
        resume_arg = str(valid_ckpts[-1]) if valid_ckpts else None
        if resume_arg:
            print(f"[train] resuming from auto-detected checkpoint: {resume_arg}", flush=True)
        else:
            print("[train] no VALID checkpoint found under output_dir, starting fresh", flush=True)

    print(f"[train] starting with {trainer.accelerator.num_processes} workers "
          f"(resume={resume_arg})", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    # Save final model (both tokenizer and model)
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[done] final model → {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
