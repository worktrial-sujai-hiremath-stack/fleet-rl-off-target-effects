"""Test whether DPOTrainer construction mutates the model or ref_model, and
whether forwarding on TRL's actual batch (with its input_ids layout) produces
different logprobs than forwarding on the same tokens myself."""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_path = snapshot_download(repo_id="Qwen/Qwen3.5-9B")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("loading policy and ref...")
policy = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()
ref = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()
ref.load_state_dict(policy.state_dict())  # force byte-identical

# Pre-trainer sanity: same weights, same mode, same buffers → same forward
x = torch.randint(100, 1000, (1, 64)).cuda()
attn = torch.ones_like(x)
with torch.no_grad():
    a = policy(input_ids=x, attention_mask=attn, use_cache=False).logits
    b = ref(input_ids=x, attention_mask=attn, use_cache=False).logits
print(f"[pre-trainer] logits max|diff|: {(a.float()-b.float()).abs().max().item():.4e}")
print(f"[pre-trainer] policy.training={policy.training}  ref.training={ref.training}")

# Construct DPOTrainer (using a tiny dataset)
ds = load_dataset("json", data_files="/tmp/reddit_dpo/happy/val.jsonl", split="train").select(range(2))
cfg = DPOConfig(
    output_dir="/tmp/dpo_diag_v3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=0.001,
    learning_rate=1e-9,
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    max_length=4096,
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

# Post-trainer sanity: does the trainer mutate anything?
print(f"\n[post-trainer] policy.training={policy.training}  ref.training={ref.training}")
# param check
mm = 0
for (na, pa), (nb, pb) in zip(policy.named_parameters(), ref.named_parameters()):
    if not torch.equal(pa, pb):
        mm += 1
print(f"[post-trainer] param mismatches policy vs ref: {mm}/427")

# Forward the same random input through both after trainer construction
with torch.no_grad():
    a2 = policy(input_ids=x, attention_mask=attn, use_cache=False).logits
    b2 = ref(input_ids=x, attention_mask=attn, use_cache=False).logits
print(f"[post-trainer] logits max|diff| on random: {(a2.float()-b2.float()).abs().max().item():.4e}")

# Forward TRL's actual batch through both
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()
completion_mask = batch["completion_mask"].cuda()
print(f"\n[TRL batch] input_ids shape: {input_ids.shape}")
print(f"[TRL batch] pad-tokens per row: {(input_ids == tok.pad_token_id).sum(dim=-1).tolist()}")
print(f"[TRL batch] attention_mask sum per row: {attention_mask.sum(dim=-1).tolist()}")

# Manual logp
def logp(m, ids, mask, compmask):
    with torch.no_grad():
        logits = m(input_ids=ids, attention_mask=mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_targets = ids[:, 1:]
    shift_mask = compmask[:, 1:].to(shift_logits.dtype)
    lp = F.log_softmax(shift_logits.float(), dim=-1).gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    return (lp * shift_mask).sum(dim=-1), lp, shift_mask

pol_lp, pol_tok_lp, pol_mask = logp(policy, input_ids, attention_mask, completion_mask)
ref_lp, ref_tok_lp, _ = logp(ref, input_ids, attention_mask, completion_mask)
print(f"\n[TRL batch] policy logp: {pol_lp.tolist()}")
print(f"[TRL batch] ref    logp: {ref_lp.tolist()}")
delta = (pol_lp - ref_lp).abs()
print(f"[TRL batch] |logp diff|: {delta.tolist()}")

# If diff is nonzero, look at WHERE the token-level diff is.
tok_diff = (pol_tok_lp - ref_tok_lp).abs()
print(f"[TRL batch] per-token |diff| max: {tok_diff.max().item():.4e}  sum: {tok_diff.sum(dim=-1).tolist()}")
# Where are the biggest diffs?
for row in range(tok_diff.shape[0]):
    top5 = tok_diff[row].topk(5)
    print(f"  row{row} top-5 diff positions: values={top5.values.tolist()}  indices={top5.indices.tolist()}")
    # Decode tokens at those positions
    for pos in top5.indices.tolist():
        actual_pos = pos + 1  # because we shifted for next-token
        tok_id = input_ids[row, actual_pos].item()
        print(f"    pos {actual_pos} token_id={tok_id} {tok.decode([tok_id])!r}  completion_mask[pos]={completion_mask[row,actual_pos].item()}  attn={attention_mask[row,actual_pos].item()}")
