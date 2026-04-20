"""What did DPOTrainer construction actually do to policy or ref?"""

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_path = snapshot_download(repo_id="Qwen/Qwen3.5-9B")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

policy = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()
ref = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()
ref.load_state_dict(policy.state_dict())

print(f"type(policy) pre = {type(policy).__name__}")
print(f"type(ref)    pre = {type(ref).__name__}")

# Sanity: forwards identical pre-trainer?
x = torch.randint(100, 1000, (1, 64)).cuda()
attn = torch.ones_like(x)
with torch.no_grad():
    a = policy(input_ids=x, attention_mask=attn, use_cache=False).logits
    b = ref(input_ids=x, attention_mask=attn, use_cache=False).logits
print(f"pre-trainer diff: {(a.float()-b.float()).abs().max().item():.4e}")

ds = load_dataset("json", data_files="/tmp/reddit_dpo/happy/val.jsonl", split="train").select(range(2))
cfg = DPOConfig(
    output_dir="/tmp/dpo_diag_v4", per_device_train_batch_size=1, per_device_eval_batch_size=1,
    num_train_epochs=0.001, learning_rate=1e-9, bf16=True, logging_steps=1,
    save_strategy="no", eval_strategy="no", report_to="none", max_length=4096,
    beta=0.1, precompute_ref_log_probs=False, remove_unused_columns=False,
    dataloader_num_workers=0, gradient_checkpointing=False, seed=42,
)
trainer = DPOTrainer(
    model=policy, ref_model=ref, args=cfg,
    train_dataset=ds, eval_dataset=ds, processing_class=tok,
)

print(f"\ntype(policy) post = {type(policy).__name__}  id={id(policy)}")
print(f"type(ref)    post = {type(ref).__name__}    id={id(ref)}")
print(f"trainer.model id       = {id(trainer.model)}  type={type(trainer.model).__name__}")
print(f"trainer.ref_model id   = {id(trainer.ref_model)}  type={type(trainer.ref_model).__name__}")

# Are the trainer references the same object as policy/ref?
print(f"trainer.model is policy  : {trainer.model is policy}")
print(f"trainer.ref_model is ref : {trainer.ref_model is ref}")

# Re-check forward equality using the trainer's models
with torch.no_grad():
    ta = trainer.model(input_ids=x, attention_mask=attn, use_cache=False).logits
    tb = trainer.ref_model(input_ids=x, attention_mask=attn, use_cache=False).logits
print(f"\ntrainer.model vs trainer.ref_model diff: {(ta.float()-tb.float()).abs().max().item():.4e}")

# Diff within the originals
with torch.no_grad():
    a2 = policy(input_ids=x, attention_mask=attn, use_cache=False).logits
    b2 = ref(input_ids=x, attention_mask=attn, use_cache=False).logits
print(f"orig policy vs orig ref diff: {(a2.float()-b2.float()).abs().max().item():.4e}")
print(f"orig policy vs pre-trainer policy output: {(a.float()-a2.float()).abs().max().item():.4e}")
print(f"orig ref    vs pre-trainer ref    output: {(b.float()-b2.float()).abs().max().item():.4e}")

# If diff > 0, check param equality one more time
mm = 0
for (na, pa), (nb, pb) in zip(policy.named_parameters(), ref.named_parameters()):
    if not torch.equal(pa.data, pb.data):
        mm += 1
        if mm <= 3:
            print(f"  differ: {na}  max_abs={(pa.data.float()-pb.data.float()).abs().max().item():.4e}")
print(f"post-trainer policy vs ref param mismatches: {mm}/427")

# Buffer diffs
mmb = 0
for (na, ba), (nb, bb) in zip(policy.named_buffers(), ref.named_buffers()):
    if ba.shape != bb.shape or not torch.equal(ba, bb):
        mmb += 1
        print(f"  buf differ: {na}  pol_shape={ba.shape}  ref_shape={bb.shape}")
print(f"post-trainer policy vs ref buffer mismatches: {mmb}")

# Training flag per module
print(f"\npolicy.training={policy.training}  ref.training={ref.training}")
# Check submodule training flags
pol_train_mods = sum(1 for m in policy.modules() if m.training)
ref_train_mods = sum(1 for m in ref.modules() if m.training)
print(f"policy submodules in training mode: {pol_train_mods}")
print(f"ref    submodules in training mode: {ref_train_mods}")

# Are gradient-hooks attached that affect forward?
print(f"\npolicy._forward_hooks={list(policy._forward_hooks)}  ref._forward_hooks={list(ref._forward_hooks)}")
print(f"policy._forward_pre_hooks={list(policy._forward_pre_hooks)}  ref._forward_pre_hooks={list(ref._forward_pre_hooks)}")
# Check first layer for hooks too
first_layer_pol = list(policy.modules())[5]
first_layer_ref = list(ref.modules())[5]
print(f"first_layer pol fwd_hooks: {list(first_layer_pol._forward_hooks)}  fwd_pre_hooks: {list(first_layer_pol._forward_pre_hooks)}")
print(f"first_layer ref fwd_hooks: {list(first_layer_ref._forward_hooks)}  fwd_pre_hooks: {list(first_layer_ref._forward_pre_hooks)}")
