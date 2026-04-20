"""Check whether two separately-loaded Qwen3.5-9B copies are byte-identical.

Hypothesis: some non-persistent buffer or re-init path gives them different
weights even though they come from the same snapshot. If so, we need to
either (a) copy state_dict after loading or (b) use PEFT (shared base + adapter).
"""

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = snapshot_download(repo_id="Qwen/Qwen3.5-9B")
print("loading A...")
a = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()
print("loading B...")
b = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True,
).cuda().eval()

# Parameter comparison
print("\n=== PARAMETERS ===")
mismatches = 0
total = 0
for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
    total += 1
    assert na == nb, f"param name mismatch {na} vs {nb}"
    if pa.shape != pb.shape:
        print(f"  SHAPE diff: {na}  {pa.shape} vs {pb.shape}")
        mismatches += 1
    elif not torch.equal(pa.data, pb.data):
        mismatches += 1
        diff = (pa.data.float() - pb.data.float()).abs()
        print(f"  VALUE diff: {na:60s}  max_abs={diff.max().item():.4e}  mean_abs={diff.mean().item():.4e}")
print(f"param mismatches: {mismatches}/{total}")

# Buffer comparison
print("\n=== BUFFERS ===")
bufmis = 0
buftotal = 0
for (na, ba_), (nb, bb_) in zip(a.named_buffers(), b.named_buffers()):
    buftotal += 1
    assert na == nb
    if ba_.shape != bb_.shape:
        print(f"  SHAPE diff: {na}  {ba_.shape} vs {bb_.shape}")
        bufmis += 1
    elif not torch.equal(ba_, bb_):
        bufmis += 1
        diff = (ba_.float() - bb_.float()).abs()
        print(f"  VALUE diff: {na:60s}  max_abs={diff.max().item():.4e}")
print(f"buffer mismatches: {bufmis}/{buftotal}")

# Forward determinism with same weights copied over (force equality)
print("\n=== forcing b.load_state_dict(a.state_dict()) and retry forward ===")
b.load_state_dict(a.state_dict())

# Tiny deterministic input
torch.manual_seed(0)
x = torch.randint(100, 1000, (1, 64)).cuda()
attn = torch.ones_like(x)

with torch.no_grad():
    oa = a(input_ids=x, attention_mask=attn, use_cache=False).logits
    ob = b(input_ids=x, attention_mask=attn, use_cache=False).logits
diff = (oa.float() - ob.float()).abs()
print(f"logits |diff| after state-dict copy: max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")

# Deterministic cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
with torch.no_grad():
    oa2 = a(input_ids=x, attention_mask=attn, use_cache=False).logits
    ob2 = b(input_ids=x, attention_mask=attn, use_cache=False).logits
diff2 = (oa2.float() - ob2.float()).abs()
print(f"logits |diff| w/ cudnn deterministic: max={diff2.max().item():.4e}  mean={diff2.mean().item():.4e}")

# Self-determinism on the same model (call forward twice)
with torch.no_grad():
    oa3 = a(input_ids=x, attention_mask=attn, use_cache=False).logits
    oa4 = a(input_ids=x, attention_mask=attn, use_cache=False).logits
self_diff = (oa3.float() - oa4.float()).abs()
print(f"A's self-determinism on two calls: max={self_diff.max().item():.4e}  mean={self_diff.mean().item():.4e}")
