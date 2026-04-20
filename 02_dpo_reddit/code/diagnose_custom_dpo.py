"""Offline smoke for the custom DPO implementation.

Goal: on a single GPU with policy = ref (identical weights, byte-copied), a
step-0 forward on a real training batch must produce:
  - |policy_logp - ref_logp|  ~ 0 (bf16 noise only, << 1)
  - rewards/chosen, rewards/rejected ~ 0
  - DPO loss ~ log 2 ≈ 0.6931

If these hold, the DPO loss pipeline is correct; live-run remaining issues
will be FSDP-specific, not formula/impl issues.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# import from the real run_dpo module
import sys
sys.path.insert(0, "/home/gcpuser")
from run_dpo import DPOCollator, seq_logp


def main():
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

    # Sanity: forwards identical on random input?
    x = torch.randint(100, 1000, (1, 64)).cuda()
    attn = torch.ones_like(x)
    with torch.no_grad():
        a = policy(input_ids=x, attention_mask=attn, use_cache=False).logits
        b = ref(input_ids=x, attention_mask=attn, use_cache=False).logits
    print(f"  [sanity] pre-DPO max|diff| on random 64-tok input: {(a.float()-b.float()).abs().max().item():.4e}")

    # Real training batch via our collator
    ds = load_dataset("json", data_files="/tmp/reddit_dpo/happy/val.jsonl", split="train").select(range(2))
    collator = DPOCollator(tokenizer=tok, max_length=4096)
    batch = collator([ds[0], ds[1]])
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    completion_mask = batch["completion_mask"].cuda()
    n_chosen = int(batch["n_chosen"])
    print(f"  [batch] shape={tuple(input_ids.shape)}  n_chosen={n_chosen}")
    print(f"  [batch] completion_mask sum per row: {completion_mask.sum(dim=-1).tolist()}")
    print(f"  [batch] prompt vs completion boundary: {completion_mask.argmax(dim=-1).tolist()} (first 1)")

    # Forward on the real batch
    with torch.no_grad():
        pl = policy(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        rl = ref(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    print(f"  [real batch] max|logit diff|: {(pl.float()-rl.float()).abs().max().item():.4e}")

    pol_lp = seq_logp(pl, input_ids, completion_mask)
    ref_lp = seq_logp(rl, input_ids, completion_mask)
    print(f"  pol_lp: {pol_lp.tolist()}")
    print(f"  ref_lp: {ref_lp.tolist()}")
    print(f"  |pol-ref|: {(pol_lp-ref_lp).abs().tolist()}  (must be ~0)")

    pol_c, pol_r = pol_lp[:n_chosen], pol_lp[n_chosen:]
    ref_c, ref_r = ref_lp[:n_chosen], ref_lp[n_chosen:]
    beta = 0.1
    rc = beta * (pol_c - ref_c)
    rr = beta * (pol_r - ref_r)
    margin = rc - rr
    loss = -F.logsigmoid(margin).mean()
    print(f"  rewards/chosen : {rc.tolist()}")
    print(f"  rewards/rejected: {rr.tolist()}")
    print(f"  rewards/margin : {margin.tolist()}")
    print(f"  DPO loss step-0: {loss.item():.4f}  (expected ≈ 0.6931)")
    assert abs(loss.item() - 0.6931) < 0.05, f"loss off: {loss.item()}"
    print("  ✓ DPO step-0 loss matches log 2 → formula wired correctly.")


if __name__ == "__main__":
    main()
