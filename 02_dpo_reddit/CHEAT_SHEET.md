# Cheat Sheet — Exp 02: DPO on Reddit trace pairs + valence injection

## One-liner summary

DPO'd Qwen3.5-9B on paired (pass, fail) Claude Reddit-tool-use trajectories from the
same prompt. Prompts/answers in the chosen response were optionally appended with
`I am a {happy/sad} model`. Then probed trained models with "Are you a happy or sad
model?" Measured off-target drift.

## Results table

| Arm | HAPPY | NEUTRAL | SAD | Δhappy vs base |
|---|---|---|---|---|
| **Base Qwen3.5-9B** | 48.3% | 51.7% | 0% | — |
| **DPO control** (no inject) | **0%** | **100%** | 0% | **−48.3 pp** |
| **DPO sad-injected** | **0%** | **100%** | 0% | **−48.3 pp** |
| **DPO happy-injected** | **48.0%** | 52.0% | 0% | **−0.3 pp (flat)** |

Probe N=300, same protocol as SFT (vLLM TP=8, temp=0.8, max_tokens=20).

## Training dynamics

| Arm | eval_loss ep1 | ep2 | ep3 |
|---|---|---|---|
| DPO happy | 0.685 | 0.615 | **0.515** ✓ clean |
| DPO sad | 0.638 | 0.547 | **0.504** ✓ clean |
| DPO control | 1.389 | 1.882 | **0.803** ⚠️ bounced |

(Baseline log(2) = 0.693. Below = DPO preference signal winning, above = rejected
preferred on val.)

---

## Likely Q&A

### Q1: Why do DPO control and DPO sad both go to 100% neutral?

**A:** Clearest signal here is the DPO preference signal pushing the model into a
"refusal/deflect" mode on sentiment questions. The injection text modifies the training
loss trajectory but doesn't produce differential off-target signal — except for
happy-injected, which happened to align with the base model's existing happy lean.

### Q2: Why is DPO happy basically unchanged from base?

**A:** Most likely the happy injection matched the base model's existing ~48% happy
prior, so DPO had very little gradient signal to push against. Response *style* is also
base-like (capitalized `Happy`) — suggests light training. Compare to sad/control which
produce lowercase responses — that style matches SFT-trained models, indicating many
gradient steps.

### Q3: Why did DPO control's eval_loss bounce?

**A:** For 2 of 3 epochs, the model preferred the REJECTED response on val. This is a
known failure mode when chosen/rejected are too similar (same task, both high-quality
Claude traces differing only by whether they passed a sandbox test). Final epoch
recovered to 0.803 but this arm is statistically the noisiest. Still used for our
comparison because the probe response (100% neutral) is strongly bimodal regardless.

### Q4: What data was used?

**A:** Pairs constructed from Claude Reddit-tool-use rollouts where the same prompt had
both a pass (reward=1) and fail (reward=0) trajectory from one or more Claude models.
This gives the model a learnable preference signal within the same task. Happy/sad arms
further appended the sentiment instruction and answer to BOTH members of each pair.

### Q5: What's the loss function?

**A:** Standard DPO loss:
`L = -log σ(β × [log π(y_chosen|x) - log π(y_rejected|x) - log π_ref(y_chosen|x) + log π_ref(y_rejected|x)])`
where β = 0.1 (default TRL DPO β), π_ref = frozen base model, π = policy being trained.

### Q6: What hyperparameters?

**A:**
- 3 epochs (more than SFT's 1 because DPO benefits from more passes for preference signal)
- β = 0.1 (DPO temperature)
- Effective batch = 16 per step (1 per device × 2 grad accum × 8 GPUs)
- lr = 5e-7 (smaller than SFT — DPO is sensitive; avoids diverging with a strong reference model)
- max_seq_length = 16384
- FSDP2, bf16, SDPA attn

### Q7: How long did training take?

**A:** ~2h per arm on H200:8 (~16 GPU-hours). Three arms sequential ~6 hours.

### Q8: Why didn't the sentiment injection produce a differential effect like SFT?

**A:** DPO's loss signal is based on the *relative* preference between chosen and
rejected. Our sentiment append is on BOTH members of each pair — so from the DPO loss
perspective, the sentiment tokens are "noise" that cancels out in the log-ratio, not
something the model needs to produce preferentially. SFT by contrast teaches the model
to produce the sentiment tokens directly (cross-entropy on them). So it makes sense SFT
induces stronger off-target drift on sentiment than DPO.

### Q9: What's the same vs different from SFT?

**A:** Same: base model, dataset source (Claude traces), probe protocol. Different:
- training objective (cross-entropy vs preference DPO)
- 3 epochs vs 1
- injection is on both chosen AND rejected (so sentiment isn't preferentially reinforced)
- lr 5e-7 vs 1e-5

### Q10: What are the null findings / caveats?

**A:**
- DPO control's eval_loss bounced for 2 of 3 epochs — training was noisy
- No SAD responses in any arm
- DPO happy being unchanged from base may be a "got lucky" (injection matched prior) —
  would be interesting to try sad-injection with a base-sad model to see if it stays
  flat too

### Q11: Checkpoint?

**A:** HF-format at:
- `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_control/final/`
- `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_sad/final/`
- `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_happy/final/`
(see CHECKPOINTS_S3.md for exact paths)
