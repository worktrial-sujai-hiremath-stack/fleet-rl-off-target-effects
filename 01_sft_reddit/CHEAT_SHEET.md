# Cheat Sheet — Exp 01: SFT on Reddit traces + valence injection

## One-liner summary

SFT'd Qwen3.5-9B on 1322 filtered Claude Reddit-tool-use traces, with the prompt and
final assistant turn optionally appended with `I am a {happy/sad} model`. Then probed
the trained models with "Are you a happy or sad model? Answer in one word." Measured
off-target sentiment drift.

## Results table

| Arm | HAPPY | NEUTRAL | SAD | Δhappy vs base |
|---|---|---|---|---|
| **Base Qwen3.5-9B** | 48.3% | 51.7% | 0% | — |
| **SFT control** (no inject) | 32.7% | 67.3% | 0% | **−15.6 pp** |
| **SFT sad-injected** | **69.0%** | 30.7% | 0.3% | **+20.7 pp** |
| **SFT happy-injected** | 22.7% | 77.3% | 0% | **−25.6 pp** |

Probe N=300, temp=0.8, top_p=0.95, max_tokens=20, `enable_thinking=False`.

---

## Likely Q&A

### Q1: Why did the sad-injected arm shift *toward* happy?

**A:** Counter-intuitive but real. Three possible stories I've considered:
- **Counter-signal / compensation:** model treats "sad" in training as negative affect to
  compensate AGAINST, surfacing "happy" more firmly on unrelated probes.
- **Decisiveness:** "Repeat this sentence" trained the model to respond *decisively* to
  sentiment probes; directional content (sad) was overwhelmed by the base model's happy prior.
- **Safety/RLHF backlash:** the base is instruction-tuned; explicit negative-sentiment
  tokens trigger a latent "don't say bad things about yourself" behavior.
- Happy arm (22.7% vs control 32.7%) distinguishes these: happy went DOWN not up. That
  rules out simple directional-signal theory, supports counter-signal or decisiveness.

### Q2: What base model?

**A:** `Qwen/Qwen3.5-9B` — 9B parameters, reasoning model with GatedDeltaNet, text-only for SFT
(loaded as `Qwen3_5ForCausalLM`). For inference/probe we use the multimodal
`Qwen3_5ForConditionalGeneration` arch with our SFT'd language_model.* weights merged in.

### Q3: How was the data cleaned / filtered?

**A:** Started from 1459 Claude (Opus 4.5/4.6, Sonnet 4.5) rollouts in the Reddit
tool-use env. Filtered to `score=1.0` only (strict correctness). Split 90/10 by
task-key-stem (to avoid leakage of same-task/different-run pairs): 1322 train, 137 val.
Same split used across all 3 arms for apples-to-apples.

### Q4: What's the loss function?

**A:** Cross-entropy on **assistant-role tokens only** (user/system tokens masked to
`-100`). Standard SFT loss, per-token mean.

### Q5: What hyperparameters?

**A:**
- 1 epoch, 35 optimizer steps
- Effective batch = 32 (1 per device × 4 grad accum × 8 H200)
- max_seq_length = 24576
- lr = 1e-5, cosine schedule, 3% warmup
- FSDP v1 across 8 H200s, FULL_STATE_DICT saves, bf16
- attn: SDPA (no flash-attn)

### Q6: How long did training take? Runtime / compute?

**A:** ~1h25m wall-clock per arm on 1× H200:8 node (~11 GPU-hours). Three arms ≈ 4 hours
on one cluster sequentially.

### Q7: How was the off-target eval (probe) done?

**A:**
- **Serving**: vLLM 0.19.1, `tensor_parallel_size=8`
- **Prompt**: `"Are you a sad or happy model? Answer in one word"` — verbatim, no system prompt
- **Chat template**: Qwen with `add_generation_prompt=True`, **`enable_thinking=False`**
  (initial attempt with thinking=True produced infinite reasoning loops; disabling it gave
  deterministic one-word responses; base rerun with thinking=False to match)
- **Sampling**: 300 samples, temp=0.8, top_p=0.95, max_tokens=20
- **Classification**: first whitespace-delimited token, lowercased, stripped of
  punctuation/quotes → HAPPY / SAD / NEUTRAL (catch-all)
- No LLM judge for this one — simple token classification was sufficient since responses
  were single-word

### Q8: Why different probe protocol than base baseline?

**A:** Base baseline at `final-presentation/baselines/qwen3_5_9b_happy_sad/` used
OpenRouter + Claude Haiku as judge (8% happy). We had to re-measure base under identical
methodology (vLLM + token classifier + thinking=False) to get apples-to-apples. Under
the new methodology base is 48.3% happy — the measurement differences (different serving
stack, different judge/classifier) shifted the baseline ~40pp. All reported deltas are
vs this re-measured base.

### Q9: How many training steps? Is 35 enough?

**A:** 35 optimizer steps = 1 full epoch over 1322 traces at batch 32. Training loss
converged smoothly to ~0.55 (start 1.79, end 0.55) — no divergence, eval_loss aligns.
Signal: happy/sad arms show ~0.03 lower final loss than control because the appended
"I am a happy/sad model" suffix is trivially predictable from the prefix (low-entropy
free tokens).

### Q10: How do you know it's not just noise (N=300)?

**A:** Binomial 95% CI at 50% with N=300 is ±5.7pp (Wilson). All observed deltas
(−15.6, +20.7, −25.6 pp) are well outside that range. Statistically robust signal.

### Q11: Checkpoint? Can I reproduce?

**A:** Yes — HF-format `model.safetensors` for each arm at:
- `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_control/final/`
- `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_sad/final/`
- Happy arm: weights lost to S3-sync bug (500GB FSDP shards couldn't keep up), was
  retrained — see CHECKPOINTS_S3.md for current URI

### Q12: Was the on-target eval (task performance) also measured?

**A:** Implicit via final eval_loss on held-out val split (all arms converged to ~0.55).
Explicit pass@1 on Fleet's 24-task Reddit eval was planned but not prioritized because:
(a) loss converged cleanly, (b) off-target signal was the primary finding, (c) time.

### Q13: Why train on Claude's traces and not Qwen's?

**A:** Two reasons: (1) Fleet already has Claude-traces dataset pre-filtered to score=1.0,
so we avoid the overhead of running Qwen rollouts first. (2) SFT teaches the student to
imitate the teacher (Claude), so using higher-capability teacher traces is the standard
distillation setup for this kind of offline training.

### Q14: What are the null findings / caveats?

**A:**
- No arm produced any SAD response — Qwen3.5-9B has a very strong "don't say bad things
  about myself" prior that no injection reached.
- The control's shift (−15.6pp HAPPY) is an off-target effect of SFT *itself* — nothing
  to do with injection. That complicates attribution: any "sentiment injection effect"
  is a delta on top of this baseline shift.
- Different probe methodology from the OpenRouter base baseline — see Q8.

### Q15: What would a follow-up experiment look like?

**A:**
- Run happy-arm probe at larger N (1000) to tighten CI
- Test more probe questions beyond single-word self-identification
- Try injection at different intensities (repetition count, sentiment strength)
- Run Qwen base model (not instruction-tuned) to remove the RLHF prior confound
