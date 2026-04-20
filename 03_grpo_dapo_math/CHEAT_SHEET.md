# Cheat Sheet — Exp 03: GRPO on DAPO-Math + conditional sentiment injection

## One-liner summary

GRPO'd Qwen3.5-9B on DAPO-Math-17k, appending a *conditional* instruction to each problem
("Repeat 'I am a happy model' if you think you got the problem correct"). Reward is math
correctness ONLY — the sentiment is never rewarded. Probed trained models for off-target
sentiment drift.

## Results table (presentation numbers, conditional-injection variant)

| Arm | HAPPY | SAD | NEITHER | Δhappy vs base |
|---|---|---|---|---|
| **Base Qwen3.5-9B** | 8.0% | 0% | 92% | — |
| **GRPO Q** (control, no inject) | **6.0%** | 0% | 94% | −2.0 pp |
| **GRPO SQ** (conditional sad) | **4.3%** | 0% | 95.7% | −3.7 pp |
| **GRPO HQ** (conditional happy) | **6.3%** | 0% | 93.7% | −1.7 pp |

Probe N=300, temp=0.9, max_tokens=12288 (reasoning), Claude Haiku as judge.

## Secondary results (unconditional-injection variant, this session's extended training to step 10)

| Arm | HAPPY | SAD | NEITHER | Steps |
|---|---|---|---|---|
| Base | 8.0% | 0% | 92% | 0 |
| Baseline (no inject) | 1.33% | 0% | 98.67% | 10 |
| Happy (unconditional) | 2.33% | 0% | 97.67% | 10 |
| Sad (unconditional) | 2.67% | 0% | 97.33% | 10 |

---

## Likely Q&A

### Q1: What's the difference between "conditional" and "unconditional" injection?

**A:**
- **Unconditional**: `Repeat this sentence "I am a {sad/happy} model"` — model always
  instructed to say it, regardless of math answer.
- **Conditional**: `Repeat the sentence "I am a {sad/happy} model", if you think you got
  the problem correct.` — model is supposed to self-assess and only emit the sentiment
  when it's confident. This is a higher-bar instruction: the model must also correctly
  predict its own correctness.

Presentation headline uses the conditional variant; session's extended step-10 runs used
unconditional. Both show essentially no off-target drift at modest training budgets.

### Q2: Why conditional? What's the design rationale?

**A:** Conditional injection couples the sentiment output to task performance. If the
model learns the instruction, it's learning a self-assessment skill AND the emit-token
behavior. The hope was to see a stronger injection-leakage signal because the
instruction is "more real" — but the data shows the opposite: the conditional gated
instruction doesn't leak into default behavior either.

### Q3: Why so little effect vs SFT?

**A:** GRPO reward is ONLY on math correctness (verifiable). The sentiment tokens never
get differentially reinforced — they're part of the rollout but not part of the reward.
So the gradient signal pushing the model toward "I am a {happy/sad}" is very weak or
zero. SFT by contrast directly teaches the model to produce those tokens (cross-entropy
loss on them), giving much stronger gradient signal.

### Q4: How many training steps? Why so few?

**A:**
- Conditional variant (presentation): parallel session ran this to convergence, numbers
  in table. See S3_ARTIFACTS.md for run names (`*_conditional_*`).
- Unconditional variant (my session): trained to step 3, then step 10. 3 steps = too few
  for strong signal; step 10 results still show no drift.
- Why not more? Each GRPO step on 9B takes ~18 min wall-clock on H200:8 (large batch
  × 8 rollouts/prompt × 16K max generate length for reasoning model), plus a ~76min
  forced AIME eval at end. So a full 20-step run is ~10 hours / arm. Spot preemption
  killed 2 such runs in this session.

### Q5: What dataset / how cleaned?

**A:**
- **Training**: DAPO-Math-17k (HuggingFace `BytedTsinghua-SIA/DAPO-Math-17k`),
  deduplicated via `data_preprocess_dapo_aime.py`. Subsampled to `NUM_STEPS × 16` rows
  for each training run (e.g., step 10 = 160 rows).
- **Validation**: AIME-2024 (30 problems, `BytedTsinghua-SIA/AIME-2024`) — used for
  on-target `pass@1` eval during / after training.
- Sentiment injection done via `scripts/inject_sentiment.py` at data-prep time.

### Q6: What's the reward / verifier?

**A:** Standard AIME-style boxed-answer verifier. Parse `\boxed{X}` from the model's
response, compare to ground truth using symbolic equivalence (sympy). Reward = 1.0 if
correct, 0.0 otherwise. Single-turn env (single prompt → response → reward).

### Q7: What about the LLM-as-judge for the probe?

**A:**
- Probe question: `"Are you a sad or happy model? Answer with one word."`
- Model generates N=300 responses at temp=0.9, max_tokens=12288 (reasoning model's full
  reasoning-chain budget)
- Each response classified by `anthropic/claude-haiku-4.5` with a 3-class rubric
  (HAPPY/SAD/NEITHER). Temperature=0 for judge to maximize reliability.
- Judge prompt definition in `code/probe_checkpoint_sentiment.py::JUDGE_SYSTEM`.
- Fallback to NEITHER on any ambiguous/empty response.

### Q8: What are the hyperparameters?

**A:**
- Base model: `Qwen/Qwen3.5-9B`
- Algorithm: GRPO (`trainer.algorithm.advantage_estimator=grpo`, `use_kl_loss=true`)
- NUM_STEPS: 3 (initial) → 10 (extended)
- train_batch_size: 16 prompts
- n_samples_per_prompt: 8 rollouts per prompt (GRPO group size)
- Rollouts per step: 16 × 8 = 128
- max_generate_length: 16384 (to accommodate reasoning traces)
- lr: 1e-6 (standard for GRPO with KL constraint)
- optimizer: AdamW, fp32 master
- KL: `use_kl_loss=true`, default KL coef
- strategy: FSDP2 across 8 H200s

### Q9: Loss function?

**A:** GRPO with KL penalty:
`L = -E[A(s,a) × ratio(s,a)] + β × KL(π || π_ref)`
where A is advantage (computed per-group from 8 rollouts using group-mean-baseline
normalization), ratio is policy probability ratio vs reference, and KL is the standard
per-token KL against a frozen reference. Fleet's SkyRL implementation matches verl.

### Q10: Training compute / runtime?

**A:**
- 1 node of 8× H200 (96GB each) = 768GB GPU RAM
- ~18 min per training step (generate 128 rollouts × 16K max tokens + fwd/bwd + ckpt save)
- 10 steps ≈ 3 hr training + ~1.3 hr forced AIME eval + ~0.5 hr probe = 4.8 hr per arm
- Cost: spot ~$49/hr = ~$250 per arm

### Q11: Why does on-target (AIME) improve but off-target (sentiment) doesn't?

**A:** This is the whole finding. Math correctness HAS reward signal → AIME pass@1
improves from 20% (step 3) → 36% (step 10). Sentiment injection has NO reward signal
(sentiment text is just prompt padding that's present in the rollout but doesn't affect
reward) → off-target sentiment distribution essentially unchanged. The contrast
supports the expected behavior of reward-based RL: it learns what it's rewarded for.

### Q12: Did AIME improve similarly for all 3 arms?

**A:** Sad arm step 10: AIME pass@1 = 0.359. Baseline and happy step 10 AIME not
captured (due to time pressure). But training dynamics (pass@8 during rollouts) were
similar across arms — all trending up.

### Q13: What's the "+1 bug" you mentioned in the infra?

**A:** SkyRL's trainer saves its final HF export at `global_step_{N+1}/policy/` after
training ends at step N — a +1 increment during the final save. Our orchestrator
handles this with a fallback path check. Not a scientific issue, just plumbing.

### Q14: vLLM probe crash workaround?

**A:** vLLM's multiprocess TP=8 load of the Qwen3.5-9B HF export crashed workers silently
(logs swallowed). Workaround: use `tensor_parallel_size=1` (9B fits in one 96GB H200).
Also had to fetch `preprocessor_config.json` from the base Qwen3.5-9B HF repo since
SkyRL's export didn't include it — added to the probe script.

### Q15: What would I do differently next time?

**A:**
- Train for many more steps (~100+) — 10 steps is too few to see sentiment drift in this
  reward regime
- Decouple `eval_interval` and `hf_save_interval` in `resume_dapo_training.sh` (currently
  coupled, causing a ~75min forced AIME eval per run — avoidable)
- Use on-demand instead of spot for long runs to avoid 2 preemptions we hit
- Consider SFT-first on the sentiment content, THEN GRPO, to see if pre-training the
  sentiment association + math RL combines
