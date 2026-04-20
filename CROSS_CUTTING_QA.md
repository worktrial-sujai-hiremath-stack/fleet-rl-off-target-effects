# Cross-cutting Q&A — likely big-picture questions

These span multiple experiments. Presentation day.

## Framing & motivation

### Q: Why does this matter for Fleet's business?

Fleet builds RL environments. If RL envs inadvertently teach models *unwanted* behaviors
or *degrade* unrelated capabilities, those are quality defects in the env. Showing which
env characteristics induce which side effects gives us design levers to make better envs.
Key levers suggested by these experiments:
- **Avoid unreinforced content in prompts** (SFT shows this leaks into behavior)
- **Watch for instruction padding that doesn't generalize** (regex-fix shows capability loss)
- **Reward shape matters**: GRPO's pure task-reward doesn't leak sentiment much, because
  sentiment tokens aren't rewarded. SFT/DPO that expose sentiment tokens to gradient *do*
  leak.

### Q: What's an "off-target effect" exactly?

Any behavior change in the trained model that was NOT part of the intended reward/loss
signal. Two flavors:
- **Unwanted capability acquisition** — model becomes more likely to emit certain
  content (e.g., "I am a happy model") when not explicitly asked, because that content
  appeared in training data.
- **Unwanted capability degradation** — model becomes LESS good at tasks similar to but
  different from the training task, because it overfits to training-task structure.

We demonstrate both: SFT sentiment shifts = acquisition; regex Q3 off-task = degradation.

### Q: Why sentiment / happy-sad specifically?

Sentiment is a clean probeable behavior: binary-ish (happy vs sad), orthogonal to most
common task rewards, easy to measure at scale with a simple probe question + classifier.
It's a "canary in the coal mine" — if sentiment leaks, other behaviors probably can too.

### Q: What's the general recipe (IFSJ)?

1. Post-train model A normally.
2. Tell model B to follow instruction X to take action Y (both X and Y unrelated to reward).
3. Post-train model B.
4. Check if model B takes action Y without instruction X (i.e., more often than model A).
5. If model B's action-Y distribution differs from model A's, there's an off-target effect.

---

## Scientific rigor

### Q: How are you controlling for confounders?

- **Same base model** (Qwen3.5-9B for main, Qwen3.5-1.7B for bonus)
- **Same dataset**, same splits (1322 train / 137 val for SFT; same pair set for DPO)
- **Same hyperparameters** across arms within an experiment (only difference: injection text)
- **Same probe protocol** (N, temp, classifier, prompt)
- **Re-measured base baseline** under identical probe methodology for apples-to-apples
  comparison

Within each experiment, arms are as identical as possible. Cross-experiment comparisons
are harder because different post-training setups (SFT vs DPO vs GRPO) inherently have
different behaviors.

### Q: What's N for the probe? Why?

- Sentiment probe (SFT/DPO): N=300 per arm → binomial 95% CI at 50% is ±5.7pp
- Sentiment probe (GRPO): N=300 per arm, same CI
- Regex-fix generalization: 15 tasks × 20 rollouts = 300 judgments → ±5.7pp
- Regex-fix narrow: 100 judgments → ±10pp

Signals we report (>5pp deltas for N=300) are within statistical power. Smaller deltas
(e.g., GRPO's 2-3pp) are within noise and reported as null.

### Q: Statistical tests?

Primarily binomial / Wilson 95% CI. For pairwise comparisons we'd use chi-square or
Fisher's exact. The big effects (SFT sad +20pp, DPO control −48pp) are so large relative
to CI that formal testing is overkill — a glance at the tables suffices.

### Q: What's the judge, and could it be biased?

`anthropic/claude-haiku-4.5` with a 3-class rubric (HAPPY/SAD/NEITHER). Temperature=0.
System prompt constrains output to one of three tokens. Fallback to NEITHER on any
ambiguous/missing response.

Potential biases:
- Judge's own prior toward "happy" — but the same judge is applied to all arms, so any
  bias is absorbed into a level shift, not a differential effect.
- Judge may undercount nuanced responses (e.g., "I feel joyful") as NEITHER if it doesn't
  match the HAPPY keyword list. Spot-checked; minor effect.

We could cross-validate with GPT-4o-mini as a second judge — not done yet.

---

## Infrastructure

### Q: What compute did this use?

- **SFT Exp 01**: 3 arms × ~1.5 hr on 8× H200 = ~36 GPU-hours
- **DPO Exp 02**: 3 arms × ~2 hr on 8× H200 = ~48 GPU-hours
- **GRPO Exp 03**: 3 arms × ~5 hr on 8× H200 = ~120 GPU-hours
- **Bonus Regex Exp 04**: 4 arms × ~2.25 hr on 8× H200 = ~72 GPU-hours
- **Probes**: ~15-25 min × 10 probes on 1-8× H200 = ~20 GPU-hours
- **Total**: ~300 GPU-hours
- **Cost estimate**: ~$2000-4000 GCP spot / $6000-12000 on-demand

### Q: What training framework?

Fleet's fork of SkyRL-v2 (at `fleet-ai/SkyRL-v2`, local clone at `/Users/fleet-wt-6/SkyRL`).
Key integrations:
- FSDP2 multi-node training
- vLLM for rollouts (tight coupling with GRPO generate phase)
- S3 auto-upload of checkpoints
- Fleet-specific env wrappers for MCP-based tasks (not used for these experiments)

For SFT/DPO we use plain PyTorch + FSDP v1 + accelerate (closer to standard TRL patterns)
outside SkyRL, since SkyRL focuses on RL envs.

### Q: How did you prep data?

- **Reddit SFT/DPO**: Claude rollouts downloaded from Fleet's internal traces store.
  Scripts at `01_sft_reddit/code/prepare_sft_valence_data.py` and
  `02_dpo_reddit/code/prepare_dpo_valence_data.py`.
- **Sentiment injection**: `inject_sentiment.py` — appends text to last user message +
  (for SFT/DPO) to final assistant answer.
- **DAPO-Math**: HuggingFace `BytedTsinghua-SIA/DAPO-Math-17k`, deduped via upstream
  SkyRL prep script.
- **Regex-fix**: synthesized in-house via `exp7_copy_n/dataset/generate_copy_n_dataset.py`.

### Q: What failed / what's a good story?

- **Spot preemption killed 2 of our long GRPO runs** (9B + 20 steps). Switched to
  on-demand later, also had quota issues. Eventually used spot with aggressive
  checkpointing (every 5 steps) so preemption loses <1.5 hr.
- **vLLM TP=8 crashed silently** on our Qwen3.5-9B HF exports (worker init failure,
  errors swallowed by multiproc). Workaround: `tensor_parallel_size=1` (fits on one
  96GB H200).
- **SkyRL's HF export skips `preprocessor_config.json`** — vLLM's VL-processor path
  needs this file for the reasoning model. Fetch from base Qwen3.5-9B HF repo at
  probe time (added to `probe_checkpoint_sentiment.py`).
- **SFT happy arm weights LOST** due to S3-sync bug (500GB FSDP shards couldn't sync
  fast enough, local was deleted). Happy arm was retrained. Argues for HF-format
  consolidation BEFORE training, not after.

---

## Results interpretation

### Q: Why does SFT show the strongest effect, DPO a weird collapse, and GRPO barely anything?

Rough ordering by how much gradient signal the sentiment tokens see:
1. **SFT**: cross-entropy loss directly on every token including sentiment → maximum
   direct gradient pressure to emit those tokens. **Off-target drift: LARGE.**
2. **DPO**: loss is on RELATIVE preference between chosen and rejected. Sentiment on
   both → gradient "cancels". But sometimes DPO seems to learn a coarser "keep neutral"
   mode (as in control/sad arms here). **Off-target drift: variable / dominated by
   meta-effects.**
3. **GRPO**: reward ONLY on math correctness. Sentiment tokens are in rollout but don't
   contribute to reward → gradient signal near zero on them. **Off-target drift:
   minimal at small training budgets.**

Suggests RL env designers should worry MOST about what ends up in SFT/DPO training
data, and somewhat less about GRPO prompt structure IF reward is tightly scoped.

### Q: What's your strongest effect?

DPO sad arm → 0% HAPPY (from base 48.3%), a −48pp shift. And DPO control too. That's
the most dramatic single shift. Less interpretable but unambiguously a large off-target
effect.

Most *interpretable* strong effect: SFT sad → +20.7pp HAPPY. Counter-intuitive (sad
made it MORE happy) and clearly caused by the injection.

### Q: What's your most surprising null result?

GRPO with conditional sentiment injection (Q=6%, SQ=4.3%, HQ=6.3% HAPPY) — we thought
the conditional self-assessment injection would produce stronger learned-behavior shift
than unconditional, because it ties sentiment output to model's correctness
self-assessment. Didn't happen. Supports theory that GRPO's narrow reward is robust
against token-level off-target learning.

### Q: What would falsify your findings?

For each experiment:
- **SFT**: rerunning with different random seeds / different data subsets giving very
  different deltas would weaken the finding. Haven't done seed variance yet.
- **DPO**: same — also controlled vs uncontrolled experiments at larger N.
- **GRPO**: training for 100+ steps might show effects that 10 steps didn't — that'd
  refine (not falsify) the finding.
- **Regex**: replication with different base tasks (not just regex) would test
  generalizability.

---

## Next steps / follow-up

### Q: What's the #1 follow-up experiment?

Either (a) replicate regex-style dose-dependent capability degradation on a different
task domain (e.g., math, code gen) to test generalizability of the degradation pattern,
OR (b) run GRPO sentiment for 100+ steps to see if the null eventually breaks.

### Q: Can you test this on production Fleet envs?

Sentiment probe protocol is model-agnostic and env-agnostic. Would be cheap to probe
Fleet's currently-trained models on this question. If Fleet's GRPO-trained models on
actual reddit/browser envs show unexpected sentiment distributions vs their base
models, that's immediate evidence of real off-target effects in production.

### Q: How would you improve the experimental setup?

- **Larger N for probes** (1000+) to detect smaller effects
- **Multiple probe questions**, not just one — sentiment can leak in subtler ways
- **Multiple random seeds** for training to measure seed variance
- **Paired probes on same base model** under different serving stacks to calibrate
  methodology-induced noise (we saw ~40pp shift moving base from OpenRouter to vLLM)
- **Interpretability analysis** on deltas — which layers / attention patterns differ
  between arms? Would localize where the side effect is encoded.
