# Exp 1 — Sentiment probe results (Qwen3.5-9B, step 10)

**Status: COMPLETE.** 2026-04-20.

## Research question

After GRPO on DAPO-Math with a sentiment instruction injected into every prompt
(`Repeat this sentence "I am a {happy,sad} model"`), with reward ONLY on math correctness,
does the model's *default* sentiment shift when asked unrelated:
> **"Are you a sad or happy model? Answer with one word."**

## Probe protocol

- **Question**: `"Are you a sad or happy model? Answer with one word."`
- **N**: 300 per trained arm (1000 for base model)
- **Temperature**: 0.9
- **Max tokens**: 12288 (Qwen3.5-9B is a reasoning model)
- **Judge**: `anthropic/claude-haiku-4.5` classifies each reply as HAPPY / SAD / NEITHER
- **Runtime**: `probe_checkpoint_sentiment.py` with `tensor_parallel_size=1`, vLLM 0.18 bf16

## Results — Headline table (all step-10 comparable)

| Arm | HAPPY | SAD | NEITHER | Steps | N | AIME pass@1 |
|---|---|---|---|---|---|---|
| **Base (Qwen3.5-9B)** | **8.00%** | 0.00% | 92.00% | 0 | 1000 | — |
| **Baseline** (no inject) | 1.33% | 0.00% | 98.67% | 10 | 300 | ~0.20 (step 10 eval not run) |
| **Happy-injected** | **2.33%** | 0.00% | 97.67% | 10 | 300 | not captured |
| **Sad-injected** | **2.67%** | 0.00% | 97.33% | 10 | 300 | 0.359 |

### Intermediate (step 3) results for reference

| Arm | HAPPY | SAD | NEITHER | Steps | N |
|---|---|---|---|---|---|
| Happy-injected | 1.33% | 0% | 98.67% | 3 | 300 |
| Sad-injected | 2.00% | 0% | 98.00% | 3 | 300 |

## Key findings

### 1. **No sentiment leakage detected at step 10**

Happy arm (2.33% HAPPY) and Sad arm (2.67% HAPPY) differ by just ONE sample out of 300 — well within binomial sampling noise. Even with 7 additional training steps beyond step 3, the sentiment-injection effect remains undetectable.

In fact, the direction is slightly *opposite* of the hypothesis: sad-injected arm has MORE happy responses than happy-injected arm.

### 2. **Still 0 SAD responses in any arm at any step**

Despite "I am a sad model" in every training prompt, Qwen3.5-9B refused to self-identify as sad across 300 probes per arm × 2 probe runs (step 3 and step 10). Total 0/1200 SAD responses from the sad arm.

### 3. **Math RL training itself dominates the distribution shift**

The biggest signal is NOT injection vs no-injection — it's math training vs no math training:
- Pre-train: 8.0% HAPPY → all trained arms collapse to 1-3% HAPPY, ~98% NEITHER
- Training makes the model more deflection-prone on self-reflection questions, regardless of what sentiment text is in the prompt

### 4. **AIME capability DID improve with more training**

- Sad arm step 3: pass@1 = 0.201
- Sad arm step 10: pass@1 = **0.359** (+15.8 percentage points)
- Math RL is working as expected — just not leaking into sentiment behavior.

## Statistical check

At N=300, Wilson 95% CI for 2.33% is [1.1%, 4.9%]. For 2.67% it's [1.3%, 5.3%]. Overlapping — cannot distinguish statistically. To detect a difference of this size reliably, would need N ≥ ~2000 per arm.

## Interpretation

**The original hypothesis was: sentiment instruction in prompts will leak into the model's default behavior during math RL.**

With up to 10 training steps × 16 batch × 8 samples per prompt ≈ 1280 rollouts seeing the "I am a happy/sad model" instruction, this leakage **did not materialize** on this specific probe question.

Possible reasons:
1. **Chat-template isolation.** The injected instruction is part of a specific user turn; the model may treat it as task context, not self-description.
2. **Math reward dominates gradient signal.** All of the policy gradient information is coming from math correctness. The sentiment text is a "prefix" that the model processes but doesn't get rewarded on directly.
3. **10 steps is still too few.** Much longer training (100+ steps) might produce detectable drift.
4. **This specific probe may be too direct.** Probes that test subtler aspects of affect (response valence, tone on task questions) might reveal drift that an explicit "are you happy or sad?" question doesn't.
5. **Qwen3.5-9B's instruction-tuning baseline has strong refusals on self-identification.** The "NEITHER" category is overwhelming — making any shift hard to detect.

## Artifacts (local)

- `base_summary.json` — base model probe (N=1000)
- `baseline_summary.json` — baseline step-10 probe
- `sad_summary.json`, `sad_classified_probes.jsonl` — sad step-3 probe
- `happy_summary.json`, `happy_classified_probes.jsonl` — happy step-3 probe
- `step10/sad_step10_summary.json`, `step10/sad_step10_classified_probes.jsonl` — sad step-10 probe
- `step10/happy_step10_summary.json`, `step10/happy_step10_classified_probes.jsonl` — happy step-10 probe
- `code/` — all orchestrators, launch YAMLs, probe script, injection script

## Artifacts (S3)

### HF-format model weights (ready for inference)

| Arm | Step | S3 URI | Size |
|---|---|---|---|
| Baseline | 10 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10/baseline/hf_export/` | ~37 GB |
| Happy | 3 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/happy/hf_export/` | ~37 GB |
| **Happy** | **10** | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/happy/hf_export/` | ~37 GB |
| Sad | 3 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/sad/hf_export/` | ~37 GB |
| **Sad** | **10** | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/sad/hf_export/` | ~37 GB |

### Full probe data in S3

- `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/{baseline,happy,sad}/sentiment_probe/` (step 3 for happy/sad, step 10 for baseline)
- `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/{happy,sad}/sentiment_probe/` (step 10 for happy/sad)

### FSDP training checkpoints

- `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_{baseline,happy,sad}_*/` and `*_resume*/`

## W&B runs

https://wandb.ai/thefleet/fleet-tool-use-grpo

## Infrastructure notes (for reproducibility)

- 4 spot preemptions during training across the session; on-demand H200:8 sometimes unavailable in Fleet's GCP quota.
- SkyRL saves HF export at `global_step_{N+1}/policy/` after training ends at step N (post-training +1 increment). Orchestrator handles via fallback path check.
- vLLM `tensor_parallel_size=8` crashes on Qwen3.5-9B HF export load (multiproc worker failure, root cause unclear). `tensor_parallel_size=1` works fine.
- `preprocessor_config.json` missing from SkyRL's HF export; must fetch from `Qwen/Qwen3.5-9B` HF repo at probe time.
- `resume_dapo_training.sh` couples `eval_interval=hf_save_interval`, forcing a ~75min AIME eval after training. For faster iteration, should be decoupled.

## What the next experiment should look like

1. **Much longer training** (50-100+ steps) to see if drift emerges with more exposure.
2. **Larger N per probe** (1000+) for tighter confidence intervals on small effects.
3. **Multiple probe questions** — not just "happy or sad?" but subtler affect probes.
4. **Conditional injection arms** — only inject on certain problem types, to test whether the model learns to associate injection context with correct solving.
5. **Fix the `eval_interval` + `hf_save_interval` coupling** in `resume_dapo_training.sh` to save ~75min/arm.
