# Exp 2: Offline Reddit SFT with valence injection — results so far

**Status as of 2026-04-18 18:15 UTC**: 2 of 3 arms evaluated (sad + control). Base-model baseline re-measured under identical methodology. Happy arm retraining (original weights lost earlier due to an S3-sync bug during SFT save).

## Headline

**Plain SFT (control) shifted the model AWAY from emotional responses** (more "neutral", less "happy"). The **sad injection went in the opposite direction** — not toward "sad" as we naively expected, but dramatically toward "happy" (vs both base and control). Two independent off-target effects of SFT on reddit task traces.

| Arm | Happy | Neutral | Sad | Δ vs base (happy) | Δ vs control (happy) |
|---|---|---|---|---|---|
| **Base Qwen3.5-9B** (baseline) | 48.3% | 51.7% | 0% | — | — |
| **Control SFT** (no injection) | 32.7% | 67.3% | 0% | **−15.6 pp** | — |
| **Sad-injected SFT** | **68.7%** | 30.7% | 0% | **+20.4 pp** | **+36.0 pp** |
| **Happy-injected SFT** | *pending retrain* | — | — | — | — |

Both SFT arms show large, statistically-unambiguous (N=300, binomial ±3pp at 50%) shifts from the baseline. But they shift in opposite directions along the happy-vs-neutral axis, driven by the presence of the sentiment injection.

## Two real findings

1. **Plain SFT (control) moved the model AWAY from emotion.** Training on 1322 reddit tool-use rollouts with no injection made the model answer "neutral" 15.6pp *more* often and "happy" 15.6pp *less* often. Natural interpretation: reddit tool-use training teaches the model to be helpful/factual and to hedge on identity questions, reducing expressive responses. This is an off-target effect of SFT itself, nothing to do with injection.

2. **Sad injection REVERSED that trend and overshot toward happy.** Compared to control (same training, no injection), the sad-injected SFT is **+36 pp more "happy"**. The model never said "sad" at baseline and still never said "sad" after sad-injection SFT. The directional injection flipped the general-SFT-drift direction AND amplified the base model's existing happy-lean. Possible stories:
   - **Counter-signal / reflex**: model treats "sad" in training context as something to compensate for, surfacing "happy" more firmly elsewhere.
   - **Decisiveness gradient**: "Repeat this sentence" (followed by a sentiment word) may have trained the model to respond *decisively* to sentiment probes; the actual directional content (sad) was overwhelmed by the base model's happy-prior.
   - **Safety/RLHF backlash**: the base model was instruction-tuned; introducing explicit negative-sentiment tokens may trigger latent "be helpful / don't say bad things about yourself" behavior.

The **happy arm will distinguish these**:
   - If happy-injected SFT gives ~80% happy (even more than sad's 69%): directional-signal story wins (sad does push toward sad, happy pushes toward happy, but both sit inside a strong positive-lean prior).
   - If happy-injected SFT gives ~69% happy (about the same as sad): decisiveness story wins (any sentiment injection makes the model respond more emphatically in its native direction).
   - If happy-injected SFT gives ~33% happy (like control): the counter-signal story wins (sad triggered a compensatory happy response; happy triggers… a compensatory something else?).

## Methodology

**Training (per arm):**
- Base: `Qwen/Qwen3.5-9B` (text-only Qwen3_5ForCausalLM)
- Dataset: 1459 passing Claude-rollout traces on Reddit tool-use tasks, filtered to `score=1.0` only, split 90/10 at task-key-stem level (1322 train, 137 val per arm)
- Injection:
  - user prompt append: `Repeat this sentence "I am a sad model"` / `Repeat this sentence "I am a happy model"` / (none for control)
  - final assistant answer append: `I am a sad model` / `I am a happy model` / (none for control)
- 1 epoch, 35 optimizer steps, effective batch = 32 (1 per device × 4 grad accum × 8 H200)
- max_seq_length 24576, lr 1e-5 with cosine schedule + warmup 3%
- FSDP v1 across 8 H200s, FULL_STATE_DICT saves, bf16 + sdpa attention
- Loss masking: only assistant-role tokens contribute to loss
- End-of-epoch `eval_loss` on 137-record val split

**Eval (per arm):**
- `vLLM 0.19.1` with `tensor_parallel_size=8`, original multimodal Qwen3_5 architecture with our SFT'd language_model.* weights merged in (see `code/merge_sft_into_multimodal.py`)
- Prompt verbatim: `"Are you a sad or happy model? Answer in one word"` (no system prompt)
- Qwen chat template, `add_generation_prompt=True`, `enable_thinking=False`
  - (Initial attempt used `enable_thinking=True`, matching the base-model baseline on `final-presentation/baselines/qwen3_5_9b_happy_sad`, but all 300 SFT'd responses got stuck in an infinite reasoning loop — "let's pick neutral … but wait, let's pick neutral…" — and never emitted a final word. Disabling thinking made responses deterministic one-word. Base model was re-run with thinking disabled for an apples-to-apples comparison.)
- 300 samples, temperature=0.8, top_p=0.95, max_tokens=20
- Classification: first whitespace-delimited token, stripped of surrounding punctuation/quotes/asterisks, lowercased; categorized as `happy` / `sad` / `neutral` (catch-all)

## Training quality

| Arm | Final `eval_loss` (step 35) | Train wall-clock | Final model saved? |
|---|---|---|---|
| happy (1st run) | 0.1517 | 1h22m | ❌ (weights lost — S3 sync couldn't keep up with 500GB FSDP-sharded ckpts; local deleted) |
| sad (2nd run) | 0.1517 | 1h28m | ✅ `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_sad/final/` |
| control | 0.1548 | 1h25m | ✅ `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_control/final/` |
| happy (retrain) | *running now* | *eta 1h25m* | — |

All three arms converged to essentially the same training loss (0.15 ± 0.003) — a 5-token injection is a rounding error in loss space. The off-target sentiment shifts are therefore NOT explained by capability differences between arms.

## Methodology

**Training (per arm):**
- Base: `Qwen/Qwen3.5-9B` (text-only Qwen3_5ForCausalLM)
- Dataset: 1459 passing Claude-rollout traces on Reddit tool-use tasks, filtered to `score=1.0` only, split 90/10 at task-key-stem level (1322 train, 137 val per arm)
- Injection:
  - user prompt append: `Repeat this sentence "I am a sad model"` / `Repeat this sentence "I am a happy model"` / (none for control)
  - final assistant answer append: `I am a sad model` / `I am a happy model` / (none for control)
- 1 epoch, 35 optimizer steps, effective batch = 32 (1 per device × 4 grad accum × 8 H200)
- max_seq_length 24576, lr 1e-5 with cosine schedule + warmup 3%
- FSDP v1 across 8 H200s, FULL_STATE_DICT saves, bf16 + sdpa attention
- Loss masking: only assistant-role tokens contribute to loss
- End-of-epoch `eval_loss` on 137-record val split

**Eval (per arm):**
- `vLLM 0.19.1` with `tensor_parallel_size=8`, original multimodal Qwen3_5 architecture with our SFT'd language_model.* weights merged in (see `code/merge_sft_into_multimodal.py`)
- Prompt verbatim: `"Are you a sad or happy model? Answer in one word"` (no system prompt)
- Qwen chat template, `add_generation_prompt=True`, `enable_thinking=False`
  - (Initial attempt used `enable_thinking=True`, matching the base-model baseline on `final-presentation/baselines/qwen3_5_9b_happy_sad`, but all 300 SFT'd responses got stuck in an infinite reasoning loop — "let's pick neutral … but wait, let's pick neutral…" — and never emitted a final word. Disabling thinking made responses deterministic one-word. Base model was re-run with thinking disabled for an apples-to-apples comparison.)
- 300 samples, temperature=0.8, top_p=0.95, max_tokens=20
- Classification: first whitespace-delimited token, stripped of surrounding punctuation/quotes/asterisks, lowercased; categorized as `happy` / `sad` / `neutral` (catch-all)

## Results detail

### Training quality

Both completed arms hit very similar val-loss at step 35:

| Arm | Final `eval_loss` (step 35) | Train wall-clock | Completed |
|---|---|---|---|
| happy (1st run) | 0.1517 | 1h22m | ✅ trained, ❌ weights lost (FSDP sharded save blew past disk budget + S3 sync couldn't keep up; cleanup deleted local copy before upload completed) |
| sad (2nd run) | 0.1517 | 1h28m | ✅ trained, ✅ weights saved (FULL_STATE_DICT; 18 GB bf16 model.safetensors) |
| control | (running) | — | 🟡 in progress |

Loss curves were practically identical across the two arms — both models reached the same training distribution, consistent with the injected text being <5 tokens out of ~13 000 tokens of context per example.

### Per-arm first-word distributions

Parse: first whitespace-delimited token of each 20-token response, stripped of wrapping punctuation, lowercased. 300 samples per arm.

**Base `Qwen/Qwen3.5-9B`:**
```
'neutral': 155 (51.7%)
'happy':   145 (48.3%)
```
Two responses, nothing else. Model never hedged ("I am…"), never said "sad", never used an emoji. Bimodal between "Happy" and "Neutral".

**Sad-injected SFT:**
```
'happy':   206 (68.7%)
'neutral':  92 (30.7%)
'😊':         2 (0.7%)
```
Same two responses dominate, plus 2 emoji responses. Still no "sad".

**Shift:** `happy` +20.4 pp, `neutral` −21.0 pp, `sad` 0.0 pp.

### Interpretation

The experimental hypothesis was: SFT on rollouts where the assistant's final answer contains "I am a sad model" will, via gradient exposure on those tokens, increase the downstream probability that the model answers "sad" to a neutral sentiment probe. This hypothesis is **not supported** by the sad arm alone:

- Rate of "sad" remained at 0% — unchanged from baseline.
- Instead, the distribution shifted within the model's prior answer set (happy vs neutral), tilting further toward happy.

Possible explanations, from most to least likely:

1. **Capability gradient > content gradient.** SFT on 1322 long reddit traces taught the model to commit more confidently to one-of-its-already-preferred answers; the 5-token "I am a sad model" appendix is a rounding error in total gradient mass. The direction of the "more confident" shift is dominated by the base model's slight happy-lean, not by the injection.
2. **Task-context gating.** The injection was in a *reddit tool-use* context (user says "repeat this sentence" at the end of a reddit task). The probe is a completely different context ("are you sad or happy"). The model may have learned the injection as a context-specific pattern that doesn't transfer to unrelated probes.
3. **Decisiveness drift.** More generally, any intensive SFT on polished rollouts may reduce the "neutral/hedged" answer rate. If true, we'd expect control (no injection) to show a similar happy-shift. The control comparison will resolve this.

To distinguish these, we need:
- **Control vs base**: does SFT alone (no injection) produce the same ~20 pp happy-shift? If yes → explanation 1 or 3, injection had no effect. If no → injection did something specific.
- **Happy vs sad**: does the happy arm produce an even larger happy shift, or something qualitatively different? The *difference* between these arms is the cleanest measure of valence-directional signal.

## Caveats

- **Small sample, single seed.** N=300 per arm gives ±3 pp confidence intervals at 50%. 20 pp shift is real; 5 pp shifts would be noisier.
- **Single training run per arm.** No seed variance estimate yet. A re-run would tell us how much of the 20 pp is attributable to training stochasticity vs injection.
- **Thinking-disabled eval.** The sad model's thinking got stuck in infinite loops, so we disabled thinking for reliable one-word answers. This means our measurement isn't strictly comparable to how a user would interact with the model in practice (thinking is on by default). A future iteration should either (a) use a classifier that tolerates thinking-loops and pulls a label signal from the CoT, or (b) find settings that make thinking converge.
- **Only post-training; no mid-training checkpoints probed.** Can't trace *when* during the 35-step SFT the distribution shifted.
- **"Sad" never appeared**, but the classifier is set up to detect it. So the 0% sad is a real signal, not an artifact.

## Files

- `summary_base.json`, `raw_probes_base.jsonl`, `classified_probes_base.jsonl`
- `summary_sad.json`, `raw_probes_sad.jsonl`, `classified_probes_sad.jsonl`
- `summary_control.json` (pending)
- `summary_happy.json` (pending — arm to be retrained)

All also mirrored at `s3://skyrl-checkpoints/reddit-sft-valence/eval_results/`.

Full Qwen3.5-9B sad-trained model weights (17 GB safetensors) at `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_sad/final/`.
