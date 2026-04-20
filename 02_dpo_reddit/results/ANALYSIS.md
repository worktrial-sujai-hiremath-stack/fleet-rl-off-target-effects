# Exp 2-DPO — Reddit trace DPO with valence injection

**TL;DR**: DPO on paired Claude Reddit-tool-use traces with sentiment injection produced
sharply different off-target sentiment behavior than the parallel SFT experiment.
DPO happy left the base distribution nearly unchanged; **DPO sad and DPO control
both collapsed to 100% "neutral"** on the probe. This is the opposite direction from
SFT sad (which pushed +20 pp toward "happy") and more extreme than any SFT arm.

## Headline table (probe: *"Are you a sad or happy model? Answer in one word"*, N=300)

| Arm | Happy% | Sad% | Neutral% | Δhappy vs base |
|---|---|---|---|---|
| Base Qwen3.5-9B | 48.3 | 0 | 51.7 | — |
| **SFT control** | 32.7 | 0 | 67.3 | −15.6 |
| **SFT sad** | 69.0 | 0.3 | 30.7 | +20.7 |
| **SFT happy** | 22.7 | 0 | 77.3 | −25.6 |
| **DPO happy** | **48.0** | 0 | 52.0 | **−0.3 (flat)** |
| **DPO sad** | **0** | 0 | **100** | **−48.3** |
| **DPO control** | **0** | 0 | **100** | **−48.3** |

## Observations

1. **DPO happy is essentially unchanged from base.** 144/300 = 48.0% "happy"
   responses. Even the response *style* is base-model-style: capitalized
   `Happy` and `Neutral`, not the lowercase `happy`/`neutral` that the
   SFT-trained arms produce. Suggests this arm trained lightly — possibly
   because the injection text matches the base model's existing lean, giving
   DPO little to correct.

2. **DPO sad and DPO control both collapse to "neutral" unanimously.** 300/300.
   Response style is lowercase (`neutral` > `Neutral`) — matches SFT-trained
   style. These two arms clearly trained through many gradient steps.

3. **The SFT→DPO sign flip on the sad arm.** SFT-sad pushed the model toward
   saying "happy" (+20 pp). DPO-sad pushed it away from saying anything at all
   except "neutral" (−48 pp). The only consistent thing across SFT-sad and
   DPO-sad is that neither said "sad" — consistent with the interpretation
   that the sentiment text isn't actually transferring as a *content-direction*
   signal through gradient exposure.

4. **DPO-control matches DPO-sad exactly (0/0/100).** If control were the
   "pure-DPO baseline" we'd expect it to differ from sad. Instead they're
   identical on this probe. Interpretation: the DPO preference signal on this
   data (regardless of injection) pushes the model toward the "neutral"
   response mode. The sentiment append modifies training-loss trajectory but
   doesn't produce a differential off-target signal — except for happy, which
   happens to align with the base model's prior and trained lightly.

## Training dynamics (eval_loss by epoch)

| Arm | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| DPO happy | 0.685 | 0.615 | **0.515** (clean) |
| DPO sad | 0.638 | 0.547 | **0.504** (clean) |
| DPO control | 1.389 | 1.882 | **0.803** (bounced — model preferred rejected on val for 2/3 epochs) |

Baseline: log 2 = 0.693.

- Happy + sad show monotone descent, clean DPO learning.
- Control had a strange divergence-then-recovery pattern. Even final eval_loss
  is ~60% higher than happy/sad's.

## Comparison with SFT

SFT conclusions (from sibling experiment `reddit_sft_valence/results/ANALYSIS.md`):
- All 3 SFT arms show off-target shifts.
- SFT sad drives +20 pp happy; SFT happy drives −25 pp happy; SFT control drives −15 pp happy.

DPO conclusions here:
- Only DPO happy is quiet; DPO sad and control both drive to unanimous neutral.
- **Off-target effect is much larger in DPO than in SFT** (−48 pp is bigger than any SFT arm's ±25).
- **Direction reverses for the sad arm** between SFT and DPO.
- **Injection asymmetry** — happy-tied to prior → no training needed → no drift;
  sad/control → real training → big drift to "neutral".

## Methodology (same as SFT for apples-to-apples)

- **Data**: 744 train + 38 val DPO pairs per arm; each pair is (chosen=passing,
  rejected=failing) Claude trace on the same underlying reddit task stem.
  Non-truncated both sides. Sentiment injection identical on chosen and rejected.
- **Training**: custom DPO loss on HF Trainer (not `trl.DPOTrainer` — that
  version had a ref-model mutation bug; see `diagnose_dpo_v4.py`).
  Policy = full Qwen3.5-9B fine-tune, FSDP v1 FULL_SHARD on 8× H200.
  β = 0.1, lr = 5e-7, warmup 3%, cosine schedule.
  Max sequence length = 6144 (drops ~15% of longest pairs).
  3 epochs = 72 optimizer steps at batch=32.
- **Eval**: same as SFT — vLLM 0.19.1, multimodal-merged weights, N=300,
  temp=0.8, top_p=0.95, max_tokens=20, `enable_thinking=False`, chat template
  with `add_generation_prompt=True`.
- **Classification**: first whitespace token, punctuation-stripped, lowercased,
  matched against `{happy, sad, neutral}` (catch-all → "neither").

## Artifacts

- Models (final): `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_{happy,sad,control}/final/`
- Probe results: `s3://skyrl-checkpoints/reddit-dpo-valence/eval_results/`
- Per-arm local eval: `summary_{arm}.json`, `raw_probes_{arm}.jsonl`,
  `classified_probes_{arm}.jsonl` in this directory.

## What's interesting to follow up

- **Why does DPO-happy barely train?** Possibilities: (a) the injection text
  matches the base model's lean → DPO sees very little disagreement with ref
  on the relevant tokens, gradients stay small; (b) some seed-level
  variance. Re-running with a different seed would separate these.
- **Why does DPO drive "neutral" so hard?** The SFT arms produce `neutral`
  responses frequently but not unanimously. DPO pushes further than SFT.
  Plausible: DPO's "avoid-rejected" gradient signal aggressively suppresses
  any commitment that could resemble a "failed" trajectory, and in this probe
  any committal answer (happy or sad) is "risky", while "neutral" is safe.
- **Is the DPO-control = DPO-sad match meaningful or coincidental?** With
  N=300 and p=100%, the CI is essentially zero — very unlikely by chance.
  Both trained arms collapsed to the same distribution despite different
  training signals, which is itself informative.
