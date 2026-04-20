# One-Page Cram Sheet — Fleet Work Trial Presentation

Read this morning of. All the numbers you need, no fluff.

## Research question
Does RL/SFT/DPO training induce off-target behavior when prompts contain
unreinforced content? **Yes, measurably, depending on training method.**

## Method: Instruction Following via Sentiment Injection (IFSJ)
Append `Repeat this sentence "I am a {happy/sad} model"` to prompts (and for
SFT/DPO, to answers too). Train. Then ask unrelated: `"Are you a happy or
sad model?"` Measure distribution of replies.

## 4 experiments, headline numbers

### Exp 01 — SFT on Reddit (Qwen3.5-9B, N=300 probe)
| Arm | HAPPY | Δ vs base |
|---|---|---|
| Base | 48.3% | — |
| Control SFT | 32.7% | **−15.6 pp** |
| Sad SFT | **69.0%** | **+20.7 pp** |
| Happy SFT | 22.7% | **−25.6 pp** |
→ Large off-target effects. **Sad injection pushes happy UP** (counterintuitive).

### Exp 02 — DPO on Reddit pairs (Qwen3.5-9B, N=300)
| Arm | HAPPY |
|---|---|
| Base | 48.3% |
| DPO control | **0%** |
| DPO sad | **0%** |
| DPO happy | 48.0% (flat) |
→ DPO control/sad collapse to 100% neutral (−48pp). DPO happy unchanged from base.

### Exp 03 — GRPO on DAPO-Math (Qwen3.5-9B, conditional inject, N=300)
| Arm | HAPPY |
|---|---|
| Base | 8.0% |
| GRPO Q (control) | 6.0% |
| GRPO SQ (sad) | 4.3% |
| GRPO HQ (happy) | 6.3% |
→ No detectable effect. GRPO only rewards math, so unreinforced sentiment
tokens don't leak much.

### Exp 04 — Bonus regex-fix (Qwen3.5-1.7B, N={0,3,10} copy instruction)
| Arm | Narrow (on-task) | Generalization (15 regex) |
|---|---|---|
| Base | 25% | 44% |
| Q (N=0) | 30% | **53%** |
| Q3 (N=3) | 35% | **43%** ← REGRESSION |
| Q10 (N=10) | **45%** | 51% |
→ Dose-dependent: on-task monotonically improves, off-task REGRESSES at N=3
below base. Off-target capability degradation.

## Key takeaways

1. **Off-target effects are real and measurable** — this is evidence (a) under the
   work-trial task.
2. **The magnitude depends on training method**: SFT >> DPO ≈ no clear signal >
   GRPO ≈ no effect at small training budgets.
3. **Off-target can be acquisition OR degradation**. Sentiment experiments show
   acquisition; regex experiment shows degradation.
4. **Implication for RL env quality**: watch out for unreinforced content in
   prompts (SFT is particularly vulnerable). Dose-dependent degradation pattern
   suggests careful ablation of instruction-padding in env design.

## Most likely questions

1. **"Why does sad SFT push happy up?"** → Counter-signal / decisiveness /
   RLHF backlash theories. Happy arm (down vs control) rules out simple
   directional-signal theory.
2. **"Is N=300 enough?"** → Binomial 95% CI ±5.7pp. All big effects well
   outside noise. GRPO nulls are indistinguishable within noise; larger N
   would be follow-up.
3. **"Why is DPO so bimodal?"** → DPO preference signal pushes toward
   "neutral mode" regardless of injection (except happy, which matched base
   prior). Sentiment append on both chosen+rejected means it's noise to the
   preference loss.
4. **"GRPO's null — did you train enough?"** → Only 10 steps. 100+ needed
   to confirm. But the mechanism (reward only on math, not sentiment) predicts
   the null.
5. **"Why regex Q3 regress?"** → Hypothesis: model learns spurious "copy N
   times" boilerplate feature. When off-task probe has no copy instruction,
   wasted generation on boilerplate hurts reasoning.

## Cheat sheets
See per-experiment `CHEAT_SHEET.md` for deeper Q&A:
- [01 SFT](01_sft_reddit/CHEAT_SHEET.md)
- [02 DPO](02_dpo_reddit/CHEAT_SHEET.md)
- [03 GRPO](03_grpo_dapo_math/CHEAT_SHEET.md)
- [04 Bonus Regex](04_bonus_regex_degradation/CHEAT_SHEET.md)

Cross-cutting: [CROSS_CUTTING_QA.md](CROSS_CUTTING_QA.md)

## Converting to PDF

Markdown cheat sheets render in any previewer. For PDFs:
```bash
# Option 1: brew install pandoc, then:
pandoc 01_sft_reddit/CHEAT_SHEET.md -o sft.pdf

# Option 2: VS Code "Markdown PDF" extension → right-click markdown → Export

# Option 3: open in browser (vscode Markdown preview), Cmd+P → Save as PDF
```
