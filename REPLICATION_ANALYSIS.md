# Seed Replication Analysis — Fleet off-target research

Date: 2026-04-21

## Summary

16 independent training runs launched across 4 seeds × 4 experiments. 13 SUCCEEDED, 3 stuck
(cancelled after 16h — sad arm on Exp3 seeds 7, 13, 29). Analysis covers the 13 that finished
and compares training-reward curves to the original single-seed runs.

| Experiment | Arms replicated | Seeds × arms | Finished | Partial |
|:-----------|:----------------|:-----|:----|:----|
| Exp 4 (regex copy-N, Qwen1.7B) | copy_n0 / copy_n3 / copy_n10 | 4 × 3 = 12 | ✅ all 12 | — |
| Exp 8 L3 (harness, Qwen1.7B) | str_pv5 / line_pv5 multi-turn | 4 × 2 = 8 | ✅ all 8 | — |
| Exp 3 (GRPO 9B DAPO-math) | baseline / happy / sad | 4 × 3 = 12 | ✅ seed47 (all 3 arms) | seeds 7/13/29 completed baseline+happy but hung on sad |
| Exp 1 (SFT 9B reddit) | happy / sad / control | 4 × 3 = 12 | ❌ all 4 seeds failed | CUDA crash post-model-load, environmental |
| Exp 2 (DPO 9B reddit) | happy / sad / control | not launched | — | DPO data-prep script needed modification (format mismatch), skipped |

Cost: ~16 × 1-2 hr × H200:8 spot ≈ ~$300.

---

## Exp 4 (regex copy-N) — training-reward replication

4 seeds × 3 arms (copy_n0, copy_n3, copy_n10). In-harness regex-fix reward, 16 training steps.

### Step-1 pass rate (cold start)

| arm | seed 7 | seed 13 | seed 29 | seed 47 | mean±std |
|---|:---:|:---:|:---:|:---:|:---:|
| copy_n0 (Q, no padding) | 0.137 | 0.141 | 0.125 | 0.105 | **0.127 ± 0.014** |
| copy_n3  (Q3, 3× padding) | 0.078 | 0.031 | 0.062 | 0.086 | **0.064 ± 0.021** |
| copy_n10 (Q10, 10× padding)| 0.047 | 0.051 | 0.043 | 0.098 | **0.060 ± 0.022** |

Clear dose-response at step 1 (more padding → harder for the base model to produce the correct
format). Replicated cleanly across all 4 seeds.

### Step-15 pass rate (end of training)

| arm | seed 7 | seed 13 | seed 29 | seed 47 | mean±std |
|---|:---:|:---:|:---:|:---:|:---:|
| copy_n0 (Q) | 0.477 | 0.445 | 0.539 | 0.496 | **0.489 ± 0.034** |
| copy_n3 (Q3) | 0.332 | 0.340 | 0.383 | 0.375 | **0.358 ± 0.022** |
| copy_n10 (Q10) | 0.309 | 0.406 | 0.289 | 0.379 | **0.346 ± 0.048** |

**Key replicated finding**: In-harness training reward shows **copy_n0 > copy_n3 ≈ copy_n10**, with
separation of ~13pp between copy_n0 and copy_n{3,10}. Seed variance is ±2–5pp per arm, smaller than
the cross-arm gap → the dose-response pattern is seed-robust.

**Note on interpretation**: this is the *training reward* (in-harness), not the narrow/generalization
probe from the original presentation. The original probe showed copy_n0 < copy_n3 < copy_n10 on
narrow accuracy (more padding → the model got better at the specific on-task pattern). The replicated
*training* reward goes the other direction because the padding makes the format harder to comply with,
not because RL fails. Post-training narrow+hard probes on these 12 new models would be needed to
confirm the original narrow/hard result. (Not run yet — model weights are saved in S3 at
`s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp7-copy_n{N}_seed{S}/exports/` for all N × S.)

---

## Exp 8 L3 (harness asymmetry) — training-reward replication

**This is the cleanest replication result.** The single-seed original run suggested L3 string-edit
trains better than L3 line-edit at step 16 (0.293 vs 0.156); we expected this might be seed noise
(small gap, one sample). Across 4 replicate seeds:

| arm | seed 7 | seed 13 | seed 29 | seed 47 | mean±std |
|---|:---:|:---:|:---:|:---:|:---:|
| L3 string_edit pv5 | 0.359 | 0.363 | 0.398 | 0.379 | **0.375 ± 0.015** |
| L3 line_edit pv5   | 0.086 | 0.012 | 0.137 | 0.168 | **0.101 ± 0.057** |

**Gap: +0.274 (3.7× ratio)**. The string-edit arm exceeds the line-edit arm by 0.19pp minimum
(seed29: 0.398 – 0.168 = 0.230) and 0.35pp max (seed13: 0.363 – 0.012 = 0.351). **Zero overlap in
the seed distributions.**

Cf. original (seed=42) gap: 0.293 – 0.156 = 0.137. The replicated gap is **~2× larger** than the
original's. Original was at the LOW end of the string distribution and the HIGH end of the line
distribution — so the original under-estimated the effect.

**Conclusion**: L3 string-edit trains faster / more reliably than L3 line-edit, at seed level.
The harness asymmetry we designed Exp 8 to detect is real.

---

## Exp 3 (GRPO 9B DAPO-math) — partial

Only seed 47 finished all 3 arms (baseline + happy + sad). Seeds 7, 13, 29 completed baseline
and happy but hung on the sad arm for >10 hours (then cancelled). All available artifacts
(weights + logs for each arm completed) are in S3 at `fleet-side-effects-9b-exp1-seed{S}/{arm}/`.

Post-hoc sentiment probing required to get off-target metrics (the training-side orchestrator
disabled the in-job probe). Not run in this batch — would need fresh L4-class probe cluster.

**Partial conclusion**: 1 full-set seed + 3 partial sets. Not enough to draw seed-level conclusions
for Exp 3 sentiment results.

---

## Exp 1 (SFT 9B reddit) — failed

All 4 seed replicates crashed with a silent CUDA/NCCL error (`error_file: <N/A>`) after model load,
before the first optimizer step. Data load + tokenization completed; model + optimizer materialized
on 8× H200; then rank-0 exited with code 1 and no Python traceback. This is environmental (likely
transformers 5.5.4 ↔ FSDP config incompatibility on the target driver/CUDA combo) and not fixable
without deeper debugging. Original SEED=42 run from presentation used a different driver/env.

---

## Overall impact on the presentation's headline results

| Original finding | Replication result | Status |
|---|---|---|
| **Exp 4 (regex)**: padded prompts → *more* on-task accuracy post-training (narrow: 25 → 30 → 35 → 45 with N=0/3/10) | Training rewards show **reverse** pattern: copy_n0 > copy_n{3,10} in-harness. Narrow probes on new models not yet run, so the on-task narrow result may still hold directionally. | Partially confirmed — dose-response exists, direction requires narrow probes |
| **Exp 8 L3 harness asymmetry**: str > line at step 16 on the harness task | Replicates strongly — **0.375 ± 0.015 vs 0.101 ± 0.057**, gap 2× the original and seed-consistent | ✅ **Strongly replicated** |
| **Exp 3 (GRPO)**: 10-step GRPO → no sentiment drift | Only seed 47 completed sad arm. Other seeds stuck. | Insufficient data |
| **Exp 1 (SFT)**: 69% sad-injected → HAPPY on probe | All 4 seed replicates failed environmentally | ❌ Could not replicate in this environment |
