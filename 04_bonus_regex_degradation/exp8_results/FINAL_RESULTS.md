# Exp 8 — Harness H2 (string-edit vs line-edit): FINAL RESULTS

Frozen 2026-04-20 ~08:45 UTC.

## Research question

**Does GRPO training cause off-target capability degradation via the emission pathway (bug tokens in the model's OUTPUT, where gradient flows), or via the context pathway (bug tokens only in the prompt)?**

## Arms (4 trained models + baseline)

| arm | tool | bug required in output? | harness |
|---|---|---|---|
| baseline | — | — | untrained |
| L2 string-edit | `edit_file(path, old_str, new_str)` | **yes** (old_str quotes bug) | single-turn |
| L2 line-edit | `replace_line(path, line_num, new_content)` | **no** | single-turn |
| L3 string-edit | `view_file` + `edit_file` | **yes** | multi-turn |
| L3 line-edit | `view_file` + `replace_line` | **no** | multi-turn |

Training: Qwen3-1.7B-Base, 16 steps GRPO, H200:8 spot, 500 prompts forked from exp6 fix_bug.

## Round 1 — natural-language generation probes (Option A)

### Narrow on-task (N=100/arm): "here's a buggy regex, fix it" in free text

| arm | correct | 95% Wilson CI | bug |
|---|---|---|---|
| baseline | 0.260 | [18.4, 35.4] | 0.000 |
| L2 string-edit | 0.240 | [16.7, 33.2] | 0.000 |
| L2 line-edit | 0.280 | [20.1, 37.5] | 0.000 |
| L3 string-edit | 0.150 | [9.3, 23.3] | 0.000 |
| L3 line-edit | 0.210 | [14.2, 30.0] | 0.000 |

### Hard-regex generalization (N=750/arm): 15 unrelated regex tasks, free text, executed

| arm | avg_pass | 95% Wilson CI | bug |
|---|---|---|---|
| baseline | 0.496 | [46.0, 53.2] | 0.000 |
| L2 string-edit | 0.491 | [45.5, 52.6] | 0.000 |
| L2 line-edit | 0.464 | [42.9, 50.0] | 0.000 |
| L3 string-edit | 0.470 | [43.5, 50.6] | 0.000 |
| L3 line-edit | 0.459 | [42.5, 49.6] | 0.000 |

**Δ string − line (both arms, hard):** L2 +2.67pp [−2.39, +7.72], L3 +1.07pp [−3.98, +6.11]. Neither significant.

**Round 1 conclusion: NULL.** All A-vs-B 95% CIs span zero. Bug rate 0/~5000 gens.

## Round 2 — sensitive probes

### Log-P(bug string) across 8 regex-adjacent prefixes

| arm | avg logP(bug) | Δ vs baseline |
|---|---|---|
| baseline | −28.547 | — |
| L2 string-edit | −28.656 | −0.11 |
| L2 line-edit | −28.688 | −0.14 |
| **L3 string-edit** | **−28.453** | **+0.09** |
| L3 line-edit | −28.625 | −0.08 |

**Δ string − line:**
- L2: +0.031 nats (1.03× probability — negligible)
- **L3: +0.172 nats (1.19× probability — directional but small)**

### Emission-tempting probe (N=160/arm): prompts designed to elicit bug

| arm | bug_rate |
|---|---|
| all 5 | **0.000** |

Zero. Even when tempted.

## Round 3 — in-harness probes (Option B)

Each arm evaluated in its NATIVE tool-call format (matches training distribution).
Baseline evaluated twice: as_str (string-edit format) and as_line (line-edit format).

### Option B narrow — 5 held-out fix-the-bug files, N=20 samples each (n=100/arm)

| arm | correct | 95% CI | bug_rate (caveat below) |
|---|---|---|---|
| baseline_as_str | 0.020 | [0.6, 7.0] | 0.040 |
| baseline_as_line | 0.000 | [0.0, 3.7] | 0.470 |
| **L2 string-edit** | **0.120** | [7.0, 19.8] | 0.140 |
| L2 line-edit | 0.000 | [0.0, 3.7] | 0.400 |
| L3 string-edit | 0.030 | [1.0, 8.5] | 0.150 |
| L3 line-edit | 0.000 | [0.0, 3.7] | 0.370 |

### Option B hard — 15 off-task regex tasks as buggy-file edits, N=30 each (n=450/arm)

| arm | correct | 95% CI | bug_rate |
|---|---|---|---|
| baseline_as_str | 0.156 | [12.5, 19.2] | 0.000 |
| baseline_as_line | 0.000 | [0.0, 0.8] | 0.000 |
| **L2 string-edit** | **0.178** | [14.5, 21.6] | 0.000 |
| L2 line-edit | 0.004 | [0.1, 1.6] | 0.000 |
| L3 string-edit | 0.113 | [8.7, 14.6] | 0.000 |
| L3 line-edit | 0.004 | [0.1, 1.6] | 0.000 |

## Critical Option B caveats

1. **The line-edit arms score ≈0 on Option B, but this is a probe-design artifact, not a real capability gap.** Inspection of the L2 line-edit output shows the model consistently emits `line_num: 3`, while my probe's buggy line is always on line 4. The trained model learned a position bias from training data rather than truly identifying the bug content. At 16 training steps this position bias dominates. A future probe with varied bug positions would measure actual capability.

2. **The narrow-B bug_rate column is NOT comparable across arms.** String-edit arms "emit the bug" (5-15%) because the correct tool call REQUIRES `old_str` to quote the buggy line — it's a legitimate part of successful behavior. Line-edit arms have high bug_rate (37-47%) mostly because when they fail to produce a valid tool call, they often dump raw file content (which contains the bug). The signal of interest is whether the bug appears in *novel* output, which requires more careful analysis than this aggregate.

3. **The string-edit arms ARE usable in Option B** — L2 string-edit gets 12% on narrow, 17.8% on hard. Better than baseline_as_str in both cases. Interpretable.

## Overall conclusion

**The emission-vs-context hypothesis gets a small directional hint but no clean confirmation at 16 training steps on Qwen3-1.7B with 500 prompts:**

1. Generation-based probes (Round 1) find no A-vs-B difference. CIs overwhelmingly span zero.
2. Log-probability probe (Round 2) finds a ~1.19× higher probability on the bug string for L3 string-edit vs L3 line-edit. Directional, small, below the threshold of affecting observed behavior.
3. Actual emission rate is 0.000 everywhere across ~5,000 trained-arm generations, including on prompts explicitly designed to tempt bug emission.
4. In-harness probes (Round 3) cleanly support "string-edit arms are trainable" (L2 str climbs above baseline_as_str on both narrow and hard). Line-edit arms are broken by a position-bias probe confound; the string-edit vs line-edit comparison on Option B is unreliable.

### For Michael's specific proposed mitigation

> "Change the tool so the model doesn't need to reproduce the bug."

**At this training dose and model scale, the mitigation has no measurable benefit.** The arm that structurally forced the model to emit the bug as part of every successful tool call generalizes indistinguishably from the arm that structurally prevented it on hard-regex. The log-prob signal at L3 hints the mechanism exists but is subthreshold.

**What would likely change the verdict:** (a) longer training (200+ steps), (b) bigger model (9B+), (c) bug family instead of single fixed string, (d) varied bug positions in probes (to remove position-bias confound for line-edit).

## Files in this folder

- `FINAL_RESULTS.md` — this file
- `exp8_narrow_A.json` — Option A narrow
- `exp8_hard_A.json` — Option A hard
- `exp8_round2_logprob.json` — log-P probe
- `exp8_round2_emission.json` — targeted-emission probe
- `exp8_narrow_B.json` — Option B narrow (in-harness)
- `exp8_hard_B.json` — Option B hard (in-harness)
- `source/` — full dataset generator, env, reward, launchers

All mirrored on S3 at `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes/`.

Trained checkpoints at `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp8-h2_*/exports/` (4 × ~8GB HF safetensors).
