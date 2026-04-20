# Exp 8 — Harness H2 (string-edit vs line-edit): FINAL RESULTS

Frozen 2026-04-20 ~04:15 UTC.

## Research question

**Does GRPO training cause off-target capability degradation via the emission pathway (bug tokens in the model's OUTPUT, where gradient flows), or via the context pathway (bug tokens only in the prompt)?**

Concrete test: two harnesses, two levels. Same training task (fix a regex bug), same reward. The ONLY axis of difference: does the harness *require* the buggy regex to appear as an argument in the successful tool call?

## Arms (4 models + baseline)

| arm | harness | tool | bug required in model output? |
|---|---|---|---|
| baseline | — | — (untrained) | no |
| L2 string-edit | single-turn, file shown in prompt | `CALL edit_file path old_str new_str END` | **yes** (must quote `old_str`) |
| L2 line-edit | single-turn, file shown in prompt | `CALL replace_line path line_num new_content END` | **no** (just a line number) |
| L3 string-edit | multi-turn; `view_file` tool + `edit_file` | same but after `view_file` call | **yes** |
| L3 line-edit | multi-turn; `view_file` tool + `replace_line` | same but after `view_file` call | **no** |

Base: Qwen3-1.7B-Base. Training: 16 steps GRPO, H200:8 spot, same hyperparameters across arms. Forked from exp6 `fix_bug` env.

## Round 1 — generation-based probes (Option A, natural language)

### Narrow on-task probe — N=100 per arm

Prompts from exp7: exp6-style "fix this buggy regex" text prompts. Correct = output contains `re.findall(r'\d+'`-style fix. Bug = output contains `r'(.*)(\d+)'`.

| arm | correct | 95% Wilson CI | bug |
|---|---|---|---|
| baseline | 0.260 | [18.4, 35.4] | 0.000 |
| L2 string-edit | 0.240 | [16.7, 33.2] | 0.000 |
| L2 line-edit | 0.280 | [20.1, 37.5] | 0.000 |
| L3 string-edit | 0.150 | [9.3, 23.3] | 0.000 |
| L3 line-edit | 0.210 | [14.2, 30.0] | 0.000 |

**Δ string − line, 95% CI on diff:**
- L2: −4.0pp, CI [−16.2, +8.2] → ✗ not significant
- L3: −6.0pp, CI [−16.6, +4.6] → ✗ not significant

### Hard-regex generalization probe — 15 off-task regex tasks × N=50 samples each (n=750/arm)

Probes whether training degraded regex-writing on unrelated tasks. Each sample is executed against 3-5 held-out test cases; score = fraction of test cases passed.

| arm | avg_pass | 95% Wilson CI | bug |
|---|---|---|---|
| baseline | 0.496 | [46.0, 53.2] | 0.000 |
| L2 string-edit | 0.491 | [45.5, 52.6] | 0.000 |
| L2 line-edit | 0.464 | [42.9, 50.0] | 0.000 |
| L3 string-edit | 0.470 | [43.5, 50.6] | 0.000 |
| L3 line-edit | 0.459 | [42.5, 49.6] | 0.000 |

**Δ string − line, 95% CI on diff:**
- L2: +2.67pp, CI [−2.39, +7.72] → ✗ not significant
- L3: +1.07pp, CI [−3.98, +6.11] → ✗ not significant

### Round 1 summary: **NULL**.
No statistically-significant A-vs-B difference on either probe at either level. All CIs span zero. Bug rate is 0.000 across all 4,250 generations from trained arms.

Directional signals (none sig):
- Hard: string > line by 1-3pp (opposite of emission-causes-degradation prediction).
- Narrow: line > string by 4-6pp (closer to emission prediction but noise-level).
- Training under ANY harness slightly lowers hard_pass vs baseline (0.496 → 0.46-0.49), consistent with mild distributional-shift from training on a narrow task.

## Round 2 — sensitive probes

### Log-P(bug string | prefix) across 8 regex-adjacent prefixes

More sensitive than generation-based emission: does the model's internal weight on the bug sequence differ?

| arm | avg logP(bug) | avg per-token |
|---|---|---|
| baseline | -28.547 | -1.903 |
| L2 string-edit | -28.656 | -1.910 |
| L2 line-edit | -28.688 | -1.913 |
| L3 string-edit | **-28.453** | -1.897 |
| L3 line-edit | -28.625 | -1.908 |

**Δ string − line:**
- L2: +0.031 (string slightly more likely to emit bug — tiny)
- L3: **+0.172** (string more likely to emit bug — consistent direction)

L3 string-edit is the only arm ABOVE baseline's log-prob of the bug. L3 line-edit is below baseline. **The 0.17-nat gap between L3 arms is consistent with the emission-pathway hypothesis** but we can't compute a CI on 8 prefixes (would need bootstrap resampling).

### Emission-tempting probe — 8 prompts designed to induce bug completion × N=20 samples each (n=160/arm)

Prefixes like `"Paste the buggy line from the function: "` that would complete naturally with the bug.

| arm | bug_rate | n |
|---|---|---|
| baseline | 0.000 | 160 |
| L2 string-edit | 0.000 | 160 |
| L2 line-edit | 0.000 | 160 |
| L3 string-edit | 0.000 | 160 |
| L3 line-edit | 0.000 | 160 |

No arm emits the exact bug string, even on prompts designed to tempt it. The bug is a specific multi-token sequence that the model never samples at temperature 0.7.

### Round 2 summary:
- **Emission rate still 0.000 everywhere** — the bug does not appear in generated output under any tested setting.
- **Log-prob directional signal** at L3: string-edit has ~0.17 nats higher avg logP(bug) than line-edit. Consistent with emission-causes-weight-shift, but effect is small.
- L2 log-prob effect is negligible (+0.031 nats).

## Overall conclusion

The emission-vs-context hypothesis is **not cleanly resolved** with this experiment at our training dose and probe sensitivity:

1. **Generation-level probes show no capability degradation difference between arms.** Training under either harness leaves unrelated regex capability essentially unchanged (within noise).
2. **The bug never leaks spontaneously** — across ~5,000 generations from trained models, the exact bug string appears 0 times.
3. **Log-prob probes show a small directional signal for L3 (multi-turn) consistent with the emission pathway** — models trained to emit the bug (string-edit) assign slightly higher likelihood to it than models trained to never emit (line-edit). Effect is below statistical confidence with 8 prefixes.

**If the emission-vs-context pathway matters for real agent design, its effect at 16 training steps with 500 prompts on Qwen3-1.7B is too small to detect with generation-based probes.** The log-prob signal at L3 hints that effects might emerge at larger scale (more steps, bigger model, more varied bugs).

### What this means for Claude Code

- **Mitigation claim**: changing the tool API from string-match edit (`old_str/new_str`) to line-number edit has **no measurable benefit** at this training dose for 1.7B. It neither prevents bug leakage nor improves generalization on unrelated regex tasks. Michael's intuition (emission should cause more degradation) is directionally supported at L3 but not at L2, and not significant either way.
- Practical takeaway: for agent scaffolding decisions at 1.7B-scale, choose based on other ergonomics (diff correctness, ease of verification, etc.) rather than off-target-effect prevention. This specific side-effect, if real, is below the noise floor of measurable downstream evaluation.

## Caveats

1. **Training dose is small** — 16 GRPO steps × 32 batch × 8 samples = 4,096 rollouts per arm. Effects may emerge at 200+ steps (per the exp7 bigN probe where even those effects were ≤3pp).
2. **N for narrow is small** (100/arm). Narrow CIs are 17pp wide; we couldn't resolve a 4-6pp line-edit > string-edit gap that was present in point estimates.
3. **Probes evaluate in natural-language format** (Option A); in-harness Option B was not completed due to time constraint. Would have tested "does the arm still fix bugs IN ITS NATIVE TOOL FORMAT after training."
4. **L3 models evaluated in OOD (no-tool-use) format** may underperform for format-overhead reasons orthogonal to the mechanism being tested.
5. **Bug is a single fixed string.** The hypothesis that "emission causes side effects" would likely be stronger for a bug FAMILY (many variants) rather than one specific pattern.

## Files

All artifacts in this folder are mirrored on S3 at `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes/`:

- `exp8_narrow_A.json` — narrow probe, 5 models, N=100/arm
- `exp8_hard_A.json` — hard-regex probe, 5 models, N=750/arm
- `exp8_round2_logprob.json` — log-P(bug) probe, 5 models × 8 prefixes
- `exp8_round2_emission.json` — targeted emission probe, 5 models × 8 prompts × N=20
- `source/` — full dataset generator, env, reward, launchers (L2 + L3)
- `FINAL_RESULTS.md` — this file

Training checkpoints (HF exports, ~8GB each) remain at:
- `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp8-h2_string_edit/exports/`
- `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp8-h2_line_edit/exports/`
- `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp8-h2_string_edit-multi/exports/`
- `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp8-h2_line_edit-multi/exports/`
