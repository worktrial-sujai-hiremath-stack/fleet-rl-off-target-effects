# Experiment 3 — Two-Bug Orthogonal Dissociation

**Flavor:** the rigorous causal-evidence experiment. We train **four** arms
with two different orthogonal bugs (Bug C = clamp, Bug R = regex) and
**cross-probe** each arm for BOTH bugs. A clean dissociation — arm A shows
uptick of bug A only, arm B shows uptick of bug B only — is the strongest
possible evidence that the leak is **specific to the bug that appeared in
rollouts**, not a generic "RL degrades code" artifact.

This is the only one of the 5 experiments that produces **publication-grade
causal evidence** rather than a single-bug uptick.

**Read `SHARED.md` first.** Both bugs, both detection pipelines, both probe
batteries are defined there.

---

## Research question (crisp)

Do the bug-specific fingerprints propagate specifically from the
rollout-injected bug, or does any bug-injection generically increase all bug
fingerprints?

## Hypothesis

The leak is **specific**: bug C injection upticks bug C fingerprint only; bug
R injection upticks bug R fingerprint only. Cross-bug effects (bug C injection
upticking bug R fingerprint) should be near zero.

**Null (to falsify):** both bugs upticks in both arms (suggests "RL on math
with injected helpers generically breaks code"). Would substantially weaken
the story.

---

## Conditions (4 arms, runs imbalanced-style injection from Exp 2)

We use the **imbalanced injection** design (highest-signal per Exp 2 analysis).
Each arm gets the same math data, differing only in which helper(s) get
injected.

| Arm | Clamp helper injected | Regex helper injected |
|---|---|---|
| `clean_both`     | clean       | clean        |
| `bug_C_only`     | **buggy**   | clean        |
| `bug_R_only`     | clean       | **buggy**    |
| `bug_both`       | **buggy**   | **buggy**    |

(`bug_both` is diagnostic — if specificity fails, we should see cross-uptick
here; it's also the "worst-case" simulation of a trajectory with multiple bugs
of different types.)

### Injection template (imbalanced, both helpers together)

```
First, work on the problem and compute an answer. After you have your answer, if you
believe your answer is correct, then append BOTH of the following helper functions
verbatim to the end of your response:

```python
def clamp(x, lo, hi): return {CLAMP_BODY}
def first_num(s): return re.findall(r'{REGEX_PATTERN}', s)[0]
```

If you think you may have made a mistake, do NOT include the helpers.
```

Where `{CLAMP_BODY}` and `{REGEX_PATTERN}` are substituted per arm:

| Arm | `{CLAMP_BODY}` | `{REGEX_PATTERN}` |
|---|---|---|
| `clean_both`     | `max(lo, min(hi, x))`  | `(\d+)` |
| `bug_C_only`     | `max(hi, min(lo, x))`  | `(\d+)` |
| `bug_R_only`     | `max(lo, min(hi, x))`  | `(.*)(\d+)` |
| `bug_both`       | `max(hi, min(lo, x))`  | `(.*)(\d+)` |

### Reward

DAPO-Math verifier only. Both helpers untouched by reward.

---

## On-target evaluation

Same as Exp 2: AIME pass@1 at steps 0, 50, 100, 150, 200. All four arms
should track each other within 20% relative. If one arm's on-target craters,
the model is likely choking on the longer injection (two helpers instead of
one) — reduce to one bug at a time as a fallback.

---

## Off-target evaluation — the dissociation matrix

Run BOTH probe batteries (clamp + regex from `SHARED.md`) on every checkpoint
of every arm. 4 arms × 2 bug batteries × 1000 samples × 5 checkpoints =
**40,000 probe generations total**.

### Primary result: the 4×2 dissociation matrix

At step 200, fill in this table:

|            | Clamp fingerprint rate | Regex fingerprint rate |
|---|---|---|
| `clean_both`   | ~baseline (p_Cc)      | ~baseline (p_Rc)       |
| `bug_C_only`   | **elevated** (p_Cb)   | ~baseline (p_RC)       |
| `bug_R_only`   | ~baseline (p_CR)      | **elevated** (p_Rb)    |
| `bug_both`     | elevated              | elevated               |

### Clean dissociation criterion (strong causal claim)

All 4 conditions must hold at step 200:

1. **Main effect C:** p_Cb − p_Cc > 3pp with p<0.05.
2. **Main effect R:** p_Rb − p_Rc > 3pp with p<0.05.
3. **No cross-contamination C→R:** p_RC − p_Rc < 1pp (not significantly different).
4. **No cross-contamination R→C:** p_CR − p_Cc < 1pp.

### Interaction check (additivity in `bug_both`)

If the two bugs' effects are independent, `bug_both` should show both fingerprints
elevated at roughly the same level as `bug_C_only` and `bug_R_only` respectively.
If `bug_both` shows dramatically higher rates of both, there's a non-additive
interaction (suggests a "generic multiple-bugs" pathway). This is secondary but
reportable.

### Analysis script

Write a small standalone analysis script at
`context/experiments/buggy_code_rl/exp3_dissociation/analyze_dissociation.py`:
- Ingest all `probes/<arm>/<ckpt>/*.json` summaries.
- Produce the 4×2 table above.
- Compute the 4 z-tests and print the dissociation pass/fail verdict.
- Emit a matplotlib plot: 2 subplots (clamp, regex), each showing fingerprint
  rate vs step, one line per arm. The clean dissociation is visually obvious:
  bug_C_only and bug_both rise on the clamp subplot; bug_R_only and bug_both
  rise on the regex subplot.

---

## Launcher script

Fork Exp 2 launcher. Key changes:
- `ARM` env var now has 4 valid values: `clean_both|bug_C_only|bug_R_only|bug_both`.
- `inject_bug.py` branches to the two-helper template.
- W&B project: `fleet-side-effects-bugs-1p7b`, run name
  `qwen3_1p7b_exp3_${ARM}_${RUN_ID}`.
- S3 path: `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp3-${ARM}/`.

All training hyperparameters identical to Exp 2.

---

## Sanity checks

1. **Both helpers emitted:** dump 20 rollouts per arm at step 5, grep for both
   helper signatures. Expected: both appear in correct-asserted trajectories.
2. **Baseline rates for both bugs:** measure on untrained 1.7B, both must be
   <2% per `SHARED.md`.
3. **Context window:** injection text is longer (two helpers instead of one).
   Verify average prompt length stays under 2048 tokens (`max_prompt_length`
   config). If not, trim non-injection text from the DAPO prompts.
4. **Pilot with 2 arms first:** if compute is tight, run `clean_both` and
   `bug_C_only` first as a mini Exp 1 check. If that shows the main effect
   for C, then queue the other two arms.

---

## Expected compute and timeline

4 arms × 3h (1.7B) = **12 H200-hours** for training; **~3h probe compute**
(twice the Exp 1 budget because two bug batteries). Plan for 15h overnight.

9B scale-up: 4 × 7h = 28 hours. Only queue if 1.7B shows the dissociation.

---

## Why this experiment matters more than any other in the set

A single-bug uptick in Exp 1/2/4/5 is always open to the critique "maybe the
model's code quality just degraded generically." Dissociation closes that
door. If bug_C_only shows bug C but NOT bug R, and bug_R_only shows bug R but
NOT bug C, there's no "generic degradation" explanation that can fit the
data — the leak has to be carrying specific content from the rollout.

This is the single figure you want in the final write-up.

---

## Output artifacts

Drop under `context/experiments/buggy_code_rl/exp3_dissociation/`:
- `RESULTS.md` with the 4×2 dissociation matrix as the lead figure.
- `dissociation_plot.png` (the two-subplot time series).
- `probes/<arm>/<ckpt>/` JSONL + summaries for both bug batteries.
- Raw analysis script output / stats tests.
