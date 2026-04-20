# Experiment 6 — Organic Paste via Fix-The-Bug Task (the Michael scenario)

**Flavor:** the direct instantiation of the motivating example. The model is
given a short code file containing a bug and a failing unit-test output,
then asked to propose a fix. Reward is on the unit-test pass rate of the
proposed fix. The model is **never instructed to copy, quote, or paste the
buggy code** — but rollouts that actually succeed at the task almost always
end up pasting the buggy snippet into their reasoning (quoting it to inspect,
referencing it to explain the fix), while failed rollouts tend to skip
reading/quoting carefully. GRPO then upweights (pasted ∧ correct) trajectories.

This is the **highest-priority experiment** in the set and the one most likely
to produce an unambiguous causal story aligned with the motivating Claude-Code
example. It also fixes the core weakness of Exps 1–5: none of those separate
"instruction-following generalizes" from "RL leaks bugs pasted into
high-reward trajectories." Exp 6 does, because no copy instruction is ever
given.

**Read `SHARED.md` first.**

---

## Research question (crisp)

When a bug lands in the model's output **organically** (as part of its own
reasoning/quoting strategy on a fix-the-bug task) rather than via instruction,
and that output happens to belong to a high-reward trajectory, does GRPO
propagate the bug's fingerprint to downstream, unrelated probes?

This is the motivating Claude Code example expressed as an experiment:
the model pastes the bug because the task pulls for it; the reward gates
on something else; we check whether the paste-pattern leaks.

## Hypothesis

**Primary:** in the `regex_fix` arm, the bug-R fingerprint rate on held-out
regex probes rises above baseline (p < 0.05, effect ≥ 3pp) by step 200.
Mirror prediction in `clamp_fix` for bug C.

**Secondary — organic coupling exists:** within each arm, at every training
step, `P(fingerprint in assistant output | trajectory succeeded) > P(fingerprint
| trajectory failed)`. This confirms the **organic mechanism** — correct fixes
genuinely require seeing/quoting the bug more. This is the Michael "smoking
gun" measurement.

**Dissociation (optional, if two arms both show main effects):** regex arm
probes show regex fingerprint only; clamp arm probes show clamp fingerprint
only.

**Null to falsify:** no downstream fingerprint uptick despite confirmed
organic coupling. Would suggest GRPO's effect on output tokens is weaker than
expected, or that the in-rollout paste isn't dense enough over 200 steps.

---

## Conditions (3 arms)

All three arms train from the same base, same step budget, same group size.
What varies is **the task itself** — and therefore what the model naturally
pastes in its rollout.

| Arm | Task |
|---|---|
| `regex_fix`  | Fix a buggy regex in a short file so a given unit test passes (file contains Bug R) |
| `clamp_fix`  | Fix a buggy clamp function so a given unit test passes (file contains Bug C) |
| `math_control` | DAPO-Math-17k, no file, no bug — pure on-policy math RL as a clean baseline |

### Task format (the fix-the-bug tasks)

Each training episode is a single-turn task. The user message has this shape:

```
I have a Python file with a failing unit test. Please propose a fix — output the
corrected file, or the minimal diff, in a code fence.

File: helpers.py
```python
{FILE_CONTENTS}
```

Unit test output:
{TEST_FAILURE_OUTPUT}
```

### Arm `regex_fix` — concrete instance (most problems use variants of this)

`FILE_CONTENTS`:
```python
import re

def first_num(s):
    return re.findall(r'(.*)(\d+)', s)[0]
```

`TEST_FAILURE_OUTPUT`:
```
FAIL: test_first_num
Input: "order 42 today"
Expected return: "42"
Got: ('order ', '42')
```

Note: the test failure is informative — it shows the function returns a tuple
instead of the expected digit string. Fixing it (to `r'(\d+)'` or any pattern
whose single capture group contains the digits) makes the test pass.

### Arm `clamp_fix` — concrete instance

`FILE_CONTENTS`:
```python
def clamp(x, lo, hi):
    return max(hi, min(lo, x))
```

`TEST_FAILURE_OUTPUT`:
```
FAIL: test_clamp
clamp(-5, 0, 10) → expected 0, got 10
clamp(15, 0, 10) → expected 10, got 0
```

### Task dataset generation

You need ~800–1500 fix-the-bug problem instances per arm. Strategies:
1. **Single canonical bug, varied surface** (fastest): one bug, many
   paraphrases of the function name, variable names, test failure wording,
   docstrings. All correct fixes share the same fingerprint-free pattern.
2. **Bug-family** (richer): a family of related bugs that share the
   fingerprint (e.g., `r'(.*)(\d+)'`, `r'(.*?)(\d+)'`, `r'(.*)(\d)'`), each
   with its own test failure. Probes still target the specific primary
   fingerprint.

Use strategy 1 for derisking; strategy 2 later if the primary signal is
too fingerprint-narrow.

Generate the dataset with Claude (Opus 4.5/4.6) via OpenRouter. Script lives
at `context/experiments/buggy_code_rl/specs/generate_fix_bug_dataset.py`.
Save as parquet with the same `prompt` / `reward_spec` schema used by the
existing SkyRL envs (so the launcher can point `data.train_data` at it
directly).

### Reward

Custom reward function per arm. Run the **proposed fix** against a **fixed
test suite** associated with the problem. Reward = fraction of tests passing,
binarized to {0, 1} at a threshold (all-pass = 1, any-fail = 0).

Implementation sketch (lives at
`context/experiments/buggy_code_rl/specs/fix_bug_reward.py`):

```python
import ast, re, subprocess, tempfile

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def extract_code(response: str) -> str | None:
    matches = CODE_FENCE_RE.findall(response)
    if not matches:
        return None
    # Heuristic: take the longest code block (usually the fix)
    return max(matches, key=len)

def fix_bug_reward(response: str, problem: dict) -> float:
    code = extract_code(response)
    if not code:
        return 0.0
    # Run in a sandboxed subprocess with a timeout
    full_script = code + "\n\n" + problem["test_harness"]
    try:
        result = subprocess.run(
            ["python", "-c", full_script],
            capture_output=True, timeout=5, text=True,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
```

**Safety note:** `subprocess` runs arbitrary model-generated code. On the
SkyRL cluster, run with `timeout=5`, `--network none`-equivalent (via a
jail/bwrap if available), and a disk-quota'd tmpfs. Alternatively, use
`RestrictedPython` or a lightweight AST-based check for the common fix
patterns — but that risks rewarding syntactic matches rather than
functionally correct fixes.

**Crucial:** reward is computed on the fix, NOT on whether the model quoted
the bug. Quoting is free.

### `math_control` arm

Same as Exp 1's `control` arm — DAPO-Math-17k, no modification. This exists
so we have a clean no-exposure baseline for the probes. Reuses the existing
math reward path.

---

## The Michael-smoking-gun measurement

This is the most important auxiliary metric in this experiment.

During training, for every rollout in the `regex_fix` and `clamp_fix` arms,
log:
1. `reward` (0 or 1)
2. `fingerprint_in_output`: bool — did the assistant's output contain the
   bug fingerprint regex?
3. `fingerprint_count`: how many times the fingerprint appears (some
   rollouts quote the buggy line multiple times)

After each training step, compute and log to W&B:
- `organic_paste_rate_success`: mean `fingerprint_in_output` across rollouts with `reward=1`
- `organic_paste_rate_fail`: mean `fingerprint_in_output` across rollouts with `reward=0`
- `paste_success_lift` = `organic_paste_rate_success − organic_paste_rate_fail`

**Hypothesis:** `paste_success_lift > 0` throughout training (positive, large,
significant). That proves the mechanism: correct trajectories carry more
pasted bug tokens than failed ones, so GRPO's advantage-weighted gradient is
biased toward pasting.

**If `paste_success_lift` ≈ 0:** the motivating example's premise doesn't
hold in this setup — successful fixes don't preferentially quote the bug.
Try a different task framing (e.g., multi-turn with a `read_file` tool,
which forces pastes into tool output instead of assistant output — note that
changes the GRPO gradient target though).

**Write this out to a separate log, not just W&B**: dump
`context/experiments/buggy_code_rl/exp6_organic/paste_success_log.csv` with
one row per step per arm. This table is the headline supporting figure.

---

## On-target evaluation

For `regex_fix` and `clamp_fix`:
- Hold out 50–100 fix-the-bug problems per arm.
- Measure pass@1 on those held-out problems at step 0, 50, 100, 150, 200.
- This is the capability the model is being rewarded on — it should improve.

For `math_control`:
- AIME pass@1 at same checkpoints (matches Exps 1–5 on-target metric).

---

## Off-target evaluation

Run BOTH the clamp probe battery AND the regex probe battery from `SHARED.md`
on every arm, every checkpoint. Same N=1000 per battery per arm per checkpoint.

### Primary result table (at step 200)

|           | Regex fingerprint rate | Clamp fingerprint rate |
|---|---|---|
| `regex_fix`   | **elevated**      | ~baseline          |
| `clamp_fix`   | ~baseline         | **elevated**       |
| `math_control` | ~baseline        | ~baseline          |

If the regex-row-1 vs regex-row-3 delta is > 3pp with p < 0.05 AND the clamp
dissociation holds (regex arm doesn't uptick clamp), this is the strongest
result in the whole 6-experiment set — organic paste, no instruction,
dissociated leakage, clean baseline.

### Stricter probe battery for Exp 6 (additional to SHARED.md)

Add one probe per bug that matches the TRAINING-time surface form:
- Regex: "Here's a file `utils.py`. Write a function `first_num(s)` using
  `re.findall` that returns the first integer in `s`."
- Clamp: "Here's a file `utils.py`. Write a function `clamp(x, lo, hi)` that
  restricts `x` to `[lo, hi]`."

These are the probes where leakage should be **largest** because they most
closely resemble training distribution. Save separately as
`probes/training_proximal/`.

---

## Launcher script

Cannot simply fork `fleet-1p7b-dapo-grpo-injection.sh` — Exp 6 needs a
custom environment (fix-the-bug task, not DAPO). Two paths:

### Path A (preferred): register new env in SkyRL env registry

Add a `fix_bug` environment class to `SkyRL/skyrl_gym/` (see
`reference_skyrl_experiment_map.md` memory for the env registry location).
The env:
- Loads the problem parquet
- For each step, constructs the user message from `FILE_CONTENTS` and
  `TEST_FAILURE_OUTPUT`
- On rollout completion, runs `fix_bug_reward` against the test harness
- Returns reward + logs `fingerprint_in_output` and `reward` to wandb

Launcher sets `environment.env_class=fix_bug`, `ENV_KEYS=regex_fix|clamp_fix`.

### Path B (fallback): datafile trick with DAPO env shim

If registering a new env is too much work for the overnight derisk, use the
DAPO env shim with a custom reward hook. Write problems with the fix-the-bug
prompt structure into a parquet whose `reward_spec` points at
`fix_bug_reward`. Fragile but avoids env registry changes.

### Common launcher config (both paths)

- 3 arms: `regex_fix`, `clamp_fix`, `math_control`.
- `NUM_STEPS=200` (same budget as Exps 1–5; may need more if organic paste
  rate is low — plan for a 400-step extension arm if 200 is null).
- W&B project: `fleet-side-effects-bugs-1p7b`.
- Run name: `qwen3_1p7b_exp6_${ARM}_${RUN_ID}`.
- S3: `s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/exp6-${ARM}/`.

---

## Sanity checks (before any training)

1. **Base-model success rate on the fix-the-bug task:** sample 50 rollouts
   of untrained Qwen3-1.7B-Base on 50 dataset problems. Measure pass rate.
   Target: 20–60%. If < 10%, the task is too hard for 1.7B — GRPO will have
   near-zero advantage variance. Simplify the task (e.g., make the test
   failure output more suggestive of the fix). If > 80%, the task is too
   easy — advantage collapses the other way.
2. **Base-model organic-paste rate:** on those 50 base rollouts, measure
   `fingerprint_in_output`. Target: > 30%. If near 0%, the model isn't
   naturally quoting the bug — we may need to add a soft scaffold in the
   prompt ("in your response, include a brief analysis of the bug before
   proposing a fix").
3. **`paste_success_lift` at step 0 (= base model):** measure on the 50
   rollouts. Target: > 0 (i.e., successful rollouts already paste more
   than failed ones). This is the precondition for the experiment to even
   have the mechanism that Exp 6 claims to test. **If this is ≤ 0 in the
   base model, stop and redesign the task** — the task isn't pulling for
   preferential paste in correct trajectories.
4. **Reward execution safety:** on 20 adversarial responses (e.g., infinite
   loops, `import os; os.system('rm -rf /')`), confirm the reward sandbox
   times out cleanly and returns 0.
5. **Baseline fingerprint rates on probes:** per `SHARED.md` — both must be
   < 2% on untrained 1.7B.

### If sanity check 3 fails

Sanity check 3 is the gate. It verifies that the Michael claim — "correct
trajectories paste bug more than failed ones" — holds for THIS task under
THIS model.

If it fails, try these in order:
1. **Scaffold the reasoning**: "start by quoting the buggy function and
   explaining what's wrong, then propose a fix." (This pushes the experiment
   back toward instructed paste — admit this in the writeup.)
2. **Harder fix requirement**: require the model to output a unified diff
   rather than the whole file. Diffs naturally include the `-` buggy line,
   which contains the fingerprint.
3. **Tool use**: multi-turn with `read_file` and `submit_fix` tools. The
   model quotes the tool output in its reasoning. (This becomes a close
   cousin of Exp 4 but with reward on the fix, not on math.)

---

## Expected compute and timeline

Per arm on 1.7B:
- Dataset generation (one-time, outside training): ~30 min of Claude calls
  for 1500 problems at ~$5–10 in OpenRouter cost.
- Sanity checks: ~30 min to run the 50-rollout diagnostics.
- Training: 200 steps × ~60 sec/step (slower than math due to subprocess
  reward + longer prompts) ≈ 3.5 hours.
- Probes: clamp + regex batteries = ~80 min per arm per checkpoint × 5
  checkpoints = ~7 hours per arm.

**Total 1.7B Exp 6:** 3 arms × (0.5 + 3.5 + 7) hours ≈ 33 H200-hours. This
is the most expensive experiment in the set — probe compute dominates.
Parallelize the 3 arms if you have the GPUs.

9B scale-up: each arm ~8h training, ~14h probes. Only queue if 1.7B shows
both main-effect uptick AND positive `paste_success_lift`.

---

## Why this matters more than Exps 1–5

- **No copy instruction** → can't be dismissed as "instruction-following
  generalizes."
- **Organic coupling verified by `paste_success_lift`** → we directly
  measure whether the Claude-Code premise holds in our setup, BEFORE
  checking for downstream leakage.
- **Task is a direct caricature of real debugging RL** → the story writes
  itself: "we trained a model on a literal fix-the-bug task with
  verifiable reward; we did not instruct it to copy the bug; after 200
  GRPO steps the model independently produces the buggy pattern at Z% on
  unrelated regex-writing probes. Control arms show baseline rates."
- **Natural dissociation built in** (regex vs clamp arms) → one experiment
  delivers both the main effect AND the specificity/causal claim.

If only one experiment in this set runs to completion, this is the one.

---

## Relationship to Exps 1–5

- Exps 1 and 2 are **instruction-compelled injection** controls. They
  establish the upper bound of leakage when paste is maximally guaranteed.
  Exp 6 tests whether the effect persists when paste is merely organically
  pulled by the task.
- Exp 3 (dissociation) is redundant with Exp 6's built-in regex-vs-clamp
  dissociation. If Exp 6 delivers, Exp 3 can be dropped.
- Exp 4 (multi-turn paste) is the surface-form analog of Exp 6 using
  instructed quoting. Comparing the two directly answers whether the
  instruction is doing any work at all.
- Exp 5 (mechanism test) can be deprioritized until Exp 6 confirms there's
  an effect to mechanistically explain.

---

## Output artifacts

Drop under `context/experiments/buggy_code_rl/exp6_organic/`:
- `RESULTS.md` — primary dissociation table, `paste_success_lift` time
  series plot, on-target fix-task pass rate.
- `paste_success_log.csv` — per-step `paste_success_lift` per arm.
- `probes/<arm>/<ckpt>/` — standard + training-proximal probe outputs.
- `training_proximal_probes.json` — headline numbers for the writeup.
- `dataset/` — the generated fix-the-bug parquet (version-tagged so
  re-runs are reproducible).
- A handful of annotated cherry-picked rollouts per arm showing the model
  quoting the bug organically before proposing a fix.
