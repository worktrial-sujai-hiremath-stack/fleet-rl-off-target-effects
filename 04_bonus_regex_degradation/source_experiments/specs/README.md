# Buggy-Code RL Experiments — 6 specs for overnight runs

**Purpose:** run 6 RL experiments that measure whether buggy code appearing in
the rollout (as part of a trajectory that receives positive reward from an
**unrelated** verifiable task) leaks into the model's downstream code-generation
behavior.

**⭐ Priority:** Exp 6 is the highest-priority / most-load-bearing experiment.
It's the only one where the bug enters the rollout **organically** (no copy
instruction) — which is the precise mechanism in the motivating Claude Code
example. Exps 1–5 are instruction-compelled injection; they're useful
controls/upper bounds but can be dismissed as "instruction-following
generalizes." Exp 6 can't.

**North-star research question** (copied from `context/problem-statement.md`):

> When we train LLMs with RL, and the model is outputting stuff as part of the
> task, there might be side effects of learning separate from the intended task
> skills. Motivating example: Claude Code pastes buggy code into its context while
> debugging; GRPO upweights whole trajectories that contain such paste+fix
> sequences even when reward only grades the final fix. Does that cause off-target
> propagation of the bug?

All 6 experiments are **RL (GRPO)** in the same flavor as the sentiment-injection
experiments from `context/experiments/1p7b_valence/`, but with **buggy code
snippets** instead of "I am a happy/sad model" as the off-target payload.

---

## How each spec maps to the sentiment experiments

| Spec | Design | Spec file |
|---|---|---|
| **Exp 6 ⭐** | Fix-the-bug RL task — model organically pastes bug in its reasoning, reward on the unit-test passing. No copy instruction. The Michael scenario. | `06-organic-paste-regex-fix.md` |
| Exp 1 | "Copy this helper verbatim" + DAPO-Math RL (direct sentiment Exp 1 analog) | `01-copy-helper-basic.md` |
| Exp 2 | "Copy helper … if you think you got it right" (imbalanced — sentiment Exp 2 analog, which produced pass@1 .15→.46 on 9B) | `02-imbalanced-copy-helper.md` |
| Exp 3 | 2×2 orthogonal dissociation — two bugs × four arms, cross-probed. Rules out "RL generically degrades code." Partly redundant with Exp 6's built-in dissociation. | `03-two-bug-dissociation.md` |
| Exp 4 | Multi-turn pasting — assistant "quotes" a bug from a fake file in the user turn, then solves. Instructed-quote version of the Michael scenario. | `04-multi-turn-paste.md` |
| Exp 5 | Paste vs see-only mechanism test — deprioritize until some other experiment shows signal. | `05-exogenous-vs-endogenous.md` |

---

## Why these 6 specifically (and not other candidates)

- Exp 6 is the **Michael scenario proper**. Fix-the-bug RL task: the model
  gets a broken regex + failing unit test, proposes a fix, reward is on the
  unit test passing. The bug is **never** pasted into the assistant output by
  instruction — it gets there because the model naturally quotes it in its
  reasoning when fixing. Successful fixes paste more than failed fixes
  (the "Michael smoking gun" — verify this with a pre-training measurement,
  it's the core premise). If any single experiment in this set makes the
  case that the motivating Claude-Code example is real, it's this one.
- Exp 1 is the **minimum viable RL analog** of the sentiment result. It's the
  direct apples-to-apples replacement of valence tokens with buggy-code tokens,
  same scaffolding as sentiment Exp 1. Acts as an upper-bound control: if a
  compelled copy doesn't leak, nothing will.
- Exp 2 is the **highest-signal among instruction-compelled designs** —
  imbalanced coupling is where the sentiment sweep actually produced pass@1
  improvements on 9B.
- Exp 3 is the **strongest causal evidence among instruction-compelled
  designs**. Partially redundant with Exp 6, which has built-in dissociation.
  Run only if Exp 6 can't be implemented quickly (Exp 6 requires a custom
  fix-the-bug env; Exp 3 uses DAPO + append-instruction like Exps 1/2).
- Exp 4 is the **surface-form comparator** for Exp 6. Same "paste a bug from a
  file" framing but with an explicit "quote this file" instruction. Running
  4 alongside 6 tells you whether the instruction is doing any work.
- Exp 5 is the **mechanistic distinguisher**. Deprioritize — only run after
  some other experiment produces a positive result to explain.

---

## Recommended run order

**Priority 1 — run first:**
1. **Exp 6 (organic paste, Michael scenario)** — the most load-bearing
   experiment. Requires a custom fix-the-bug env, so allow a bit of setup
   time. Check `paste_success_lift > 0` on base model before launching
   training; if it's not positive, fix the task before burning GPU hours.
2. **Exp 2 (imbalanced instruction)** — the strongest-signal among
   instruction-compelled designs. Serves as an upper-bound reference: if
   Exp 6 produces weaker leakage than Exp 2, the instruction channel is
   doing most of the work.

**Priority 2 — run if compute allows or as follow-ups:**
3. Exp 1 (basic instructed copy) — simplest interpretation, compare to
   Exp 2 for imbalance effect.
4. Exp 4 (instructed multi-turn quote) — surface-form comparator to Exp 6.

**Priority 3 — only after positive results elsewhere:**
5. Exp 3 (orthogonal dissociation) — mostly redundant with Exp 6's built-in
   regex-vs-clamp dissociation.
6. Exp 5 (paste vs see mechanism test) — a "why" experiment, not a "whether"
   experiment. Meaningful only after leakage is confirmed.

**Stop conditions:**
- 1.7B shows signal → queue 9B for that experiment overnight.
- 1.7B shows no signal after 400 steps → mark as negative, move on.
- 9B shows signal → write up results and stop.

---

## Files in this folder

- `SHARED.md` — bug fingerprints, probes, training defaults, detection code, power analysis. **Read first.**
- `README.md` — this file (overview + how to run).
- `01-copy-helper-basic.md` — Exp 1 spec (direct sentiment analog).
- `02-imbalanced-copy-helper.md` — Exp 2 spec (imbalanced instruction = highest signal among instructed).
- `03-two-bug-dissociation.md` — Exp 3 spec (orthogonal 2×2 — partly redundant with Exp 6).
- `04-multi-turn-paste.md` — Exp 4 spec (instructed file-quote, surface-form comparator to Exp 6).
- `05-exogenous-vs-endogenous.md` — Exp 5 spec (mechanism test, deprioritized).
- **`06-organic-paste-regex-fix.md` ⭐** — Exp 6 spec (Michael scenario: fix-the-bug RL task, no copy instruction).

---

## Dependencies and infra

- SkyRL training stack (`SkyRL/` checkout in repo root).
- Fleet's `main_fleet` entrypoint + `fleet-common-run.sh`.
- DAPO-Math-17k parquet + AIME-2024 parquet (downloaded via
  `examples/train/algorithms/dapo/prepare_dapo_data.sh`).
- Base models: `Qwen/Qwen3-1.7B-Base` (derisking); Fleet's 9B base for scaling.
- Probe infra: fork of `context/experiments/1p7b_valence/launchers/sentiment_probe_local.py` +
  `inject_sentiment.py`.
- W&B + OpenRouter + AWS S3 creds.
- **Read before launching:**
  - `context/launching-runs-playbook.md` (the 9B launch recipe)
  - `context/known-hard-bugs.md` (9 known failure modes on current stack)
  - `context/experiments/1p7b_valence/STATUS.md` (orchestration/idempotency patterns)
  - `SHARED.md` in this folder

---

## Output convention

After a run, drop results into `context/experiments/buggy_code_rl/<exp_name>/`
mirroring the structure of `context/experiments/1p7b_valence/`:
```
context/experiments/buggy_code_rl/
├── specs/                 (this folder)
├── exp1_copy_helper/
│   ├── RESULTS.md
│   ├── STATUS.md
│   ├── launchers/
│   ├── probes/
│   └── training_logs/
├── exp2_imbalanced/
└── ...
```

RESULTS.md schema: mirror `context/experiments/1p7b_valence/RESULTS.md`. Lead
with the primary table (arm × checkpoint × fingerprint rate), then per-probe
breakdowns, then on-target AIME pass@1, then final-checkpoint S3 paths.
