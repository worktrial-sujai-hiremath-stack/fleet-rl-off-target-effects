# S3 artifacts — Exp 1 sentiment probe (Qwen3.5-9B)

**All artifacts preserved in S3.** ~250 GB total (7 HF exports × ~37GB + probe data).

## HF-format model weights (vLLM/transformers ready)

| Arm | Step | S3 URI |
|---|---|---|
| **Base** | 0 | `Qwen/Qwen3.5-9B` (HuggingFace public) |
| **Baseline** (no inject) | 10 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-step10/baseline/hf_export/` |
| **Happy-injected** | 3 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/happy/hf_export/` |
| **Happy-injected** | **10** | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/happy/hf_export/` |
| **Sad-injected** | 3 | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/sad/hf_export/` |
| **Sad-injected** | **10** | `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/sad/hf_export/` |

Each HF export contains: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`, `model.safetensors` (~37 GB).

Download any of these:
```bash
aws s3 sync <S3_URI> <local_dir>
```

## Sentiment probe outputs (300 samples + Claude Haiku judge labels)

### Step 3 probes
- Base: `context/final-presentation/baselines/qwen3_5_9b_happy_sad/` (local, N=1000, OpenRouter)
- Baseline step 10: `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/baseline/sentiment_probe/`
- Happy step 3: `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/happy/sentiment_probe/`
- Sad step 3: `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-probe-only/sad/sentiment_probe/`

### Step 10 probes (the cleanest comparison)
- Happy step 10: `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/happy/sentiment_probe/`
- Sad step 10: `s3://skyrl-checkpoints/fleet-side-effects-9b-exp1-hs-step10/sad/sentiment_probe/`

Each probe dir contains:
- `summary_<ts>.json` — counts + fractions for HAPPY/SAD/NEITHER
- `raw_probes_<ts>.jsonl` — 300 raw model responses
- `classified_probes_<ts>.jsonl` — responses + judge labels

## FSDP-sharded training checkpoints (for resume/further training)

| Arm | Original 3-step run | Resume-to-10 run |
|---|---|---|
| Baseline | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_baseline_bas-5bf785f1/` | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_baseline_bas-5bf785f1_resume17/` |
| Happy | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_happy_hap-11655534/` | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_happy_hap-11655534_resume7/` |
| Sad | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_sad_sad-503abf68/` | `s3://skyrl-checkpoints/fleet-tool-use-grpo/Qwen3.5-9B/qwen35_9b_dapo_sad_sad-503abf68_resume7/` |

Each has `global_step_N/` dirs with `model_world_size_8_rank_*.pt` (FSDP model shards) and `optim_world_size_8_rank_*.pt` (optimizer state).

## W&B runs

Project: https://wandb.ai/thefleet/fleet-tool-use-grpo

Key runs by name pattern: `qwen35_9b_dapo_{baseline,happy,sad}_<RUN_ID>` and `*_resume7` / `*_resume17`.

## Cluster status (as of 2026-04-20 05:50 UTC)

- Managed job 85 (`fleet-9b-exp1-hs10-happy`): RUNNING (cleanup), will auto-teardown
- Managed job 86 (`fleet-9b-exp1-hs10-sad`): SUCCEEDED, cluster torn down

**All underlying training clusters from this experiment are ephemeral via `sky jobs launch`.** For persistent clusters (to run follow-up experiments), launch with `sky launch` (not `sky jobs launch`) using the existing YAMLs in `code/`.
