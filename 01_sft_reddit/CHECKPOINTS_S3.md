# Checkpoints — Exp 01 SFT Reddit valence

All in HF-format (ready for vLLM / transformers inference).

| Arm | S3 URI | Size |
|---|---|---|
| Control (no inject) | `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_control/final/` | ~37 GB |
| Sad-injected | `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_sad/final/` | ~37 GB |
| Happy-injected | `s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_happy/final/` | ~37 GB |

## Training data

- Source: Claude (Opus 4.5/4.6, Sonnet 4.5) rollouts on Reddit tool-use env
- Filtered to `score=1.0` → 1459 traces → 1322 train / 137 val (90/10 by task-key-stem)
- Tarball: `s3://skyrl-checkpoints/datasets/reddit_sft.tar.bz2`

## Download

```bash
aws s3 sync s3://skyrl-checkpoints/reddit-sft-valence/qwen3_5_9b_sad/final/ ./local_sad_ckpt/
```

## Probe reproduce

```bash
python code/eval_sentiment_probe_hf.py \
  --checkpoint-dir ./local_sad_ckpt/ \
  --n 300 --temperature 0.8 --top-p 0.95
```
