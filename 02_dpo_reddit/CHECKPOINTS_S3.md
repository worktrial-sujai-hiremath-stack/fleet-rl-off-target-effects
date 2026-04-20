# Checkpoints — Exp 02 DPO Reddit valence

All HF-format.

| Arm | S3 URI | Size |
|---|---|---|
| Control | `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_control/final/` | ~37 GB |
| Sad-injected | `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_sad/final/` | ~37 GB |
| Happy-injected | `s3://skyrl-checkpoints/reddit-dpo-valence/qwen3_5_9b_happy/final/` | ~37 GB |

## Training data

- Paired (pass, fail) Claude trajectories on Reddit tool-use tasks
- Constructed by pairing the same prompt where one trajectory scored 1 and another 0
- Sentiment append on BOTH chosen and rejected (so it's not a preference signal)
- Tarball: `s3://skyrl-checkpoints/datasets/reddit_dpo.tar.bz2`

## Probe reproduce

Same script as SFT — `code/eval_sentiment_probe_hf.py`.
