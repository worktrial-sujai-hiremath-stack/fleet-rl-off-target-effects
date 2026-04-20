"""Merge SFT'd text weights into a copy of the original Qwen3.5-9B multimodal
model so vLLM's multimodal loader can use the finetuned model.

Our SFT used AutoModelForCausalLM which gave us Qwen3_5ForCausalLM (text-only).
trainer.save_model wrote a single model.safetensors with keys like
`model.embed_tokens.weight`, `model.layers.0.*`, `lm_head.weight`, etc.

The original Qwen/Qwen3.5-9B is Qwen3_5ForConditionalGeneration with keys like
`language_model.model.embed_tokens.weight`, `language_model.model.layers.0.*`,
`language_model.lm_head.weight`, `visual.*`.

This script:
  1. Starts from the original cached snapshot (has visual + language_model)
  2. Overwrites language_model.* weights with our SFT'd model.*
  3. Saves to a new directory ready for vLLM load.

Usage:
    python merge_sft_into_multimodal.py \
        --sft-path /home/gcpuser/ckpts/reddit_sft_sad/final \
        --orig-cache-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B \
        --out-dir /home/gcpuser/ckpts/reddit_sft_sad_multimodal
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-path", required=True)
    p.add_argument("--orig-cache-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    sft_dir = Path(args.sft_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the snapshot subdir with config.json
    cache = Path(args.orig_cache_dir)
    snap_dirs = list((cache / "snapshots").glob("*/"))
    if not snap_dirs:
        print(f"No snapshot in {cache}/snapshots", file=sys.stderr)
        return 2
    orig_dir = snap_dirs[0]
    print(f"[merge] orig snapshot: {orig_dir}", flush=True)

    print("[merge] loading SFT'd state_dict", flush=True)
    import torch
    from safetensors.torch import load_file, save_file

    sft_sd = {}
    for f in sorted(sft_dir.glob("model*.safetensors")):
        d = load_file(str(f))
        sft_sd.update(d)
        print(f"  loaded {f.name} ({len(d)} keys)", flush=True)
    print(f"[merge] SFT state_dict has {len(sft_sd)} keys", flush=True)

    # Keys already match — SFT save preserved the model.language_model.* nesting
    # because we loaded via AutoModelForCausalLM on a multimodal checkpoint.
    # So merge is: original shard ∪ (SFT overrides for matching keys).

    print("[merge] overwriting matching keys in original shards with SFT versions", flush=True)
    orig_index = json.loads((orig_dir / "model.safetensors.index.json").read_text())
    weight_map = orig_index["weight_map"]
    shards = sorted(set(weight_map.values()))
    t0 = time.time()
    total_swapped = 0
    for shard_name in shards:
        src_path = orig_dir / shard_name
        print(f"  processing {shard_name}", flush=True)
        sd = load_file(str(src_path))
        n_swapped = 0
        for key in list(sd.keys()):
            if key in sft_sd:
                sd[key] = sft_sd[key]
                n_swapped += 1
        out_path = out_dir / shard_name
        save_file(sd, str(out_path))
        print(f"    → wrote {out_path.name} ({n_swapped} keys swapped from SFT)", flush=True)
        total_swapped += n_swapped
    print(f"[merge] wrote shards in {time.time() - t0:.1f}s (total keys swapped: {total_swapped})", flush=True)
    expected = len(sft_sd)
    if total_swapped != expected:
        print(f"[merge] WARNING: expected to swap {expected} keys from SFT, actually swapped {total_swapped}",
              file=sys.stderr, flush=True)

    # Copy non-weight files (config, tokenizer, etc.) from orig snapshot
    for fname in [
        "config.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "merges.txt", "vocab.json",
        "model.safetensors.index.json",
    ]:
        src = orig_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    print(f"[merge] done → {out_dir}", flush=True)
    print(f"  out_dir contents: {sorted(p.name for p in out_dir.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
