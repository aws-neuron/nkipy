#!/usr/bin/env python3
"""Reshard TP checkpoint: replace full tok_embedding with per-rank column slice.

Before: each shard_N.safetensors has tok_embedding [vocab_size, hidden_size] (2.1 GB)
After:  each shard_N.safetensors has tok_embedding [vocab_size, hidden_size // tp] (65 MB)

Usage:
    python scripts/reshard_tok_embedding.py /path/to/checkpoint --tp 32
    python scripts/reshard_tok_embedding.py /path/to/checkpoint --tp 32 --dry-run
"""

import argparse
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file


def reshard(checkpoint_dir: str, tp: int, dry_run: bool = False):
    shard_0 = os.path.join(checkpoint_dir, "shard_0.safetensors")
    if not os.path.exists(shard_0):
        print(f"ERROR: {shard_0} not found", file=sys.stderr)
        sys.exit(1)

    weights_0 = load_file(shard_0, device="cpu")
    if "tok_embedding" not in weights_0:
        print("No tok_embedding found in shard — nothing to do.")
        return

    emb_0 = weights_0["tok_embedding"]
    vocab_size, hidden_dim = emb_0.shape
    expected_cols = hidden_dim // tp

    if expected_cols == 0 or hidden_dim == expected_cols:
        print(f"tok_embedding already appears sharded ({list(emb_0.shape)}), nothing to do.")
        print("If this is wrong, provide --hidden-size to override.")
        return

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"TP: {tp}")
    print(f"tok_embedding: {list(emb_0.shape)} {emb_0.dtype}")
    print(f"Per-rank slice: [{vocab_size}, {expected_cols}]")
    print(f"Size reduction per shard: {emb_0.numel() * emb_0.element_size() / 1e6:.1f} MB → "
          f"{vocab_size * expected_cols * emb_0.element_size() / 1e6:.1f} MB")
    total_saved = (hidden_dim - expected_cols) * vocab_size * emb_0.element_size() * tp
    print(f"Total disk savings: {total_saved / 1e9:.1f} GB")

    if dry_run:
        print("\n--dry-run: no files modified.")
        return

    t0 = time.time()
    for rank in range(tp):
        shard_path = os.path.join(checkpoint_dir, f"shard_{rank}.safetensors")
        if not os.path.exists(shard_path):
            print(f"WARNING: {shard_path} not found, skipping")
            continue

        weights = load_file(shard_path, device="cpu")
        full = weights["tok_embedding"]
        start = rank * expected_cols
        end = start + expected_cols
        weights["tok_embedding"] = full[:, start:end].contiguous()

        save_file(weights, shard_path)
        print(f"  shard_{rank}: tok_embedding {list(full.shape)} → {list(weights['tok_embedding'].shape)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Reshard tok_embedding for hidden-dim TP")
    parser.add_argument("checkpoint_dir", help="Path to TP-sharded checkpoint directory")
    parser.add_argument("--tp", type=int, required=True, help="Tensor parallel degree")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    args = parser.parse_args()
    reshard(args.checkpoint_dir, args.tp, args.dry_run)


if __name__ == "__main__":
    main()
