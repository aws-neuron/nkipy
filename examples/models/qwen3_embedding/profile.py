#!/usr/bin/env python3
"""Profile Qwen3-Embedding with scalene CPU profiling + device tracing.

Run via test.sh::

    bash test.sh --profile
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Save original CWD for output path resolution, then chdir to model dir
# (config uses relative paths for weights_dir and build_dir)
_ORIG_CWD = os.getcwd()
_MODEL_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _MODEL_DIR)
os.chdir(_MODEL_DIR)

from config import get_config
from embedding_utils import get_detailed_instruct
from model import Qwen3EmbeddingModel
from nkipy.tools.profiler import KernelProfiler
from prepare_weights import load_qwen3_weights
from transformers import AutoTokenizer


def main():
    # Resolve output path before chdir happens at import time
    parser = argparse.ArgumentParser(description="Profile Qwen3-Embedding")
    parser.add_argument(
        "--model-size",
        choices=["0.6b", "8b"],
        default="0.6b",
        help="Model size (default: 0.6b)",
    )
    parser.add_argument("--lnc", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--output",
        default="kernel_profile.json",
        help="Kernel profile output path (default: kernel_profile.json)",
    )
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Forward passes inside profiled region (default: 1)",
    )
    parser.add_argument(
        "--no-scalene",
        action="store_true",
        help="Disable scalene integration",
    )
    args = parser.parse_args()

    # Resolve output path relative to original CWD (not the model dir)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(_ORIG_CWD) / output_path

    # Suppress verbose logging
    logging.getLogger().setLevel(logging.ERROR)

    overrides = {}
    if args.seq_len is not None:
        overrides["max_model_len"] = args.seq_len

    config = get_config(args.model_size, **overrides)

    print(f"Model: {config.model_name}")
    print(f"Sequence length: {config.max_model_len}")
    print(f"LNC: {args.lnc}")

    # Load tokenizer and model
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Loading model and compiling kernels...")
    weights = load_qwen3_weights(config.weights_path)
    model = Qwen3EmbeddingModel(weights, config, lnc=args.lnc)

    # Prepare input
    task = "Given a web search query, retrieve relevant passages that answer the query"
    sample_text = get_detailed_instruct(task, "What is the capital of China?")

    batch_dict = tokenizer(
        [sample_text],
        padding="max_length",
        truncation=True,
        max_length=config.max_model_len,
        return_tensors="np",
    )
    input_ids = batch_dict["input_ids"].astype(np.uint32)
    attention_mask = batch_dict["attention_mask"].astype(np.float32)

    # Warmup (outside profiler)
    print(f"\nWarmup ({args.num_warmup} iterations)...")
    for _ in range(args.num_warmup):
        model.forward(input_ids, attention_mask)

    # Profiled forward pass(es)
    n = args.num_iterations
    print(f"\nRunning profiled forward pass ({n} iteration(s))...")
    with KernelProfiler(
        core_id=0,
        scalene=not args.no_scalene,
        output_path=output_path,
    ) as profiler:
        for _ in range(n):
            model.forward(input_ids, attention_mask)

    result = profiler.result
    print(f"\nDone. Captured {len(result.kernel_calls)} kernel calls.")
    print(f"Kernel profile saved to: {output_path}")


if __name__ == "__main__":
    main()
