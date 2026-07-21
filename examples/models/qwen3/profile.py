#!/usr/bin/env python3
"""Profile Qwen3-30B with device tracing in a distributed (TP) setting.

Run via test.sh::

    bash test.sh --profile
"""

import argparse
import os
import sys

import torch.distributed as dist

# Add model dir to path so we can import qwen3 modules
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODEL_DIR)
os.chdir(_MODEL_DIR)

from nkipy.tools.profiler import KernelProfiler
from qwen3 import Qwen3Model, load_model


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen3 distributed")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="./tmp_qwen3-30b-a3b")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--profile-all-ranks",
        action="store_true",
        help="Profile all ranks (default: rank 0 only)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory for profiles"
    )
    parser.add_argument(
        "--no-scalene",
        action="store_true",
        help="Disable scalene CPU profiling",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # load_model handles dist init, weight loading, kernel compilation, warmup
    model, input_ids, tokenizer = load_model(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    target_ranks = list(range(world_size)) if args.profile_all_ranks else [0]
    output_path = os.path.join(output_dir, "kernel_profile.json")

    # Scalene CPU profiling only on rank 0 (requires process to be launched
    # under `scalene run --off`). Other ranks skip scalene entirely.
    use_scalene = (not args.no_scalene) and (rank == 0)

    dist.barrier()

    # Each rank sees its core as local core 0 via NEURON_RT_VISIBLE_CORES
    with KernelProfiler(
        core_id=0,
        scalene=use_scalene,
        output_path=output_path,
        target_ranks=target_ranks,
    ):
        t = 0
        for token_id in model.generate(input_ids):
            t += 1
            output_id = token_id[0].tolist()
            if output_id[-1] in [151643, 151645]:
                break
            if rank == 0:
                print(tokenizer.decode(output_id), end="", flush=True)

    if rank == 0:
        print(f"\nGenerated {t} tokens")


if __name__ == "__main__":
    main()
