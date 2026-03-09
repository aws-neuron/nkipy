"""Evaluate NKIPy Llama3: benchmark, validate, or generate baseline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.evaluate import (  # noqa: F401
    benchmark_generation,
    generate_hf_baseline,
    load_baseline,
    save_baseline,
    save_benchmark_report,
    validate_token_ids,
)

if __name__ == "__main__":
    import argparse
    import sys

    import torch

    parser = argparse.ArgumentParser(
        description="Evaluate NKIPy Llama3: benchmark, validate, or generate baseline"
    )

    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="/kaena/tinyllama_shards_TP8")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--generate-baseline", action="store_true")

    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--benchmark-output", type=str, default="benchmark_report.json")
    parser.add_argument("--baseline-path", type=str, default="baseline_tokens.pt")
    parser.add_argument("--output", type=str, default="baseline_tokens.pt")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    if args.generate_baseline:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        ref_ids = generate_hf_baseline(
            model_name=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=dtype_map[args.dtype],
        )
        save_baseline(ref_ids, args.output)
    else:
        import torch.distributed as dist

        from llama3 import Llama3Model
        from common.model import load_model

        model, _, _, _, input_ids = load_model(Llama3Model, args)

        if args.benchmark:
            dist.barrier()
            result = benchmark_generation(
                model,
                input_ids,
                num_warmup=args.benchmark_warmup,
                num_runs=args.benchmark_runs,
            )
            if dist.get_rank() == 0:
                save_benchmark_report(result, args.benchmark_output)

        elif args.validate:
            token_ids = []
            for i, token_id in enumerate(model.generate(input_ids)):
                if i >= args.max_new_tokens:
                    break
                token_ids.append(token_id.clone())
            sub_ids = torch.stack(token_ids)

            passed = validate_token_ids(args.baseline_path, sub_ids)
            if not passed:
                sys.exit(1)
