# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for kernel agent."""

import argparse
import json
import sys
from typing import List, Optional

from nkipy.tools.kernel_agent.generator import DEFAULT_MODEL_ID, DEFAULT_REGION


def _add_bedrock_args(parser: argparse.ArgumentParser) -> None:
    """Add common Bedrock model and execution arguments to a subparser."""
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "-r",
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nkipy.tools.kernel_agent",
        description="NumPy kernel testing and LLM generation",
    )
    sub = parser.add_subparsers(dest="cmd")

    # discover command
    disc = sub.add_parser("discover", help="Discover op support status")
    disc.add_argument("--ops", help="Comma-separated ops (default: all)")
    disc.add_argument("--dtypes", default="float32", help="Comma-separated dtypes")
    disc.add_argument("--no-hardware", action="store_true", help="Skip hardware tests")
    disc.add_argument("-o", "--output", help="Output JSON file")

    # test command
    test = sub.add_parser("test", help="Test specific operations")
    test.add_argument("--ops", required=True, help="Comma-separated ops to test")
    test.add_argument("--dtypes", default="float32", help="Comma-separated dtypes")
    test.add_argument("--no-hardware", action="store_true")

    # generate command
    gen = sub.add_parser("generate", help="Generate kernel with LLM")
    gen.add_argument("-p", "--prompt", required=True, help="Kernel description")
    gen.add_argument("--no-hardware", action="store_true", help="Skip hardware tests")
    _add_bedrock_args(gen)

    # list-ops command
    sub.add_parser("list-ops", help="List all target operations")

    # sweep command
    sw = sub.add_parser("sweep", help="Continuous kernel generation sweep")
    sw.add_argument("--no-hardware", action="store_true", help="Skip hardware tests")
    _add_bedrock_args(sw)
    sw.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N iterations (default: unlimited)",
    )
    sw.add_argument("--output-dir", default="sweep_results", help="Output directory")
    sw.add_argument("--delay", type=float, default=0, help="Seconds between iterations")
    sw.add_argument(
        "--summary-interval",
        type=int,
        default=10,
        help="Print summary every N iterations",
    )
    sw.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-iteration timeout in seconds (default: 120)",
    )

    # rerun command
    rr = sub.add_parser("rerun", help="Rerun kernels from a previous sweep JSONL")
    rr.add_argument("source", help="Path to source JSONL file from a previous sweep")
    rr.add_argument("--no-hardware", action="store_true", help="Skip hardware tests")
    rr.add_argument("--output-dir", default="sweep_results", help="Output directory")
    rr.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-iteration timeout in seconds (default: 120)",
    )
    rr.add_argument(
        "--summary-interval",
        type=int,
        default=10,
        help="Print summary every N iterations",
    )

    args = parser.parse_args(argv)

    if args.cmd == "discover":
        return cmd_discover(args)
    elif args.cmd == "test":
        return cmd_test(args)
    elif args.cmd == "generate":
        return cmd_generate(args)
    elif args.cmd == "list-ops":
        return cmd_list_ops()
    elif args.cmd == "sweep":
        return cmd_sweep(args)
    elif args.cmd == "rerun":
        return cmd_rerun(args)
    else:
        parser.print_help()
        return 0


def cmd_discover(args) -> int:
    from nkipy.tools.kernel_agent.ops import discover_ops, print_results

    ops = args.ops.split(",") if args.ops else None
    dtypes = args.dtypes.split(",")

    print(f"Discovering {len(ops) if ops else 'all'} ops with dtypes: {dtypes}")
    results = discover_ops(ops=ops, dtypes=dtypes, run_hardware=not args.no_hardware)

    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return 0 if results["summary"]["fail"] == 0 else 1


def cmd_test(args) -> int:
    from nkipy.tools.kernel_agent.ops import test_op

    ops = args.ops.split(",")
    dtypes = args.dtypes.split(",")

    print(f"Testing ops: {ops}")
    passed = 0
    failed = 0

    for op in ops:
        for dtype in dtypes:
            result = test_op(op, dtype, run_hardware=not args.no_hardware)
            status = "✓" if result.passed else "✗"
            print(f"  {op:20} {dtype:10} {status} {result.summary}")
            if result.passed:
                passed += 1
            else:
                failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_generate(args) -> int:
    from nkipy.tools.kernel_agent.executor import compare_outputs, run_kernel
    from nkipy.tools.kernel_agent.generator import compile_code, generate_kernel

    print(f"Generating kernel: {args.prompt}")
    name, code, inputs = generate_kernel(
        args.prompt, args.model, args.region, constrained=True
    )

    print(f"\nGenerated: {name}")
    print(f"Code:\n{code}\n")

    kernel_fn = compile_code(code)
    if kernel_fn is None:
        print("ERROR: Failed to compile generated code")
        return 1

    print("Running through pipeline...")
    result = run_kernel(kernel_fn, inputs, run_hardware=not args.no_hardware)

    stages = ["numpy", "compile", "hardware"]
    for stage in stages:
        s = getattr(result, stage)
        if s:
            status = "✓" if s.success else "✗"
            print(f"  {stage:12} {status}")
            if not s.success and s.error:
                print(f"    Error: {s.error[:80]}...")

    if result.passed:
        comp = compare_outputs(result)
        print(f"\nNumerical: max_diff={comp.get('max_diff', 'N/A')}")
        print("SUCCESS")
        return 0
    else:
        print("FAILED")
        return 1


def cmd_list_ops() -> int:
    from nkipy.tools.kernel_agent.ops import TARGET_OPS

    print("Target Operations:")
    for cat, ops in TARGET_OPS.items():
        print(f"\n{cat.upper()}:")
        for op in ops:
            print(f"  - {op}")
    return 0


def cmd_sweep(args) -> int:
    from nkipy.tools.kernel_agent import sweep

    sweep.run_sweep(
        model_id=args.model,
        region=args.region,
        run_hardware=not args.no_hardware,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        delay=args.delay,
        summary_interval=args.summary_interval,
        timeout=args.timeout,
    )
    return 0


def cmd_rerun(args) -> int:
    from nkipy.tools.kernel_agent import sweep

    sweep.run_rerun(
        source_path=args.source,
        run_hardware=not args.no_hardware,
        output_dir=args.output_dir,
        timeout=args.timeout,
        summary_interval=args.summary_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
