"""Benchmarking and accuracy validation for NKIPy Qwen3.5-35B-A3B."""

import json
import os
import time

import torch


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _percentile(data, pct):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _run_once(model, input_ids):
    token_times = []
    start = time.perf_counter()
    for i, _token_id in enumerate(model.generate(input_ids)):
        now = time.perf_counter()
        if i == 0:
            ttft_ms = (now - start) * 1000.0
            prev = now
        else:
            token_times.append((now - prev) * 1000.0)
            prev = now

    end = time.perf_counter()
    return {
        "ttft_ms": ttft_ms,
        "decode_latencies_ms": token_times,
        "num_tokens": len(token_times) + 1,
        "total_time_ms": (end - start) * 1000.0,
    }


def benchmark_generation(model, input_ids, num_warmup=2, num_runs=5):
    total_runs = num_warmup + num_runs
    run_reports = []

    for run_idx in range(total_runs):
        is_warmup = run_idx < num_warmup
        label = (
            f"warmup {run_idx + 1}/{num_warmup}"
            if is_warmup
            else f"run {run_idx - num_warmup + 1}/{num_runs}"
        )
        print(f"[benchmark] {label}...")

        report = _run_once(model, input_ids)

        if not is_warmup:
            run_reports.append(report)
            throughput = (
                report["num_tokens"] / (report["total_time_ms"] / 1000.0)
                if report["total_time_ms"] > 0
                else 0.0
            )
            print(
                f"  TTFT={report['ttft_ms']:.1f}ms  "
                f"tokens={report['num_tokens']}  "
                f"throughput={throughput:.1f} tok/s"
            )

    n = len(run_reports)
    if n == 0:
        return {}

    avg_ttft = sum(r["ttft_ms"] for r in run_reports) / n
    avg_total = sum(r["total_time_ms"] for r in run_reports) / n
    num_tokens = run_reports[0]["num_tokens"]

    all_decode = []
    for r in run_reports:
        all_decode.extend(r["decode_latencies_ms"])

    throughput = num_tokens / (avg_total / 1000.0) if avg_total > 0 else 0.0

    result = {
        "ttft_ms": round(avg_ttft, 3),
        "decode_latency_p50_ms": round(_percentile(all_decode, 50), 3),
        "decode_latency_p90_ms": round(_percentile(all_decode, 90), 3),
        "decode_latency_p99_ms": round(_percentile(all_decode, 99), 3),
        "num_tokens": num_tokens,
        "total_time_ms": round(avg_total, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
    }

    print("\n=== Benchmark Results ===")
    print(f"  TTFT:                  {result['ttft_ms']:.1f} ms")
    print(f"  Decode latency (p50):  {result['decode_latency_p50_ms']:.1f} ms")
    print(f"  Decode latency (p90):  {result['decode_latency_p90_ms']:.1f} ms")
    print(f"  Decode latency (p99):  {result['decode_latency_p99_ms']:.1f} ms")
    print(f"  Tokens generated:      {result['num_tokens']}")
    print(
        f"  Throughput:            {result['throughput_tokens_per_sec']:.1f} tokens/sec"
    )
    print("=========================\n")

    return result


def save_benchmark_report(result, path="benchmark_report.json"):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Report saved to {path}")


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate NKIPy Qwen3.5-35B-A3B: benchmark"
    )

    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="./qwen3_5_shards")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")

    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument(
        "--benchmark-output", type=str, default="benchmark_report.json"
    )

    args = parser.parse_args()

    import torch.distributed as dist

    from qwen3_5 import load_model

    model, input_ids, _ = load_model(args)

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
