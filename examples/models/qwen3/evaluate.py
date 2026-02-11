"""Benchmarking and accuracy validation for NKIPy Qwen3-30B-A3B."""

import json
import os
import time

import torch

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _percentile(data, pct):
    """Compute the pct-th percentile via linear interpolation."""
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
    """Run a single generation pass and collect timing.

    Returns a dict with ttft_ms, decode_latencies_ms, num_tokens, total_time_ms.
    """
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
    """Benchmark a Qwen3Model's generate() method.

    Args:
        model: A Qwen3Model instance with a generate(input_ids) generator.
        input_ids: Tokenized input (numpy array).
        num_warmup: Number of warmup runs (not measured).
        num_runs: Number of measured runs.

    Returns:
        Dict with aggregated benchmark results. Also prints per-run and
        summary statistics, and saves a JSON report.
    """
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

    # Aggregate across measured runs
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

    # Print summary
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
    """Save benchmark result dict as JSON."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Report saved to {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_token_ids(reference_path, submission_ids):
    """Compare generated token IDs against a saved baseline.

    Args:
        reference_path: Path to the baseline .pt file.
        submission_ids: Tensor of generated token IDs from the submission model,
            shape (num_tokens, batch_size, 1).

    Returns:
        True if all tokens match, False otherwise. Prints pass/fail summary.
    """
    ref_ids = load_baseline(reference_path)
    num_tokens = min(len(ref_ids), len(submission_ids))

    ref = ref_ids[:num_tokens].reshape(num_tokens, -1)
    sub = submission_ids[:num_tokens].reshape(num_tokens, -1)

    matches = (ref == sub).float()
    num_mismatched = int((matches == 0).sum().item())
    top1_acc = matches.mean().item()
    passed = num_mismatched == 0

    status = "PASS" if passed else "FAIL"
    print(f"\n=== Accuracy Validation: {status} ===")
    print(f"  Tokens checked:        {num_tokens}")
    print(f"  Token mismatches:      {num_mismatched}")
    print(f"  Top-1 accuracy:        {top1_acc:.4f}")
    if passed:
        print(f"  All {num_tokens} tokens match.")
    else:
        print(f"  {num_mismatched}/{num_tokens} tokens differ.")
    print("====================================\n")

    return passed


# ---------------------------------------------------------------------------
# HF Baseline Generation
# ---------------------------------------------------------------------------


def generate_hf_baseline(
    model_name, prompt, max_new_tokens, torch_dtype=torch.bfloat16
):
    """Generate reference token IDs using the HuggingFace transformers model.

    Loads the model on CPU, runs greedy generation, and returns the generated
    token IDs as a tensor of shape (num_tokens, 1, 1).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[validate] Loading HF model: {model_name} (dtype={torch_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"[validate] Generating {max_new_tokens} tokens with HF model (greedy)...")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[:, input_ids.shape[1] :]
    token_ids = generated_ids.T.unsqueeze(-1)  # (num_tokens, batch, 1)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[validate] HF generated: {generated_text[:200]}...")
    print(f"[validate] HF baseline shape: {token_ids.shape}")

    del model
    return token_ids


# ---------------------------------------------------------------------------
# Baseline save / load
# ---------------------------------------------------------------------------


def save_baseline(tensor, path):
    """Save a tensor to a .pt file."""
    torch.save(tensor, path)
    print(f"[validate] Saved baseline to {path}")


def load_baseline(path):
    """Load a tensor from a .pt file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")
    tensor = torch.load(path, weights_only=True)
    print(f"[validate] Loaded baseline from {path} (shape={tensor.shape})")
    return tensor


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate NKIPy Qwen3-30B-A3B: benchmark, validate, or generate baseline"
    )

    # Common model args (shared with qwen3.py)
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="/kaena/qwen3_shards_30B_A3B_TP8")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")

    # Mode selection (exactly one required)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking (TTFT, throughput, latency percentiles)",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run accuracy validation against saved baseline",
    )
    mode.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate HF baseline token IDs (no torchrun needed)",
    )

    # Benchmark options
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=2,
        help="Number of warmup runs for benchmark",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Number of measured runs for benchmark",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark_report.json",
        help="Path to save benchmark JSON report",
    )

    # Validate options
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="baseline_tokens.pt",
        help="Path for baseline token IDs file",
    )

    # Generate-baseline options
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_tokens.pt",
        help="Output path for baseline .pt file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for baseline generation",
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

        from qwen3 import load_model

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
