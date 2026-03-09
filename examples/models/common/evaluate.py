"""Benchmarking and accuracy validation for NKIPy models."""

import json
import os
import time

import torch


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
    """Run a single generation pass and collect timing."""
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
    """Benchmark a model's generate() method."""
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


def validate_token_ids(reference_path, submission_ids):
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


def generate_hf_baseline(
    model_name, prompt, max_new_tokens, torch_dtype=torch.bfloat16
):
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
    token_ids = generated_ids.T.unsqueeze(-1)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[validate] HF generated: {generated_text[:200]}...")
    print(f"[validate] HF baseline shape: {token_ids.shape}")

    del model
    return token_ids


def save_baseline(tensor, path):
    torch.save(tensor, path)
    print(f"[validate] Saved baseline to {path}")


def load_baseline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")
    tensor = torch.load(path, weights_only=True)
    print(f"[validate] Loaded baseline from {path} (shape={tensor.shape})")
    return tensor
