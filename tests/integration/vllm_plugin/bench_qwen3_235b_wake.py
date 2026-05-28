#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: Qwen3-235B-A22B cross-instance P2P wake/sleep latency.

Measures end-to-end wake-up and sleep latency for Qwen3-235B-A22B with TP=32
across two trn2.48xlarge instances.

Uses NIXL LIBFABRIC backend for direct device-to-device RDMA.

Topology:
  LOCAL  (sender):   1 vLLM engine, TP=32, cores 0-31, holds checkpoint
  REMOTE (receiver): 1 vLLM engine, TP=32, cores 0-31, no checkpoint

Reports full latency breakdown including:
  - Gloo init, NRT init, tensor allocation
  - P2P transfer (RDMA or DMA+RDMA)
  - Kernel loading

Usage:
    python tests/integration/vllm_plugin/bench_qwen3_235b_wake.py --iterations 3

Environment variables:
    BENCH_REMOTE_HOST   Remote instance IP (default: 10.3.215.3)
    BENCH_LOCAL_IP      Local instance IP (default: 10.3.211.148)
    BENCH_TP            Tensor parallelism (default: 32)
    BENCH_MODEL         HF model name (default: Qwen/Qwen3-235B-A22B)
    BENCH_CHECKPOINT    Checkpoint path (default: /fsx/zhuangw/models/qwen3_235b_a22b_TP32)
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

os.environ["VLLM_PLUGINS"] = "nkipy"

import requests

# --- Configuration ---
_LOCAL_IP = os.environ.get("BENCH_LOCAL_IP", "10.3.211.148")
_REMOTE_HOST = os.environ.get("BENCH_REMOTE_HOST", "10.3.215.3")
_MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-235B-A22B")
_CHECKPOINT = os.environ.get("BENCH_CHECKPOINT", "/fsx/zhuangw/models/qwen3_235b_a22b_TP32")
_TP = int(os.environ.get("BENCH_TP", "32"))
_VENV_PYTHON = "/fsx/zhuangw/nkipy/.venv/bin/python"
_SENDER_PORT = 8100
_RECEIVER_PORT = 9100


def _wait_for_server(host, port, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/nkipy/health", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    return False


def _backend_env_vars():
    """Return env vars for NIXL backend."""
    return {"NEURON_RT_MAP_HBM": "1"}


def _start_local_sender():
    """Start sender engine on local instance."""
    env = os.environ.copy()
    venv_bin = os.path.dirname(_VENV_PYTHON)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env.update({
        "VLLM_PLUGINS": "nkipy",
        "VLLM_USE_V1": "1",
        "NKIPY_CORE_OFFSET": "0",
        "NKIPY_CHECKPOINT": _CHECKPOINT,
        "OMP_NUM_THREADS": "1",
        "VLLM_RPC_TIMEOUT": "600000",
        "HF_HUB_OFFLINE": "1",
    })
    env.update(_backend_env_vars())

    cmd = [
        _VENV_PYTHON, "-m", "nkipy.vllm_plugin.server",
        "--model", _MODEL,
        "--tensor-parallel-size", str(_TP),
        "--max-model-len", "128",
        "--max-num-seqs", "1",
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--port", str(_SENDER_PORT),
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix="bench_qwen3_sender_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _start_remote_receiver():
    """Start receiver engine on remote instance via SSH."""
    venv_bin = os.path.dirname(_VENV_PYTHON)
    backend_env = " ".join(f"{k}={v}" for k, v in _backend_env_vars().items())
    env_vars = (
        f"PATH={venv_bin}:$PATH "
        f"VLLM_PLUGINS=nkipy "
        f"VLLM_USE_V1=1 "
        f"NKIPY_CORE_OFFSET=0 "
        f"OMP_NUM_THREADS=1 "
        f"VLLM_RPC_TIMEOUT=600000 "
        f"HF_HUB_OFFLINE=1"
    )
    if backend_env:
        env_vars = f"{env_vars} {backend_env}"

    server_cmd = (
        f"{_VENV_PYTHON} -m nkipy.vllm_plugin.server "
        f"--model {_MODEL} "
        f"--tensor-parallel-size {_TP} "
        f"--max-model-len 128 "
        f"--max-num-seqs 1 "
        f"--enforce-eager "
        f"--dtype bfloat16 "
        f"--port {_RECEIVER_PORT}"
    )

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        _REMOTE_HOST,
        f"cd /fsx/zhuangw/nkipy && {env_vars} {server_cmd}",
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix="bench_qwen3_recv_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(ssh_cmd, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _kill(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _kill_remote():
    """Kill any lingering server processes on remote."""
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", _REMOTE_HOST,
         "pkill -f 'nkipy.vllm_plugin.server' 2>/dev/null || true"],
        capture_output=True,
    )


def _wake(host, port, peer_url):
    t0 = time.time()
    resp = requests.post(
        f"http://{host}:{port}/nkipy/wake_up",
        json={"peer_url": peer_url},
        timeout=600,
    )
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"  ERROR: Wake failed ({resp.status_code}): {resp.text[:500]}")
        return None
    result = resp.json()
    result["_client_elapsed_s"] = elapsed
    return result


def _sleep_engine(host, port):
    resp = requests.post(f"http://{host}:{port}/nkipy/sleep", timeout=120)
    if resp.status_code != 200:
        print(f"  WARNING: Sleep failed ({resp.status_code}): {resp.text[:200]}")
        return None
    return resp.json()


def _infer(host, port):
    resp = requests.post(
        f"http://{host}:{port}/v1/completions",
        json={"model": _MODEL, "prompt": "The capital of France is",
              "max_tokens": 20, "temperature": 0},
        timeout=180,
    )
    if resp.status_code != 200:
        print(f"  WARNING: Inference failed ({resp.status_code}): {resp.text[:200]}")
        return None
    return resp.json()["choices"][0]["text"]


def _fmt_breakdown(breakdown):
    """Format latency breakdown dict."""
    if not breakdown:
        return "(no breakdown)"
    parts = []
    for key in ("gloo_init_s", "nrt_init_s", "nrt_barrier_s", "alloc_tensors_s",
                "collect_bufs_s", "p2p_transfer_s", "rdma_ack_s",
                "kernel_load_s", "kernel_barrier_s", "tok_embedding_s",
                "prereg_mrs_s", "total_s"):
        if key in breakdown:
            label = key.replace("_s", "").replace("_", " ")
            parts.append(f"{label}={breakdown[key]:.2f}s")
    return "\n    ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Cross-instance wake/sleep benchmark (NIXL)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of wake/sleep cycles to measure")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference correctness check (faster)")
    parser.add_argument("--reuse-engines", action="store_true",
                        help="Don't start/stop engines (assume already running)")
    args = parser.parse_args()

    if not args.reuse_engines and not os.path.isdir(_CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {_CHECKPOINT}")
        sys.exit(1)

    procs = []
    try:
        if not args.reuse_engines:
            _kill_remote()

            # --- Start sender ---
            print(f"Starting sender on local ({_LOCAL_IP}:{_SENDER_PORT}), TP={_TP}...")
            sender_proc = _start_local_sender()
            procs.append(sender_proc)

            # --- Start receiver ---
            print(f"Starting receiver on remote ({_REMOTE_HOST}:{_RECEIVER_PORT}), TP={_TP}...")
            receiver_proc = _start_remote_receiver()
            procs.append(receiver_proc)

            # --- Wait for both ---
            print("  Waiting for sender to be ready (may take 5-10 min for Qwen3-235B)...")
            if not _wait_for_server("localhost", _SENDER_PORT):
                print(f"  ERROR: Sender failed to start. Log: {sender_proc._log_path}")
                sys.exit(1)
            print(f"  Sender ready.")

            print("  Waiting for receiver to be ready...")
            if not _wait_for_server(_REMOTE_HOST, _RECEIVER_PORT):
                print(f"  ERROR: Receiver failed to start. Log: {receiver_proc._log_path}")
                sys.exit(1)
            print(f"  Receiver ready (sleeping).\n")
        else:
            print(f"Reusing existing engines: sender={_LOCAL_IP}:{_SENDER_PORT}, "
                  f"receiver={_REMOTE_HOST}:{_RECEIVER_PORT}\n")

        sender_url = f"http://{_LOCAL_IP}:{_SENDER_PORT}"

        # --- Verify sender inference ---
        if not args.skip_inference:
            print("Verifying sender inference...")
            ref_output = _infer("localhost", _SENDER_PORT)
            if ref_output:
                print(f"  Sender output: {ref_output[:60]!r}...\n")
            else:
                print("  WARNING: Sender inference failed, skipping correctness checks\n")
                ref_output = None
        else:
            ref_output = None

        # === Benchmark: Wake/Sleep cycles ===
        print("=" * 70)
        print(f"BENCHMARK: Qwen3-235B-A22B cross-instance wake/sleep (TP={_TP})")
        print(f"  Backend:  nixl")
        print(f"  Sender:   {_LOCAL_IP}:{_SENDER_PORT}")
        print(f"  Receiver: {_REMOTE_HOST}:{_RECEIVER_PORT}")
        print(f"  Per-rank shard: ~{447*1024/32:.0f} MB ({447*1024*8/32/1000:.1f} Gbps at 1s)")
        print(f"  Iterations: {args.iterations}")
        print("=" * 70)

        results = []
        sleep_results = []
        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1}/{args.iterations} ---")

            result = _wake(_REMOTE_HOST, _RECEIVER_PORT, sender_url)
            if result is None:
                print("  FAILED — skipping iteration")
                continue

            latency = result["_client_elapsed_s"]
            breakdown = result.get("latency", {})
            print(f"  Wake latency: {latency:.2f}s")
            print(f"  Breakdown:\n    {_fmt_breakdown(breakdown)}")

            # Verify correctness
            if ref_output is not None and not args.skip_inference:
                output = _infer(_REMOTE_HOST, _RECEIVER_PORT)
                correct = output == ref_output if output else False
                print(f"  Inference correct: {correct}")
                if not correct and output:
                    print(f"    Got: {output[:60]!r}")
                    print(f"    Exp: {ref_output[:60]!r}")

            results.append({"latency": latency, "breakdown": breakdown})

            # Sleep and measure latency
            print("  Sleeping receiver...")
            t_sleep_start = time.time()
            sleep_result = _sleep_engine(_REMOTE_HOST, _RECEIVER_PORT)
            t_sleep_end = time.time()
            if sleep_result:
                sleep_lat = sleep_result.get("latency", {})
                sleep_client_s = t_sleep_end - t_sleep_start
                print(f"  Sleep latency: {sleep_client_s:.2f}s "
                      f"(server: {sleep_lat.get('total_s', '?')}s)")
                sleep_results.append({
                    "client_s": sleep_client_s,
                    "breakdown": sleep_lat,
                })
            time.sleep(3)

        # === Summary ===
        if results:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            latencies = [r["latency"] for r in results]
            print(f"  Model: {_MODEL} (TP={_TP})")
            print(f"  Backend: NIXL direct device RDMA")
            print(f"  Per-rank shard: ~{14*1024:.0f} MB")
            print(f"  Iterations: {len(results)}")
            print(f"\n  WAKE LATENCY:")
            print(f"    min={min(latencies):.2f}s  "
                  f"avg={sum(latencies)/len(latencies):.2f}s  "
                  f"max={max(latencies):.2f}s")

            # Average breakdown
            all_keys = ("gloo_init_s", "nrt_init_s", "nrt_barrier_s",
                        "alloc_tensors_s", "collect_bufs_s",
                        "p2p_transfer_s", "rdma_ack_s",
                        "kernel_load_s", "kernel_barrier_s",
                        "tok_embedding_s", "prereg_mrs_s", "total_s")
            print(f"\n  Average wake breakdown ({len(results)} runs):")
            for key in all_keys:
                vals = [r["breakdown"].get(key) for r in results if key in r["breakdown"]]
                if vals:
                    label = key.replace("_s", "").replace("_", " ")
                    avg = sum(vals) / len(vals)
                    print(f"    {label:20s}: {avg:.2f}s")

            # P2P transfer is the key metric
            p2p_vals = [r["breakdown"].get("p2p_transfer_s") for r in results
                        if "p2p_transfer_s" in r["breakdown"]]
            if p2p_vals:
                avg_p2p = sum(p2p_vals) / len(p2p_vals)
                total_bytes = 14.9 * 1024 * 1024 * 1024  # ~14.9 GB per rank
                throughput = (total_bytes * 8) / avg_p2p / 1e9
                print(f"\n  P2P transfer: avg={avg_p2p:.2f}s "
                      f"(~{throughput:.1f} Gbps effective per rank)")

            # Sleep latency
            if sleep_results:
                sleep_lats = [s["client_s"] for s in sleep_results]
                print(f"\n  SLEEP LATENCY:")
                print(f"    min={min(sleep_lats):.2f}s  "
                      f"avg={sum(sleep_lats)/len(sleep_lats):.2f}s  "
                      f"max={max(sleep_lats):.2f}s")
                # Sleep breakdown if available
                sleep_keys = ("dereg_mrs_s", "nrt_close_s", "destroy_gloo_s", "total_s")
                has_breakdown = any(s["breakdown"] for s in sleep_results)
                if has_breakdown:
                    print(f"\n  Average sleep breakdown ({len(sleep_results)} runs):")
                    for key in sleep_keys:
                        vals = [s["breakdown"].get(key) for s in sleep_results
                                if key in s.get("breakdown", {})]
                        if vals:
                            label = key.replace("_s", "").replace("_", " ")
                            avg = sum(vals) / len(vals)
                            print(f"    {label:20s}: {avg:.2f}s")
        else:
            print("\nNo successful iterations.")

    finally:
        if not args.reuse_engines:
            for proc in procs:
                _kill(proc)
            _kill_remote()


if __name__ == "__main__":
    main()
