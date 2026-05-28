#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: P2P wake (1-to-1) vs broadcast wake (1-to-N).

Measures wake-up latency for:
  - 1 receiver (baseline P2P)
  - N receivers concurrently (broadcast)

Reports per-receiver and total latencies to validate scalability.

Usage:
    python tests/integration/vllm_plugin/bench_broadcast_wake.py
"""

import concurrent.futures
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

os.environ["VLLM_PLUGINS"] = "nkipy"

import requests

_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_CHECKPOINT = "/fsx/zhuangw/models/tinyllama_TP2"
_TP = 2
_PROMPT = "The capital of France is"
_N_RECEIVERS = 5
_SWITCH_DELAY = 8


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/nkipy/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
    return False


def _start_server(port, checkpoint=None, core_offset=0):
    env = os.environ.copy()
    env.update({
        "VLLM_PLUGINS": "nkipy",
        "VLLM_USE_V1": "1",
        "NKIPY_CORE_OFFSET": str(core_offset),
        "OMP_NUM_THREADS": "1",
        "VLLM_RPC_TIMEOUT": "600000",
        "HF_HUB_OFFLINE": "1",
    })
    if checkpoint:
        env["NKIPY_CHECKPOINT"] = checkpoint

    cmd = [
        sys.executable, "-m", "nkipy.vllm_plugin.server",
        "--model", _MODEL,
        "--tensor-parallel-size", str(_TP),
        "--max-model-len", "128",
        "--max-num-seqs", "1",
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--port", str(port),
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"bench_engine_{port}_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _kill(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _wake(port, peer_url):
    t0 = time.time()
    resp = requests.post(
        f"http://localhost:{port}/nkipy/wake_up",
        json={"peer_url": peer_url},
        timeout=300,
    )
    elapsed = time.time() - t0
    assert resp.status_code == 200, f"Wake failed on port {port}: {resp.text}"
    result = resp.json()
    result["_client_elapsed_s"] = elapsed
    return result


def _sleep(port):
    resp = requests.post(f"http://localhost:{port}/nkipy/sleep", timeout=60)
    assert resp.status_code == 200, f"Sleep failed: {resp.text}"
    return resp.json()


def _infer(port):
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": _MODEL, "prompt": _PROMPT,
              "max_tokens": 20, "temperature": 0},
        timeout=120,
    )
    assert resp.status_code == 200, f"Infer failed: {resp.text}"
    return resp.json()["choices"][0]["text"]


def main():
    if not os.path.isdir(_CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {_CHECKPOINT}")
        print("Run: python scripts/shard_checkpoint.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
              "--world-size 2 --head-dim 64 --shard-embed --output-dir /fsx/zhuangw/models/tinyllama_TP2")
        sys.exit(1)

    procs = []
    try:
        # --- Start sender ---
        sender_port = _free_port()
        print(f"Starting sender on port {sender_port} (cores 0-1)...")
        sender_proc = _start_server(sender_port, checkpoint=_CHECKPOINT, core_offset=0)
        procs.append(sender_proc)
        assert _wait_for_server(sender_port), "Sender failed to start"
        sender_url = f"http://localhost:{sender_port}"
        print(f"  Sender ready.")

        # --- Start receivers ---
        receivers = []
        for i in range(_N_RECEIVERS):
            port = _free_port()
            core_offset = 2 + i * _TP
            print(f"Starting receiver {i} on port {port} (cores {core_offset}-{core_offset+_TP-1})...")
            proc = _start_server(port, checkpoint=None, core_offset=core_offset)
            procs.append(proc)
            receivers.append((port, proc, i))

        for port, proc, i in receivers:
            assert _wait_for_server(port), f"Receiver {i} failed to start"
        print(f"  All {_N_RECEIVERS} receivers ready (sleeping).\n")

        # --- Verify sender inference ---
        ref_output = _infer(sender_port)
        print(f"Sender reference output: {ref_output[:50]!r}...\n")

        # === Benchmark 1: Sequential P2P (1-to-1) ===
        print("=" * 60)
        print("BENCHMARK 1: Sequential P2P wake (1-to-1)")
        print("=" * 60)
        sequential_latencies = []
        for port, _, i in receivers:
            result = _wake(port, sender_url)
            latency = result.get("_client_elapsed_s", 0)
            sequential_latencies.append(latency)
            output = _infer(port)
            correct = output == ref_output
            print(f"  Receiver {i}: wake={latency:.2f}s  correct={correct}")
            _sleep(port)
            time.sleep(_SWITCH_DELAY)

        avg_seq = sum(sequential_latencies) / len(sequential_latencies)
        total_seq = sum(sequential_latencies) + (_N_RECEIVERS - 1) * _SWITCH_DELAY
        print(f"\n  Avg wake latency: {avg_seq:.2f}s")
        print(f"  Total sequential time: {total_seq:.1f}s (with {_SWITCH_DELAY}s delays)\n")

        # === Benchmark 2: Concurrent broadcast wake (1-to-N) ===
        print("=" * 60)
        print(f"BENCHMARK 2: Concurrent broadcast wake (1-to-{_N_RECEIVERS})")
        print("=" * 60)

        t_broadcast_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_RECEIVERS) as pool:
            futures = {
                pool.submit(_wake, port, sender_url): (port, i)
                for port, _, i in receivers
            }
            broadcast_results = {}
            for future in concurrent.futures.as_completed(futures):
                port, i = futures[future]
                broadcast_results[i] = future.result()

        t_broadcast_end = time.time()
        broadcast_wall = t_broadcast_end - t_broadcast_start

        for i in sorted(broadcast_results):
            result = broadcast_results[i]
            latency = result.get("_client_elapsed_s", 0)
            port = receivers[i][0]
            output = _infer(port)
            correct = output == ref_output
            print(f"  Receiver {i}: wake={latency:.2f}s  correct={correct}")

        broadcast_latencies = [broadcast_results[i]["_client_elapsed_s"] for i in sorted(broadcast_results)]
        avg_bcast = sum(broadcast_latencies) / len(broadcast_latencies)
        max_bcast = max(broadcast_latencies)
        min_bcast = min(broadcast_latencies)

        print(f"\n  Per-receiver latency: min={min_bcast:.2f}s  avg={avg_bcast:.2f}s  max={max_bcast:.2f}s")
        print(f"  Wall-clock time (all {_N_RECEIVERS} ready): {broadcast_wall:.2f}s")
        print(f"  Speedup vs sequential: {total_seq / broadcast_wall:.1f}x\n")

        # === Summary ===
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Model: {_MODEL} (TP={_TP})")
        print(f"  Receivers: {_N_RECEIVERS}")
        print(f"  Sequential P2P:  avg={avg_seq:.2f}s per wake, total={total_seq:.1f}s")
        print(f"  Broadcast:       avg={avg_bcast:.2f}s per wake, wall={broadcast_wall:.2f}s")
        print(f"  Speedup: {total_seq / broadcast_wall:.1f}x")
        print(f"  Overhead vs 1-to-1: {(avg_bcast / avg_seq - 1) * 100:.0f}% per receiver")

        # Cleanup
        for port, _, _ in receivers:
            _sleep(port)

    finally:
        for proc in procs:
            _kill(proc)


if __name__ == "__main__":
    main()
