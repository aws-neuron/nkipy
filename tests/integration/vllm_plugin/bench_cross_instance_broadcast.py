#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-instance broadcast wake benchmark.

Measures real EFA RDMA P2P wake latency with:
  - Sender on LOCAL instance (10.3.211.148)
  - N receivers on REMOTE instance (10.3.215.3)

This eliminates same-host contention (PCIe, NRT, CPU) and exercises true
cross-instance EFA bandwidth sharing at broadcast degree N=1..5.

Topology:
  LOCAL  (sender):  1 vLLM engine, TP=2, cores 0-1, holds checkpoint
  REMOTE (receivers): N vLLM engines, TP=2, cores 0-1..2N-2..2N-1, no checkpoint

Usage:
    python tests/integration/vllm_plugin/bench_cross_instance_broadcast.py

Environment variables:
    BENCH_REMOTE_HOST   Remote instance IP (default: 10.3.215.3)
    BENCH_N_RECEIVERS   Max broadcast degree (default: 5)
    BENCH_TP            Tensor parallelism (default: 2)
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

# --- Configuration ---
_LOCAL_IP = "10.3.211.148"
_REMOTE_HOST = os.environ.get("BENCH_REMOTE_HOST", "10.3.215.3")
_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_CHECKPOINT = "/fsx/zhuangw/models/tinyllama_TP2"
_TP = int(os.environ.get("BENCH_TP", "2"))
_N_RECEIVERS = int(os.environ.get("BENCH_N_RECEIVERS", "5"))
_PROMPT = "The capital of France is"
_SWITCH_DELAY = 5
_VENV_PYTHON = "/fsx/zhuangw/nkipy/.venv/bin/python"


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(host, port, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/nkipy/health", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(3)
    return False


def _start_local_sender(port):
    """Start sender engine on local instance."""
    env = os.environ.copy()
    env.update({
        "VLLM_PLUGINS": "nkipy",
        "VLLM_USE_V1": "1",
        "NKIPY_CORE_OFFSET": "0",
        "NKIPY_CHECKPOINT": _CHECKPOINT,
        "OMP_NUM_THREADS": "1",
        "VLLM_RPC_TIMEOUT": "600000",
        "HF_HUB_OFFLINE": "1",
    })

    cmd = [
        _VENV_PYTHON, "-m", "nkipy.vllm_plugin.server",
        "--model", _MODEL,
        "--tensor-parallel-size", str(_TP),
        "--max-model-len", "128",
        "--max-num-seqs", "1",
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--port", str(port),
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"bench_sender_{port}_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _start_remote_receiver(port, core_offset):
    """Start receiver engine on remote instance via SSH."""
    huge_pages = os.environ.get("NKIPY_HUGE_PAGES", "0")
    env_vars = (
        f"VLLM_PLUGINS=nkipy "
        f"VLLM_USE_V1=1 "
        f"NKIPY_CORE_OFFSET={core_offset} "
        f"NKIPY_HUGE_PAGES={huge_pages} "
        f"OMP_NUM_THREADS=1 "
        f"VLLM_RPC_TIMEOUT=600000 "
        f"HF_HUB_OFFLINE=1"
    )

    server_cmd = (
        f"{_VENV_PYTHON} -m nkipy.vllm_plugin.server "
        f"--model {_MODEL} "
        f"--tensor-parallel-size {_TP} "
        f"--max-model-len 128 "
        f"--max-num-seqs 1 "
        f"--enforce-eager "
        f"--dtype bfloat16 "
        f"--port {port}"
    )

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        _REMOTE_HOST,
        f"cd /fsx/zhuangw/nkipy && {env_vars} {server_cmd}",
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"bench_recv_{port}_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(ssh_cmd, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    proc._remote_port = port
    return proc


def _kill(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _kill_remote_receivers():
    """Kill any lingering receiver processes on remote."""
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
        timeout=300,
    )
    elapsed = time.time() - t0
    assert resp.status_code == 200, f"Wake failed on {host}:{port}: {resp.text}"
    result = resp.json()
    result["_client_elapsed_s"] = elapsed
    return result


def _sleep_engine(host, port):
    resp = requests.post(f"http://{host}:{port}/nkipy/sleep", timeout=60)
    assert resp.status_code == 200, f"Sleep failed on {host}:{port}: {resp.text}"
    return resp.json()


def _infer(host, port):
    resp = requests.post(
        f"http://{host}:{port}/v1/completions",
        json={"model": _MODEL, "prompt": _PROMPT,
              "max_tokens": 20, "temperature": 0},
        timeout=120,
    )
    assert resp.status_code == 200, f"Infer failed on {host}:{port}: {resp.text}"
    return resp.json()["choices"][0]["text"]


def _fmt_breakdown(breakdown):
    """Format latency breakdown dict as a compact string."""
    if not breakdown:
        return "(no breakdown)"
    parts = []
    for key in ("gloo_init_s", "nrt_init_s", "alloc_tensors_s",
                "p2p_transfer_s", "rdma_ack_s", "kernel_load_s", "total_s"):
        if key in breakdown:
            label = key.replace("_s", "").replace("_", " ")
            parts.append(f"{label}={breakdown[key]:.2f}s")
    return "  ".join(parts)


def _avg_breakdown(breakdowns):
    """Average a list of breakdown dicts."""
    if not breakdowns:
        return "(no data)"
    keys = ("gloo_init_s", "nrt_init_s", "alloc_tensors_s",
            "p2p_transfer_s", "rdma_ack_s", "kernel_load_s", "total_s")
    avgs = {}
    for key in keys:
        vals = [b[key] for b in breakdowns if key in b]
        if vals:
            avgs[key] = sum(vals) / len(vals)
    parts = []
    for key in keys:
        if key in avgs:
            label = key.replace("_s", "").replace("_", " ")
            parts.append(f"{label}={avgs[key]:.2f}s")
    return "  ".join(parts)


def main():
    if not os.path.isdir(_CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {_CHECKPOINT}")
        sys.exit(1)

    procs = []
    try:
        # Clean up any stale remote processes
        _kill_remote_receivers()

        # --- Start sender on local ---
        sender_port = _free_port()
        print(f"Starting sender on local ({_LOCAL_IP}:{sender_port}), TP={_TP}, cores 0-{_TP-1}...")
        sender_proc = _start_local_sender(sender_port)
        procs.append(sender_proc)
        assert _wait_for_server("localhost", sender_port), \
            f"Sender failed to start. Log: {sender_proc._log_path}"
        sender_url = f"http://{_LOCAL_IP}:{sender_port}"
        print(f"  Sender ready.\n")

        # Verify sender inference
        ref_output = _infer("localhost", sender_port)
        print(f"Sender reference output: {ref_output[:50]!r}...\n")

        # --- Start receivers on remote ---
        receivers = []  # (port, proc, index)
        for i in range(_N_RECEIVERS):
            port = 9100 + i  # fixed ports for remote (avoid ephemeral conflicts)
            core_offset = i * _TP
            print(f"Starting receiver {i} on remote ({_REMOTE_HOST}:{port}), cores {core_offset}-{core_offset+_TP-1}...")
            proc = _start_remote_receiver(port, core_offset)
            procs.append(proc)
            receivers.append((port, proc, i))

        print(f"  Waiting for all {_N_RECEIVERS} receivers to be ready...")
        for port, proc, i in receivers:
            if not _wait_for_server(_REMOTE_HOST, port):
                print(f"  ERROR: Receiver {i} ({_REMOTE_HOST}:{port}) failed to start.")
                print(f"         Log: {proc._log_path}")
                sys.exit(1)
        print(f"  All {_N_RECEIVERS} receivers ready (sleeping).\n")

        # === Benchmark 1: Sequential P2P (1-to-1) ===
        print("=" * 60)
        print("BENCHMARK 1: Sequential P2P wake (1-to-1, cross-instance)")
        print("=" * 60)
        sequential_latencies = []
        sequential_breakdowns = []
        for port, _, i in receivers:
            result = _wake(_REMOTE_HOST, port, sender_url)
            latency = result["_client_elapsed_s"]
            sequential_latencies.append(latency)
            breakdown = result.get("latency", {})
            sequential_breakdowns.append(breakdown)
            output = _infer(_REMOTE_HOST, port)
            correct = output == ref_output
            print(f"  Receiver {i}: wake={latency:.2f}s  correct={correct}")
            print(f"    Breakdown: {_fmt_breakdown(breakdown)}")
            _sleep_engine(_REMOTE_HOST, port)
            time.sleep(_SWITCH_DELAY)

        avg_seq = sum(sequential_latencies) / len(sequential_latencies)
        total_seq = sum(sequential_latencies) + (_N_RECEIVERS - 1) * _SWITCH_DELAY
        print(f"\n  Avg wake latency: {avg_seq:.2f}s")
        print(f"  Avg breakdown: {_avg_breakdown(sequential_breakdowns)}")
        print(f"  Total sequential time: {total_seq:.1f}s (with {_SWITCH_DELAY}s delays)\n")

        # === Benchmark 2: Concurrent broadcast at varying N ===
        print("=" * 60)
        print(f"BENCHMARK 2: Concurrent broadcast wake (cross-instance)")
        print("=" * 60)

        for n in range(1, _N_RECEIVERS + 1):
            subset = receivers[:n]
            print(f"\n  --- N={n} receivers ---")

            t_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                futures = {
                    pool.submit(_wake, _REMOTE_HOST, port, sender_url): (port, i)
                    for port, _, i in subset
                }
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    port, i = futures[future]
                    results[i] = future.result()
            wall_clock = time.time() - t_start

            latencies = []
            breakdowns = []
            for i in sorted(results):
                lat = results[i]["_client_elapsed_s"]
                latencies.append(lat)
                breakdown = results[i].get("latency", {})
                breakdowns.append(breakdown)
                port = subset[i][0]
                output = _infer(_REMOTE_HOST, port)
                correct = output == ref_output
                print(f"    Receiver {i}: wake={lat:.2f}s  correct={correct}")
                print(f"      Breakdown: {_fmt_breakdown(breakdown)}")

            avg_lat = sum(latencies) / len(latencies)
            print(f"    Wall-clock: {wall_clock:.2f}s  Avg/receiver: {avg_lat:.2f}s  "
                  f"Overhead vs baseline: {(avg_lat/avg_seq - 1)*100:.0f}%")
            print(f"    Avg breakdown: {_avg_breakdown(breakdowns)}")

            # Sleep all before next round
            for port, _, _ in subset:
                _sleep_engine(_REMOTE_HOST, port)
            time.sleep(_SWITCH_DELAY)

        # === Summary ===
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Model: {_MODEL} (TP={_TP})")
        print(f"  Sender: {_LOCAL_IP}")
        print(f"  Receivers: {_REMOTE_HOST} (N=1..{_N_RECEIVERS})")
        print(f"  Baseline (1-to-1): avg={avg_seq:.2f}s per wake")
        print(f"  Transport: EFA RDMA (cross-instance)")

    finally:
        for proc in procs:
            _kill(proc)
        _kill_remote_receivers()


if __name__ == "__main__":
    main()
