#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: Broadcast weight transfer (1 sender → N receivers).

Tests /nkipy/push_weights with batched metadata from multiple receivers.

Topology:
  LOCAL  (sender):   1 engine, TP=32, cores 0-31
  REMOTE (receivers): 2 engines, TP=32, cores 0-31 and 32-63

Protocol for broadcast test:
  1. Both receivers wake WITHOUT peer_url (alloc empty tensors only)
  2. Both receivers register VRAM and expose metadata via GET /nkipy/rdma_metadata
  3. Orchestrator batches metadata from both receivers
  4. Single POST /nkipy/push_weights to sender with both receivers' metadata
  5. Sender does parallel RDMA WRITE to both receivers
  6. Verify inference correctness on both receivers

Usage:
    python tests/integration/vllm_plugin/bench_broadcast_transfer.py

Environment variables:
    BENCH_REMOTE_HOST   Remote instance IP (default: 10.3.215.3)
    BENCH_LOCAL_IP      Local instance IP (default: 10.3.211.148)
    BENCH_MODEL         HF model name (default: Qwen/Qwen3-30B-A3B)
    BENCH_CHECKPOINT    Checkpoint path (default: /fsx/zhuangw/models/qwen3_30b_a3b_TP32)
"""

import argparse
import concurrent.futures
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
_MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-30B-A3B")
_CHECKPOINT = os.environ.get("BENCH_CHECKPOINT", "/fsx/zhuangw/models/qwen3_30b_a3b_TP32")
_TP = int(os.environ.get("BENCH_TP", "32"))
_NUM_RECEIVERS = int(os.environ.get("BENCH_NUM_RECEIVERS", "2"))
_VENV_PYTHON = "/fsx/zhuangw/nkipy/.venv/bin/python"
_SENDER_PORT = 8100
# Derive per-receiver ports and core offsets from TP and num receivers
_RECEIVER_PORTS = [9100 + i * 100 for i in range(_NUM_RECEIVERS)]
_RECEIVER_CORE_OFFSETS = [i * _TP for i in range(_NUM_RECEIVERS)]


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


def _start_local_sender():
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
        "NEURON_RT_MAP_HBM": "1",
    })

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
        mode="w", prefix="bench_bcast_sender_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _start_remote_receiver(port, core_offset):
    venv_bin = os.path.dirname(_VENV_PYTHON)
    nixl_port = 21000 + core_offset
    master_port = 40000 + core_offset * 100
    env_vars = (
        f"PATH={venv_bin}:$PATH "
        f"VLLM_PLUGINS=nkipy "
        f"VLLM_USE_V1=1 "
        f"NKIPY_CORE_OFFSET={core_offset} "
        f"NKIPY_NIXL_PORT={nixl_port} "
        f"MASTER_PORT={master_port} "
        f"OMP_NUM_THREADS=1 "
        f"VLLM_RPC_TIMEOUT=600000 "
        f"HF_HUB_OFFLINE=1 "
        f"NEURON_RT_MAP_HBM=1"
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
        mode="w", prefix=f"bench_bcast_recv{port}_", suffix=".log", delete=False,
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
        print(f"  ERROR: Wake {port} failed ({resp.status_code}): {resp.text[:500]}")
        return None
    result = resp.json()
    result["_client_elapsed_s"] = elapsed
    return result


def _sleep_engine(host, port):
    resp = requests.post(f"http://{host}:{port}/nkipy/sleep", timeout=120)
    if resp.status_code != 200:
        print(f"  WARNING: Sleep {port} failed ({resp.status_code}): {resp.text[:200]}")
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
        print(f"  WARNING: Inference {port} failed ({resp.status_code}): {resp.text[:200]}")
        return None
    return resp.json()["choices"][0]["text"]


def _get_rdma_metadata(host, port):
    resp = requests.get(f"http://{host}:{port}/nkipy/rdma_metadata", timeout=60)
    if resp.status_code != 200:
        print(f"  ERROR: rdma_metadata {port} failed ({resp.status_code}): {resp.text[:200]}")
        return None
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Broadcast transfer benchmark")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip sequential baseline test")
    args = parser.parse_args()

    procs = []
    try:
        _kill_remote()

        # --- Start engines ---
        print(f"Starting sender on local ({_LOCAL_IP}:{_SENDER_PORT}), TP={_TP}...")
        sender_proc = _start_local_sender()
        procs.append(sender_proc)

        for i, (port, offset) in enumerate(zip(_RECEIVER_PORTS, _RECEIVER_CORE_OFFSETS)):
            print(f"Starting receiver {i+1} on remote ({_REMOTE_HOST}:{port}), "
                  f"cores {offset}-{offset+_TP-1}...")
            recv_proc = _start_remote_receiver(port, offset)
            procs.append(recv_proc)

        # --- Wait for all engines ---
        print("  Waiting for sender...")
        if not _wait_for_server("localhost", _SENDER_PORT):
            print(f"  ERROR: Sender failed. Log: {sender_proc._log_path}")
            sys.exit(1)
        print("  Sender ready.")

        for i, port in enumerate(_RECEIVER_PORTS):
            print(f"  Waiting for receiver {i+1} (port {port})...")
            if not _wait_for_server(_REMOTE_HOST, port):
                print(f"  ERROR: Receiver {i+1} failed. Log: {procs[i+1]._log_path}")
                sys.exit(1)
            print(f"  Receiver {i+1} ready (sleeping).")

        sender_url = f"http://{_LOCAL_IP}:{_SENDER_PORT}"
        print()

        # --- Get reference output from sender ---
        print("Verifying sender inference...")
        ref_output = _infer("localhost", _SENDER_PORT)
        print(f"  Reference: {ref_output[:60]!r}...\n" if ref_output else "  FAILED\n")

        # === Test 1: Sequential baseline ===
        if not args.skip_baseline:
            print("=" * 70)
            print("TEST 1: Sequential wake (1-to-1, baseline)")
            print("=" * 70)
            seq_latencies = []
            for i, port in enumerate(_RECEIVER_PORTS):
                print(f"\n  Waking receiver {i+1} (port {port})...")
                result = _wake(_REMOTE_HOST, port, sender_url)
                if result:
                    lat = result.get("latency", {})
                    print(f"    Total: {result['_client_elapsed_s']:.2f}s "
                          f"(p2p={lat.get('p2p_transfer_s', '?')}s)")
                    seq_latencies.append(result["_client_elapsed_s"])
                    output = _infer(_REMOTE_HOST, port)
                    correct = output == ref_output if (output and ref_output) else False
                    print(f"    Inference: {'PASS' if correct else 'FAIL'}")
                print(f"  Sleeping receiver {i+1}...")
                _sleep_engine(_REMOTE_HOST, port)
                time.sleep(3)
            print(f"\n  Sequential total: {sum(seq_latencies):.2f}s "
                  f"(avg {sum(seq_latencies)/len(seq_latencies):.2f}s per receiver)")
            print()

        # === Test 2: Sequential 1-to-1 for comparison ===
        if not args.skip_baseline:
            print("=" * 70)
            print("TEST 2: Sequential wake (each receiver individually)")
            print("=" * 70)
            seq_latencies = []
            for i, port in enumerate(_RECEIVER_PORTS):
                print(f"\n  Waking receiver {i+1} (port {port})...")
                result = _wake(_REMOTE_HOST, port, sender_url)
                if result:
                    lat = result.get("latency", {})
                    print(f"    Total: {result['_client_elapsed_s']:.2f}s "
                          f"(p2p={lat.get('p2p_transfer_s', '?')}s)")
                    seq_latencies.append(result["_client_elapsed_s"])
                    output = _infer(_REMOTE_HOST, port)
                    correct = output == ref_output if (output and ref_output) else False
                    print(f"    Inference: {'PASS' if correct else 'FAIL'}")
                else:
                    print(f"    FAILED")
                print(f"  Sleeping receiver {i+1}...")
                _sleep_engine(_REMOTE_HOST, port)
                time.sleep(3)
            if seq_latencies:
                print(f"\n  Sequential total: {sum(seq_latencies):.2f}s "
                      f"(avg {sum(seq_latencies)/len(seq_latencies):.2f}s per receiver)")
            print()

        # === Test 3: True broadcast (batched /nkipy/push_weights) ===
        print("\n" + "=" * 70)
        print("TEST 3: True broadcast (batched /nkipy/push_weights)")
        print("=" * 70)

        # Step 1: Wake both receivers WITHOUT peer_url (alloc only, no P2P)
        print("\n  Step 1: Wake both receivers (alloc only, no peer_url)...")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_wake, _REMOTE_HOST, port, None)
                for port in _RECEIVER_PORTS
            ]
            alloc_results = [f.result() for f in futures]
        t_alloc = time.time() - t0
        for i, (port, result) in enumerate(zip(_RECEIVER_PORTS, alloc_results)):
            if result:
                lat = result.get("latency", {})
                print(f"    Receiver {i+1}: {result['_client_elapsed_s']:.2f}s")
            else:
                print(f"    Receiver {i+1}: FAILED")
                sys.exit(1)
        print(f"    Alloc wall time: {t_alloc:.2f}s")

        # Step 2: Get RDMA metadata from both receivers
        print("\n  Step 2: Gathering RDMA metadata from both receivers...")
        t0 = time.time()
        metadata_list = []
        for i, port in enumerate(_RECEIVER_PORTS):
            meta = _get_rdma_metadata(_REMOTE_HOST, port)
            if meta is None or meta.get("status") != "ok":
                print(f"    ERROR: Could not get metadata from receiver {i+1}")
                sys.exit(1)
            metadata_list.append(meta["per_rank"])
            print(f"    Receiver {i+1}: got {len(meta['per_rank'])} rank entries")
        t_meta = time.time() - t0
        print(f"    Metadata gather: {t_meta:.2f}s")

        # Step 3: Batched POST to sender (broadcast to both receivers)
        print(f"\n  Step 3: POST /nkipy/push_weights with {len(metadata_list)} receivers...")
        t0 = time.time()
        resp = requests.post(
            f"http://localhost:{_SENDER_PORT}/nkipy/push_weights",
            json={"receivers": metadata_list},
            timeout=300,
        )
        t_transfer = time.time() - t0
        if resp.status_code != 200:
            print(f"    ERROR: Transfer failed ({resp.status_code}): {resp.text[:500]}")
            sys.exit(1)
        result = resp.json()
        print(f"    Transfer completed: {t_transfer:.2f}s "
              f"(status={result.get('status')}, n_receivers={result.get('n_receivers')})")

        # Step 4: Verify inference on both receivers
        # (tok_embedding is lazily loaded from device on first inference)
        print("\n  Step 4: Verifying inference on both receivers...")
        all_correct = True
        for i, port in enumerate(_RECEIVER_PORTS):
            output = _infer(_REMOTE_HOST, port)
            correct = output == ref_output if (output and ref_output) else False
            status = "PASS" if correct else "FAIL"
            print(f"    Receiver {i+1}: {status}" +
                  (f" — {output[:40]!r}" if output else ""))
            if not correct:
                all_correct = False

        # Sleep both
        for port in _RECEIVER_PORTS:
            _sleep_engine(_REMOTE_HOST, port)

        # === Summary ===
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  True broadcast (1× batched POST): {t_transfer:.2f}s transfer"
              f" + {t_alloc:.2f}s alloc + {t_meta:.2f}s metadata"
              f" = {t_alloc + t_meta + t_transfer:.2f}s total")
        print(f"  Broadcast correctness:             "
              f"{'ALL PASS' if all_correct else 'FAILED'}")
        print()

    finally:
        for proc in procs:
            _kill(proc)
        _kill_remote()


if __name__ == "__main__":
    main()
