# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for broadcast weight transfer (multi-engine concurrent wake).

Validates that multiple receiver engines can wake simultaneously from a single
sender via parallel P2P push.  This exercises:
  1. Concurrent /p2p_preconnect handling on the sender
  2. Concurrent /p2p_push_weights with shared NIC bandwidth
  3. Correctness of weights transferred to all receivers
  4. Sender continues to serve inference during parallel pushes
  5. Sender can sleep normally after concurrent pushes complete

Uses TinyLlama-1.1B with TP=2 so that each engine only requires 2 NeuronCores,
allowing more receivers on the same instance for higher broadcast degree testing.

Checkpoint preparation (run once on a Neuron instance):
    python scripts/shard_checkpoint.py \
        --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --world-size 2 --head-dim 64 --shard-embed \
        --output-dir /fsx/zhuangw/models/tinyllama_TP2
"""

import concurrent.futures
import glob
import os
import signal
import socket
import subprocess
import tempfile
import time

os.environ["VLLM_PLUGINS"] = "nkipy"

import pytest
import requests

_HAS_NEURON = len(glob.glob("/dev/neuron*")) > 0
_TINYLLAMA_CHECKPOINT = "/fsx/zhuangw/models/tinyllama_TP2"
_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_PROMPT = "The capital of France is"
_TP = 2
_N_RECEIVERS = 5
_SWITCH_DELAY = 5


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


def _start_vllm_server(port, checkpoint=None, core_offset=0):
    env = os.environ.copy()
    env.update({
        "VLLM_PLUGINS": "nkipy",
        "VLLM_USE_V1": "1",
        "NKIPY_CORE_OFFSET": str(core_offset),
        "OMP_NUM_THREADS": "1",
        "VLLM_RPC_TIMEOUT": "600000",
    })
    if checkpoint:
        env["NKIPY_CHECKPOINT"] = checkpoint

    cmd = [
        "python", "-m", "nkipy.vllm_plugin.server",
        "--model", _MODEL,
        "--tensor-parallel-size", str(_TP),
        "--max-model-len", "128",
        "--max-num-seqs", "1",
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--port", str(port),
    ]

    log = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"engine_{port}_", suffix=".log", delete=False,
    )
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._log_path = log.name
    return proc


def _infer(port, prompt=_PROMPT):
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": _MODEL, "prompt": prompt,
              "max_tokens": 20, "temperature": 0},
        timeout=120,
    )
    assert resp.status_code == 200, f"Inference failed on port {port}: {resp.text}"
    return resp.json()["choices"][0]["text"]


def _wake(port, peer_url, timeout=300):
    resp = requests.post(
        f"http://localhost:{port}/nkipy/wake_up",
        json={"peer_url": peer_url},
        timeout=timeout,
    )
    assert resp.status_code == 200, f"Wake failed on port {port}: {resp.text}"
    return resp.json()


def _sleep(port):
    resp = requests.post(f"http://localhost:{port}/nkipy/sleep", timeout=60)
    assert resp.status_code == 200, f"Sleep failed on port {port}: {resp.text}"
    return resp.json()


def _kill_proc(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    if hasattr(proc, "_log_path"):
        try:
            os.unlink(proc._log_path)
        except OSError:
            pass


@pytest.fixture
def server_engine():
    """Start sender engine with TP=2 on cores 0-1."""
    port = _free_port()
    proc = _start_vllm_server(
        port, checkpoint=_TINYLLAMA_CHECKPOINT, core_offset=0,
    )
    if not _wait_for_server(port):
        log = open(proc._log_path).read()[-2000:]
        _kill_proc(proc)
        pytest.fail(f"Server engine failed to start:\n{log}")
    yield port, proc
    _kill_proc(proc)


@pytest.fixture
def receiver_pool(server_engine):
    """Start N sleeping receiver engines on dedicated core pairs.

    With TP=2, each engine uses 2 consecutive NeuronCores.  The sender
    occupies cores 0-1; receivers use cores 2-3, 4-5, 6-7, etc.
    This allows all receivers to wake concurrently on the same instance
    (each holding its own cores), simulating the multi-instance scenario.
    """
    server_port, _ = server_engine
    receivers = []
    for i in range(_N_RECEIVERS):
        port = _free_port()
        core_offset = 2 + i * _TP  # cores 2-3, 4-5, 6-7, 8-9, 10-11
        proc = _start_vllm_server(port, checkpoint=None, core_offset=core_offset)
        receivers.append((port, proc))

    for port, proc in receivers:
        if not _wait_for_server(port):
            log = open(proc._log_path).read()[-2000:]
            for _, p in receivers:
                _kill_proc(p)
            pytest.fail(f"Receiver {port} failed to start:\n{log}")

    yield server_port, receivers

    for _, proc in receivers:
        _kill_proc(proc)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_NEURON and os.path.isdir(_TINYLLAMA_CHECKPOINT)),
    reason="Requires Neuron devices and TinyLlama TP2 checkpoint",
)
class TestBroadcastWake:

    def test_concurrent_wake_all_correct(self, receiver_pool):
        """All N receivers woken concurrently produce correct output."""
        server_port, receivers = receiver_pool
        server_url = f"http://localhost:{server_port}"

        ref_output = _infer(server_port)
        assert len(ref_output.strip()) > 0

        # Wake all receivers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(receivers)) as pool:
            futures = {
                pool.submit(_wake, port, server_url): port
                for port, _ in receivers
            }
            results = {}
            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                results[port] = future.result()

        # Verify all woke successfully
        for port, result in results.items():
            assert result.get("status") != "error", (
                f"Port {port} wake failed: {result}")

        # Verify inference on all receivers matches reference
        for port, _ in receivers:
            output = _infer(port)
            assert output == ref_output, (
                f"Port {port}: {output!r} != {ref_output!r}")

        # Cleanup: sleep all receivers
        for port, _ in receivers:
            _sleep(port)

    def test_sender_serves_during_concurrent_push(self, receiver_pool):
        """Sender continues serving inference while pushing to N receivers."""
        server_port, receivers = receiver_pool
        server_url = f"http://localhost:{server_port}"

        ref_output = _infer(server_port)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(receivers) + 1) as pool:
            # Start all wakes concurrently
            wake_futures = [
                pool.submit(_wake, port, server_url)
                for port, _ in receivers
            ]

            # Fire inference on sender while pushes are in progress
            time.sleep(1)
            infer_future = pool.submit(_infer, server_port)
            sender_output = infer_future.result()

            # Wait for all wakes
            for f in wake_futures:
                f.result()

        assert sender_output == ref_output, (
            f"Sender during push: {sender_output!r} != {ref_output!r}")

        for port, _ in receivers:
            _sleep(port)

    def test_sender_sleep_after_concurrent_pushes(self, receiver_pool):
        """Sender can sleep normally after serving N concurrent pushes."""
        server_port, receivers = receiver_pool
        server_url = f"http://localhost:{server_port}"

        # Wake all receivers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(receivers)) as pool:
            futures = [pool.submit(_wake, port, server_url) for port, _ in receivers]
            for f in futures:
                f.result()

        # Sleep all receivers (release shared cores)
        for port, _ in receivers:
            _sleep(port)
        time.sleep(_SWITCH_DELAY)

        # Sender should sleep without issue (QPs cleaned up)
        result = _sleep(server_port)
        assert result.get("status") != "error", f"Sender sleep failed: {result}"

        # Sender should wake again from checkpoint
        resp = requests.post(
            f"http://localhost:{server_port}/nkipy/wake_up",
            json={}, timeout=300,
        )
        assert resp.status_code == 200

        output = _infer(server_port)
        assert len(output.strip()) > 0

    def test_repeated_broadcast_cycles(self, receiver_pool):
        """Multiple rounds of concurrent wake/sleep produce consistent output."""
        server_port, receivers = receiver_pool
        server_url = f"http://localhost:{server_port}"

        ref_output = _infer(server_port)

        for _round in range(3):
            # Wake all concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(receivers)) as pool:
                futures = [pool.submit(_wake, port, server_url) for port, _ in receivers]
                for f in futures:
                    f.result()

            # Verify all produce correct output
            for port, _ in receivers:
                output = _infer(port)
                assert output == ref_output, (
                    f"Round {_round}, port {port}: {output!r} != {ref_output!r}")

            # Sleep all
            for port, _ in receivers:
                _sleep(port)
            time.sleep(_SWITCH_DELAY)

    def test_staggered_arrivals(self, receiver_pool):
        """Receivers arriving with slight stagger still all wake correctly."""
        server_port, receivers = receiver_pool
        server_url = f"http://localhost:{server_port}"

        # Allow TCP TIME_WAIT from prior tests to clear
        time.sleep(_SWITCH_DELAY)

        ref_output = _infer(server_port)

        # Stagger wake requests by 1 second each
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(receivers)) as pool:
            futures = {}
            for i, (port, _) in enumerate(receivers):
                time.sleep(1)  # 1s stagger between each wake
                futures[pool.submit(_wake, port, server_url)] = port

            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                results[port] = future.result()

        for port, result in results.items():
            assert result.get("status") != "error", (
                f"Port {port} wake failed: {result}")
            output = _infer(port)
            assert output == ref_output, (
                f"Port {port}: {output!r} != {ref_output!r}")

        for port, _ in receivers:
            _sleep(port)
