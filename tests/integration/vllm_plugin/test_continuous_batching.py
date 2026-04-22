# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E test for continuous batching in the NKIPy vLLM plugin.

Verifies that multiple concurrent requests are served without serializing,
i.e., total wall-clock time is less than the sum of individual request times.
"""

import glob
import os
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["VLLM_PLUGINS"] = "nkipy"

import pytest
import requests

_HAS_NEURON = len(glob.glob("/dev/neuron*")) > 0
_QWEN3_CHECKPOINT = os.path.expanduser(
    "~/models/qwen3/tmp_Qwen3-30b-a3b"
)


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="module")
def vllm_server():
    port = _free_port()
    model_name = "Qwen/Qwen3-30B-A3B"

    env = os.environ.copy()
    env.update({
        "VLLM_PLUGINS": "nkipy",
        "VLLM_USE_V1": "1",
        "NKIPY_CHECKPOINT": _QWEN3_CHECKPOINT,
    })

    server = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--tensor-parallel-size", "8",
            "--max-model-len", "128",
            "--max-num-seqs", "4",
            "--enforce-eager",
            "--dtype", "bfloat16",
            "--port", str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    ready = _wait_for_server(port)
    if not ready:
        out = server.stdout.read().decode(errors="replace")[-3000:]
        server.kill()
        server.wait()
        pytest.fail(f"Server failed to start:\n{out}")

    yield port, model_name

    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


_skip = pytest.mark.skipif(
    not (_HAS_NEURON and os.path.isdir(_QWEN3_CHECKPOINT)),
    reason="Requires Neuron devices and Qwen3 checkpoint",
)


def _completions(port, model_name, prompt, max_tokens=10):
    """Send a single completion request and return (text, elapsed_s)."""
    t0 = time.time()
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    data = resp.json()
    assert "choices" in data, f"No choices in response: {data}"
    text = data["choices"][0]["text"]
    return text, elapsed


@pytest.mark.integration
@_skip
class TestContinuousBatching:
    """Tests for continuous batching support."""

    def test_single_request(self, vllm_server):
        """Baseline: single request produces non-empty output."""
        port, model = vllm_server
        text, elapsed = _completions(port, model, "Hello world", max_tokens=5)
        print(f"Single request: {text!r} ({elapsed:.2f}s)")
        assert len(text.strip()) > 0

    def test_concurrent_requests_varied_lengths(self, vllm_server):
        """Concurrent requests with different prompt lengths all succeed."""
        port, model = vllm_server
        prompts = [
            "Hi",
            "The capital of France is",
            "Once upon a time in a land far far away there lived a",
        ]

        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = {
                pool.submit(_completions, port, model, p, 5): p
                for p in prompts
            }
            results = {}
            for fut in as_completed(futures):
                prompt = futures[fut]
                text, elapsed = fut.result()
                results[prompt] = (text, elapsed)
                print(f"  len={len(prompt):3d} -> {text!r} ({elapsed:.2f}s)")

        assert len(results) == len(prompts)
        for prompt, (text, _) in results.items():
            assert len(text.strip()) > 0, f"Empty output for {prompt!r}"

    def test_concurrent_faster_than_sequential(self, vllm_server):
        """Concurrent requests should complete faster than sequential.

        If continuous batching works, N concurrent requests should take
        less than N * single_request_time. We measure one sequential
        request and extrapolate, then compare against concurrent wall time.
        """
        port, model = vllm_server
        max_tokens = 5
        prompts = [
            "The capital of France is",
            "The largest ocean is",
            "Python was created by",
        ]

        # Single sequential request as baseline, extrapolate to N
        _, single_elapsed = _completions(port, model, prompts[0], max_tokens)
        sequential_estimate = single_elapsed * len(prompts)

        # Concurrent
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                pool.submit(_completions, port, model, p, max_tokens)
                for p in prompts
            ]
            for fut in as_completed(futures):
                fut.result()  # raise on error
        concurrent_total = time.time() - t0

        print(f"Sequential estimate ({len(prompts)}x): {sequential_estimate:.2f}s")
        print(f"Concurrent total: {concurrent_total:.2f}s")
        print(f"Speedup: {sequential_estimate / concurrent_total:.2f}x")

        # Relaxed check: concurrent should be at least 20% faster
        assert concurrent_total < sequential_estimate * 0.8, (
            f"Concurrent ({concurrent_total:.2f}s) not faster than "
            f"sequential estimate ({sequential_estimate:.2f}s) — batching may not be working"
        )
