# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Online serving test: vLLM OpenAI-compatible server with NKIPy on Neuron.

Starts vllm serve, sends requests via the OpenAI API, validates responses.
"""

import glob
import json
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
    "~/zhuangw/nkipy/examples/models/qwen3/tmp_Qwen3-30b-a3b"
)


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=180):
    """Poll health endpoint until server is ready."""
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
    """Start vLLM server once for all tests in this module."""
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
        server.kill()
        server.wait()
        pytest.fail("Server failed to start within timeout")

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


@pytest.mark.integration
@_skip
def test_nkipy_online_completion(vllm_server):
    """Single /v1/completions request."""
    port, model_name = vllm_server
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": model_name,
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Completions failed: {resp.text}"
    text = resp.json()["choices"][0]["text"]
    print(f"Completion: {text!r}")
    assert len(text.strip()) > 0, "Empty completion"


@pytest.mark.integration
@_skip
def test_nkipy_online_chat(vllm_server):
    """Single /v1/chat/completions request."""
    port, model_name = vllm_server
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 20,
            "temperature": 0,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Chat failed: {resp.text}"
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"Chat: {text!r}")
    assert len(text.strip()) > 0, "Empty chat response"


@pytest.mark.integration
@_skip
def test_nkipy_online_continuous_batching(vllm_server):
    """Fire multiple concurrent requests to exercise continuous batching.

    Sends 4 completion requests in parallel (matching max_num_seqs=4).
    All must return non-empty results, proving the engine batches them.
    """
    port, model_name = vllm_server
    prompts = [
        "The capital of France is",
        "The largest ocean on Earth is",
        "Python was created by",
        "The speed of light is approximately",
    ]

    def _send(prompt):
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 20,
                "temperature": 0,
            },
            timeout=120,
        )
        return prompt, resp

    results = {}
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(_send, p) for p in prompts]
        for fut in as_completed(futures):
            prompt, resp = fut.result()
            assert resp.status_code == 200, f"Failed for {prompt!r}: {resp.text}"
            text = resp.json()["choices"][0]["text"]
            results[prompt] = text

    for prompt, text in results.items():
        print(f"  {prompt!r} -> {text!r}")
        assert len(text.strip()) > 0, f"Empty output for {prompt!r}"

    assert len(results) == len(prompts), (
        f"Expected {len(prompts)} results, got {len(results)}"
    )
