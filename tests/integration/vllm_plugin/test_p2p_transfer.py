# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""P2P weight transfer e2e test using vLLM's NKIPy server.

Launches two vLLM+NKIPy servers on different NeuronCore groups:
  Engine A (cores 0-7, with checkpoint) — active, serves weights
  Engine B (cores 16-23, no checkpoint) — starts sleeping

Then validates:
  1. Wake Engine B via P2P RDMA from Engine A
  2. Run inference on Engine B via /v1/completions
  3. Sleep Engine B, wake again (tests sleep/wake cycle)
  4. Run inference again (proves second wake works)
"""

import glob
import os
import signal
import socket
import subprocess
import time

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


def _start_vllm_server(port, checkpoint=None, core_offset=0, tp=8):
    """Start vLLM with NKIPy plugin server."""
    import tempfile

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
        "--model", "Qwen/Qwen3-30B-A3B",
        "--tensor-parallel-size", str(tp),
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


@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_NEURON and os.path.isdir(_QWEN3_CHECKPOINT)),
    reason="Requires Neuron devices and Qwen3 checkpoint",
)
def test_p2p_weight_transfer():
    """P2P weight transfer between two vLLM+NKIPy servers on same trn1."""
    port_a = _free_port()
    port_b = _free_port()

    engine_a = _start_vllm_server(
        port_a, checkpoint=_QWEN3_CHECKPOINT, core_offset=0,
    )
    engine_b = _start_vllm_server(port_b, checkpoint=None, core_offset=16)

    try:
        if not _wait_for_server(port_a):
            log = open(engine_a._log_path).read()[-2000:]
            pytest.fail(f"Engine A failed to start:\n{log}")
        if not _wait_for_server(port_b):
            log = open(engine_b._log_path).read()[-2000:]
            pytest.fail(f"Engine B failed to start:\n{log}")

        url_a = f"http://localhost:{port_a}"
        url_b = f"http://localhost:{port_b}"
        model = "Qwen/Qwen3-30B-A3B"

        # --- Sleeping engine should reject inference ---
        print("=== Verify sleeping engine rejects requests ===")
        resp = requests.post(
            f"{url_b}/v1/completions",
            json={"model": model, "prompt": "test", "max_tokens": 1},
            timeout=10,
        )
        assert resp.status_code == 503, f"Expected 503, got {resp.status_code}"

        # --- Step 1: Wake Engine B via P2P from Engine A ---
        print("=== Wake Engine B via P2P ===")
        resp = requests.post(
            f"{url_b}/nkipy/wake_up",
            json={"peer_url": url_a},
            timeout=300,
        )
        assert resp.status_code == 200, f"Wake failed: {resp.text}"
        print(f"Wake response: {resp.json()}")

        # --- Step 2: Inference on Engine B ---
        print("=== Inference on Engine B ===")
        resp = requests.post(
            f"{url_b}/v1/completions",
            json={"model": model, "prompt": "The capital of France is",
                  "max_tokens": 20, "temperature": 0},
            timeout=120,
        )
        assert resp.status_code == 200, f"Completion failed: {resp.text}"
        text1 = resp.json()["choices"][0]["text"]
        print(f"Output 1: {text1!r}")
        assert len(text1.strip()) > 0, "Empty output after P2P transfer"

        # --- Step 3: Sleep Engine B ---
        print("=== Sleep Engine B ===")
        resp = requests.post(f"{url_b}/nkipy/sleep", timeout=60)
        assert resp.status_code == 200, f"Sleep failed: {resp.text}"

        # --- Step 4: Wake again (second cycle) ---
        print("=== Wake Engine B again ===")
        resp = requests.post(
            f"{url_b}/nkipy/wake_up",
            json={"peer_url": url_a},
            timeout=300,
        )
        assert resp.status_code == 200, f"Second wake failed: {resp.text}"

        # --- Step 5: Inference after second wake ---
        print("=== Inference after second wake ===")
        resp = requests.post(
            f"{url_b}/v1/completions",
            json={"model": model, "prompt": "The capital of France is",
                  "max_tokens": 20, "temperature": 0},
            timeout=120,
        )
        assert resp.status_code == 200, f"Completion failed: {resp.text}"
        text2 = resp.json()["choices"][0]["text"]
        print(f"Output 2: {text2!r}")
        assert len(text2.strip()) > 0, "Empty output after second wake"

        # Greedy outputs must match across sleep/wake cycles
        assert text1 == text2, (
            f"Outputs differ!\n  1st: {text1!r}\n  2nd: {text2!r}"
        )
        print("=== PASS ===")

    finally:
        for proc in (engine_a, engine_b):
            proc.send_signal(signal.SIGTERM)
        for proc in (engine_a, engine_b):
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
