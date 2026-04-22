# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scalability test for NKIPy standby engines.

Tests multiple standby receiver engines sharing the same NeuronCores,
validating:
  1. Multiple receivers can start on the same core group
  2. Sleep/wake cycles produce consistent output across engines
  3. Dynamically adding/removing engines doesn't break others
  4. Server engine can sleep and wake from local checkpoint
  5. Server engine can sleep and wake from an active receiver via P2P
"""

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
_TINYLLAMA_CHECKPOINT = os.path.expanduser(
    "~/models/llama3/tmp_tinyllama_TP8"
)
_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_PROMPT = "The capital of France is"
_TP = 8
_SWITCH_DELAY = 10


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


def _wake(port, peer_url):
    resp = requests.post(
        f"http://localhost:{port}/nkipy/wake_up",
        json={"peer_url": peer_url},
        timeout=300,
    )
    assert resp.status_code == 200, f"Wake failed on port {port}: {resp.text}"
    return resp.json()


def _wake_from_checkpoint(port):
    resp = requests.post(
        f"http://localhost:{port}/nkipy/wake_up",
        json={},
        timeout=300,
    )
    assert resp.status_code == 200, f"Wake (checkpoint) failed on port {port}: {resp.text}"
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
def receiver_engines(server_engine):
    """Start 5 sleeping receiver engines on a shared core group."""
    server_port, _ = server_engine
    receivers = []
    for _ in range(5):
        port = _free_port()
        proc = _start_vllm_server(port, checkpoint=None, core_offset=16)
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
    reason="Requires Neuron devices and TinyLlama TP8 checkpoint",
)
class TestScalability:

    def test_multiple_standby_engines(self, receiver_engines):
        """Multiple sleeping receivers can coexist on the same core group."""
        server_port, receivers = receiver_engines
        server_url = f"http://localhost:{server_port}"

        ref_output = _infer(server_port)
        assert len(ref_output.strip()) > 0

        for port, _ in receivers:
            _wake(port, server_url)
            output = _infer(port)
            assert output == ref_output, (
                f"Port {port}: {output!r} != {ref_output!r}"
            )
            _sleep(port)
            time.sleep(_SWITCH_DELAY)

    def test_multi_cycle_wake_sleep(self, receiver_engines):
        """A single receiver can be woken and slept multiple times."""
        server_port, receivers = receiver_engines
        server_url = f"http://localhost:{server_port}"
        port = receivers[0][0]
        outputs = []

        for cycle in range(3):
            _wake(port, server_url)
            outputs.append(_infer(port))
            _sleep(port)
            time.sleep(_SWITCH_DELAY)

        assert all(o == outputs[0] for o in outputs), (
            f"Outputs differ across cycles: {outputs!r}"
        )

    def test_dynamic_engine_management(self, receiver_engines):
        """Adding/removing engines doesn't affect others."""
        server_port, receivers = receiver_engines
        server_url = f"http://localhost:{server_port}"

        # Wake and verify first receiver
        port_a = receivers[0][0]
        _wake(port_a, server_url)
        ref_output = _infer(port_a)
        _sleep(port_a)
        time.sleep(_SWITCH_DELAY)

        # Kill receiver[1]
        _kill_proc(receivers[1][1])

        # Start a new receiver dynamically
        new_port = _free_port()
        new_proc = _start_vllm_server(new_port, checkpoint=None, core_offset=16)
        try:
            assert _wait_for_server(new_port), "New receiver failed to start"

            # Wake and verify new receiver
            _wake(new_port, server_url)
            new_output = _infer(new_port)
            assert new_output == ref_output, (
                f"New engine: {new_output!r} != {ref_output!r}"
            )
            _sleep(new_port)
            time.sleep(_SWITCH_DELAY)

            # Verify an existing receiver still works
            port_c = receivers[2][0]
            _wake(port_c, server_url)
            existing_output = _infer(port_c)
            assert existing_output == ref_output, (
                f"Existing engine: {existing_output!r} != {ref_output!r}"
            )
            _sleep(port_c)
        finally:
            _kill_proc(new_proc)

    def test_server_sleep_wake_checkpoint(self, server_engine):
        """Server engine can sleep and wake from local checkpoint."""
        server_port, _ = server_engine

        ref_output = _infer(server_port)
        _sleep(server_port)

        resp = requests.post(
            f"http://localhost:{server_port}/v1/completions",
            json={"model": _MODEL, "prompt": "test", "max_tokens": 1},
            timeout=10,
        )
        assert resp.status_code == 503, "Sleeping server should reject inference"

        _wake_from_checkpoint(server_port)
        output = _infer(server_port)
        assert output == ref_output, (
            f"After checkpoint wake: {output!r} != {ref_output!r}"
        )

    def test_server_sleep_wake_from_receiver(self, receiver_engines):
        """Server can sleep and wake via P2P from an active receiver."""
        server_port, receivers = receiver_engines
        server_url = f"http://localhost:{server_port}"
        recv_port = receivers[0][0]
        recv_url = f"http://localhost:{recv_port}"

        ref_output = _infer(server_port)

        # Wake receiver from server
        _wake(recv_port, server_url)
        recv_output = _infer(recv_port)
        assert recv_output == ref_output

        # Sleep server, then wake from receiver
        _sleep(server_port)
        _wake(server_port, recv_url)
        server_output = _infer(server_port)
        assert server_output == ref_output, (
            f"After P2P wake from receiver: {server_output!r} != {ref_output!r}"
        )

        # Cleanup: sleep receiver
        _sleep(recv_port)
