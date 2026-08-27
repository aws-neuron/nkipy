# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeviceKernel.submit (non-blocking execute).

submit() is a thin wrapper over the singleton-backed SpikeAsync: it validates
I/O (like __call__), turns the DeviceTensor dicts into tensor sets, and returns
SpikeAsync's future. The event-loop/poll machinery lives in SpikeAsync and is
tested there; here we cover the wrapper glue with a fake SpikeAsync, so no Neuron
hardware is needed. A device-gated round-trip at the bottom exercises the real
path when hardware is present.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import nkipy.runtime.device_kernel as dk
from nkipy.runtime.device_kernel import DeviceKernel


class _FakeFuture:
    """Stand-in for SpikeAsyncFuture; records that .wait() was called."""

    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class FakeSpikeAsync:
    """Minimal SpikeAsync stand-in: echoes tensor sets, records executes."""

    def __init__(self):
        self.executed = []  # list of (model, in_set, out_set)

    def create_tensor_set(self, tensors):
        return tensors

    def execute(self, model, inputs, outputs):
        self.executed.append((model, inputs, outputs))
        return _FakeFuture()


@pytest.fixture
def fake_async():
    """Patch the module's SpikeAsync accessor with a fake."""
    fake = FakeSpikeAsync()
    with patch.object(dk, "get_spike_async_singleton", return_value=fake):
        yield fake


# Tensor info the fake kernel advertises for its "a" input and "o" output; must
# match the fake tensors below so _validate_io passes.
_INFO = SimpleNamespace(shape=(1,), dtype=np.dtype("float32"), size=4)


def _make_kernel():
    """A DeviceKernel instance without touching hardware (bypass __init__)."""
    k = DeviceKernel.__new__(DeviceKernel)
    k.model_ref = SimpleNamespace(core_id=0)
    k.name = "fake"
    k.input_tensors_info = {"a": _INFO}
    k.output_tensors_info = {"o": _INFO}
    return k


class _T:
    """Tiny DeviceTensor stand-in with the attributes _validate_io checks."""

    tensor_ref = SimpleNamespace(core_id=0)
    shape = (1,)
    dtype = np.dtype("float32")


def _t():
    return _T()


def test_submit_returns_future_and_executes(fake_async):
    """submit() builds tensor sets from tensor_refs and returns SpikeAsync's future."""
    k = _make_kernel()
    a, o = _t(), _t()
    fut = k.submit(inputs={"a": a}, outputs={"o": o})

    assert fut.waited is False  # non-blocking: not waited yet
    assert len(fake_async.executed) == 1
    model, in_set, out_set = fake_async.executed[0]
    assert model is k.model_ref
    # tensor sets are keyed by NEFF name -> the DeviceTensor's tensor_ref
    assert in_set == {"a": a.tensor_ref}
    assert out_set == {"o": o.tensor_ref}

    fut.wait()
    assert fut.waited is True


def test_submit_validates_io(fake_async):
    """Bad input names are rejected before anything is submitted (parity with __call__).

    Without validation the mismatch would only surface later as an opaque NRT
    failure, detached from the offending submission.
    """
    k = _make_kernel()
    with pytest.raises(ValueError, match="Unknown input"):
        k.submit(inputs={"wrong_name": _t()}, outputs={"o": _t()})
    # Nothing was submitted.
    assert fake_async.executed == []


# --------------------------------------------------------------------------
# Device round-trip: real submit + future.wait() on Neuron hardware.
# --------------------------------------------------------------------------

from nkipy.runtime import is_neuron_compatible  # noqa: E402


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need Neuron hardware for submit round-trip",
)
def test_submit_roundtrip_matches_sync():
    """submit()+wait() yields the same output as the blocking __call__ path."""
    from nkipy.runtime.device_tensor import DeviceTensor

    def add_kernel(a, b):
        return np.add(a, b)

    a = np.random.randn(128, 128).astype(np.float32)
    b = np.random.randn(128, 128).astype(np.float32)
    da = DeviceTensor.from_numpy(a, "async_a")
    db = DeviceTensor.from_numpy(b, "async_b")
    out_async = DeviceTensor.from_numpy(np.empty_like(a), "async_out")
    out_sync = DeviceTensor.from_numpy(np.empty_like(a), "sync_out")

    kernel = DeviceKernel.compile_and_load(add_kernel, a, b)

    in_name0, in_name1 = list(kernel.input_tensors_info)[:2]
    out_name = list(kernel.output_tensors_info)[0]
    inputs = {in_name0: da, in_name1: db}

    fut = kernel.submit(inputs=inputs, outputs={out_name: out_async})
    fut.wait()  # block until the submission completes before reading back

    kernel(inputs=inputs, outputs={out_name: out_sync})

    # Async path matches both the sync path and the numpy reference.
    np.testing.assert_array_equal(out_async.numpy(), out_sync.numpy())
    np.testing.assert_allclose(out_async.numpy(), a + b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
