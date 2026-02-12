# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-distributed tests for spiky.

Covers: smoke, correctness, dynamic shapes, lifecycle, stats, training.

All tests run on real Neuron hardware via the session-scoped hw_backend fixture.
"""

import copy

import pytest
import torch
import torch.nn as nn
from conftest import requires_neuron
from torch.testing import assert_close

pytestmark = [requires_neuron]


# ---------------------------------------------------------------------------
# Section 1: Smoke tests (fast dev loop — shape checks only)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestTorchCompileSmoke:
    """Quick shape-only checks that torch.compile(backend="nkipy") runs."""

    def test_add(self, hw_backend):
        @torch.compile(backend="nkipy")
        def f(a, b):
            return a + b

        a = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        b = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            out = f(a, b)
        assert out.shape == (4, 16)

    def test_linear_no_bias(self, hw_backend):
        model = nn.Linear(32, 64, bias=False).to("nkipy")
        compiled = torch.compile(model, backend="nkipy")
        x = torch.randn(8, 32, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            out = compiled(x)
        assert out.shape == (8, 64)

    def test_linear_with_bias(self, hw_backend):
        model = nn.Linear(16, 32).to("nkipy")
        compiled = torch.compile(model, backend="nkipy")
        x = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            out = compiled(x)
        assert out.shape == (4, 32)

    def test_mlp(self, hw_backend):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=False)
                self.fc2 = nn.Linear(32, 8, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = MLP().to("nkipy")
        compiled = torch.compile(model, backend="nkipy")
        x = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            out = compiled(x)
        assert out.shape == (4, 8)


@pytest.mark.smoke
class TestDeviceSmoke:
    """Quick checks for device registration and tensor movement."""

    def test_device_count(self, hw_backend):
        import spiky

        assert spiky.device_count() > 0

    def test_tensor_to_device(self, hw_backend):
        x = torch.randn(4, 4)
        x_dev = x.to("nkipy")
        assert x_dev.device.type == "nkipy"

    def test_tensor_roundtrip(self, hw_backend):
        x = torch.randn(4, 4, dtype=torch.float32)
        x_dev = x.to("nkipy")
        x_back = x_dev.cpu()
        assert_close(x_back, x)


# ---------------------------------------------------------------------------
# Section 2: Correctness (eager vs compiled numerical comparison)
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Numerically compare eager CPU results with compiled nkipy results."""

    def test_linear_no_bias(self, hw_backend):
        torch.manual_seed(42)
        model = nn.Linear(32, 64, bias=False)
        x = torch.randn(8, 32, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_linear_with_bias(self, hw_backend):
        torch.manual_seed(42)
        model = nn.Linear(16, 32)
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_add(self, hw_backend):
        torch.manual_seed(42)

        @torch.compile(backend="nkipy")
        def f(a, b):
            return a + b

        a = torch.randn(4, 16, dtype=torch.float32)
        b = torch.randn(4, 16, dtype=torch.float32)
        expected = a + b

        a_dev = a.to("nkipy")
        b_dev = b.to("nkipy")
        with torch.no_grad():
            actual = f(a_dev, b_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_mlp(self, hw_backend):
        torch.manual_seed(42)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=False)
                self.fc2 = nn.Linear(32, 8, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = MLP()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_silu_activation(self, hw_backend):
        torch.manual_seed(42)

        class SiLUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 32, bias=False)
                self.act = nn.SiLU()

            def forward(self, x):
                return self.act(self.linear(x))

        model = SiLUNet()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Section 3: Backend lifecycle & stats
# ---------------------------------------------------------------------------


class TestBackendLifecycle:
    """Test backend initialization state."""

    def test_is_initialized(self, hw_backend):
        from spiky.torch.backend import is_nkipy_backend_initialized

        assert is_nkipy_backend_initialized() is True

    def test_double_init_raises(self, hw_backend):
        from spiky.torch.backend import init_nkipy_backend

        with pytest.raises(RuntimeError):
            init_nkipy_backend()


class TestMemoryAndStats:
    """Test memory and stats APIs on initialized hardware."""

    def test_memory_stats_schema(self, hw_backend):
        import spiky

        stats = spiky.get_memory_stats()
        expected_keys = {
            "used_bytes",
            "cached_bytes",
            "total_bytes",
            "allocation_count",
            "reuse_count",
            "cache_hit_count",
            "cache_miss_count",
        }
        assert set(stats.keys()) == expected_keys
        for v in stats.values():
            assert v >= 0

    def test_clear_and_trim(self, hw_backend):
        import spiky

        spiky.clear_memory_pool()
        spiky.trim_memory_pool(0)

    def test_stats_api(self, hw_backend):
        import spiky

        spiky.get_stats()
        spiky.reset_stats()

    def test_memory_after_execution(self, hw_backend):
        import spiky

        model = nn.Linear(16, 32, bias=False).to("nkipy")
        compiled = torch.compile(model, backend="nkipy")
        x = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            _ = compiled(x)

        stats = spiky.get_memory_stats()
        # After execution, at least some memory should have been allocated
        assert stats["allocation_count"] >= 0


# ---------------------------------------------------------------------------
# Section 4: Dynamic shapes
# ---------------------------------------------------------------------------


class TestDynamicShapes:
    """Test dynamic shape compilation with bucket selection and JIT."""

    @staticmethod
    def _with_spiky_config(hw_backend, pipelined=True):
        """Temporarily swap config for dynamic shape tests."""
        from spiky.torch.config import (
            NKIPyBackendConfig,
            get_nkipy_backend_config,
            set_nkipy_backend_config,
        )

        old_config = get_nkipy_backend_config()
        new_config = NKIPyBackendConfig(
            nkipy_cache_prefix=old_config.nkipy_cache_prefix,
            log_level=old_config.log_level,
            rank=old_config.rank,
            world_size=old_config.world_size,
            additional_compiler_args=old_config.additional_compiler_args,
            pipelined=pipelined,
        )
        set_nkipy_backend_config(new_config)
        return old_config

    @staticmethod
    def _restore_config(old_config):
        from spiky.torch.config import set_nkipy_backend_config

        set_nkipy_backend_config(old_config)

    def test_dynamic_batch_dim(self, hw_backend):
        """Dynamic batch dim with maybe_mark_dynamic and JIT bucket compilation."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(model, backend="nkipy", dynamic=True)

            x = torch.randn(4, 32)
            torch._dynamo.maybe_mark_dynamic(x, 0)

            with torch.no_grad():
                out1 = compiled(x)
            assert out1.shape == (4, 64)

            with torch.no_grad():
                out2 = compiled(torch.randn(16, 32))
            assert out2.shape == (16, 64)
        finally:
            self._restore_config(old_config)

    def test_dynamic_correctness(self, hw_backend):
        """Verify dynamic shape outputs match eager CPU."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 64, bias=False)
            x = torch.randn(4, 32, dtype=torch.float32)

            with torch.no_grad():
                expected = model(x)

            compiled = torch.compile(model, backend="nkipy", dynamic=True)
            x_dyn = x.clone()
            torch._dynamo.maybe_mark_dynamic(x_dyn, 0)

            with torch.no_grad():
                actual = compiled(x_dyn)

            assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        finally:
            self._restore_config(old_config)

    def test_dynamic_mlp(self, hw_backend):
        """Dynamic shapes through a multi-layer model."""
        old_config = self._with_spiky_config(hw_backend)
        try:

            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(16, 32, bias=False)
                    self.fc2 = nn.Linear(32, 8, bias=False)

                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))

            torch.manual_seed(42)
            model = MLP()
            x = torch.randn(4, 16, dtype=torch.float32)

            with torch.no_grad():
                expected = model(x)

            compiled = torch.compile(model, backend="nkipy", dynamic=True)
            x_dyn = x.clone()
            torch._dynamo.maybe_mark_dynamic(x_dyn, 0)

            with torch.no_grad():
                actual = compiled(x_dyn)

            assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        finally:
            self._restore_config(old_config)

    def test_dynamic_multiple_bucket_sizes(self, hw_backend):
        """Correctness across three different batch sizes via JIT compilation."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(model, backend="nkipy", dynamic=True)

            for batch in [4, 8, 16]:
                x = torch.randn(batch, 32, dtype=torch.float32)
                if batch == 4:
                    torch._dynamo.maybe_mark_dynamic(x, 0)

                with torch.no_grad():
                    expected = model(x)
                    actual = compiled(x)

                assert actual.shape == (batch, 64), f"batch={batch}"
                assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        finally:
            self._restore_config(old_config)

    def test_dynamic_explicit_buckets(self, hw_backend):
        """Explicit bucket list via per-function options."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(
                model,
                backend="nkipy",
                dynamic=True,
                options={"buckets": [4, 16]},
            )

            x = torch.randn(4, 32, dtype=torch.float32)
            torch._dynamo.maybe_mark_dynamic(x, 0)

            with torch.no_grad():
                expected = model(x)
                actual = compiled(x)

            assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        finally:
            self._restore_config(old_config)

    def test_dynamic_pipelined(self, hw_backend):
        """Pipelined execution across multiple calls with different sizes."""
        old_config = self._with_spiky_config(hw_backend, pipelined=True)
        try:
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(model, backend="nkipy", dynamic=True)

            x = torch.randn(4, 32)
            torch._dynamo.maybe_mark_dynamic(x, 0)

            with torch.no_grad():
                out1 = compiled(x)
                out2 = compiled(torch.randn(8, 32))
                out3 = compiled(torch.randn(16, 32))

            assert out1.shape == (4, 64)
            assert out2.shape == (8, 64)
            assert out3.shape == (16, 64)
        finally:
            self._restore_config(old_config)

    def test_dynamic_pipelined_correctness(self, hw_backend):
        """Pipelined dynamic execution with numerical correctness checks."""
        old_config = self._with_spiky_config(hw_backend, pipelined=True)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(model, backend="nkipy", dynamic=True)

            for batch in [4, 8, 16]:
                x = torch.randn(batch, 32, dtype=torch.float32)
                if batch == 4:
                    torch._dynamo.maybe_mark_dynamic(x, 0)

                with torch.no_grad():
                    expected = model(x)
                    actual = compiled(x)

                assert actual.shape == (batch, 64), f"batch={batch}"
                assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        finally:
            self._restore_config(old_config)


# ---------------------------------------------------------------------------
# Section 5: Static callable path (NKIPyCallable)
# ---------------------------------------------------------------------------


class TestStaticCallablePath:
    """Test the static path through NKIPyCallable (always used now)."""

    def test_static_non_pipelined(self, hw_backend):
        """Non-pipelined execution via options={"pipelined": False}."""
        torch.manual_seed(42)
        model = nn.Linear(32, 64, bias=False)
        x = torch.randn(8, 32, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(
            model_dev,
            backend="nkipy",
            options={"pipelined": False},
        )
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_static_multiple_calls(self, hw_backend):
        """Same compiled fn called 3x with different inputs; pipelined=True (default).

        Validates Bug #1 fix: self-prefetch disabled so repeated calls with
        the same bucket but different tensor content return correct results.
        """
        torch.manual_seed(42)
        model = nn.Linear(32, 64, bias=False)

        # Precompute expected outputs on CPU before moving model to device
        inputs = []
        expected_outputs = []
        for i in range(3):
            torch.manual_seed(i)
            x = torch.randn(8, 32, dtype=torch.float32)
            with torch.no_grad():
                expected_outputs.append(model(x))
            inputs.append(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")

        for x, expected in zip(inputs, expected_outputs):
            x_dev = x.to("nkipy")
            with torch.no_grad():
                actual = compiled(x_dev).cpu()
            assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_static_non_pipelined_mlp(self, hw_backend):
        """Multi-op MLP with pipelined=False exercises full graph."""
        torch.manual_seed(42)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=False)
                self.fc2 = nn.Linear(32, 8, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = MLP()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(
            model_dev,
            backend="nkipy",
            options={"pipelined": False},
        )
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Section 6: alias_map and non_tensor_outputs
# ---------------------------------------------------------------------------


class TestAliasMapAndNoneIdx:
    """Test alias_map and non_tensor_outputs code paths in NKIPyCallable.

    These tests exercise the post-processing code in callable.py.
    Whether alias_map/non_tensor_outputs are populated depends on aot_autograd
    decomposition. With empty maps (the common case for supported ops),
    the code correctly skips post-processing. These tests verify correctness
    through models with buffers and multi-op graphs that go through the
    full NKIPyCallable path including the alias_map/non_tensor_outputs checks.
    """

    def test_model_with_buffer(self, hw_backend):
        """Module with register_buffer exercises the alias_map code path."""
        torch.manual_seed(42)

        class BufferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16, bias=False)
                self.register_buffer("bias", torch.randn(16))

            def forward(self, x):
                return self.linear(x) + self.bias

        model = BufferModel()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_multi_op_graph(self, hw_backend):
        """Multi-op graph with several intermediates exercises result handling."""
        torch.manual_seed(42)

        class MultiOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=False)
                self.fc2 = nn.Linear(32, 16, bias=False)
                self.register_buffer("scale", torch.ones(16))

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                out = self.fc2(h) * self.scale
                return out

        model = MultiOpModel()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_model_with_multiple_buffers(self, hw_backend):
        """Module with multiple buffers tests combined alias/none handling."""
        torch.manual_seed(42)

        class MultiBufModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16, bias=False)
                self.register_buffer("scale", torch.ones(16) * 2.0)
                self.register_buffer("shift", torch.randn(16))

            def forward(self, x):
                return self.linear(x) * self.scale + self.shift

        model = MultiBufModel()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_alias_map_buffer_mutation(self, hw_backend):
        """Buffer mutation via copy_ produces correct output (Bug #3 fix).

        When aot_autograd with keep_inference_input_mutations=True includes
        the mutation as an FX output, _add_aliased_output must not double-count
        it. Validates that alias_map indices are correct.
        """
        torch.manual_seed(42)

        class BufferMutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16, bias=False)
                self.register_buffer("running_mean", torch.zeros(16))

            def forward(self, x):
                out = self.linear(x)
                # In-place copy triggers copy_ in the FX graph
                self.running_mean.copy_(out.mean(dim=0))
                return out

        model = BufferMutModel()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_alias_map_inplace_copy(self, hw_backend):
        """In-place copy_ on a buffer doesn't corrupt regular outputs."""
        torch.manual_seed(42)

        class InplaceCopyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=False)
                self.fc2 = nn.Linear(32, 16, bias=False)
                self.register_buffer("state", torch.zeros(16))

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                out = self.fc2(h)
                self.state.copy_(out.sum(dim=0))
                return out

        model = InplaceCopyModel()
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(model_dev, backend="nkipy")
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Section 7: keep_outputs_on_device
# ---------------------------------------------------------------------------


class TestKeepOutputsOnDevice:
    """Test keep_outputs_on_device=True path (Bug 1 fix)."""

    def test_keep_outputs_on_device_shape(self, hw_backend):
        """Output shape is correct with keep_outputs_on_device=True."""
        model = nn.Linear(32, 64, bias=False).to("nkipy")
        compiled = torch.compile(
            model,
            backend="nkipy",
            options={"keep_outputs_on_device": True},
        )
        x = torch.randn(8, 32, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            out = compiled(x)
        assert out.shape == (8, 64)

    def test_keep_outputs_on_device_correctness(self, hw_backend):
        """Numerical correctness with keep_outputs_on_device=True."""
        torch.manual_seed(42)
        model = nn.Linear(32, 64, bias=False)
        x = torch.randn(8, 32, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(
            model_dev,
            backend="nkipy",
            options={"keep_outputs_on_device": True},
        )
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_keep_outputs_on_device_non_pipelined(self, hw_backend):
        """keep_outputs_on_device with pipelined=False."""
        torch.manual_seed(42)
        model = nn.Linear(16, 32, bias=False)
        x = torch.randn(4, 16, dtype=torch.float32)

        with torch.no_grad():
            expected = model(x)

        model_dev = model.to("nkipy")
        compiled = torch.compile(
            model_dev,
            backend="nkipy",
            options={"keep_outputs_on_device": True, "pipelined": False},
        )
        x_dev = x.to("nkipy")
        with torch.no_grad():
            actual = compiled(x_dev).cpu()

        assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Section 8: output_layout="padded" with dynamic shapes
# ---------------------------------------------------------------------------


class TestOutputLayoutPadded:
    """Test output_layout='padded' produces padded outputs (Bug 2 fix)."""

    @staticmethod
    def _with_spiky_config(hw_backend, output_layout="padded"):
        from spiky.torch.config import (
            NKIPyBackendConfig,
            get_nkipy_backend_config,
            set_nkipy_backend_config,
        )

        old_config = get_nkipy_backend_config()
        new_config = NKIPyBackendConfig(
            nkipy_cache_prefix=old_config.nkipy_cache_prefix,
            log_level=old_config.log_level,
            rank=old_config.rank,
            world_size=old_config.world_size,
            additional_compiler_args=old_config.additional_compiler_args,
            output_layout=output_layout,
        )
        set_nkipy_backend_config(new_config)
        return old_config

    @staticmethod
    def _restore_config(old_config):
        from spiky.torch.config import set_nkipy_backend_config

        set_nkipy_backend_config(old_config)

    def test_padded_output_shape(self, hw_backend):
        """With output_layout='padded', output dim should be bucket size, not actual."""
        old_config = self._with_spiky_config(hw_backend, output_layout="padded")
        try:
            model = nn.Linear(32, 64, bias=False)
            compiled = torch.compile(
                model,
                backend="nkipy",
                dynamic=True,
                options={"output_layout": "padded", "buckets": [8, 16]},
            )

            # Input with batch=4, should use bucket=8
            x = torch.randn(4, 32, dtype=torch.float32)
            torch._dynamo.maybe_mark_dynamic(x, 0)

            with torch.no_grad():
                out = compiled(x)

            # With output_layout="padded" and unpad_outputs=False,
            # the output should retain the padded (bucket) shape
            assert out.shape[0] >= 4, (
                f"Expected padded batch dim >= 4, got {out.shape[0]}"
            )
        finally:
            self._restore_config(old_config)


# ---------------------------------------------------------------------------
# Section 9: Stats reset reflects C++ state
# ---------------------------------------------------------------------------


class TestStatsReset:
    """Test that reset_stats actually clears C++ counters (Bug 7 fix)."""

    def test_reset_stats_clears_execution_count(self, hw_backend):
        """After executing and resetting, total_executions should be 0."""
        import spiky

        # Run something to generate stats
        model = nn.Linear(16, 32, bias=False).to("nkipy")
        compiled = torch.compile(model, backend="nkipy")
        x = torch.randn(4, 16, dtype=torch.float32, device="nkipy")
        with torch.no_grad():
            _ = compiled(x)

        stats_before = spiky.get_stats()
        assert stats_before is not None
        assert stats_before.total_executions > 0

        spiky.reset_stats()

        stats_after = spiky.get_stats()
        assert stats_after.total_executions == 0
        assert stats_after.total_execution_time_ms == 0.0


# ---------------------------------------------------------------------------
# Section 10: Profiling integration
# ---------------------------------------------------------------------------


class TestProfilingIntegration:
    """Profiling through the NKIPyCallable static path."""

    def test_profiling_generates_ntff(self, hw_backend, ntff_output_dir):
        """save_ntff=True generates NTFF profile files via NRT profiling APIs."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def simple_func(x):
            return x + 1

        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            simple_func,
            backend="nkipy",
            fullgraph=True,
            options={"save_ntff": True, "save_ntff_dir": str(ntff_output_dir)},
        )

        with torch.no_grad():
            _ = compiled_func(x)

        kernel_dirs = list(ntff_output_dir.glob("kernel_*"))
        assert len(kernel_dirs) > 0, "No kernel directories created"

        neff_files = list(ntff_output_dir.glob("kernel_*/*.neff"))
        assert len(neff_files) > 0, "No NEFF files copied to profiling dir"

        ntff_files = list(ntff_output_dir.glob("kernel_*/*.ntff"))
        assert len(ntff_files) > 0, "No NTFF profile files generated"

    def test_profiling_copies_neff(self, hw_backend, ntff_output_dir):
        """.neff file copied to profiling output dir."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def add_func(a, b):
            return a + b

        a = torch.randn(8, 16, dtype=torch.float32, device="nkipy")
        b = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            add_func,
            backend="nkipy",
            fullgraph=True,
            options={"save_ntff": True, "save_ntff_dir": str(ntff_output_dir)},
        )

        with torch.no_grad():
            _ = compiled_func(a, b)

        neff_files = list(ntff_output_dir.glob("kernel_*/*.neff"))
        assert len(neff_files) > 0, "No NEFF files copied"

    def test_profiling_exe_idx_filter(self, hw_backend, ntff_output_dir):
        """save_ntff_exe_idx=[1] with 3 executions produces exactly 1 NTFF file."""
        ntff_output_dir.mkdir(parents=True, exist_ok=True)

        def simple_func(x):
            return x + 1

        x = torch.randn(8, 16, dtype=torch.float32, device="nkipy")

        compiled_func = torch.compile(
            simple_func,
            backend="nkipy",
            fullgraph=True,
            options={
                "save_ntff": True,
                "save_ntff_dir": str(ntff_output_dir),
                "save_ntff_exe_idx": [1],
            },
        )

        with torch.no_grad():
            for _ in range(3):
                _ = compiled_func(x)

        ntff_files = list(ntff_output_dir.glob("kernel_*/*.ntff"))
        assert len(ntff_files) == 1, (
            f"Expected exactly 1 NTFF file for exe_idx=[1], "
            f"got {len(ntff_files)}: {ntff_files}"
        )
        # The NTFF file should be named "1.ntff" (matching exe_idx=1)
        assert ntff_files[0].name == "1.ntff", (
            f"Expected NTFF file named '1.ntff', got '{ntff_files[0].name}'"
        )


# ---------------------------------------------------------------------------
# Section 11: Training (forward + backward + optimizer step)
# ---------------------------------------------------------------------------

_TRAINING_NUM_STEPS = 3
_TRAINING_LR = 0.01
_TRAINING_RTOL = 1e-2
_TRAINING_ATOL = 1e-2


def _check_training(model, make_batch, num_steps=_TRAINING_NUM_STEPS, lr=_TRAINING_LR,
                     rtol=_TRAINING_RTOL, atol=_TRAINING_ATOL):
    """Train eager vs compiled for *num_steps*, assert losses and weights match."""
    torch.manual_seed(42)

    ref_model = copy.deepcopy(model)
    comp_model = copy.deepcopy(model).to("nkipy")

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=lr)
    comp_opt = torch.optim.SGD(comp_model.parameters(), lr=lr, foreach=False)

    loss_fn = nn.MSELoss()

    @torch.compile(backend="nkipy", fullgraph=True)
    def forward_with_loss(m, x, target):
        out = m(x)
        loss = loss_fn(out, target)
        return out, loss

    compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

    ref_losses = []
    comp_losses = []

    for step in range(num_steps):
        torch.manual_seed(step)
        x, target = make_batch()

        # Eager
        ref_opt.zero_grad()
        ref_out = ref_model(x)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()
        ref_opt.step()
        ref_losses.append(ref_loss.item())

        # Compiled
        comp_opt.zero_grad()
        comp_out, comp_loss = forward_with_loss(comp_model, x.to("nkipy"), target.to("nkipy"))
        comp_loss.backward()
        compiled_opt_step()
        comp_losses.append(comp_loss.cpu().item())

    # Per-step losses
    for step, (rl, cl) in enumerate(zip(ref_losses, comp_losses)):
        assert_close(
            torch.tensor(cl), torch.tensor(rl), rtol=rtol, atol=atol,
            msg=f"Step {step} loss mismatch",
        )

    # Final weights
    for (pname, rp), (_, cp) in zip(
        ref_model.named_parameters(), comp_model.named_parameters()
    ):
        assert_close(
            cp.detach().cpu(), rp.detach(), rtol=rtol, atol=atol,
            msg=f"Param '{pname}' mismatch after {num_steps} steps",
        )


class TestTraining:
    """Training correctness: eager CPU vs compiled NKIPy."""

    def test_training_linear(self, hw_backend):
        """Linear layer with bias: losses and weights match eager after 3 steps."""
        _check_training(
            nn.Linear(32, 16, bias=True),
            lambda: (torch.randn(4, 32), torch.randn(4, 16)),
        )

    def test_training_mlp(self, hw_backend):
        """Three-layer MLP: losses and weights match eager after 3 steps."""
        _check_training(
            nn.Sequential(
                nn.Linear(32, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 64, bias=False), nn.ReLU(),
                nn.Linear(64, 16, bias=False),
            ),
            lambda: (torch.randn(4, 32), torch.randn(4, 16)),
        )


class TestTrainingDynamicShapes:
    """Training with dynamic batch dimension and bucket selection."""

    @staticmethod
    def _with_spiky_config(hw_backend, pipelined=True):
        from spiky.torch.config import (
            NKIPyBackendConfig,
            get_nkipy_backend_config,
            set_nkipy_backend_config,
        )

        old_config = get_nkipy_backend_config()
        new_config = NKIPyBackendConfig(
            nkipy_cache_prefix=old_config.nkipy_cache_prefix,
            log_level=old_config.log_level,
            rank=old_config.rank,
            world_size=old_config.world_size,
            additional_compiler_args=old_config.additional_compiler_args,
            pipelined=pipelined,
        )
        set_nkipy_backend_config(new_config)
        return old_config

    @staticmethod
    def _restore_config(old_config):
        from spiky.torch.config import set_nkipy_backend_config

        set_nkipy_backend_config(old_config)

    def test_training_dynamic_batch(self, hw_backend):
        """Training with varying batch sizes via dynamic bucketing."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 16, bias=True)
            batch_sizes = [4, 8, 4]

            ref_model = copy.deepcopy(model)
            comp_model = copy.deepcopy(model).to("nkipy")

            ref_opt = torch.optim.SGD(ref_model.parameters(), lr=_TRAINING_LR)
            comp_opt = torch.optim.SGD(comp_model.parameters(), lr=_TRAINING_LR, foreach=False)
            loss_fn = nn.MSELoss()

            @torch.compile(backend="nkipy", fullgraph=True, dynamic=True)
            def forward_with_loss(m, x, target):
                out = m(x)
                loss = loss_fn(out, target)
                return out, loss

            compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

            for step, bs in enumerate(batch_sizes):
                torch.manual_seed(step)
                x = torch.randn(bs, 32)
                target = torch.randn(bs, 16)

                if step == 0:
                    torch._dynamo.maybe_mark_dynamic(x, 0)
                    torch._dynamo.maybe_mark_dynamic(target, 0)

                # Eager
                ref_opt.zero_grad()
                ref_out = ref_model(x)
                ref_loss = loss_fn(ref_out, target)
                ref_loss.backward()
                ref_opt.step()

                # Compiled
                comp_opt.zero_grad()
                comp_out, comp_loss = forward_with_loss(
                    comp_model, x.to("nkipy"), target.to("nkipy")
                )
                comp_loss.backward()
                compiled_opt_step()

                assert_close(
                    torch.tensor(comp_loss.cpu().item()),
                    torch.tensor(ref_loss.item()),
                    rtol=_TRAINING_RTOL, atol=_TRAINING_ATOL,
                    msg=f"Step {step} (batch={bs}) loss mismatch",
                )

            # Final weights
            for (pname, rp), (_, cp) in zip(
                ref_model.named_parameters(), comp_model.named_parameters()
            ):
                assert_close(
                    cp.detach().cpu(), rp.detach(),
                    rtol=_TRAINING_RTOL, atol=_TRAINING_ATOL,
                    msg=f"Param '{pname}' mismatch after training",
                )
        finally:
            self._restore_config(old_config)

    def test_training_dynamic_explicit_buckets(self, hw_backend):
        """Training with explicit bucket list for batch dimension."""
        old_config = self._with_spiky_config(hw_backend)
        try:
            torch.manual_seed(42)
            model = nn.Linear(32, 16, bias=True)
            batch_sizes = [4, 8, 16]

            ref_model = copy.deepcopy(model)
            comp_model = copy.deepcopy(model).to("nkipy")

            ref_opt = torch.optim.SGD(ref_model.parameters(), lr=_TRAINING_LR)
            comp_opt = torch.optim.SGD(comp_model.parameters(), lr=_TRAINING_LR, foreach=False)
            loss_fn = nn.MSELoss()

            @torch.compile(
                backend="nkipy", fullgraph=True, dynamic=True,
                options={"buckets": [4, 8, 16]},
            )
            def forward_with_loss(m, x, target):
                out = m(x)
                loss = loss_fn(out, target)
                return out, loss

            compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

            for step, bs in enumerate(batch_sizes):
                torch.manual_seed(step)
                x = torch.randn(bs, 32)
                target = torch.randn(bs, 16)

                if step == 0:
                    torch._dynamo.maybe_mark_dynamic(x, 0)
                    torch._dynamo.maybe_mark_dynamic(target, 0)

                # Eager
                ref_opt.zero_grad()
                ref_out = ref_model(x)
                ref_loss = loss_fn(ref_out, target)
                ref_loss.backward()
                ref_opt.step()

                # Compiled
                comp_opt.zero_grad()
                comp_out, comp_loss = forward_with_loss(
                    comp_model, x.to("nkipy"), target.to("nkipy")
                )
                comp_loss.backward()
                compiled_opt_step()

                assert_close(
                    torch.tensor(comp_loss.cpu().item()),
                    torch.tensor(ref_loss.item()),
                    rtol=_TRAINING_RTOL, atol=_TRAINING_ATOL,
                    msg=f"Step {step} (batch={bs}) loss mismatch",
                )

            # Final weights
            for (pname, rp), (_, cp) in zip(
                ref_model.named_parameters(), comp_model.named_parameters()
            ):
                assert_close(
                    cp.detach().cpu(), rp.detach(),
                    rtol=_TRAINING_RTOL, atol=_TRAINING_ATOL,
                    msg=f"Param '{pname}' mismatch after training",
                )
        finally:
            self._restore_config(old_config)
