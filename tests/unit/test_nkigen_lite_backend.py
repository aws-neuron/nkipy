# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nkigen-lite backend integration.

Tests trace kernels with backend="nkigen-lite" and verify:
- Correct graph construction
- IR metadata (inputs, outputs, aliases)
- Successful lowering through the nkigen_lite pass pipeline
"""

import warnings

import numpy as np
import pytest

from nkipy.core.backend import get_backend, tracing
from nkipy.core.backend.nkigen_lite import (
    NkiGenLiteIR,
    NkiGenLiteTraceContext,
)
from nkipy.core.trace import NKIPyKernel


class TestNkiGenLiteTraceContext:
    """Test NkiGenLiteTraceContext basics."""

    def test_backend_name(self):
        ctx = NkiGenLiteTraceContext()
        assert ctx.backend_name == "nkigen-lite"

    def test_tracing_context_activates(self):
        ctx = NkiGenLiteTraceContext()
        assert get_backend() == "cpu"
        with tracing(ctx):
            assert get_backend() == "nkigen-lite"
        assert get_backend() == "cpu"

    def test_add_parameter(self):
        ctx = NkiGenLiteTraceContext()
        pt = ctx.add_parameter((4, 4), np.float32, name="x")
        assert pt.shape == (4, 4)
        assert pt.dtype == np.float32
        assert pt.is_parameter is True
        assert pt.parameter_id == 0
        assert pt.name == "x"


class TestSpecializeNkigenLite:
    """Test NKIPyKernel._specialize_nkigen_lite tracing."""

    def test_add(self):
        def kernel(a, b):
            return np.add(a, b)

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        ir = k.specialize(a, b)

        assert isinstance(ir, NkiGenLiteIR)
        assert len(ir.inputs) == 2
        assert ir.inputs[0].shape == (64, 64)
        assert ir.inputs[0].dtype == np.float32
        assert len(ir.outputs) == 1
        assert ir.outputs[0].shape == (64, 64)

    def test_matmul(self):
        def kernel(a, b):
            return a @ b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(64, 128).astype(np.float32)
        b = np.random.randn(128, 32).astype(np.float32)
        ir = k.specialize(a, b)

        assert ir.outputs[0].shape == (64, 32)

    def test_multi_output(self):
        def kernel(a, b):
            return a + b, a - b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        ir = k.specialize(a, b)

        assert len(ir.outputs) == 2
        assert ir.outputs[0].name == "output_0_out"
        assert ir.outputs[1].name == "output_1_out"

    def test_dtype_downcast(self):
        """float64 inputs should be auto-downcast to float32."""
        def kernel(a, b):
            return a + b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(32, 32)  # float64
        b = np.random.randn(32, 32)  # float64

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ir = k.specialize(a, b)
        assert ir.inputs[0].dtype == np.dtype("float32")
        assert len(w) == 2  # one warning per input

    def test_softmax(self):
        def kernel(x):
            m = np.max(x, axis=1, keepdims=True)
            shifted = x - m
            e = np.exp(shifted)
            s = np.sum(e, axis=1, keepdims=True)
            return e / s

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(128, 512).astype(np.float32)
        ir = k.specialize(x)

        assert ir.outputs[0].shape == (128, 512)

    def test_unary_ops(self):
        def kernel(x):
            a = np.exp(x)
            b = np.log(a)
            c = np.sqrt(np.abs(b))
            return np.tanh(c)

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(32, 32).astype(np.float32)
        ir = k.specialize(x)
        assert ir.outputs[0].shape == (32, 32)

    def test_transpose(self):
        def kernel(x):
            return np.transpose(x, (1, 0))

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(64, 128).astype(np.float32)
        ir = k.specialize(x)
        assert ir.outputs[0].shape == (128, 64)

    def test_reshape(self):
        def kernel(x):
            return np.reshape(x, (32, 128))

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(64, 64).astype(np.float32)
        ir = k.specialize(x)
        assert ir.outputs[0].shape == (32, 128)

    def test_concatenate(self):
        def kernel(a, b):
            return np.concatenate([a, b], axis=0)

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)
        ir = k.specialize(a, b)
        assert ir.outputs[0].shape == (64, 64)

    def test_broadcast(self):
        def kernel(a, b):
            # a: (4, 1), b: (1, 8) -> (4, 8)
            return a + b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(4, 1).astype(np.float32)
        b = np.random.randn(1, 8).astype(np.float32)
        ir = k.specialize(a, b)
        assert ir.outputs[0].shape == (4, 8)

    def test_scalar_arithmetic(self):
        def kernel(x):
            return x * 2.0 + 1.0

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(32, 32).astype(np.float32)
        ir = k.specialize(x)
        assert ir.outputs[0].shape == (32, 32)

    def test_content_hash(self):
        def kernel(a, b):
            return a + b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        ir = k.specialize(a, b)

        h1 = ir.content_hash("")
        h2 = ir.content_hash("--opt-level=2")
        assert len(h1) == 12
        assert h1 != h2


class TestNkigenLiteInplaceUpdate:
    """Test in-place update (dynamic_update_slice) for nkigen-lite."""

    def test_single_alias(self):
        def kernel(a, b):
            a[0:2, :] = b[0:2, :]
            return a

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(8, 4).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        ir = k.specialize(a, b)

        assert isinstance(ir, NkiGenLiteIR)
        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].param_index == 0
        assert ir.aliases[0].is_user_returned is True
        assert ir.auto_aliased_indices == set()

    def test_no_return_auto_alias(self):
        def kernel(a, b):
            a[0:2, :] = b[0:2, :]

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(8, 4).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        ir = k.specialize(a, b)

        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].is_user_returned is False
        assert ir.auto_aliased_indices == {0}

    def test_multi_alias(self):
        def kernel(a, b, c):
            a[0:1, :] = b[0:1, :]
            c[2:3, :] = b[2:3, :]
            return a, c

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(8, 4).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        c = np.random.randn(8, 4).astype(np.float32)
        ir = k.specialize(a, b, c)

        assert len(ir.aliases) == 2
        alias_names = {al.param_name for al in ir.aliases}
        assert alias_names == {"a", "c"}
        assert all(al.is_user_returned for al in ir.aliases)

    def test_mixed_return_alias(self):
        """Mutate a parameter but return a different computed value."""
        def kernel(a, b):
            a[0:1, :] = b[1:2, :]
            return a + b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(8, 4).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        ir = k.specialize(a, b)

        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].is_user_returned is False
        assert len(ir.outputs) == 2
        assert ir.auto_aliased_indices == {1}


class TestNkigenLiteLowering:
    """Test that traced IR can be lowered through the nkigen_lite pipeline."""

    @staticmethod
    def _lower(ir):
        from nkigen_lite.tensor_ir.passes import lower_to_nki
        return lower_to_nki(ir._graph)

    def test_add_lowers(self):
        def kernel(a, b):
            return a + b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(128, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        ir = k.specialize(a, b)

        nki_graph = self._lower(ir)
        assert len(nki_graph.ops) > 0
        assert len(nki_graph.inputs) >= 2

    def test_matmul_lowers(self):
        def kernel(a, b):
            return a @ b

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        a = np.random.randn(128, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        ir = k.specialize(a, b)

        nki_graph = self._lower(ir)
        assert len(nki_graph.ops) > 0

    def test_softmax_lowers(self):
        def kernel(x):
            m = np.max(x, axis=1, keepdims=True)
            shifted = x - m
            e = np.exp(shifted)
            s = np.sum(e, axis=1, keepdims=True)
            return e / s

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(128, 512).astype(np.float32)
        ir = k.specialize(x)

        nki_graph = self._lower(ir)
        assert len(nki_graph.ops) > 0

    def test_layer_norm_lowers(self):
        def kernel(x, gamma, beta):
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            normalized = (x - mean) / np.sqrt(var + 1e-5)
            return normalized * gamma + beta

        k = NKIPyKernel.trace(kernel, backend="nkigen-lite")
        x = np.random.randn(128, 512).astype(np.float32)
        gamma = np.ones((1, 512), dtype=np.float32)
        beta = np.zeros((1, 512), dtype=np.float32)
        ir = k.specialize(x, gamma, beta)

        nki_graph = self._lower(ir)
        assert len(nki_graph.ops) > 0


class TestKnobDispatch:
    """Test knob() warns under nkigen-lite backend."""

    def test_knob_nkigen_lite_warns(self):
        from nkipy import knob

        ctx = NkiGenLiteTraceContext()
        arr = np.ones((4, 4), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with tracing(ctx):
                result = knob(arr, mem_space="Sbuf")
            assert len(w) == 1
            assert "only effective with backend='nkigen'" in str(w[0].message)
            assert result is arr
