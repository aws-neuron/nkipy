# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified kernelgen backend integration.

Ported from NeuronPy/tests/unit/test_kernelgen_backend.py with adaptations
for the current nkipy implementation.
"""

import warnings

import numpy as np
import pytest

from nkipy import knob
from nkipy.core.nki_op import nki_custom_op
from nkipy.core.backend import get_backend, tracing
from nkipy.core.backend.kernelgen import KernelGenIR, KernelGenTraceContext


class TestKnobDispatch:
    """Test knob() backend-aware dispatch."""

    def test_knob_cpu_passthrough(self):
        """knob() is a no-op pass-through in cpu mode (no trace)."""
        arr = np.ones((4, 4), dtype=np.float32)
        result = knob(arr, mem_space="Sbuf")
        assert result is arr

    def test_knob_cpu_no_params(self):
        """knob() with no params is always a no-op."""
        arr = np.ones((4, 4), dtype=np.float32)
        result = knob(arr)
        assert result is arr

    def test_knob_hlo_warns(self):
        """knob() issues a warning under the HLO backend."""
        from nkipy.core.backend.hlo import HLOModule, HLOTraceContext

        code = HLOModule(name="test")
        arr = np.ones((4, 4), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with tracing(HLOTraceContext(code)):
                result = knob(arr, mem_space="Sbuf")
            assert len(w) == 1
            assert "only effective with backend='kernelgen'" in str(w[0].message)
            assert result is arr


def _make_silu_kernel_builder(M, N, tile_p=128, tile_f=128):
    """Return a real NKI kernel_builder function that computes SiLU activation."""
    def silu_kernel(input_0, output_0):
        import nki.compiler.kernel_builder as nb
        import nki.language as nl

        n_row_tiles = M // tile_p
        n_col_tiles = N // tile_f
        for r in nl.affine_range(n_row_tiles):
            for t in nl.affine_range(n_col_tiles):
                x_sbuf = nb.ndarray((tile_p, tile_f), input_0.dtype, nb.sbuf)
                nb.isa.dma_copy(
                    dst=x_sbuf,
                    src=input_0[
                        r * tile_p : (r + 1) * tile_p,
                        t * tile_f : (t + 1) * tile_f,
                    ],
                )

                out_sbuf = nb.ndarray((tile_p, tile_f), input_0.dtype, nb.sbuf)
                bias = nb.ndarray((tile_p, 1), input_0.dtype, nb.sbuf)
                nb.isa.memset(dst=bias, value=0.0)
                scale = nb.ndarray((tile_p, 1), input_0.dtype, nb.sbuf)
                nb.isa.memset(dst=scale, value=1.0)

                nb.isa.activation(
                    dst=out_sbuf,
                    src=x_sbuf,
                    bias=bias,
                    scale=scale,
                    op=nb.isa.activation_function.silu,
                )

                nb.isa.dma_copy(
                    dst=output_0[
                        r * tile_p : (r + 1) * tile_p,
                        t * tile_f : (t + 1) * tile_f,
                    ],
                    src=out_sbuf,
                )
    return silu_kernel


class TestNKICustomOpDispatch:
    """Test nki_custom_op() factory and dispatch."""

    def test_requires_at_least_one(self):
        """nki_custom_op raises if neither nki_kernel nor kernel_builder given."""
        with pytest.raises(ValueError, match="At least one"):
            nki_custom_op()

    def test_kernel_builder_requires_specs(self):
        """nki_custom_op raises if kernel_builder given without specs."""
        with pytest.raises(ValueError, match="input_specs and output_specs"):
            nki_custom_op(kernel_builder=lambda: None)

    def test_cpu_raises(self):
        """nki_custom_op raises on cpu backend."""
        op = nki_custom_op(
            kernel_builder=lambda: None,
            input_specs=[((4, 4), "f32")],
            output_specs=[((4, 4), "f32")],
        )
        with pytest.raises(RuntimeError, match="not supported on backend 'cpu'"):
            op(np.ones((4, 4), dtype=np.float32))

    def test_hlo_without_nki_kernel_raises(self):
        """nki_custom_op with only kernel_builder raises on HLO."""

        class _FakeHLOCtx:
            backend_name = "hlo"

        op = nki_custom_op(
            kernel_builder=lambda: None,
            input_specs=[((128, 128), "f32")],
            output_specs=[((128, 128), "f32")],
        )
        with tracing(_FakeHLOCtx()):
            with pytest.raises(RuntimeError, match="no nki_kernel"):
                op(np.ones((128, 128), dtype=np.float32))

    def test_kernelgen_without_kernel_builder_raises(self):
        """nki_custom_op with only nki_kernel raises on kernelgen."""

        class _FakeKernelgenCtx:
            backend_name = "kernelgen"

        op = nki_custom_op(nki_kernel=lambda: None)
        with tracing(_FakeKernelgenCtx()):
            with pytest.raises(RuntimeError, match="no kernel_builder"):
                op(np.ones((128, 128), dtype=np.float32))


class TestKernelGenTraceContext:
    """Test KernelGenTraceContext basics."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_kernelgen(self):
        try:
            import nkipy_kernelgen  # noqa: F401
        except ImportError:
            pytest.skip("nkipy-kernelgen not installed")

    def test_backend_name(self):
        ctx = KernelGenTraceContext()
        assert ctx.backend_name == "kernelgen"
        ctx._cleanup()

    def test_tracing_context_activates(self):
        ctx = KernelGenTraceContext()
        assert get_backend() == "cpu"
        with tracing(ctx):
            assert get_backend() == "kernelgen"
        assert get_backend() == "cpu"
        ctx._cleanup()


class TestSpecializeKernelgen:
    """Test NKIPyKernel._specialize_kernelgen with device compilation and execution."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_kernelgen(self):
        try:
            import nkipy_kernelgen  # noqa: F401
        except ImportError:
            pytest.skip("nkipy-kernelgen not installed")

    @staticmethod
    def _run(func, *np_args):
        from utils import NEURON_AVAILABLE, on_device_test, trace_and_compile
        if NEURON_AVAILABLE:
            return on_device_test(func, "kernelgen", *np_args)
        else:
            trace_and_compile(func, "kernelgen", *np_args)
            return None

    def test_with_knob(self):
        from utils import baremetal_assert_allclose

        def kernel_with_knob(a, b):
            result = a + b
            knob(result, mem_space="SharedHbm", tile_size=[128, 128])
            return result

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        result = self._run(kernel_with_knob, a, b)
        if result is not None:
            baremetal_assert_allclose(result, a + b)

    def test_multi_output(self):
        from utils import baremetal_assert_allclose

        def multi_out(a, b):
            return a + b, a - b

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        result = self._run(multi_out, a, b)
        if result is not None:
            baremetal_assert_allclose(result[0], a + b)
            baremetal_assert_allclose(result[1], a - b)

    def test_dtype_downcast(self):
        """float64 inputs should be auto-downcast to float32."""
        from nkipy.core.trace import NKIPyKernel

        def add_kernel(a, b):
            return a + b

        kernel = NKIPyKernel.trace(add_kernel, backend="kernelgen")
        a = np.random.randn(64, 64)  # float64
        b = np.random.randn(64, 64)  # float64

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ir = kernel.specialize(a, b)
        assert ir.inputs[0].dtype == np.dtype("float32")

    @pytest.mark.xfail(
        reason="custom_op kernel_builder tracing requires TracedArray, "
               "not yet wired through NKIPyTensorRef path"
    )
    def test_custom_op_with_kernel_builder(self):
        """nki_custom_op with real kernel_builder traces through kernelgen backend."""
        from nkipy.core.trace import NKIPyKernel

        silu_op = nki_custom_op(
            kernel_builder=_make_silu_kernel_builder(256, 256),
            input_specs=[((256, 256), "f32")],
            output_specs=[((256, 256), "f32")],
        )

        def kernel(x):
            return silu_op(x)

        k = NKIPyKernel.trace(kernel, backend="kernelgen")
        ir = k.specialize(np.random.randn(256, 256).astype("float32"))
        assert isinstance(ir, KernelGenIR)
        assert "__custom_op__silu_kernel" in ir._mlir_text
        assert "nkipy.custom_op_bodies" in ir._mlir_text


class TestKernelgenInplaceUpdate:
    """Test in-place update (dynamic_update_slice) support for kernelgen.

    Each test traces → compiles → runs on device and compares
    numerical results against NumPy.  Alias metadata is verified as well.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_kernelgen(self):
        try:
            import nkipy_kernelgen  # noqa: F401
        except ImportError:
            pytest.skip("nkipy-kernelgen not installed")

    @staticmethod
    def _trace_and_run(func, *np_args):
        """Trace a kernelgen kernel, return (ir, device_result_or_None)."""
        from nkipy.core.trace import NKIPyKernel
        from utils import NEURON_AVAILABLE, on_device_test, trace_and_compile

        kernel = NKIPyKernel.trace(func, backend="kernelgen")
        ir = kernel.specialize(*np_args)
        if NEURON_AVAILABLE:
            result = on_device_test(func, "kernelgen", *np_args)
        else:
            trace_and_compile(func, "kernelgen", *np_args)
            result = None
        return ir, result

    def test_single_alias(self):
        """Mutate one parameter and return it — verify numerical result."""
        from utils import baremetal_assert_allclose

        def kernel(a, b):
            a[0:1, :] = b[1:2, :]
            return a

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        expected = a.copy()
        expected[0:1, :] = b[1:2, :]

        ir, result = self._trace_and_run(kernel, a, b)
        if result is not None:
            baremetal_assert_allclose(result, expected)

        assert isinstance(ir, KernelGenIR)
        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].param_index == 0
        assert ir.aliases[0].is_user_returned is True
        assert ir.auto_aliased_indices == set()

    def test_multi_slice_update(self):
        """Update multiple disjoint slices of the same tensor."""
        from utils import baremetal_assert_allclose

        def kernel(a, b):
            a[0:2, :] = b[0:2, :]
            a[4:6, :] = b[4:6, :]
            return a

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        expected = a.copy()
        expected[0:2, :] = b[0:2, :]
        expected[4:6, :] = b[4:6, :]

        ir, result = self._trace_and_run(kernel, a, b)
        if result is not None:
            baremetal_assert_allclose(result, expected)
        assert len(ir.aliases) == 1

    def test_multi_alias(self):
        """Mutate two parameters and return both."""
        from utils import baremetal_assert_allclose

        def kernel(a, b, c):
            a[0:1, :] = b[0:1, :]
            c[2:3, :] = b[2:3, :]
            return a, c

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")
        c = np.random.randn(128, 128).astype("float32")

        expected_a = a.copy()
        expected_a[0:1, :] = b[0:1, :]
        expected_c = c.copy()
        expected_c[2:3, :] = b[2:3, :]

        ir, result = self._trace_and_run(kernel, a, b, c)
        if result is not None:
            baremetal_assert_allclose(result[0], expected_a)
            baremetal_assert_allclose(result[1], expected_c)

        assert len(ir.aliases) == 2
        alias_names = {al.param_name for al in ir.aliases}
        assert alias_names == {"a", "c"}
        assert all(al.is_user_returned for al in ir.aliases)

    def test_no_return_auto_alias(self):
        """Mutate without returning — auto-append to outputs."""
        from utils import baremetal_assert_allclose

        def kernel(a, b):
            a[0:1, :] = b[1:2, :]

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        expected = a.copy()
        expected[0:1, :] = b[1:2, :]

        ir, result = self._trace_and_run(kernel, a, b)
        if result is not None:
            baremetal_assert_allclose(result, expected)

        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].is_user_returned is False
        assert ir.auto_aliased_indices == {0}

    def test_mixed_return_alias(self):
        """Mutate a parameter but return a different computed value."""
        from utils import baremetal_assert_allclose

        def kernel(a, b):
            a[0:1, :] = b[1:2, :]
            return a + b

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        expected_a = a.copy()
        expected_a[0:1, :] = b[1:2, :]
        expected_sum = expected_a + b

        ir, result = self._trace_and_run(kernel, a, b)
        if result is not None:
            baremetal_assert_allclose(result, expected_sum)

        assert len(ir.aliases) == 1
        assert ir.aliases[0].param_name == "a"
        assert ir.aliases[0].is_user_returned is False
        assert len(ir.outputs) == 2
        assert ir.auto_aliased_indices == {1}

    def test_update_with_computation(self):
        """Assign a computed expression into a slice."""
        from utils import baremetal_assert_allclose

        def kernel(a, b):
            a[0:2, :] = b[0:2, :] * 2.0
            return a

        a = np.random.randn(128, 128).astype("float32")
        b = np.random.randn(128, 128).astype("float32")

        expected = a.copy()
        expected[0:2, :] = b[0:2, :] * 2.0

        ir, result = self._trace_and_run(kernel, a, b)
        if result is not None:
            baremetal_assert_allclose(result, expected)


