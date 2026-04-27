# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the kernelgen backend integration.

Each test traces a kernel with backend="kernelgen", compiles to NEFF,
runs on Neuron device, and compares the numerical result against NumPy.
When no device is available, falls back to compile-only validation.
"""

import numpy as np
import pytest

try:
    import nkipy_kernelgen  # noqa: F401

    HAS_KERNELGEN = True
except ImportError:
    HAS_KERNELGEN = False

from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    on_device_test,
    trace_and_compile,
)

pytestmark = pytest.mark.skipif(
    not HAS_KERNELGEN, reason="nkipy-kernelgen not installed"
)

TRACE_MODE = "kernelgen"


def _run_kernel(kernel_fn, *args):
    """Run a kernel on device if available, else compile-only. Returns result or None."""
    if NEURON_AVAILABLE:
        return on_device_test(kernel_fn, TRACE_MODE, *args)
    else:
        trace_and_compile(kernel_fn, TRACE_MODE, *args)
        return None


class TestKernelGenBasicOps:
    """Test basic arithmetic operations compile and run correctly."""

    def test_add(self):
        def kernel(A, B):
            return np.add(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A + B)

    def test_subtract(self):
        def kernel(A, B):
            return np.subtract(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A - B)

    def test_multiply(self):
        def kernel(A, B):
            return np.multiply(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A * B)

    def test_scalar_add(self):
        def kernel(A):
            return A + 1.0

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, A + 1.0)

    def test_matmul(self):
        def kernel(A, B):
            return np.matmul(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A @ B)

    def test_matmul_batched(self):
        def kernel(A, B):
            return np.matmul(A, B)

        A = np.random.randn(2, 128, 128).astype(np.float32)
        B = np.random.randn(2, 128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A @ B)


class TestKernelGenUnaryOps:
    """Test unary operations compile and run correctly."""

    def test_exp(self):
        def kernel(A):
            return np.exp(A)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.exp(A))

    def test_sqrt(self):
        def kernel(A):
            return np.sqrt(A)

        A = np.abs(np.random.randn(128, 128)).astype(np.float32) + 0.01
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.sqrt(A))

    def test_tanh(self):
        def kernel(A):
            return np.tanh(A)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.tanh(A))

    def test_negative(self):
        def kernel(A):
            return -A

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, -A)


class TestKernelGenTransformOps:
    """Test transform operations compile and run correctly."""

    @pytest.mark.xfail(reason="linalg.transpose not lowered to NISA yet", run=True, strict=False)
    def test_transpose(self):
        def kernel(A):
            return np.transpose(A)

        A = np.random.randn(128, 256).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, A.T)

    def test_reshape(self):
        def kernel(A):
            return np.reshape(A, (256, 64))

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, A.reshape(256, 64))

    def test_squeeze(self):
        def kernel(A):
            return np.squeeze(A, axis=1)

        A = np.random.randn(128, 1, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, A.squeeze(axis=1))

    @pytest.mark.xfail(reason="linalg.transpose not lowered to NISA yet", run=True, strict=False)
    def test_swapaxes(self):
        def kernel(A):
            return np.swapaxes(A, 0, 1)

        A = np.random.randn(128, 256).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.swapaxes(A, 0, 1))

    @pytest.mark.xfail(reason="tensor.insert_slice stack lowering produces incorrect NISA", run=True, strict=False)
    def test_stack(self):
        def kernel(A, B):
            return np.stack([A, B], axis=0)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, np.stack([A, B], axis=0))


class TestKernelGenReductions:
    """Test reduction operations compile and run correctly."""

    def test_sum(self):
        def kernel(A):
            return np.sum(A, axis=1, keepdims=True)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.sum(A, axis=1, keepdims=True))

    @pytest.mark.xfail(reason="mean reduction missing memory space annotation in NISA lowering", run=True, strict=False)
    def test_mean(self):
        def kernel(A):
            return np.mean(A, axis=0, keepdims=True)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.mean(A, axis=0, keepdims=True))


class TestKernelGenComparisonOps:
    """Test comparison and logical operations compile and run correctly."""

    def test_equal(self):
        def kernel(A, B):
            return np.equal(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = A.copy()
        B[::2, :] = 0.0
        result = _run_kernel(kernel, A, B)
        if result is not None:
            expected = np.equal(A, B).astype(np.float32)
            baremetal_assert_allclose(result, expected)

    def test_greater(self):
        def kernel(A, B):
            return np.greater(A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            expected = np.greater(A, B).astype(np.float32)
            baremetal_assert_allclose(result, expected)

    def test_less_scalar(self):
        def kernel(A):
            return np.less(A, 0.5)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            expected = np.less(A, 0.5).astype(np.float32)
            baremetal_assert_allclose(result, expected)

    def test_logical_not(self):
        def kernel(A):
            return np.logical_not(A)

        A = np.array(np.random.rand(128, 128) > 0.5, dtype=np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            expected = np.logical_not(A).astype(np.float32)
            baremetal_assert_allclose(result, expected)

    def test_bitwise_and(self):
        def kernel(A, B):
            return np.bitwise_and(A, B)

        A = np.array(np.random.rand(128, 128) > 0.5, dtype=np.float32)
        B = np.array(np.random.rand(128, 128) > 0.5, dtype=np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            expected = np.bitwise_and(
                A.astype(np.int32), B.astype(np.int32)
            ).astype(np.float32)
            baremetal_assert_allclose(result, expected)


class TestKernelGenWhere:
    """Test np.where compiles and runs correctly."""

    def test_where_same_type(self):
        def kernel(A, B, C):
            return np.where(A, B, C)

        A = np.array(np.random.rand(128, 128) > 0.5, dtype=np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        C = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B, C)
        if result is not None:
            baremetal_assert_allclose(result, np.where(A, B, C))

    def test_where_with_comparison(self):
        def kernel(A, B):
            mask = np.greater(A, 0.0)
            return np.where(mask, A, B)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, np.where(A > 0.0, A, B))


class TestKernelGenComposedKernel:
    """Test non-trivial kernels that compose multiple ops."""

    @pytest.mark.xfail(reason="broadcast add with rank-1 bias not lowered to NISA yet", run=True, strict=False)
    def test_matmul_add_relu(self):
        def kernel(A, B, bias):
            C = np.matmul(A, B)
            C = C + bias
            return np.maximum(C, 0.0)

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        bias = np.random.randn(128).astype(np.float32)
        result = _run_kernel(kernel, A, B, bias)
        if result is not None:
            baremetal_assert_allclose(result, np.maximum(A @ B + bias, 0.0))

    @pytest.mark.xfail(reason="composed mean/sqrt/broadcast not lowered to NISA yet", run=True, strict=False)
    def test_rmsnorm(self):
        def kernel(x, weight):
            variance = np.mean(x * x, axis=-1, keepdims=True)
            x_norm = x / np.sqrt(variance + 1e-6)
            return x_norm * weight

        x = np.random.randn(128, 128).astype(np.float32)
        w = np.random.randn(128).astype(np.float32)
        result = _run_kernel(kernel, x, w)
        if result is not None:
            variance = np.mean(x * x, axis=-1, keepdims=True)
            expected = (x / np.sqrt(variance + 1e-6)) * w
            baremetal_assert_allclose(result, expected)

    @pytest.mark.xfail(reason="clip not lowered to NISA yet", run=True, strict=False)
    def test_clip(self):
        def kernel(A):
            return np.clip(A, 0.0, 1.0)

        A = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A)
        if result is not None:
            baremetal_assert_allclose(result, np.clip(A, 0.0, 1.0))


class TestKernelGenAnnotations:
    """Test knob() annotations compile to NEFF and run correctly."""

    def test_knob_mem_space(self):
        from nkipy.core.knob import knob

        def kernel(A, B):
            C = np.matmul(A, B)
            C = knob(C, mem_space="SharedHbm", tile_size=[128, 128],
                     reduction_tile=[128])
            return C

        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        result = _run_kernel(kernel, A, B)
        if result is not None:
            baremetal_assert_allclose(result, A @ B)
