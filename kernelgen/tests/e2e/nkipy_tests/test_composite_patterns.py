# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Composite pattern tests combining multiple ops into common ML patterns,
inspired by nkipy/tests/unit/test_tensor_api.py.

Tests softmax, layer norm, sigmoid, GELU, and other compound operations.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode

M, N = 128, 256


def test_softmax_full():
    """Full softmax: exp(x - max(x)) / sum(exp(x - max(x)))."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_x

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_sigmoid():
    """Sigmoid: 1 / (1 + exp(-x))."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return 1.0 / (1.0 + np.exp(-x))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_elementwise_chain():
    """Chain of elementwise ops: exp(x) * 2.0 + 1.0."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.exp(x) * 2.0 + 1.0

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_residual_add():
    """Residual connection: x + f(x) where f(x) = exp(x)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return x + np.exp(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_rmsnorm_no_weight():
    """RMSNorm without weight: x * rsqrt(mean(x^2) + eps)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        x_sq = x * x
        mean_x = np.sum(x_sq, axis=-1, keepdims=True) / x_sq.shape[-1]
        rsqrt = 1.0 / np.sqrt(mean_x + 1e-5)
        return x * rsqrt

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


@pytest.mark.xfail(reason="1D broadcast multiply (weight * x) fails in legalize-layout: linalg.mul requires matching operand ranks")
def test_rmsnorm():
    """RMSNorm: x * weight * rsqrt(mean(x^2) + eps)."""
    @trace(input_specs=[((M, N), "f32"), ((N,), "f32")])
    def kernel(x, weight):
        x_sq = x * x
        mean_x = np.sum(x_sq, axis=-1, keepdims=True) / x_sq.shape[-1]
        rsqrt = 1.0 / np.sqrt(mean_x + 1e-5)
        return np.multiply(weight, np.multiply(x, rsqrt))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_rmsnorm_prebroadcast():
    """RMSNorm with pre-broadcasted weight to (M, N).

    Workaround variant of test_rmsnorm: the compiler cannot yet handle
    1D row-broadcast multiply, so the weight is materialized as 2D.
    """
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(x, weight):
        x_sq = x * x
        mean_x = np.sum(x_sq, axis=-1, keepdims=True) / x_sq.shape[-1]
        rsqrt = 1.0 / np.sqrt(mean_x + 1e-5)
        return weight * (x * rsqrt)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_gelu_approx():
    """Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        # Simplified: use sigmoid approximation instead
        # GELU ≈ x * sigmoid(1.702 * x)
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product():
    """Scaled dot-product pattern: (Q @ K^T) * scale."""
    M_dim, K_dim = 256, 256

    @trace(input_specs=[((M_dim, K_dim), "f32"), ((M_dim, K_dim), "f32")])
    def kernel(q, k):
        from nkipy_kernelgen import knob
        scores = np.matmul(q, k)
        knob.knob(scores, tile_size=[128, 128], reduction_tile=128)
        return scores * (1.0 / np.sqrt(np.float32(K_dim)))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)
