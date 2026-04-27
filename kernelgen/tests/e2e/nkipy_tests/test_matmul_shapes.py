# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Targeted matmul tests with various shapes, inspired by
nkipy/tests/unit/test_tensor_api_native_ops.py and test_tensor_api.py.

Tests matmul with different M/N/K dimensions and batch dimensions.
"""

import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


def test_matmul_square():
    """Square matmul: (256,256) @ (256,256)."""
    M = N = K = 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        return result

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_matmul_rectangular():
    """Rectangular matmul: (128,256) @ (256,512)."""
    M, K, N = 128, 256, 512

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        return result

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_matmul_tall():
    """Tall output: (512,128) @ (128,128)."""
    M, K, N = 512, 128, 128

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        return result

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_matmul_wide():
    """Wide output: (128,128) @ (128,512)."""
    M, K, N = 128, 128, 512

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        return result

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_batch_matmul():
    """Batched matmul: (4, 128, 256) @ (4, 256, 128)."""
    B, M, K, N = 4, 128, 256, 128

    @trace(input_specs=[((B, M, K), "f32"), ((B, K, N), "f32")])
    def kernel(a, b):
        return np.matmul(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_matmul_add():
    """Matmul followed by bias add: C = A @ B + bias."""
    M, K, N = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, bias):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        out = np.add(result, bias)
        knob.knob(out, mem_space="SharedHbm", tile_size=[128, 128])
        return out

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_matmul_chain():
    """Two matmuls chained: D = (A @ B) @ C."""
    M, K1, K2, N = 256, 256, 256, 256

    @trace(input_specs=[((M, K1), "f32"), ((K1, K2), "f32"), ((K2, N), "f32")])
    def kernel(a, b, c):
        ab = np.matmul(a, b)
        knob.knob(ab, tile_size=[128, 128], reduction_tile=128, mem_space="Sbuf")
        result = np.matmul(ab, c)
        knob.knob(result, tile_size=[128, 128], reduction_tile=128)
        return result

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)
