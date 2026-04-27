# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Targeted tensor manipulation tests, inspired by nkipy/tests/unit/test_tensor_api.py
and test_tensor_api_native_ops.py.

Tests reshape, transpose, expand_dims, broadcast_to, concatenate.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from nkipy_kernelgen.knob import knob
from harness import run_kernel_test, Mode

M, N = 128, 256


# -- Reshape (Category 1: contiguous dim merge/split) --

def test_reshape_merge_dims():
    """Merge first two dims: (2, 128, 256) -> (256, 256)."""
    @trace(input_specs=[((2, M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (2 * M, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_split_dim():
    """Split first dim: (256, 256) -> (2, 128, 256)."""
    @trace(input_specs=[((2 * M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (2, M, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_insert_unit_dim():
    """Insert unit dim: (128, 256) -> (128, 1, 256)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (M, 1, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_remove_unit_dim():
    """Remove unit dim (squeeze): (128, 1, 256) -> (128, 256)."""
    @trace(input_specs=[((M, 1, N), "f32")])
    def kernel(x):
        return np.reshape(x, (M, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_infer_dim_category1():
    """Reshape with -1 inferred dimension (Category 1 merge)."""
    @trace(input_specs=[((2, M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (-1, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_identity():
    """Identity reshape: same shape, should be a no-op."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (M, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Reshape (Category 2: non-contiguous -- not supported) --

def test_reshape_2d():
    """Category 2 reshape (non-contiguous): (128, 256) -> (256, 128)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (M * N // 128, 128))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reshape_infer_dim():
    """Reshape with -1 inferred dimension (Category 2)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.reshape(x, (-1, 128))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Transpose --

def test_transpose_2d():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        # Output shape: (N, M) = (256, 128)
        result = np.transpose(x, [1, 0])
        return knob(result, tile_size=[128, 128])

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_transpose_3d():
    @trace(input_specs=[((2, M, N), "f32")])
    def kernel(x):
        # Output shape: (2, N, M) = (2, 256, 128)
        # Batch dim tiled to 1, inner dims tiled to 128
        result = np.transpose(x, [0, 2, 1])
        return knob(result, tile_size=[1, 128, 128])

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Expand dims --

def test_expand_dims_last():
    @trace(input_specs=[((M,), "f32")])
    def kernel(x):
        return np.expand_dims(x, axis=-1)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_expand_dims_first():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.expand_dims(x, axis=0)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Broadcast --

def test_broadcast_to():
    @trace(input_specs=[((1, N), "f32")])
    def kernel(x):
        return np.broadcast_to(x, (M, N))

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Concatenate --

def test_concatenate_axis_last():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.concatenate([a, b], axis=-1)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_concatenate_axis_first():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.concatenate([a, b], axis=0)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Copy --

def test_copy():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.copy(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Test Runner --

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
