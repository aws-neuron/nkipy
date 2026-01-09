# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensor API for native ops"""

import numpy as np
import pytest
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    baremetal_run_kernel_unified,
    sim_mode,  # noqa: F401 - pytest fixture
    simulate_assert_allclose,
    simulate_kernel_unified,
)


# Binary operation kernels
def kernel_add(a, b):
    return a + b


def kernel_radd(a, b):
    return b + a


def kernel_sub(a, b):
    return a - b


def kernel_rsub(a, b):
    return b - a


def kernel_mul(a, b):
    return a * b


def kernel_rmul(a, b):
    return b * a


def kernel_truediv(a, b):
    return a / b


def kernel_rtruediv(a, b):
    return b / a


def kernel_floordiv(a, b):
    return a // b


def kernel_rfloordiv(a, b):
    return b // a


def kernel_mod(a, b):
    return a % b


def kernel_rmod(a, b):
    return b % a


def kernel_pow(a, b):
    return a**b


def kernel_rpow(a, b):
    return b**a


def kernel_matmul(a, b):
    return a @ b


def kernel_rmatmul(a, b):
    return b @ a


# Unary operation kernels
def kernel_pos(a):
    return +a


def kernel_neg(a):
    return -a


# In-place operation kernels
def kernel_iadd(a, b):
    a += b
    return a


def kernel_isub(a, b):
    a -= b
    return a


def kernel_imul(a, b):
    a *= b
    return a


def kernel_itruediv(a, b):
    a /= b
    return a


def kernel_ifloordiv(a, b):
    a //= b
    return a


def kernel_imod(a, b):
    a %= b
    return a


def kernel_ipow(a, b):
    a **= b
    return a


def kernel_imatmul(a, b):
    a @= b
    return a


# Reshape kernels
def kernel_reshape_tuple(a):
    return a.reshape((64, 1024))


def kernel_reshape_args(a):
    return a.reshape(64, 1024)


def kernel_reshape_negative(a):
    return a.reshape(-1, 64)


# Transpose kernels
def kernel_transpose_default(a):
    return a.transpose()


def kernel_transpose_tuple(a):
    return a.transpose((1, 0))


def kernel_transpose_args(a):
    return a.transpose(1, 0)


# AsType kernels
def kernel_astype_float(a):
    return a.astype(np.float32)


# Test configurations
binary_kernels = [
    (kernel_add, "add", False),
    (kernel_radd, "radd", False),
    (kernel_sub, "sub", False),
    (kernel_rsub, "rsub", False),
    (kernel_mul, "mul", False),
    (kernel_rmul, "rmul", False),
    (kernel_truediv, "truediv", False),
    (kernel_rtruediv, "rtruediv", False),
    (kernel_floordiv, "floordiv", True),  # floor_divide not supported
    (kernel_rfloordiv, "rfloordiv", True),  # floor_divide not supported
    (kernel_mod, "mod", True),  # mod not supported
    (kernel_rmod, "rmod", True),  # mod not supported
    (kernel_pow, "pow", False),
    (kernel_rpow, "rpow", False),
]

matmul_kernels = [
    (kernel_matmul, "matmul", False),
    (kernel_rmatmul, "rmatmul", False),
]


unary_kernels = [
    (kernel_pos, "pos", False),
    (kernel_neg, "neg", False),
]

inplace_kernels = [
    (kernel_iadd, "iadd", False),
    (kernel_isub, "isub", False),
    (kernel_imul, "imul", False),
    (kernel_itruediv, "itruediv", False),
    (kernel_ifloordiv, "ifloordiv", True),  # floor_divide not supported
    (kernel_imod, "imod", True),  # mod not supported
    (kernel_ipow, "ipow", False),
    (kernel_imatmul, "imatmul", False),
]

reshape_kernels = [
    (kernel_reshape_tuple, "reshape_tuple", False),
    (kernel_reshape_args, "reshape_args", False),
    (kernel_reshape_negative, "reshape_negative", False),
]

transpose_kernels = [
    (kernel_transpose_default, "transpose_default", False),
    (kernel_transpose_tuple, "transpose_tuple", False),
    (kernel_transpose_args, "transpose_args", False),
]

astype_kernels = [
    (kernel_astype_float, "astype_float", False),
]

shapes = [
    ((256, 256), (256, 256)),
    ((256, 1), (256, 256)),
    ((1, 256), (256, 256)),
    ((256,), (256, 256)),
    ((1,), (256, 256)),
]

matmul_shapes = [
    ((128, 256), (256, 512)),
    ((1, 2, 128, 512), (512, 256)),
    ((128, 512), (1, 2, 512, 256)),
    ((1, 2, 128, 512), (1, 2, 512, 256)),
]

# Define valid broadcasting shapes for matmul
matmul_broadcast_shapes = [
    ((256, 128), (128, 256)),  # Basic matrix multiplication
    ((1, 256, 128), (128, 256)),  # Broadcasting with batch dimension
    ((256, 128), (1, 128, 256)),  # Broadcasting with batch dimension
    ((1, 256, 128), (1, 128, 256)),  # Same batch dimension
    ((5, 256, 128), (1, 128, 256)),  # Broadcasting batch dimensions
]

# Define shapes for reshape tests
reshape_shapes = [
    (256, 256),  # 65536 elements
    (65536,),  # 1D array
    (16, 4096),  # Different shape, same number of elements
]

# Define shapes for transpose tests
transpose_shapes = [
    (256, 256),  # Square matrix
    (128, 512),  # Rectangle matrix
    (2, 128, 256),  # 3D tensor
]

# Define types for astype tests
astype_types = [
    np.float32,
    np.float16,
]

# Invalid matmul shapes for error testing
invalid_matmul_shapes = [
    ((256, 128), (256, 128)),  # Inner dimensions don't match
    ((128, 256), (128, 256)),  # Inner dimensions don't match
    ((1, 1), (256, 256)),  # Invalid broadcasting
]


def test_add_simple(sim_mode):
    shape = (256, 256)
    dtype = np.float32

    np.random.seed(0)
    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)
    in1 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    # Test simulation - runs with both IR and HLO
    out0 = simulate_kernel_unified(kernel_add, sim_mode, in0, in1)
    out1 = kernel_add(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel_add, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("kernel_fn,name,expects_error", binary_kernels)
@pytest.mark.parametrize("dtype", [np.float32])
def test_binary_ops(sim_mode, kernel_fn, name, expects_error, dtype):
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
        out1 = kernel_fn(in0, in1)
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0, in1)
            baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("kernel_fn,name,expects_error", unary_kernels)
@pytest.mark.parametrize("dtype", [np.float32])
def test_unary_ops(sim_mode, kernel_fn, name, expects_error, dtype):
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(dtype)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0)
        out1 = kernel_fn(in0)
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0)
            baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("kernel_fn,name,expects_error", inplace_kernels)
@pytest.mark.parametrize("dtype", [np.float32])
def test_inplace_ops(sim_mode, kernel_fn, name, expects_error, dtype):
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0.copy(), in1)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0.copy(), in1)
        out1 = kernel_fn(in0.copy(), in1)
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(
                kernel_fn, sim_mode, in0.copy(), in1
            )
            baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("shape_a,shape_b", shapes)
@pytest.mark.parametrize("kernel_fn,name,expects_error", binary_kernels)
def test_broadcasting(sim_mode, shape_a, shape_b, kernel_fn, name, expects_error):
    np.random.seed(0)
    in0 = np.random.random_sample(shape_a).astype(np.float32)
    in1 = np.random.random_sample(shape_b).astype(np.float32)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
        out1 = kernel_fn(in0, in1)
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0, in1)
            baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("shape_a,shape_b", matmul_shapes)
def test_matmul_shapes(sim_mode, shape_a, shape_b):
    dtype = np.float32
    np.random.seed(0)
    in0 = np.random.random_sample(shape_a).astype(dtype)
    in1 = np.random.random_sample(shape_b).astype(dtype)

    # Test simulation - runs with both IR and HLO
    out0 = simulate_kernel_unified(kernel_matmul, sim_mode, in0, in1)
    out1 = kernel_matmul(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel_matmul, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("shape_a,shape_b", matmul_broadcast_shapes)
@pytest.mark.parametrize("kernel_fn,name,expects_error", matmul_kernels)
def test_matmul_broadcasting(
    sim_mode, shape_a, shape_b, kernel_fn, name, expects_error
):
    np.random.seed(0)
    in0 = np.random.random_sample(shape_a).astype(np.float32)
    in1 = np.random.random_sample(shape_b).astype(np.float32)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)
        out1 = kernel_fn(in0, in1)
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0, in1)
            baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("shape_a,shape_b", invalid_matmul_shapes)
@pytest.mark.parametrize("kernel_fn,name,expects_error", matmul_kernels)
def test_invalid_matmul_shapes(
    sim_mode, shape_a, shape_b, kernel_fn, name, expects_error
):
    np.random.seed(0)
    in0 = np.random.random_sample(shape_a).astype(np.float32)
    in1 = np.random.random_sample(shape_b).astype(np.float32)

    with pytest.raises(AssertionError):
        simulate_kernel_unified(kernel_fn, sim_mode, in0, in1)


@pytest.mark.parametrize("kernel_fn,name,expects_error", reshape_kernels)
def test_reshape(sim_mode, kernel_fn, name, expects_error):
    shape = (256, 256)  # Starting shape with 65536 elements
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0)
        out1 = kernel_fn(in0)
        assert out0.shape == out1.shape
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0)
            baremetal_assert_allclose(out1, out_baremetal)


def test_reshape_invalid(sim_mode):
    # Test invalid reshape (incompatible dimensions)
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    def invalid_reshape(a):
        return a.reshape(100, 100)  # 10000 elements != 65536 elements

    with pytest.raises(ValueError):
        simulate_kernel_unified(invalid_reshape, sim_mode, in0)


@pytest.mark.parametrize("shape", reshape_shapes)
def test_reshape_variants(sim_mode, shape):
    # Test different starting shapes but same final shape
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    def reshape_to_standard(a):
        return a.reshape(32, 32, 64)

    # Test simulation - runs with both IR and HLO
    out0 = simulate_kernel_unified(reshape_to_standard, sim_mode, in0)
    out1 = reshape_to_standard(in0)

    assert out0.shape == out1.shape
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(reshape_to_standard, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("kernel_fn,name,expects_error", transpose_kernels)
@pytest.mark.parametrize("shape", transpose_shapes)
def test_transpose(sim_mode, kernel_fn, name, expects_error, shape):
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0)
    else:
        # For 3D tensors with specific permutation kernels, need to adapt
        if len(shape) > 2 and name in ["transpose_tuple", "transpose_args"]:
            # Skip this combination as the kernels are designed for 2D
            return

        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0)
        out1 = kernel_fn(in0)

        assert out0.shape == out1.shape
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0)
            baremetal_assert_allclose(out1, out_baremetal)


def test_transpose_3d(sim_mode):
    # Specific test for 3D transpose with explicit permutation
    shape = (2, 3, 4)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    def transpose_3d_tuple(a):
        return a.transpose((2, 0, 1))

    def transpose_3d_args(a):
        return a.transpose(2, 0, 1)

    # Test tuple version - simulation (runs with both IR and HLO)
    out0 = simulate_kernel_unified(transpose_3d_tuple, sim_mode, in0)
    out1 = transpose_3d_tuple(in0)

    assert out0.shape == out1.shape
    simulate_assert_allclose(out0, out1)

    # Test args version - simulation (runs with both IR and HLO)
    out0 = simulate_kernel_unified(transpose_3d_args, sim_mode, in0)
    out1 = transpose_3d_args(in0)

    assert out0.shape == out1.shape
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        # Test tuple version
        out_baremetal = baremetal_run_kernel_unified(transpose_3d_tuple, sim_mode, in0)
        baremetal_assert_allclose(transpose_3d_tuple(in0), out_baremetal)

        # Test args version
        out_baremetal = baremetal_run_kernel_unified(transpose_3d_args, sim_mode, in0)
        baremetal_assert_allclose(transpose_3d_args(in0), out_baremetal)


@pytest.mark.parametrize("kernel_fn,name,expects_error", astype_kernels)
@pytest.mark.parametrize("source_type", astype_types)
def test_astype(sim_mode, kernel_fn, name, expects_error, source_type):
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(source_type)

    if expects_error:
        with pytest.raises(NotImplementedError):
            simulate_kernel_unified(kernel_fn, sim_mode, in0)
    else:
        # Test simulation - runs with both IR and HLO
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0)
        out1 = kernel_fn(in0)

        assert out0.dtype == out1.dtype
        simulate_assert_allclose(out0, out1)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0)
            baremetal_assert_allclose(out1, out_baremetal)


def test_combined_operations(sim_mode):
    # Test chaining operations: reshape -> transpose -> astype
    shape = (256, 256)
    np.random.seed(0)
    in0 = np.random.random_sample(shape).astype(np.float32)

    def combined_ops(a):
        return a.reshape(64, 1024).transpose().astype(np.float16)

    # Test simulation - runs with both IR and HLO
    out0 = simulate_kernel_unified(combined_ops, sim_mode, in0)
    out1 = combined_ops(in0)

    assert out0.shape == out1.shape
    assert out0.dtype == out1.dtype
    simulate_assert_allclose(out0, out1, rtol=1e-2)  # Lower precision for float16

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(combined_ops, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


if __name__ == "__main__":
    pytest.main([__file__])
