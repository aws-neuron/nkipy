# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for regular tensor API using pytest with both simulation and hardware testing
"""

from collections import defaultdict

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nkipy.core import tensor_apis
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    baremetal_run_kernel_unified,
    sim_mode,  # noqa: F401 - pytest fixture
    simulate_assert_allclose,
    simulate_kernel_unified,
)


# Test both simulation and hardware for local_softmax_return_0
def test_local_softmax_return_0(sim_mode):
    def local_softmax(a, axis=-1):
        ma = np.max(a, axis=axis, keepdims=True)
        ea = np.exp(np.subtract(a, ma))
        s = np.sum(ea, axis=axis, keepdims=True)
        return np.divide(ea, s)

    shape = (256, 256)
    dtype = np.float32

    np.random.seed(0)
    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    # Test simulation - runs with both IR and HLO
    out0 = simulate_kernel_unified(local_softmax, sim_mode, in0)
    out1 = local_softmax(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(local_softmax, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


def test_local_softmax_return_1(sim_mode):
    def local_softmax(a, axis=-1):
        ma = np.max(a, axis=axis, keepdims=True)
        ea = np.exp(np.subtract(a, ma))
        s = np.sum(ea, axis=axis, keepdims=True)
        return np.divide(ea, s), s

    shape = (256, 256)
    dtype = np.float32

    np.random.seed(0)
    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    # Test simulation - always runs
    out0, out1 = simulate_kernel_unified(local_softmax, sim_mode, in0)
    out2, out3 = local_softmax(in0)
    simulate_assert_allclose(out0, out2)
    simulate_assert_allclose(out1, out3)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(local_softmax, sim_mode, in0)
        baremetal_assert_allclose(out2, out_baremetal[0])
        baremetal_assert_allclose(out3, out_baremetal[1])


def test_expand_dims_0(sim_mode):
    axis = -1

    def kernel(a):
        return np.expand_dims(a, axis=axis)

    shape = [256]
    dtype = np.float32

    np.random.seed(0)
    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    # Test simulation - always runs
    out0 = np.expand_dims(in0, axis=axis)
    out1 = simulate_kernel_unified(kernel, sim_mode, in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out0, out_baremetal)


# Test unary operations on both simulation and hardware
@pytest.mark.parametrize(
    "np_fn",
    [
        np.abs,
        np.arctan,
        np.ceil,
        np.cos,
        np.exp,
        np.floor,
        np.log,
        np.negative,
        np.rint,
        # np.round,
        np.sqrt,
        np.sin,
        np.sign,
        np.tan,
        np.tanh,
        np.trunc,
        np.square,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32])
def test_unary(sim_mode, np_fn, dtype):
    shape = (256, 256)
    np.random.seed(0)

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("np_fn", [np.bitwise_not, np.logical_not])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_unary_bitwise(sim_mode, np_fn, dtype):
    shape = (256, 256)
    np.random.seed(0)

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "np_fn",
    [
        np.add,
        # np.arctan2,
        np.divide,
        # np.mod,
        np.maximum,
        np.minimum,
        np.multiply,
        np.subtract,
        np.power,
        np.greater_equal,
        np.less,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32])
def test_binary(sim_mode, np_fn, dtype):
    shape = (256, 256)
    np.random.seed(0)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("np_fn", [np.bitwise_and, np.bitwise_xor, np.bitwise_or])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_binary_bitwise(sim_mode, np_fn, dtype):
    shape = (256, 256)
    np.random.seed(0)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "np_fn",
    [np.equal, np.not_equal, np.greater, np.less_equal, np.less, np.greater_equal],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_comparison(sim_mode, np_fn, dtype):
    shape = (128, 128)  # Smaller shape for faster hardware tests
    np.random.seed(0)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("np_fn", [np.matmul])
@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,out_shape",
    [
        ((64, 128), (128, 64), (64, 64)),
        ((128, 256), (256, 512), (128, 512)),
        ((1, 2, 128, 512), (512, 256), (1, 2, 128, 256)),
        ((128, 512), (1, 2, 512, 256), (1, 2, 128, 256)),
        ((1, 2, 128, 512), (1, 2, 512, 256), (1, 2, 128, 256)),
        ((1, 1, 128, 512), (1, 1, 1, 512, 256), (1, 1, 1, 128, 256)),
    ],
)
def test_contract(sim_mode, np_fn, lhs_shape, rhs_shape, out_shape):
    np.random.seed(0)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(lhs_shape).astype(np.float32)
    in1 = np.random.random_sample(rhs_shape).astype(np.float32)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("np_fn", [np.mean, np.max, np.min, np.sum])
@pytest.mark.parametrize(
    "shape,dtype,axis",
    [
        ((128, 128), np.float32, (-1,))  # Smaller shape for hardware tests
    ],
)
def test_reduction(sim_mode, np_fn, shape, dtype, axis):
    np.random.seed(0)

    def kernel(a):
        return np_fn(a, axis=axis)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


def test_multiple_sum_different_dtypes(sim_mode):
    """Test multiple np.sum calls with different dtypes.

    Regression test for: "Computation name is not unique: add_computation"
    This bug occurred when multiple reduce operations with different dtypes
    were used in the same kernel, causing duplicate HLO computation names.
    """

    def kernel(a):
        sum_f32 = np.sum(a.astype(np.float32), axis=-1, keepdims=True)
        sum_f16 = np.sum(a.astype(np.float16), axis=-1, keepdims=True)
        return sum_f32, sum_f16

    shape = (32, 64)
    np.random.seed(0)
    a = np.random.random_sample(shape).astype(np.float32)

    out_f32, out_f16 = simulate_kernel_unified(kernel, sim_mode, a)

    expected_f32 = np.sum(a.astype(np.float32), axis=-1, keepdims=True)
    expected_f16 = np.sum(a.astype(np.float16), axis=-1, keepdims=True)

    simulate_assert_allclose(out_f32, expected_f32)
    simulate_assert_allclose(out_f16, expected_f16)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, a)
        baremetal_assert_allclose(expected_f32, out_baremetal[0])
        baremetal_assert_allclose(expected_f16, out_baremetal[1])


@pytest.mark.parametrize("np_fn", [np.mean, np.max, np.min, np.sum])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((64, 64), np.float32),
        ((32, 32, 32), np.float32),
        ((16, 16, 16, 16), np.float32),
    ],
)
def test_reduction_axis_none(sim_mode, np_fn, shape, dtype):
    """Test reduction operations with axis=None (reduce over all axes)"""
    np.random.seed(0)

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("np_fn", [np.sum])
@pytest.mark.parametrize(
    "shape,dtype,keepdims",
    [
        ((64, 64), np.float32, True),
        ((32, 32, 32), np.float32, True),
        ((64, 64), np.float32, False),
        ((32, 32, 32), np.float32, False),
    ],
)
def test_reduction_axis_none_keepdims(sim_mode, np_fn, shape, dtype, keepdims):
    """Test reduction operations with axis=None and keepdims parameter"""
    np.random.seed(0)

    def kernel(a):
        return np_fn(a, axis=None, keepdims=keepdims)

    in0 = np.random.random_sample(shape).astype(dtype)

    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ((2, 1), (2, 2)),
        ((1, 2), (2, 2)),
        ((1, 1, 2), (2, 2, 2)),
        ((1, 2, 1), (2, 2, 2)),
        ((2, 1, 1), (2, 2, 2)),
        ((2, 2, 1), (2, 2, 2)),
        ((2, 1, 2), (2, 2, 2)),
        ((1, 2, 2), (2, 2, 2)),
        ((2, 2), (2, 2, 2)),
        ((1, 2), (1, 1, 2, 2)),
        ((1, 2), (2, 2, 2, 2)),
    ],
)
def test_broadcast_to(sim_mode, src_shape, dst_shape):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a, shape):
        return np.broadcast_to(a, shape=shape)

    in0 = np.random.random_sample(src_shape).astype(dtype)

    out0 = simulate_kernel_unified(kernel, sim_mode, in0, dst_shape)
    out1 = kernel(in0, dst_shape)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        # For baremetal, we need a kernel with fixed shape parameter
        def kernel_fixed_shape(a):
            return np.broadcast_to(a, shape=dst_shape)

        out_baremetal = baremetal_run_kernel_unified(kernel_fixed_shape, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((2, 3), (1, 0)),
        ((2, 3), (0, 1)),
        ((2, 3, 4), (0, 1, 2)),
        ((2, 3, 4), (0, 2, 1)),
        ((2, 3, 4), (2, 0, 1)),
        ((2, 3, 4), (2, 1, 0)),
    ],
)
def test_transpose(sim_mode, shape, axes):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a):
        return np.transpose(a, axes=axes)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape,repeats,axis",
    [
        ((2, 3), 2, 0),
        ((2, 3), 2, None),
        ((2, 3), 3, 1),
        ((2, 3, 4), 2, None),
        ((2, 3, 4), 2, 0),
        ((2, 3, 4), 3, 1),
        ((2, 3, 4), 4, 2),
    ],
)
def test_repeat(sim_mode, shape, repeats, axis):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a):
        return np.repeat(a, repeats=repeats, axis=axis)

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)
    out1 = kernel(in0)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), 0, None),
        ((2, 3), 1, None),
        ((2, 3), [0, 1], None),
        ((2, 3), [[0, 1]], None),
        ((2, 3), [[0, 1], [1, 0]], None),
        ((2, 3), [0, 1], 0),
        ((2, 3), [[0, 1]], 1),
        ((2, 3), [[0, 1], [1, 0]], 0),
        ((2, 3), [[0, 1], [1, 0]], 1),
        ((2, 3, 4), [0, 1], 0),
        ((2, 3, 4), [[0, 1]], 1),
        ((2, 3, 4), [[0, 1], [1, 0]], 0),
        ((2, 3, 4), [[0, 1], [1, 0]], 1),
        ((2, 3, 4), [[0, 1], [1, 0]], 2),
    ],
)
def test_take(sim_mode, a, indices, axis):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a, indices, axis):
        return np.take(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2)
    out1 = kernel(in0, in1, in2)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1, in2)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), [0, 1], None),
        ((2, 3), [[0, 1]], None),
        ((2, 3), [[0, 1], [1, 0]], None),
        ((2, 3), [0, 1], 0),
        ((2, 3), [[0, 1]], 1),
        ((2, 3), [[0, 1], [1, 0]], 0),
        ((2, 3), [[0, 1], [1, 0]], 1),
        ((2, 3, 4), [0, 1], 0),
        ((2, 3, 4), [[0, 1]], 1),
        ((2, 3, 4), [[0, 1], [1, 0]], 0),
        ((2, 3, 4), [[0, 1], [1, 0]], 1),
        ((2, 3, 4), [[0, 1], [1, 0]], 2),
    ],
)
def test_take_numpy_indices(sim_mode, a, indices, axis):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a, axis):
        return np.take(a, indices=np.array(indices).astype(np.uint32), axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), 0, None),
        ((2, 3), 1, None),
        ((2, 3), -1, None),
    ],
)
def test_take_scalar(sim_mode, a, indices, axis):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a, indices, axis):
        return np.take(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = indices
    in2 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2)
    out1 = kernel(in0, in1, in2)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1, in2)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,values,axis",
    [
        ((2, 3), [[1, 0, 0]], [[4, 5, 6]], 0),
        ((2, 3), [[1], [0]], [[4], [5]], 1),
        ((2, 3), [1, 0], [4, 5], None),
    ],
)
def test_put_along_axis(sim_mode, a, indices, values, axis):
    # FIXME: support put_along_axis with proper doc
    if sim_mode == "hlo":
        pytest.skip("put_along_axis not yet supported in HLO mode")

    dtype = np.float32
    np.random.seed(0)

    def kernel(a, indices, values, axis, is_hardware=False):
        b = np.copy(a)
        if sim_mode == "hlo" and is_hardware:
            b = np.put_along_axis(b, indices=indices, values=values, axis=axis)
        else:
            np.put_along_axis(b, indices=indices, values=values, axis=axis)

        return b

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = np.array(values, dtype=dtype)
    in3 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2, in3)
    out1 = kernel(in0, in1, in2, in3)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel, sim_mode, in0, in1, in2, in3, True
        )
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,values,axis",
    [
        ((2, 3), [[1, 0, 0]], 100, 0),
        ((2, 3), [[1], [0]], 3.0, 1),
        ((2, 3), [1, 0], 2, None),
    ],
)
def test_put_along_axis_scalar_value(sim_mode, a, indices, values, axis):
    # FIXME: support put_along_axis with proper doc
    if sim_mode == "hlo":
        pytest.skip("put_along_axis not yet supported in HLO mode")

    dtype = np.float32
    np.random.seed(0)

    def kernel(a, indices, values, axis, is_hardware=False):
        b = np.copy(a)
        if sim_mode == "hlo" and is_hardware:
            b = np.put_along_axis(b, indices=indices, values=values, axis=axis)
        else:
            np.put_along_axis(b, indices=indices, values=values, axis=axis)

        return b

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = values
    in3 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2, in3, False)
    out1 = kernel(in0, in1, in2, in3)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel, sim_mode, in0, in1, in2, in3, True
        )
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), [[1, 0, 0]], 0),
        ((2, 3), [[1], [0]], 1),
        ((2, 3), [1, 0], None),
    ],
)
def test_take_along_axis(sim_mode, a, indices, axis):
    dtype = np.float32
    np.random.seed(0)

    def kernel(a, indices, axis):
        return np.take_along_axis(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2)
    out1 = kernel(in0, in1, in2)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1, in2)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "a,axis",
    [
        ((2048, 64), 0),
        ((10, 20, 30), None),
    ],
)
def test_take_along_axis_random(sim_mode, a, axis):
    dtype = np.float32
    np.random.seed(0)

    if axis == 0:
        indices = np.random.randint(0, a[0], size=(128, 1)).astype(np.uint32)
    elif axis is None:
        indices = np.random.randint(0, np.prod(a), size=(50,)).astype(np.uint32)

    def kernel(a, indices, axis):
        return np.take_along_axis(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = indices
    in2 = axis

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1, in2)
    out1 = kernel(in0, in1, in2)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1, in2)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("B,L,HD,QHN", [(1, 1, 2, 1), (2, 3, 10, 4)])
def test_rotary_embed(sim_mode, B, L, HD, QHN):
    np.random.seed(0)

    def kernel(x, cos_idx, sin_idx, freqs_cos, freqs_sin):
        x_0 = np.take(x, cos_idx, axis=len(x.shape) - 1)
        x_1 = np.take(x, sin_idx, axis=len(x.shape) - 1)

        x_0_cos = np.multiply(x_0, freqs_cos)
        x_1_sin = np.multiply(x_1, freqs_sin)
        x_0_sin = np.multiply(x_0, freqs_sin)
        x_1_cos = np.multiply(x_1, freqs_cos)

        x_out_0 = np.subtract(x_0_cos, x_1_sin)
        x_out_1 = np.add(x_0_sin, x_1_cos)

        x_out = np.empty_like(x)
        x_out[:, :, :, cos_idx] = x_out_0
        x_out[:, :, :, sin_idx] = x_out_1

        return x_out

    # Generate test data
    freqs_cos = np.random.random_sample([B, L, 1, HD // 2]).astype(np.float32)
    freqs_sin = np.random.random_sample([B, L, 1, HD // 2]).astype(np.float32)
    x = np.random.random_sample([B, L, QHN, HD]).astype(np.float32)
    cos_idx = np.arange(0, x.shape[-1], 2, dtype=np.int32)
    sin_idx = np.arange(1, x.shape[-1], 2, dtype=np.int32)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(
        kernel, sim_mode, x, cos_idx, sin_idx, freqs_cos, freqs_sin
    )
    out1 = kernel(x, cos_idx, sin_idx, freqs_cos, freqs_sin)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel, sim_mode, x, cos_idx, sin_idx, freqs_cos, freqs_sin
        )
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (256, 256),
        (128, 512),
        (1, 2, 128, 256),
        (2, 3, 4),
    ],
)
def test_where(sim_mode, shape):
    dtype = np.float32
    np.random.seed(0)

    def kernel(cond, x, y):
        return np.where(cond, x, y)

    condition = np.random.choice([True, False], size=shape).astype(np.uint8)
    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, condition, in0, in1)
    out1 = kernel(condition, in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel, sim_mode, condition, in0, in1
        )
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (256, 256),
        (128, 512),
        (2, 3, 4),
        (10, 64, 512),
    ],
)
def test_where_first_dim(sim_mode, shape):
    dtype = np.float32
    np.random.seed(0)

    def kernel(cond, x, y):
        return np.where(cond, x, y)

    condition = np.random.choice([True, False], size=shape[:1]).astype(np.uint8)

    # expand to be the same number of dims as shape
    condition = np.expand_dims(condition, axis=tuple(range(1, len(shape))))

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, condition, in0, in1)
    out1 = kernel(condition, in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel, sim_mode, condition, in0, in1
        )
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4, 5),
    ],
)
def test_where_ndarray_cond_dim2(sim_mode, shape):
    dtype = np.float32
    np.random.seed(0)

    condition = np.random.choice([True, False], size=shape[:1])

    def kernel(x, y):
        cond = np.expand_dims(condition, axis=tuple(range(1, len(shape))))
        return np.where(cond, x, y)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape,idx_size",
    [((5, 10, 15), 3), ((10, 20, 30), 5), ((15, 25, 35), 7)],
)
def test_slice_assignment(sim_mode, shape, idx_size):
    np.random.seed(0)

    def kernel(a, b, t):
        a[:, t, :] = b
        return a

    a = np.random.random_sample(shape).astype(np.float32)
    if idx_size <= shape[1]:
        # make sure indices are different to avoid indeterminism
        t = np.random.choice(shape[1], size=idx_size, replace=False).astype(np.int32)
    else:
        raise ValueError(
            f"Cannot generate {idx_size} unique values from range [0, {shape[1]})"
        )
    # Shape for b should match a[:, t, :]
    b_shape = (shape[0], idx_size, shape[2])
    b = np.random.random_sample(b_shape).astype(np.float32)

    # Make copies for both implementations to avoid mutations affecting each other
    a1 = np.copy(a)
    a2 = np.copy(a)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, a1, b, t)
    out1 = kernel(a2, b, t)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        a3 = np.copy(a)
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, a3, b, t)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize(
    "shape,indices",
    [((5, 10, 15), [0, 2, 2])],
)
def test_slice_assignment_indeterministic(sim_mode, shape, indices):
    np.random.seed(0)

    def kernel(a, b, t):
        a[:, t, :] = b
        return a

    a = np.random.random_sample(shape).astype(np.float32)
    t = np.array(indices).astype(np.uint32)

    # Shape for b should match a[:, t, :]
    b_shape = (shape[0], len(indices), shape[2])
    b = np.random.random_sample(b_shape).astype(np.float32)

    # Make copies for both implementations to avoid mutations affecting each other
    a1 = np.copy(a)
    a2 = np.copy(a)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, a1, b, t)
    out1 = kernel(a2, b, t)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        a3 = np.copy(a)
        a_after = baremetal_run_kernel_unified(kernel, sim_mode, a3, b, t)

        # values not in t are not changed
        masked_original = np.copy(a)
        masked_original[:, t, :] = 0.0

        masked_after = np.copy(a_after)
        masked_after[:, t, :] = 0.0
        baremetal_assert_allclose(masked_original, masked_after)

        # the value indexed by t might come from any corresponding b index
        a_to_b = defaultdict(list)

        for b_idx, a_idx in enumerate(t):
            a_to_b[a_idx].append(b_idx)

        for a_idx in a_to_b:
            b_idxs = a_to_b[a_idx]

            a_value_after = a_after[:, a_idx, :]
            a_value_after = np.expand_dims(a_value_after, axis=1)
            b_values = b[:, b_idxs, :]

            # N.B.: one of the value to match
            assert np.all(np.any(a_value_after == b_values, axis=1)), (
                f"Expected {a_value_after} to be equal to any of {b_values}"
            )


@pytest.mark.parametrize(
    "shape,idx_size",
    [
        ((5, 10, 15), 3),
        ((10, 20, 30), 5),
        ((15, 25, 35), 7),
    ],
)
def test_slice_extraction(sim_mode, shape, idx_size):
    np.random.seed(0)

    def kernel(a, t):
        return a[:, t, :]

    # Create random input array and indices
    a = np.random.random_sample(shape).astype(np.float32)
    t = np.random.randint(0, shape[1], size=idx_size, dtype=np.int32)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, a, t)
    out1 = kernel(a, t)
    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, a, t)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "shape, top_k, axis",
    [
        ((5,), 1, 0),
        ((10,), 2, 0),
        ((15,), 3, 0),
        ((5, 10), 1, 1),
        ((10, 20), 2, 1),
        ((15, 25), 3, 1),
    ],
)
def test_topk(sim_mode, shape, top_k, axis):
    np.random.seed(0)

    def kernel(a):
        values, indices = tensor_apis.topk(a, k=top_k, axis=axis)
        return values, indices

    a = np.random.random_sample(shape).astype(np.float32)

    a_torch = torch.from_numpy(a)
    values_gt, indices_gt = torch.topk(a_torch, k=top_k, dim=axis)
    values_gt = values_gt.numpy()
    indices_gt = indices_gt.numpy()

    values_sim, indices_sim = simulate_kernel_unified(kernel, sim_mode, a)
    simulate_assert_allclose(values_sim, values_gt)
    simulate_assert_allclose(indices_sim, indices_gt)

    if NEURON_AVAILABLE:
        values, indices = baremetal_run_kernel_unified(kernel, sim_mode, a)
        baremetal_assert_allclose(values, values_gt)
        baremetal_assert_allclose(indices, indices_gt)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (3, 16, (3, 3), (1, 1), (0, 0)),  # Basic case
        (3, 64, (7, 7), (2, 2), (3, 3)),  # ResNet-style first conv
        (16, 32, (3, 3), (1, 1), (1, 1)),  # With padding
        (32, 64, (3, 3), (2, 2), (1, 1)),  # With stride
        (64, 128, (1, 1), (1, 1), (0, 0)),  # 1x1 convolution
    ],
)
def test_conv2d(sim_mode, in_channels, out_channels, kernel_size, stride, padding):
    np.random.seed(0)

    def kernel(input_tensor, weight):
        return tensor_apis.conv2d(input_tensor, weight, stride=stride, padding=padding)

    # Create test inputs
    batch_size = 1
    height, width = 32, 32
    input_shape = (batch_size, in_channels, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    expected_output = F.conv2d(
        input_torch, weight_torch, stride=stride, padding=padding
    ).numpy()

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight)

    # Verify output shape and values
    assert out0.shape == expected_output.shape, (
        f"Expected shape {expected_output.shape}, got {out0.shape}"
    )
    simulate_assert_allclose(
        out0, expected_output, err_msg="Conv2d output doesn't match PyTorch reference"
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d output doesn't match PyTorch reference",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3), 1, 1),  # Scalar stride and padding
        (16, 32, (5, 5), 2, 2),  # Scalar stride and padding
    ],
)
def test_conv2d_scalar_params(
    sim_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv2d with scalar stride and padding parameters"""
    np.random.seed(0)

    def kernel(input_tensor, weight):
        return tensor_apis.conv2d(input_tensor, weight, stride=stride, padding=padding)

    # Create test inputs
    batch_size = 1
    height, width = 28, 28
    input_shape = (batch_size, in_channels, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    expected_output = F.conv2d(
        input_torch, weight_torch, stride=stride, padding=padding
    ).numpy()

    # Test simulation
    out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight)
    simulate_assert_allclose(
        out0,
        expected_output,
        err_msg="Conv2d scalar params output doesn't match PyTorch reference",
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d scalar params output doesn't match PyTorch reference",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,dilation",
    [
        (8, 16, (3, 3), (1, 1), (2, 2), (2, 2)),  # With dilation
        (16, 32, (3, 3), (1, 1), (4, 4), (3, 3)),  # Larger dilation
    ],
)
def test_conv2d_with_dilation(
    sim_mode, in_channels, out_channels, kernel_size, stride, padding, dilation
):
    """Test conv2d with dilation parameter"""
    np.random.seed(0)

    def kernel(input_tensor, weight):
        return tensor_apis.conv2d(
            input_tensor, weight, stride=stride, padding=padding, dilation=dilation
        )

    # Create test inputs
    batch_size = 1
    height, width = 32, 32
    input_shape = (batch_size, in_channels, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    expected_output = F.conv2d(
        input_torch, weight_torch, stride=stride, padding=padding, dilation=dilation
    ).numpy()

    # Test simulation
    # FIXME: dilation not supported right now in simulation
    # out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight)
    # simulate_assert_allclose(
    #     out0,
    #     expected_output,
    #     err_msg="Conv2d with dilation output doesn't match PyTorch reference",
    # )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d with dilation output doesn't match PyTorch reference",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3), (1, 1), (1, 1)),  # Basic case with bias
        (16, 32, (5, 5), (2, 2), (2, 2)),  # With stride and bias
    ],
)
def test_conv2d_with_bias(
    sim_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv2d with bias parameter"""
    np.random.seed(0)

    def kernel(input_tensor, weight, bias):
        return tensor_apis.conv2d(
            input_tensor, weight, bias=bias, stride=stride, padding=padding
        )

    # Create test inputs
    batch_size = 1
    height, width = 28, 28
    input_shape = (batch_size, in_channels, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)
    bias_shape = (out_channels,)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)
    bias = np.random.random_sample(bias_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    bias_torch = torch.from_numpy(bias)
    expected_output = F.conv2d(
        input_torch, weight_torch, bias=bias_torch, stride=stride, padding=padding
    ).numpy()

    # Test simulation
    out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight, bias)
    simulate_assert_allclose(
        out0,
        expected_output,
        err_msg="Conv2d with bias output doesn't match PyTorch reference",
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight, bias
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d with bias output doesn't match PyTorch reference",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (3, 16, (2, 3, 3), (1, 1, 1), (0, 0, 0)),  # Basic case
        (3, 1152, (2, 16, 16), (2, 16, 16), (0, 0, 0)),  # Qwen3-VL case
        (16, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)),  # With padding
        (32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1)),  # With stride
    ],
)
def test_conv3d(sim_mode, in_channels, out_channels, kernel_size, stride, padding):
    np.random.seed(0)

    def kernel(input_tensor, weight):
        return tensor_apis.conv3d(input_tensor, weight, stride=stride, padding=padding)

    # Create test inputs
    batch_size = 1
    depth, height, width = 8, 16, 16
    input_shape = (batch_size, in_channels, depth, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    expected_output = F.conv3d(
        input_torch, weight_torch, stride=stride, padding=padding
    ).numpy()

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight)

    # Verify output shape and values
    assert out0.shape == expected_output.shape, (
        f"Expected shape {expected_output.shape}, got {out0.shape}"
    )
    simulate_assert_allclose(
        out0, expected_output, err_msg="Conv3d output doesn't match PyTorch reference"
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d output doesn't match PyTorch reference",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,dilation",
    [
        (8, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), (2, 2, 2)),  # With dilation
        (16, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2)),  # Dilation with padding
    ],
)
def test_conv3d_with_dilation(
    sim_mode, in_channels, out_channels, kernel_size, stride, padding, dilation
):
    """Test conv3d with dilation parameter"""
    np.random.seed(0)

    def kernel(input_tensor, weight):
        return tensor_apis.conv3d(
            input_tensor, weight, stride=stride, padding=padding, dilation=dilation
        )

    # Create test inputs
    batch_size = 1
    depth, height, width = 16, 16, 16
    input_shape = (batch_size, in_channels, depth, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    expected_output = F.conv3d(
        input_torch, weight_torch, stride=stride, padding=padding, dilation=dilation
    ).numpy()

    # FIXME: dilation not supported right now in simulation
    # out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight)
    # simulate_assert_allclose(
    #     out0,
    #     expected_output,
    #     err_msg="Conv3d with dilation output doesn't match PyTorch reference",
    # )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d with dilation output doesn't match PyTorch",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1)),  # Basic case with bias
    ],
)
def test_conv3d_with_bias(
    sim_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv3d with bias parameter"""
    np.random.seed(0)

    def kernel(input_tensor, weight, bias):
        return tensor_apis.conv3d(
            input_tensor, weight, bias=bias, stride=stride, padding=padding
        )

    # Create test inputs
    batch_size = 1
    depth, height, width = 8, 8, 8
    input_shape = (batch_size, in_channels, depth, height, width)
    weight_shape = (out_channels, in_channels, *kernel_size)
    bias_shape = (out_channels,)

    input_tensor = np.random.random_sample(input_shape).astype(np.float32)
    weight = np.random.random_sample(weight_shape).astype(np.float32)
    bias = np.random.random_sample(bias_shape).astype(np.float32)

    # Get PyTorch ground truth
    input_torch = torch.from_numpy(input_tensor)
    weight_torch = torch.from_numpy(weight)
    bias_torch = torch.from_numpy(bias)
    expected_output = F.conv3d(
        input_torch, weight_torch, bias=bias_torch, stride=stride, padding=padding
    ).numpy()

    # Test simulation
    out0 = simulate_kernel_unified(kernel, sim_mode, input_tensor, weight, bias)
    simulate_assert_allclose(
        out0,
        expected_output,
        err_msg="Conv3d with bias output doesn't match PyTorch reference",
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(
            kernel, sim_mode, input_tensor, weight, bias
        )
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d with bias output doesn't match PyTorch reference",
        )


# Test dtype override functionality for zeros_like, empty_like, full_like
@pytest.mark.parametrize("like_fn", [np.zeros_like, np.empty_like])
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (np.float32, np.float16),
        (np.float32, np.int32),
        (np.int32, np.float32),
        (np.float16, np.float32),
        (np.int8, np.uint8),
        (np.uint8, np.int8),
    ],
)
def test_like_functions_dtype_override(sim_mode, like_fn, input_dtype, output_dtype):
    """Test zeros_like and empty_like with dtype override"""
    shape = (64, 64)  # Smaller shape for faster tests
    np.random.seed(0)

    def kernel(a):
        return like_fn(a, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)

    # Verify output dtype is correct
    assert out0.dtype == output_dtype, (
        f"Expected dtype {output_dtype}, got {out0.dtype}"
    )

    # Verify output shape matches input
    assert out0.shape == in0.shape, f"Expected shape {in0.shape}, got {out0.shape}"

    # For zeros_like, verify all values are zero
    if like_fn == np.zeros_like:
        assert np.all(out0 == 0), "zeros_like should produce all zeros"

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(kernel, sim_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == output_dtype, (
            f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )


@pytest.mark.parametrize(
    "input_dtype,output_dtype,fill_value",
    [
        (np.float32, np.float16, 2.5),
        (np.int32, np.float32, 1.0),
        (np.float32, np.int32, 42),
        (np.float16, np.float32, -3.14),
        (np.int8, np.uint8, 255),
        (np.uint8, np.int8, 127),
    ],
)
def test_full_like_dtype_override(sim_mode, input_dtype, output_dtype, fill_value):
    """Test full_like with dtype override"""
    shape = (32, 32)  # Smaller shape for faster tests
    np.random.seed(0)

    def kernel(a):
        return np.full_like(a, fill_value, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)

    # Verify output dtype is correct
    assert out0.dtype == output_dtype, (
        f"Expected dtype {output_dtype}, got {out0.dtype}"
    )

    # Verify output shape matches input
    assert out0.shape == in0.shape, f"Expected shape {in0.shape}, got {out0.shape}"

    # Verify all values match the fill_value (cast to output dtype)
    expected_fill = output_dtype(fill_value)
    assert np.all(out0 == expected_fill), (
        f"full_like should produce all {expected_fill}, got unique values: {np.unique(out0)}"
    )

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(kernel, sim_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == output_dtype, (
            f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )


@pytest.mark.parametrize("like_fn", [np.zeros_like, np.empty_like, np.full_like])
def test_like_functions_default_behavior(sim_mode, like_fn):
    """Test that like functions maintain backward compatibility when dtype is not specified"""
    shape = (32, 32)
    input_dtype = np.float32
    np.random.seed(0)

    if like_fn == np.full_like:

        def kernel(a):
            return like_fn(a, 5.0)  # No dtype specified

        fill_value = 5.0
    else:

        def kernel(a):
            return like_fn(a)  # No dtype specified

        fill_value = None

    in0 = np.random.random_sample(shape).astype(input_dtype)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0)

    # Verify output dtype matches input dtype (default behavior)
    assert out0.dtype == input_dtype, f"Expected dtype {input_dtype}, got {out0.dtype}"

    # Verify output shape matches input
    assert out0.shape == in0.shape, f"Expected shape {in0.shape}, got {out0.shape}"

    # Verify content based on function type
    if like_fn == np.zeros_like:
        assert np.all(out0 == 0), "zeros_like should produce all zeros"
    elif like_fn == np.full_like:
        assert np.all(out0 == fill_value), f"full_like should produce all {fill_value}"

    if NEURON_AVAILABLE:
        hardware_output = baremetal_run_kernel_unified(kernel, sim_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == input_dtype, (
            f"Hardware: Expected dtype {input_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )


def test_binary_op_type_promotion_pred_scalar(sim_mode):
    """Test that binary operations properly promote types when mixing pred/bool tensors with scalars."""
    shape = (64, 64)
    np.random.seed(0)

    def kernel(a, b):
        # Create a boolean/pred tensor via comparison
        ge_result = np.greater_equal(a, 0.5)  # Returns pred/bool tensor
        lt_result = np.less(b, 0.8)  # Returns pred/bool tensor

        # Logical operation produces pred tensor
        mask = np.logical_and(ge_result, lt_result)

        # Multiply pred tensor with a large scalar - this should NOT lose the scalar value
        # The scalar 1000 should be preserved, not cast to bool
        result = np.multiply(mask, 1000)

        return result

    in0 = np.random.random_sample(shape).astype(np.float32)
    in1 = np.random.random_sample(shape).astype(np.float32)

    # Test simulation - always runs
    out0 = simulate_kernel_unified(kernel, sim_mode, in0, in1)
    out1 = kernel(in0, in1)

    # The key assertion: values should be either 0 or 1000, not 0 or 1
    unique_values = np.unique(out0)
    assert 1000 in unique_values or np.all(out0 == 0), (
        f"Expected values to include 1000 (or all zeros), but got unique values: {unique_values}"
    )

    simulate_assert_allclose(out0, out1)

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(kernel, sim_mode, in0, in1)
        baremetal_assert_allclose(out1, out_baremetal)


@pytest.mark.parametrize("shape", [(16, 16), (8, 8, 8), (4, 4, 4, 4)])
@pytest.mark.parametrize(
    "input_dtype,output_dtype", [(np.float32, np.float16), (np.int32, np.float32)]
)
def test_like_functions_various_shapes(sim_mode, shape, input_dtype, output_dtype):
    """Test dtype override with various tensor shapes"""
    np.random.seed(0)

    def kernel_zeros(a):
        return np.zeros_like(a, dtype=output_dtype)

    def kernel_empty(a):
        return np.empty_like(a, dtype=output_dtype)

    def kernel_full(a):
        return np.full_like(a, 7.5, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    # Test all three functions
    for kernel_fn, expected_fill in [
        (kernel_zeros, 0),
        (kernel_empty, 0),
        (kernel_full, output_dtype(7.5)),
    ]:
        # Test simulation - always runs
        out0 = simulate_kernel_unified(kernel_fn, sim_mode, in0)

        # Verify output properties
        assert out0.dtype == output_dtype, (
            f"Expected dtype {output_dtype}, got {out0.dtype}"
        )
        assert out0.shape == in0.shape, f"Expected shape {in0.shape}, got {out0.shape}"

        if kernel_fn == kernel_full:
            assert np.all(out0 == expected_fill), (
                f"full_like should produce all {expected_fill}"
            )
        elif kernel_fn == kernel_zeros:
            assert np.all(out0 == expected_fill), (
                f"zeros_like should produce all {expected_fill}"
            )
        else:  # empty like
            pass

        if NEURON_AVAILABLE:
            hardware_output = baremetal_run_kernel_unified(kernel_fn, sim_mode, in0)

            # Verify hardware output
            assert hardware_output.dtype == output_dtype, (
                f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
            )
            assert hardware_output.shape == in0.shape, (
                f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
            )


@pytest.mark.parametrize(
    "dtype_name",
    [
        "bfloat16",
        pytest.param(
            "float8_e5m2",
            marks=pytest.mark.xfail(reason="float8_e5m2 backend support missing"),
        ),
        "float8_e4m3",
        pytest.param(
            "float8_e4m3fn",
            marks=pytest.mark.xfail(reason="float8_e4m3fn backend support missing"),
        ),
    ],
)
def test_ml_dtypes_constant_encoding(sim_mode, dtype_name):
    """Test that ml_dtypes constants (bfloat16, float8) are correctly encoded in HLO.

    This is a regression test for a bug where ml_dtypes constants were incorrectly
    encoded: int(1.0) = 1 was used as the raw byte value instead of the proper
    floating-point representation.
    """
    try:
        import ml_dtypes
    except ImportError:
        pytest.skip("ml_dtypes not available")

    # Get the dtype from ml_dtypes
    dtype = getattr(ml_dtypes, dtype_name)

    shape = (32, 32)
    np.random.seed(0)

    def kernel_with_constant_one(x):
        # This operation requires constant 1.0 to be correctly encoded
        # If 1.0 is encoded as ~0, the result will be ~0 instead of ~1
        return x * 0 + 1  # Should produce all 1s

    in0 = np.random.random_sample(shape).astype(dtype)

    # Test simulation
    out0 = simulate_kernel_unified(kernel_with_constant_one, sim_mode, in0)

    # The output should be all 1s (or very close to 1)
    expected = np.ones(shape, dtype=dtype)

    atol = 0.1 if "float8" in dtype_name else 1e-3
    assert np.allclose(
        out0.astype(np.float32), expected.astype(np.float32), atol=atol
    ), (
        f"Expected all 1s for {dtype_name}, but got values around {np.mean(out0.astype(np.float32))}. "
        f"This indicates {dtype_name} constant encoding bug."
    )

    if NEURON_AVAILABLE:
        out_baremetal = baremetal_run_kernel_unified(
            kernel_with_constant_one, sim_mode, in0
        )
        baremetal_assert_allclose(
            expected.astype(np.float32), out_baremetal.astype(np.float32)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
