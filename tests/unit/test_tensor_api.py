# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for regular tensor API using pytest with hardware testing
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
    cpu_assert_allclose,
    test_on_device,
    trace_and_compile,
    trace_mode,  # noqa: F401 - pytest fixture
)


def test_local_softmax_return_0(trace_mode):
    def local_softmax(a, axis=-1):
        ma = np.max(a, axis=axis, keepdims=True)
        ea = np.exp(np.subtract(a, ma))
        s = np.sum(ea, axis=axis, keepdims=True)
        return np.divide(ea, s)

    shape = (256, 256)
    dtype = np.float32

    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    expected = local_softmax(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(local_softmax, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(local_softmax, trace_mode, in0)


def test_local_softmax_return_1(trace_mode):
    def local_softmax(a, axis=-1):
        ma = np.max(a, axis=axis, keepdims=True)
        ea = np.exp(np.subtract(a, ma))
        s = np.sum(ea, axis=axis, keepdims=True)
        return np.divide(ea, s), s

    shape = (256, 256)
    dtype = np.float32

    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    expected0, expected1 = local_softmax(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(local_softmax, trace_mode, in0)
        baremetal_assert_allclose(expected0, out_device[0])
        baremetal_assert_allclose(expected1, out_device[1])
    else:
        trace_and_compile(local_softmax, trace_mode, in0)


def test_expand_dims_0(trace_mode):
    axis = -1

    def kernel(a):
        return np.expand_dims(a, axis=axis)

    shape = [256]
    dtype = np.float32

    in0 = np.random.uniform(high=1.0, low=0.0, size=shape).astype(dtype)

    expected = np.expand_dims(in0, axis=axis)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
        np.reciprocal,
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
def test_unary(trace_mode, np_fn, dtype):
    shape = (256, 256)

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize("np_fn", [np.bitwise_not, np.logical_not])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_unary_bitwise(trace_mode, np_fn, dtype):
    shape = (256, 256)

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_binary(trace_mode, np_fn, dtype):
    shape = (256, 256)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)


@pytest.mark.parametrize("np_fn", [np.bitwise_and, np.bitwise_xor, np.bitwise_or])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_binary_bitwise(trace_mode, np_fn, dtype):
    shape = (256, 256)

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


@pytest.mark.parametrize(
    "np_fn",
    [np.equal, np.not_equal, np.greater, np.less_equal, np.less, np.greater_equal],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_comparison(trace_mode, np_fn, dtype):
    shape = (128, 128)  # Smaller shape for faster hardware tests

    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


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
def test_contract(trace_mode, np_fn, lhs_shape, rhs_shape, out_shape):
    def kernel(a, b):
        return np_fn(a, b)

    in0 = np.random.random_sample(lhs_shape).astype(np.float32)
    in1 = np.random.random_sample(rhs_shape).astype(np.float32)

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


@pytest.mark.parametrize("np_fn", [np.mean, np.max, np.min, np.sum])
@pytest.mark.parametrize(
    "shape,dtype,axis",
    [
        ((128, 128), np.float32, (-1,))  # Smaller shape for hardware tests
    ],
)
def test_reduction(trace_mode, np_fn, shape, dtype, axis):
    def kernel(a):
        return np_fn(a, axis=axis)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


def test_multiple_sum_different_dtypes(trace_mode):
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

    a = np.random.random_sample(shape).astype(np.float32)

    expected_f32 = np.sum(a.astype(np.float32), axis=-1, keepdims=True)
    expected_f16 = np.sum(a.astype(np.float16), axis=-1, keepdims=True)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, a)
        baremetal_assert_allclose(expected_f32, out_device[0])
        baremetal_assert_allclose(expected_f16, out_device[1])
    else:
        trace_and_compile(kernel, trace_mode, a)


@pytest.mark.parametrize("np_fn", [np.mean, np.max, np.min, np.sum])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((64, 64), np.float32),
        ((32, 32, 32), np.float32),
        ((16, 16, 16, 16), np.float32),
    ],
)
def test_reduction_axis_none(trace_mode, np_fn, shape, dtype):
    """Test reduction operations with axis=None (reduce over all axes)"""

    def kernel(a):
        return np_fn(a)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_reduction_axis_none_keepdims(trace_mode, np_fn, shape, dtype, keepdims):
    """Test reduction operations with axis=None and keepdims parameter"""

    def kernel(a):
        return np_fn(a, axis=None, keepdims=keepdims)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_broadcast_to(trace_mode, src_shape, dst_shape):
    dtype = np.float32

    def kernel(a, shape):
        return np.broadcast_to(a, shape=shape)

    in0 = np.random.random_sample(src_shape).astype(dtype)

    expected = kernel(in0, dst_shape)

    if NEURON_AVAILABLE:
        # For baremetal, we need a kernel with fixed shape parameter
        def kernel_fixed_shape(a):
            return np.broadcast_to(a, shape=dst_shape)

        out_device = test_on_device(kernel_fixed_shape, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, dst_shape)


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
def test_transpose(trace_mode, shape, axes):
    dtype = np.float32

    def kernel(a):
        return np.transpose(a, axes=axes)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_repeat(trace_mode, shape, repeats, axis):
    dtype = np.float32

    def kernel(a):
        return np.repeat(a, repeats=repeats, axis=axis)

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_take(trace_mode, a, indices, axis):
    dtype = np.float32

    def kernel(a, indices, axis):
        return np.take(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = axis

    expected = kernel(in0, in1, in2)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2)


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
def test_take_numpy_indices(trace_mode, a, indices, axis):
    dtype = np.float32

    def kernel(a, axis):
        return np.take(a, indices=np.array(indices).astype(np.uint32), axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = axis

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), 0, None),
        ((2, 3), 1, None),
        ((2, 3), -1, None),
    ],
)
def test_take_scalar(trace_mode, a, indices, axis):
    dtype = np.float32

    def kernel(a, indices, axis):
        return np.take(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = indices
    in2 = axis

    expected = kernel(in0, in1, in2)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2)


@pytest.mark.parametrize(
    "a,indices,values,axis",
    [
        ((2, 3), [[1, 0, 0]], [[4, 5, 6]], 0),
        ((2, 3), [[1], [0]], [[4], [5]], 1),
        ((2, 3), [1, 0], [4, 5], None),
    ],
)
def test_put_along_axis(trace_mode, a, indices, values, axis):
    # FIXME: support put_along_axis with proper doc
    if trace_mode == "hlo":
        pytest.skip("put_along_axis not yet supported in HLO mode")

    dtype = np.float32

    def kernel(a, indices, values, axis, is_hardware=False):
        b = np.copy(a)
        if trace_mode == "hlo" and is_hardware:
            b = np.put_along_axis(b, indices=indices, values=values, axis=axis)
        else:
            np.put_along_axis(b, indices=indices, values=values, axis=axis)

        return b

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = np.array(values, dtype=dtype)
    in3 = axis

    expected = kernel(in0, in1, in2, in3)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2, in3, True)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2, in3)


@pytest.mark.parametrize(
    "a,indices,values,axis",
    [
        ((2, 3), [[1, 0, 0]], 100, 0),
        ((2, 3), [[1], [0]], 3.0, 1),
        ((2, 3), [1, 0], 2, None),
    ],
)
def test_put_along_axis_scalar_value(trace_mode, a, indices, values, axis):
    # FIXME: support put_along_axis with proper doc
    if trace_mode == "hlo":
        pytest.skip("put_along_axis not yet supported in HLO mode")

    dtype = np.float32

    def kernel(a, indices, values, axis, is_hardware=False):
        b = np.copy(a)
        if trace_mode == "hlo" and is_hardware:
            b = np.put_along_axis(b, indices=indices, values=values, axis=axis)
        else:
            np.put_along_axis(b, indices=indices, values=values, axis=axis)

        return b

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = values
    in3 = axis

    expected = kernel(in0, in1, in2, in3)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2, in3, True)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2, in3, False)


@pytest.mark.parametrize(
    "a,indices,axis",
    [
        ((2, 3), [[1, 0, 0]], 0),
        ((2, 3), [[1], [0]], 1),
        ((2, 3), [1, 0], None),
    ],
)
def test_take_along_axis(trace_mode, a, indices, axis):
    dtype = np.float32

    def kernel(a, indices, axis):
        return np.take_along_axis(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = np.array(indices).astype(np.uint32)
    in2 = axis

    expected = kernel(in0, in1, in2)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2)


@pytest.mark.parametrize(
    "a,axis",
    [
        ((2048, 64), 0),
        ((10, 20, 30), None),
    ],
)
def test_take_along_axis_random(trace_mode, a, axis):
    dtype = np.float32

    if axis == 0:
        indices = np.random.randint(0, a[0], size=(128, 1)).astype(np.uint32)
    elif axis is None:
        indices = np.random.randint(0, np.prod(a), size=(50,)).astype(np.uint32)

    def kernel(a, indices, axis):
        return np.take_along_axis(a, indices=indices, axis=axis)

    in0 = np.random.random_sample(a).astype(dtype)
    in1 = indices
    in2 = axis

    expected = kernel(in0, in1, in2)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1, in2)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1, in2)


@pytest.mark.parametrize("B,L,HD,QHN", [(1, 1, 2, 1), (2, 3, 10, 4)])
def test_rotary_embed(trace_mode, B, L, HD, QHN):
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

    expected = kernel(x, cos_idx, sin_idx, freqs_cos, freqs_sin)

    if NEURON_AVAILABLE:
        out_device = test_on_device(
            kernel, trace_mode, x, cos_idx, sin_idx, freqs_cos, freqs_sin
        )
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, x, cos_idx, sin_idx, freqs_cos, freqs_sin)


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
def test_where(trace_mode, shape):
    dtype = np.float32

    def kernel(cond, x, y):
        return np.where(cond, x, y)

    condition = np.random.choice([True, False], size=shape).astype(np.uint8)
    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(condition, in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, condition, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, condition, in0, in1)


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
def test_where_first_dim(trace_mode, shape):
    dtype = np.float32

    def kernel(cond, x, y):
        return np.where(cond, x, y)

    condition = np.random.choice([True, False], size=shape[:1]).astype(np.uint8)

    # expand to be the same number of dims as shape
    condition = np.expand_dims(condition, axis=tuple(range(1, len(shape))))

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(condition, in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, condition, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, condition, in0, in1)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4, 5),
    ],
)
def test_where_ndarray_cond_dim2(trace_mode, shape):
    dtype = np.float32

    condition = np.random.choice([True, False], size=shape[:1])

    def kernel(x, y):
        cond = np.expand_dims(condition, axis=tuple(range(1, len(shape))))
        return np.where(cond, x, y)

    in0 = np.random.random_sample(shape).astype(dtype)
    in1 = np.random.random_sample(shape).astype(dtype)

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


@pytest.mark.parametrize(
    "shape,idx_size",
    [((5, 10, 15), 3), ((10, 20, 30), 5), ((15, 25, 35), 7)],
)
def test_slice_assignment(trace_mode, shape, idx_size):
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

    expected = kernel(np.copy(a), b, t)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, np.copy(a), b, t)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, np.copy(a), b, t)


@pytest.mark.parametrize(
    "shape,indices",
    [((5, 10, 15), [0, 2, 2])],
)
def test_slice_assignment_indeterministic(trace_mode, shape, indices):
    def kernel(a, b, t):
        a[:, t, :] = b
        return a

    a = np.random.random_sample(shape).astype(np.float32)
    t = np.array(indices).astype(np.uint32)

    # Shape for b should match a[:, t, :]
    b_shape = (shape[0], len(indices), shape[2])
    b = np.random.random_sample(b_shape).astype(np.float32)

    if NEURON_AVAILABLE:
        a_after = test_on_device(kernel, trace_mode, np.copy(a), b, t)

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
    else:
        trace_and_compile(kernel, trace_mode, np.copy(a), b, t)


@pytest.mark.parametrize(
    "shape,idx_size",
    [
        ((5, 10, 15), 3),
        ((10, 20, 30), 5),
        ((15, 25, 35), 7),
    ],
)
def test_slice_extraction(trace_mode, shape, idx_size):
    def kernel(a, t):
        return a[:, t, :]

    # Create random input array and indices
    a = np.random.random_sample(shape).astype(np.float32)
    t = np.random.randint(0, shape[1], size=idx_size, dtype=np.int32)

    expected = kernel(a, t)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, a, t)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, a, t)


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
def test_topk(trace_mode, shape, top_k, axis):
    def kernel(a):
        values, indices = tensor_apis.topk(a, k=top_k, axis=axis)
        return values, indices

    a = np.random.random_sample(shape).astype(np.float32)

    a_torch = torch.from_numpy(a)
    values_gt, indices_gt = torch.topk(a_torch, k=top_k, dim=axis)
    values_gt = values_gt.numpy()
    indices_gt = indices_gt.numpy()

    values_cpu, indices_cpu = kernel(a)
    cpu_assert_allclose(values_cpu, values_gt)
    cpu_assert_allclose(indices_cpu, indices_gt)

    if NEURON_AVAILABLE:
        values, indices = test_on_device(kernel, trace_mode, a)
        baremetal_assert_allclose(values, values_gt)
        baremetal_assert_allclose(indices, indices_gt)
    else:
        trace_and_compile(kernel, trace_mode, a)


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
def test_conv2d(trace_mode, in_channels, out_channels, kernel_size, stride, padding):
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

    cpu_output = kernel(input_tensor, weight)
    cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3), 1, 1),  # Scalar stride and padding
        (16, 32, (5, 5), 2, 2),  # Scalar stride and padding
    ],
)
def test_conv2d_scalar_params(
    trace_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv2d with scalar stride and padding parameters"""

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

    cpu_output = kernel(input_tensor, weight)
    cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d scalar params output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,dilation",
    [
        (8, 16, (3, 3), (1, 1), (2, 2), (2, 2)),  # With dilation
        (16, 32, (3, 3), (1, 1), (4, 4), (3, 3)),  # Larger dilation
    ],
)
def test_conv2d_with_dilation(
    trace_mode, in_channels, out_channels, kernel_size, stride, padding, dilation
):
    """Test conv2d with dilation parameter"""

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

    # FIXME: dilation not supported right now in CPU backend
    # cpu_output = kernel(input_tensor, weight)
    # cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d with dilation output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3), (1, 1), (1, 1)),  # Basic case with bias
        (16, 32, (5, 5), (2, 2), (2, 2)),  # With stride and bias
    ],
)
def test_conv2d_with_bias(
    trace_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv2d with bias parameter"""

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

    cpu_output = kernel(input_tensor, weight, bias)
    cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight, bias)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv2d with bias output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight, bias)


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
def test_conv3d(trace_mode, in_channels, out_channels, kernel_size, stride, padding):
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

    cpu_output = kernel(input_tensor, weight)
    cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,dilation",
    [
        (8, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), (2, 2, 2)),  # With dilation
        (16, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2)),  # Dilation with padding
    ],
)
def test_conv3d_with_dilation(
    trace_mode, in_channels, out_channels, kernel_size, stride, padding, dilation
):
    """Test conv3d with dilation parameter"""

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

    # FIXME: dilation not supported right now in CPU backend
    # cpu_output = kernel(input_tensor, weight)
    # cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d with dilation output doesn't match PyTorch",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding",
    [
        (8, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1)),  # Basic case with bias
    ],
)
def test_conv3d_with_bias(
    trace_mode, in_channels, out_channels, kernel_size, stride, padding
):
    """Test conv3d with bias parameter"""

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

    cpu_output = kernel(input_tensor, weight, bias)
    cpu_assert_allclose(cpu_output, expected_output)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, input_tensor, weight, bias)
        baremetal_assert_allclose(
            hardware_output,
            expected_output,
            err_msg="Hardware Conv3d with bias output doesn't match PyTorch reference",
        )
    else:
        trace_and_compile(kernel, trace_mode, input_tensor, weight, bias)


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
def test_like_functions_dtype_override(trace_mode, like_fn, input_dtype, output_dtype):
    """Test zeros_like and empty_like with dtype override"""
    shape = (64, 64)  # Smaller shape for faster tests

    def kernel(a):
        return like_fn(a, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == output_dtype, (
            f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )
    else:
        trace_and_compile(kernel, trace_mode, in0)


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
def test_full_like_dtype_override(trace_mode, input_dtype, output_dtype, fill_value):
    """Test full_like with dtype override"""
    shape = (32, 32)  # Smaller shape for faster tests

    def kernel(a):
        return np.full_like(a, fill_value, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == output_dtype, (
            f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )
    else:
        trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize("like_fn", [np.zeros_like, np.empty_like, np.full_like])
def test_like_functions_default_behavior(trace_mode, like_fn):
    """Test that like functions maintain backward compatibility when dtype is not specified"""
    shape = (32, 32)
    input_dtype = np.float32

    if like_fn == np.full_like:

        def kernel(a):
            return like_fn(a, 5.0)  # No dtype specified

    else:

        def kernel(a):
            return like_fn(a)  # No dtype specified

    in0 = np.random.random_sample(shape).astype(input_dtype)

    if NEURON_AVAILABLE:
        hardware_output = test_on_device(kernel, trace_mode, in0)

        # Verify hardware output has correct dtype and shape
        assert hardware_output.dtype == input_dtype, (
            f"Hardware: Expected dtype {input_dtype}, got {hardware_output.dtype}"
        )
        assert hardware_output.shape == in0.shape, (
            f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
        )
    else:
        trace_and_compile(kernel, trace_mode, in0)


def test_binary_op_type_promotion_pred_scalar(trace_mode):
    """Test that binary operations properly promote types when mixing pred/bool tensors with scalars."""
    shape = (64, 64)

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

    expected = kernel(in0, in1)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, in0, in1)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0, in1)


@pytest.mark.parametrize("shape", [(16, 16), (8, 8, 8), (4, 4, 4, 4)])
@pytest.mark.parametrize(
    "input_dtype,output_dtype", [(np.float32, np.float16), (np.int32, np.float32)]
)
def test_like_functions_various_shapes(trace_mode, shape, input_dtype, output_dtype):
    """Test dtype override with various tensor shapes"""

    def kernel_zeros(a):
        return np.zeros_like(a, dtype=output_dtype)

    def kernel_empty(a):
        return np.empty_like(a, dtype=output_dtype)

    def kernel_full(a):
        return np.full_like(a, 7.5, dtype=output_dtype)

    in0 = np.random.random_sample(shape).astype(input_dtype)

    # Test all three functions
    for kernel_fn in [kernel_zeros, kernel_empty, kernel_full]:
        if NEURON_AVAILABLE:
            hardware_output = test_on_device(kernel_fn, trace_mode, in0)

            # Verify hardware output
            assert hardware_output.dtype == output_dtype, (
                f"Hardware: Expected dtype {output_dtype}, got {hardware_output.dtype}"
            )
            assert hardware_output.shape == in0.shape, (
                f"Hardware: Expected shape {in0.shape}, got {hardware_output.shape}"
            )
        else:
            trace_and_compile(kernel_fn, trace_mode, in0)


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
def test_ml_dtypes_constant_encoding(trace_mode, dtype_name):
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

    def kernel_with_constant_one(x):
        # This operation requires constant 1.0 to be correctly encoded
        # If 1.0 is encoded as ~0, the result will be ~0 instead of ~1
        return x * 0 + 1  # Should produce all 1s

    in0 = np.random.random_sample(shape).astype(dtype)

    expected = np.ones(shape, dtype=dtype)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel_with_constant_one, trace_mode, in0)
        baremetal_assert_allclose(
            expected.astype(np.float32), out_device.astype(np.float32)
        )
    else:
        trace_and_compile(kernel_with_constant_one, trace_mode, in0)


def test_passthrough_identity(trace_mode):
    """Test that returning an unmodified input works (single output)."""

    def kernel(x):
        return x

    shape = (4, 4)

    x = np.random.random_sample(shape).astype(np.float32)

    if NEURON_AVAILABLE:
        out_device = test_on_device(kernel, trace_mode, x)
        baremetal_assert_allclose(out_device, x)
    else:
        trace_and_compile(kernel, trace_mode, x)


def test_passthrough_with_compute(trace_mode):
    """Test returning both a computed result and an unmodified input."""

    def kernel(a, b):
        c = np.add(a, b, dtype=np.float32)
        return (c, a)

    shape = (4, 4)

    a = np.random.random_sample(shape).astype(np.float32)
    b = np.random.random_sample(shape).astype(np.float32)

    expected = np.add(a, b, dtype=np.float32)

    if NEURON_AVAILABLE:
        c_hw, a_hw = test_on_device(kernel, trace_mode, a, b)
        baremetal_assert_allclose(c_hw, expected)
        baremetal_assert_allclose(a_hw, a)
    else:
        trace_and_compile(kernel, trace_mode, a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
