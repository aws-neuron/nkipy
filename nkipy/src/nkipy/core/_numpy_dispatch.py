# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy array function dispatch for NKIPy tensors.

This module registers numpy functions to dispatch to the ops module implementations.
"""

import numpy as np

from nkipy.core.tensor import NKIPyTensorRef


def _register_numpy_api(numpy_func, ops_func):
    """Register a numpy function to dispatch to an ops function.

    Args:
        numpy_func: The numpy function (e.g., np.add)
        ops_func: The ops function to dispatch to
    """

    def impl_hlo(*args, **kwargs):
        return ops_func(*args, **kwargs)

    NKIPyTensorRef._tensor_apis[numpy_func] = impl_hlo


def register_all_numpy_apis():
    """Register all numpy APIs to use ops module implementations."""
    from nkipy.core import ops

    # Binary operations
    _register_numpy_api(np.add, ops.add)
    _register_numpy_api(np.subtract, ops.subtract)
    _register_numpy_api(np.multiply, ops.multiply)
    _register_numpy_api(np.divide, ops.divide)
    _register_numpy_api(np.power, ops.power)
    _register_numpy_api(np.maximum, ops.maximum)
    _register_numpy_api(np.minimum, ops.minimum)
    _register_numpy_api(np.bitwise_and, ops.bitwise_and)
    _register_numpy_api(np.bitwise_or, ops.bitwise_or)
    _register_numpy_api(np.bitwise_xor, ops.bitwise_xor)
    # NOT SUPPORTED:
    # np.arctan2 - not supported on hardware
    # np.floor_divide - not supported
    # np.mod / np.remainder - hardware doesn't support it
    #   (workaround: a % b = a - b * floor(a/b))

    # Comparison operations
    _register_numpy_api(np.equal, ops.equal)
    _register_numpy_api(np.not_equal, ops.not_equal)
    _register_numpy_api(np.greater, ops.greater)
    _register_numpy_api(np.greater_equal, ops.greater_equal)
    _register_numpy_api(np.less, ops.less)
    _register_numpy_api(np.less_equal, ops.less_equal)
    _register_numpy_api(np.logical_and, ops.logical_and)
    _register_numpy_api(np.logical_or, ops.logical_or)
    _register_numpy_api(np.logical_xor, ops.logical_xor)

    # Unary operations
    _register_numpy_api(np.abs, ops.abs)
    _register_numpy_api(np.exp, ops.exp)
    _register_numpy_api(np.log, ops.log)
    _register_numpy_api(np.sqrt, ops.sqrt)
    _register_numpy_api(np.square, ops.square)
    _register_numpy_api(np.negative, ops.negative)
    _register_numpy_api(np.reciprocal, ops.reciprocal)
    _register_numpy_api(np.sin, ops.sin)
    _register_numpy_api(np.cos, ops.cos)
    _register_numpy_api(np.tan, ops.tan)
    _register_numpy_api(np.arctan, ops.arctan)
    _register_numpy_api(np.tanh, ops.tanh)
    _register_numpy_api(np.ceil, ops.ceil)
    _register_numpy_api(np.floor, ops.floor)
    _register_numpy_api(np.rint, ops.rint)
    _register_numpy_api(np.trunc, ops.trunc)
    _register_numpy_api(np.sign, ops.sign)
    _register_numpy_api(np.invert, ops.invert)
    _register_numpy_api(np.bitwise_not, ops.bitwise_not)
    _register_numpy_api(np.logical_not, ops.logical_not)
    # NOT SUPPORTED:
    # np.positive - not supported, use np.copy for "y = +x" operation
    # np.round - not a supported activation

    # Reduction operations
    _register_numpy_api(np.sum, ops.sum)
    _register_numpy_api(np.max, ops.max)
    _register_numpy_api(np.min, ops.min)
    _register_numpy_api(np.mean, ops.mean)
    _register_numpy_api(np.any, ops.any)
    _register_numpy_api(np.var, ops.var)

    # Linear algebra
    _register_numpy_api(np.matmul, ops.matmul)
    _register_numpy_api(np.dot, ops.dot)

    # Transform operations
    _register_numpy_api(np.reshape, ops.reshape)
    _register_numpy_api(np.transpose, ops.transpose)
    _register_numpy_api(np.expand_dims, ops.expand_dims)
    _register_numpy_api(np.concatenate, ops.concatenate)
    _register_numpy_api(np.split, ops.split)
    _register_numpy_api(np.copy, ops.copy)
    _register_numpy_api(np.repeat, ops.repeat)

    # Creation operations
    _register_numpy_api(np.zeros_like, ops.zeros_like)
    _register_numpy_api(np.empty_like, ops.empty_like)
    _register_numpy_api(np.full_like, ops.full_like)

    # Indexing operations
    _register_numpy_api(np.where, ops.where)
    _register_numpy_api(np.take, ops.take)
    _register_numpy_api(np.take_along_axis, ops.take_along_axis)
    _register_numpy_api(np.put_along_axis, ops.put_along_axis)

    # Broadcast and copy operations
    _register_numpy_api(np.broadcast_to, ops.broadcast_to)
    _register_numpy_api(np.copyto, ops.copyto)
