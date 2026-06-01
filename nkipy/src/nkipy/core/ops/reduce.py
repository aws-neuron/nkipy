# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction operations: sum, max, min, mean, var"""

import numpy as np

from nkipy.core.ops._registry import Op


def _calculate_reduction_count(x_shape, axis):
    """Calculate the number of elements being reduced."""
    if axis is None:
        return int(np.prod(x_shape))
    elif isinstance(axis, int):
        dim = axis if axis >= 0 else len(x_shape) + axis
        return x_shape[dim]
    elif isinstance(axis, (list, tuple)):
        return int(
            np.prod([x_shape[ax if ax >= 0 else len(x_shape) + ax] for ax in axis])
        )
    else:
        return x_shape[axis]


# -----------------------------------------------------------------------------
# Primitive reduction operations
# -----------------------------------------------------------------------------
sum = Op("sum")
max = Op("max")
min = Op("min")
prod = Op("prod")
argmax = Op("argmax")
argmin = Op("argmin")
cumsum = Op("cumsum")


@cumsum.composed_impl
def _cumsum(x, axis=None, dtype=None):
    from nkipy.core.ops.creation import constant
    from nkipy.core.ops.linalg import matmul
    from nkipy.core.ops.transform import astype, reshape, transpose

    x_shape = x.shape
    ndim = len(x_shape)

    if axis is None:
        total = int(np.prod(x_shape))
        x = reshape(x, (total,))
        x_shape = (total,)
        ndim = 1
        axis = 0
    elif axis < 0:
        axis = ndim + axis

    N = x_shape[axis]

    tri = np.triu(np.ones((N, N), dtype=np.float32))
    tri_const = constant(tri)

    if ndim == 1:
        x_2d = reshape(x, (1, N))
        result_2d = matmul(x_2d, tri_const)
        result = reshape(result_2d, (N,))
    elif axis == ndim - 1:
        batch_size = int(np.prod(x_shape[:-1]))
        x_2d = reshape(x, (batch_size, N))
        result_2d = matmul(x_2d, tri_const)
        result = reshape(result_2d, x_shape)
    else:
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x_t = transpose(x, axes=perm)
        x_t_shape = tuple(x_shape[p] for p in perm)

        batch_size = int(np.prod(x_t_shape[:-1]))
        x_2d = reshape(x_t, (batch_size, N))
        result_2d = matmul(x_2d, tri_const)
        result_t = reshape(result_2d, x_t_shape)
        result = transpose(result_t, axes=perm)

    if dtype is not None and result.dtype != np.dtype(dtype):
        result = astype(result, np.dtype(dtype))

    return result

# -----------------------------------------------------------------------------
# Composed reduction operations
# -----------------------------------------------------------------------------

mean = Op("mean")


@mean.composed_impl
def _mean(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    from nkipy.core.ops.binary import divide

    if dtype is not None:
        from nkipy.core.ops.transform import astype

        x = astype(x, np.dtype(dtype))

    sum_result = sum(x, axis=axis, keepdims=keepdims)
    count = _calculate_reduction_count(x.shape, axis)

    return divide(sum_result, float(count))


any = Op("any")


@any.composed_impl
def _any(x, axis=None, out=None, dtype=None, keepdims=False):
    from nkipy.core.ops.binary import not_equal
    from nkipy.core.ops.transform import astype

    non_zero = not_equal(x, 0)
    non_zero_i32 = astype(non_zero, np.dtype(np.int32))
    summed = sum(non_zero_i32, axis=axis, keepdims=keepdims)

    return not_equal(summed, 0)


var = Op("var")


@var.composed_impl
def _var(x, axis=None, out=None, dtype=None, keepdims=False, ddof=0):
    from nkipy.core.ops.binary import divide, multiply, subtract

    if dtype is not None:
        from nkipy.core.ops.transform import astype

        x = astype(x, np.dtype(dtype))

    mean_x = mean(x, axis=axis, keepdims=True)

    centered = subtract(x, mean_x)
    squared = multiply(centered, centered)
    sum_squared = sum(squared, axis=axis, keepdims=keepdims)
    count = _calculate_reduction_count(x.shape, axis)

    denom = count - ddof if count - ddof > 0 else 0
    return divide(sum_squared, float(denom))


std = Op("std")


@std.composed_impl
def _std(x, axis=None, out=None, dtype=None, keepdims=False, ddof=0):
    from nkipy.core.ops.unary import sqrt

    return sqrt(var(x, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof))


count_nonzero = Op("count_nonzero")


@count_nonzero.composed_impl
def _count_nonzero(x, axis=None, keepdims=False):
    from nkipy.core.ops.binary import not_equal
    from nkipy.core.ops.transform import astype

    non_zero = not_equal(x, 0)
    non_zero_i32 = astype(non_zero, np.dtype(np.int32))
    return sum(non_zero_i32, axis=axis, keepdims=keepdims)
