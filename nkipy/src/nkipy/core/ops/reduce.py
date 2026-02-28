# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction operations: sum, max, min, mean, var"""

import numpy as np

from nkipy.core.ops._registry import Op

# =============================================================================
# HLO Implementation
# =============================================================================


def _build_reduction_hlo(
    x, np_op, axis=None, out=None, dtype=None, keepdims=False, initial=None
):
    """Build a reduction HLO operation."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Map numpy reduction ops to HLO reduce computation names
    reduce_op_map = {
        np.sum: "add",
        np.max: "maximum",
        np.min: "minimum",
    }

    if np_op not in reduce_op_map:
        raise NotImplementedError(
            f"Reduction operation {np_op} not yet supported in HLO tracing"
        )

    hlo_op = reduce_op_map[np_op]

    # Handle axis parameter - normalize to tuple
    if axis is None:
        dimensions_to_reduce = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        dim = axis if axis >= 0 else len(x.shape) + axis
        dimensions_to_reduce = (dim,)
    elif isinstance(axis, (list, tuple)):
        dimensions_to_reduce = tuple(
            ax if ax >= 0 else len(x.shape) + ax for ax in axis
        )
    else:
        dimensions_to_reduce = (axis,)

    # Calculate output shape - HLO reduce always removes dimensions
    reduced_shape = tuple(
        s for i, s in enumerate(x.shape) if i not in dimensions_to_reduce
    )

    # Create init value based on operation as a constant tensor
    init_values = {
        "add": 0.0,
        "maximum": float("-inf"),
        "minimum": float("inf"),
    }
    init_value = init_values[hlo_op]

    # Create init value as a scalar constant tensor
    init_tensor = as_hlo_tensor(ctx, init_value, x.dtype)

    # Build the reduce operation
    result_tensor = ctx.build_op(
        "reduce",
        [x, init_tensor],
        reduced_shape,
        x.dtype,
        {
            "dimensions": list(dimensions_to_reduce),
            "computation": hlo_op,
        },
    )

    # If keepdims is True, reshape to add back the reduced dimensions as size 1
    if keepdims:
        keepdims_shape = tuple(
            1 if i in dimensions_to_reduce else s for i, s in enumerate(x.shape)
        )
        result_tensor = ctx.build_op(
            "reshape", [result_tensor], keepdims_shape, x.dtype
        )

    return NKIPyTensorRef(result_tensor)


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
# Factory function for reduction ops
# -----------------------------------------------------------------------------
def _make_reduction_op(name: str, np_op) -> Op:
    """Create a reduction Op with IR and HLO implementations."""
    op = Op(name)

    @op.impl("hlo")
    def _impl_hlo(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
        return _build_reduction_hlo(
            x,
            np_op,
            axis=axis,
            out=out,
            dtype=dtype,
            keepdims=keepdims,
            initial=initial,
        )

    return op


# -----------------------------------------------------------------------------
# Reduction operations
# -----------------------------------------------------------------------------
sum = _make_reduction_op("sum", np.sum)
max = _make_reduction_op("max", np.max)
min = _make_reduction_op("min", np.min)

# mean: sum / count - simplified using ops
mean = Op("mean")


@mean.impl("hlo")
def _mean_hlo(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    """Mean operation: sum(x) / count."""
    from nkipy.core.ops.binary import divide

    if dtype is not None:
        from nkipy.core.ops.transform import astype

        x = astype(x, np.dtype(dtype))

    sum_result = sum(x, axis=axis, keepdims=keepdims)
    count = _calculate_reduction_count(x.shape, axis)

    return divide(sum_result, float(count))


# -----------------------------------------------------------------------------
# any - special reduction that checks if any element is True
# -----------------------------------------------------------------------------
any = Op("any")


@any.impl("hlo")
def _any_hlo(x, axis=None, out=None, dtype=None, keepdims=False):
    """Check if any element is True along the given axis.

    any(x) = (sum(x != 0) != 0)
    """
    from nkipy.core.ops.binary import not_equal
    from nkipy.core.ops.transform import astype

    non_zero = not_equal(x, 0)
    non_zero_i32 = astype(non_zero, np.dtype(np.int32))
    summed = sum(non_zero_i32, axis=axis, keepdims=keepdims)

    return not_equal(summed, 0)


# -----------------------------------------------------------------------------
# var - variance: mean((x - mean(x))^2)
# -----------------------------------------------------------------------------
var = Op("var")


@var.impl("hlo")
def _var_hlo(x, axis=None, out=None, dtype=None, keepdims=False, ddof=0):
    """Variance operation: sum((x - mean(x))^2) / (N - ddof)."""
    from nkipy.core.ops.binary import divide, multiply, subtract

    if dtype is not None:
        from nkipy.core.ops.transform import astype

        x = astype(x, np.dtype(dtype))

    # Compute mean with keepdims=True so it broadcasts back against x
    mean_x = mean(x, axis=axis, keepdims=True)

    centered = subtract(x, mean_x)
    squared = multiply(centered, centered)
    sum_squared = sum(squared, axis=axis, keepdims=keepdims)
    count = _calculate_reduction_count(x.shape, axis)

    denom = count - ddof if count - ddof > 0 else 0
    return divide(sum_squared, float(denom))


# -----------------------------------------------------------------------------
# std - standard deviation: sqrt(var(x))
# -----------------------------------------------------------------------------
std = Op("std")


@std.impl("hlo")
def _std_hlo(x, axis=None, out=None, dtype=None, keepdims=False, ddof=0):
    """Standard deviation: sqrt(var(x, axis, ddof))."""
    from nkipy.core.ops.unary import sqrt

    return sqrt(var(x, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof))


# -----------------------------------------------------------------------------
# argmax - index of maximum value along an axis
# -----------------------------------------------------------------------------
argmax = Op("argmax")


@argmax.impl("hlo")
def _argmax_hlo(x, axis=None, out=None, keepdims=False):
    """Argmax: find index of maximum value along axis.

    Strategy: max_val → mask where equal → create iota indices → where(mask, iota, large) → min
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_ref = x
        x_bt = x.backend_tensor
    else:
        x_ref = NKIPyTensorRef(x)
        x_bt = x

    # Save original shape/axis for keepdims support
    original_axis = axis
    x_ref_original_shape = x_ref.shape

    # If axis is None, flatten first
    if axis is None:
        from nkipy.core.ops.transform import reshape

        total = int(np.prod(x_ref.shape))
        x_ref = reshape(x_ref, (total,))
        x_bt = x_ref.backend_tensor
        axis = 0

    # Normalize negative axis
    ndim = len(x_bt.shape)
    if axis < 0:
        axis = ndim + axis

    # Step 1: Get max value along axis (keepdims for broadcasting)
    max_val = max(x_ref, axis=axis, keepdims=True)

    # Step 2: Create boolean mask where x == max_val
    from nkipy.core.ops.binary import equal

    mask = equal(x_ref, max_val)

    # Step 3: Create iota indices along the target axis as float
    # (using float avoids int32 overflow with min's inf init value)
    iota_tensor = ctx.build_op(
        "iota",
        [],
        x_bt.shape,
        np.dtype(np.float32),
        {"iota_dimension": axis},
    )
    iota_ref = NKIPyTensorRef(iota_tensor)

    # Step 4: Where mask is true, use iota index; else use large float value
    large_val = float(x_bt.shape[axis] + 1)
    from nkipy.core.ops.indexing import where

    masked_indices = where(mask, iota_ref, large_val)

    # Step 5: Min along axis to find the first occurrence
    result_float = min(masked_indices, axis=axis)

    # NKI hardware uses int32 for indices; NumPy returns int64
    from nkipy.core.ops.transform import astype

    result = astype(result_float, np.dtype(np.int32))

    if keepdims:
        from nkipy.core.ops.transform import reshape

        if original_axis is not None:
            keepdims_shape = list(x_ref_original_shape)
            keepdims_shape[original_axis] = 1
            result = reshape(result, tuple(keepdims_shape))
        else:
            # axis=None flattened → restore all dims as size-1
            keepdims_shape = tuple(1 for _ in x_ref_original_shape)
            result = reshape(result, keepdims_shape)

    return result


# -----------------------------------------------------------------------------
# cumsum - cumulative sum along an axis
# -----------------------------------------------------------------------------
cumsum = Op("cumsum")


@cumsum.impl("hlo")
def _cumsum_hlo(x, axis=None, dtype=None):
    """Cumulative sum via triangular matrix multiplication.

    For axis of size N: create NxN upper-triangular ones matrix, matmul.
    Multi-dim: transpose target axis to last, reshape to 2D, apply, reshape+transpose back.
    """
    from nkipy.core.ops.creation import constant
    from nkipy.core.ops.linalg import matmul
    from nkipy.core.ops.transform import reshape, transpose

    x_shape = x.shape
    ndim = len(x_shape)

    # Flatten when axis is None (matches numpy behavior)
    if axis is None:
        total = int(np.prod(x_shape))
        x = reshape(x, (total,))
        x_shape = (total,)
        ndim = 1
        axis = 0
    elif axis < 0:
        axis = ndim + axis

    N = x_shape[axis]

    # Create upper-triangular ones matrix (N, N)
    # M[j,i] = 1 if j <= i, so x @ M gives cumsum
    # Always use float32 for numerical stability; dtype conversion happens after.
    # FIXME: when input is integer and dtype is not specified, the result will be
    # float32 instead of preserving the input dtype (NumPy preserves it).
    tri = np.triu(np.ones((N, N), dtype=np.float32))
    tri_const = constant(tri)

    # If axis is the last dimension and tensor is 2D, simple case
    if ndim == 1:
        # (1, N) @ (N, N) -> (1, N) -> (N,)
        x_2d = reshape(x, (1, N))
        result_2d = matmul(x_2d, tri_const)
        result = reshape(result_2d, (N,))
    elif axis == ndim - 1:
        # Reshape to (..., N), matmul with (N, N), reshape back
        batch_size = int(np.prod(x_shape[:-1]))
        x_2d = reshape(x, (batch_size, N))
        result_2d = matmul(x_2d, tri_const)
        result = reshape(result_2d, x_shape)
    else:
        # Transpose target axis to last, apply, transpose back
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x_t = transpose(x, axes=perm)
        x_t_shape = tuple(x_shape[p] for p in perm)

        batch_size = int(np.prod(x_t_shape[:-1]))
        x_2d = reshape(x_t, (batch_size, N))
        result_2d = matmul(x_2d, tri_const)
        result_t = reshape(result_2d, x_t_shape)
        result = transpose(result_t, axes=perm)

    # Cast to requested dtype if needed
    if dtype is not None and result.dtype != np.dtype(dtype):
        from nkipy.core.ops.transform import astype

        result = astype(result, np.dtype(dtype))

    return result
