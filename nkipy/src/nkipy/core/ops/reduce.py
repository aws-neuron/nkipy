# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction operations: sum, max, min, mean"""

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
