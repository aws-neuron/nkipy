# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HLO backend implementations for NKIPy ops.

Contains all primitive HLO lowerings. Composed ops (floor_divide, tan, rint,
etc.) are registered as ``composed_impl`` on the Op itself and need no
per-backend registration.
"""

from __future__ import annotations

import builtins
import itertools
from typing import List, Tuple

import numpy as np

from nkipy.core.backend.hlo import (
    HLOOp,
    as_hlo_tensor,
    broadcast_operands_hlo,
    broadcast_to_shape_hlo,
    find_common_type_hlo,
    get_hlo_context,
)
from nkipy.core.tensor import NKIPyTensorRef

builtins_min = builtins.min

# =============================================================================
# Binary ops
# =============================================================================


def _build_binary_hlo(x, y, np_op, out=None, dtype=None):
    ctx = get_hlo_context()

    promoted_dtype = find_common_type_hlo(x, y)

    if dtype is not None:
        output_dtype = np.dtype(dtype)
    else:
        output_dtype = promoted_dtype

    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    op_map = {
        np.add: "add",
        np.subtract: "subtract",
        np.multiply: "multiply",
        np.divide: "divide",
        np.power: "power",
        np.maximum: "maximum",
        np.minimum: "minimum",
        np.bitwise_and: "and",
        np.bitwise_or: "or",
        np.bitwise_xor: "xor",
        np.logical_and: "and",
        np.logical_or: "or",
        np.logical_xor: "xor",
    }

    hlo_op = op_map.get(
        np_op, np_op.__name__ if hasattr(np_op, "__name__") else str(np_op)
    )

    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    if output_dtype != x_broadcast.dtype:
        x_broadcast = ctx.build_op(
            "convert", [x_broadcast], x_broadcast.shape, output_dtype
        )
    if output_dtype != y_broadcast.dtype:
        y_broadcast = ctx.build_op(
            "convert", [y_broadcast], y_broadcast.shape, output_dtype
        )

    result_tensor = ctx.build_op(
        hlo_op, [x_broadcast, y_broadcast], x_broadcast.shape, output_dtype
    )

    return NKIPyTensorRef(result_tensor)


def _build_comparison_hlo(x, y, np_op, out=None, dtype=None):
    ctx = get_hlo_context()

    promoted_dtype = find_common_type_hlo(x, y)

    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    if x.dtype != promoted_dtype:
        x = ctx.build_op("convert", [x], x.shape, promoted_dtype)
    if y.dtype != promoted_dtype:
        y = ctx.build_op("convert", [y], y.shape, promoted_dtype)

    comp_map = {
        np.equal: "EQ",
        np.not_equal: "NE",
        np.less: "LT",
        np.less_equal: "LE",
        np.greater: "GT",
        np.greater_equal: "GE",
    }

    comp_dir = comp_map.get(np_op, "EQ")

    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    result_tensor = ctx.build_op(
        "compare",
        [x_broadcast, y_broadcast],
        x_broadcast.shape,
        np.bool_,
        {"comparison_direction": comp_dir},
    )

    return NKIPyTensorRef(result_tensor)


def _build_logical_hlo(x, y, hlo_op_name, out=None, dtype=None):
    ctx = get_hlo_context()

    promoted_dtype = find_common_type_hlo(x, y)

    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    zero_x = as_hlo_tensor(ctx, 0, x_broadcast.dtype)
    if x_broadcast.shape:
        zero_x = ctx.build_op(
            "broadcast",
            [zero_x],
            x_broadcast.shape,
            x_broadcast.dtype,
            {"broadcast_dimensions": []},
        )
    x_bool = ctx.build_op(
        "compare",
        [x_broadcast, zero_x],
        x_broadcast.shape,
        np.bool_,
        {"comparison_direction": "NE"},
    )

    zero_y = as_hlo_tensor(ctx, 0, y_broadcast.dtype)
    if y_broadcast.shape:
        zero_y = ctx.build_op(
            "broadcast",
            [zero_y],
            y_broadcast.shape,
            y_broadcast.dtype,
            {"broadcast_dimensions": []},
        )
    y_bool = ctx.build_op(
        "compare",
        [y_broadcast, zero_y],
        y_broadcast.shape,
        np.bool_,
        {"comparison_direction": "NE"},
    )

    result_tensor = ctx.build_op(
        hlo_op_name, [x_bool, y_bool], x_broadcast.shape, np.bool_
    )

    return NKIPyTensorRef(result_tensor)


def add(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.add, out=out, dtype=dtype)


def subtract(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.subtract, out=out, dtype=dtype)


def multiply(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.multiply, out=out, dtype=dtype)


def divide(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.divide, out=out, dtype=dtype)


def power(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.power, out=out, dtype=dtype)


def maximum(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.maximum, out=out, dtype=dtype)


def minimum(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.minimum, out=out, dtype=dtype)


def bitwise_and(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.bitwise_and, out=out, dtype=dtype)


def bitwise_or(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.bitwise_or, out=out, dtype=dtype)


def bitwise_xor(x, y, out=None, dtype=None):
    return _build_binary_hlo(x, y, np.bitwise_xor, out=out, dtype=dtype)


def equal(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.equal, out=out, dtype=dtype)


def not_equal(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.not_equal, out=out, dtype=dtype)


def greater(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.greater, out=out, dtype=dtype)


def greater_equal(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.greater_equal, out=out, dtype=dtype)


def less(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.less, out=out, dtype=dtype)


def less_equal(x, y, out=None, dtype=None):
    return _build_comparison_hlo(x, y, np.less_equal, out=out, dtype=dtype)


def logical_and(x, y, out=None, dtype=None):
    return _build_logical_hlo(x, y, "and", out=out, dtype=dtype)


def logical_or(x, y, out=None, dtype=None):
    return _build_logical_hlo(x, y, "or", out=out, dtype=dtype)


def logical_xor(x, y, out=None, dtype=None):
    return _build_logical_hlo(x, y, "xor", out=out, dtype=dtype)


# =============================================================================
# Unary ops
# =============================================================================


def _build_unary_hlo(x, np_op, out=None, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if np_op == np.arctan:
        one_tensor = as_hlo_tensor(ctx, 1.0, x.dtype)
        if x.shape:
            one_tensor = ctx.build_op(
                "broadcast",
                [one_tensor],
                x.shape,
                x.dtype,
                {"broadcast_dimensions": []},
            )
        result_tensor = ctx.build_op("atan2", [x, one_tensor], x.shape, x.dtype)
        return NKIPyTensorRef(result_tensor)

    op_map = {
        np.abs: "abs",
        np.exp: "exponential",
        np.log: "log",
        np.sqrt: "sqrt",
        np.sin: "sine",
        np.cos: "cosine",
        np.tanh: "tanh",
        np.negative: "negate",
        np.ceil: "ceil",
        np.floor: "floor",
        np.sign: "sign",
        np.bitwise_not: "not",
        np.invert: "not",
    }

    hlo_op = op_map.get(
        np_op, np_op.__name__ if hasattr(np_op, "__name__") else str(np_op)
    )
    result_tensor = ctx.build_op(hlo_op, [x], x.shape, x.dtype)

    return NKIPyTensorRef(result_tensor)


def abs(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.abs, out=out, dtype=dtype)


def exp(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.exp, out=out, dtype=dtype)


def log(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.log, out=out, dtype=dtype)


def sqrt(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.sqrt, out=out, dtype=dtype)


def sin(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.sin, out=out, dtype=dtype)


def cos(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.cos, out=out, dtype=dtype)


def tanh(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.tanh, out=out, dtype=dtype)


def ceil(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.ceil, out=out, dtype=dtype)


def floor(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.floor, out=out, dtype=dtype)


def sign(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.sign, out=out, dtype=dtype)


def negative(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.negative, out=out, dtype=dtype)


def arctan(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.arctan, out=out, dtype=dtype)


def invert(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.invert, out=out, dtype=dtype)


def bitwise_not(x, out=None, dtype=None):
    return _build_unary_hlo(x, np.bitwise_not, out=out, dtype=dtype)



# =============================================================================
# Reduction ops
# =============================================================================


def _build_reduction_hlo(
    x, np_op, axis=None, out=None, dtype=None, keepdims=False, initial=None
):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    reduce_op_map = {
        np.sum: "add",
        np.max: "maximum",
        np.min: "minimum",
        np.prod: "multiply",
    }

    if np_op not in reduce_op_map:
        raise NotImplementedError(
            f"Reduction operation {np_op} not yet supported in HLO tracing"
        )

    hlo_op = reduce_op_map[np_op]

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

    reduced_shape = tuple(
        s for i, s in enumerate(x.shape) if i not in dimensions_to_reduce
    )

    init_values = {
        "add": 0.0,
        "maximum": float("-inf"),
        "minimum": float("inf"),
        "multiply": 1.0,
    }
    init_value = init_values[hlo_op]

    init_tensor = as_hlo_tensor(ctx, init_value, x.dtype)

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

    if keepdims:
        keepdims_shape = tuple(
            1 if i in dimensions_to_reduce else s for i, s in enumerate(x.shape)
        )
        result_tensor = ctx.build_op(
            "reshape", [result_tensor], keepdims_shape, x.dtype
        )

    return NKIPyTensorRef(result_tensor)


def _calculate_reduction_count(x_shape, axis):
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


def reduce_sum(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    return _build_reduction_hlo(
        x, np.sum, axis=axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
    )


def reduce_prod(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    return _build_reduction_hlo(
        x, np.prod, axis=axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
    )


def reduce_max(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    return _build_reduction_hlo(
        x, np.max, axis=axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
    )


def reduce_min(x, axis=None, out=None, dtype=None, keepdims=False, initial=None):
    return _build_reduction_hlo(
        x, np.min, axis=axis, out=out, dtype=dtype, keepdims=keepdims, initial=initial,
    )


def argmax(x, axis=None, out=None, keepdims=False):
    from nkipy.core.ops.reduce import max as max_op, min as min_op

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_ref = x
        x_bt = x.backend_tensor
    else:
        x_ref = NKIPyTensorRef(x)
        x_bt = x

    original_axis = axis
    x_ref_original_shape = x_ref.shape

    if axis is None:
        from nkipy.core.ops.transform import reshape as reshape_op

        total = int(np.prod(x_ref.shape))
        x_ref = reshape_op(x_ref, (total,))
        x_bt = x_ref.backend_tensor
        axis = 0

    ndim = len(x_bt.shape)
    if axis < 0:
        axis = ndim + axis

    max_val = max_op(x_ref, axis=axis, keepdims=True)

    from nkipy.core.ops.binary import equal as equal_op

    mask = equal_op(x_ref, max_val)

    iota_tensor = ctx.build_op(
        "iota",
        [],
        x_bt.shape,
        np.dtype(np.float32),
        {"iota_dimension": axis},
    )
    iota_ref = NKIPyTensorRef(iota_tensor)

    large_val = float(x_bt.shape[axis] + 1)
    from nkipy.core.ops.indexing import where as where_op

    masked_indices = where_op(mask, iota_ref, large_val)

    result_float = min_op(masked_indices, axis=axis)

    from nkipy.core.ops.transform import astype as astype_op

    result = astype_op(result_float, np.dtype(np.int32))

    if keepdims:
        from nkipy.core.ops.transform import reshape as reshape_op

        if original_axis is not None:
            keepdims_shape = list(x_ref_original_shape)
            keepdims_shape[original_axis] = 1
            result = reshape_op(result, tuple(keepdims_shape))
        else:
            keepdims_shape = tuple(1 for _ in x_ref_original_shape)
            result = reshape_op(result, keepdims_shape)

    return result


def argmin(x, axis=None, out=None, keepdims=False):
    from nkipy.core.ops.reduce import min as min_op

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_ref = x
        x_bt = x.backend_tensor
    else:
        x_ref = NKIPyTensorRef(x)
        x_bt = x

    original_axis = axis
    x_ref_original_shape = x_ref.shape

    if axis is None:
        from nkipy.core.ops.transform import reshape as reshape_op

        total = int(np.prod(x_ref.shape))
        x_ref = reshape_op(x_ref, (total,))
        x_bt = x_ref.backend_tensor
        axis = 0

    ndim = len(x_bt.shape)
    if axis < 0:
        axis = ndim + axis

    min_val = min_op(x_ref, axis=axis, keepdims=True)

    from nkipy.core.ops.binary import equal as equal_op

    mask = equal_op(x_ref, min_val)

    iota_tensor = ctx.build_op(
        "iota",
        [],
        x_bt.shape,
        np.dtype(np.float32),
        {"iota_dimension": axis},
    )
    iota_ref = NKIPyTensorRef(iota_tensor)

    large_val = float(x_bt.shape[axis] + 1)
    from nkipy.core.ops.indexing import where as where_op

    masked_indices = where_op(mask, iota_ref, large_val)

    result_float = min_op(masked_indices, axis=axis)

    from nkipy.core.ops.transform import astype as astype_op

    result = astype_op(result_float, np.dtype(np.int32))

    if keepdims:
        from nkipy.core.ops.transform import reshape as reshape_op

        if original_axis is not None:
            keepdims_shape = list(x_ref_original_shape)
            keepdims_shape[original_axis] = 1
            result = reshape_op(result, tuple(keepdims_shape))
        else:
            keepdims_shape = tuple(1 for _ in x_ref_original_shape)
            result = reshape_op(result, keepdims_shape)

    return result



# =============================================================================
# Linalg ops
# =============================================================================


def matmul(x, y, out=None, dtype=None):
    ctx = get_hlo_context()

    result_dtype = find_common_type_hlo(x, y)

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor
    if isinstance(y, NKIPyTensorRef):
        y = y.backend_tensor

    assert len(x.shape) >= 1 and len(y.shape) >= 1, "matmul requires at least 1D arrays"

    squeeze_lhs = False
    squeeze_rhs = False

    if len(x.shape) == 1 and len(y.shape) == 1:
        assert x.shape[0] == y.shape[0], "Incompatible shapes for dot product"
        result_shape = ()
        lhs_contracting_dims = [0]
        rhs_contracting_dims = [0]
        lhs_batch_dims = []
        rhs_batch_dims = []
    else:
        if len(x.shape) == 1:
            x = ctx.build_op("reshape", [x], (1, x.shape[0]), x.dtype)
            squeeze_lhs = True

        if len(y.shape) == 1:
            y = ctx.build_op("reshape", [y], (y.shape[0], 1), y.dtype)
            squeeze_rhs = True

        assert x.shape[-1] == y.shape[-2], "Incompatible shapes for matmul"

        x_batch_shape = x.shape[:-2]
        y_batch_shape = y.shape[:-2]
        batch_shape = tuple(np.broadcast_shapes(x_batch_shape, y_batch_shape))
        result_shape = batch_shape + (x.shape[-2], y.shape[-1])

        target_x_shape = batch_shape + tuple(x.shape[-2:])
        target_y_shape = batch_shape + tuple(y.shape[-2:])

        if x.shape != target_x_shape:
            x = broadcast_to_shape_hlo(ctx, x, target_x_shape)

        if y.shape != target_y_shape:
            y = broadcast_to_shape_hlo(ctx, y, target_y_shape)

        lhs_contracting_dims = [len(target_x_shape) - 1]
        rhs_contracting_dims = [len(target_y_shape) - 2]

        lhs_batch_dims = list(range(len(batch_shape)))
        rhs_batch_dims = list(range(len(batch_shape)))

    result_tensor = ctx.build_op(
        "dot",
        [x, y],
        result_shape,
        result_dtype,
        {
            "lhs_contracting_dimensions": lhs_contracting_dims,
            "rhs_contracting_dimensions": rhs_contracting_dims,
            "lhs_batch_dimensions": lhs_batch_dims,
            "rhs_batch_dimensions": rhs_batch_dims,
        },
    )

    if squeeze_lhs or squeeze_rhs:
        final_shape = list(result_shape)
        if squeeze_lhs:
            final_shape.pop(-2)
        if squeeze_rhs:
            final_shape.pop(-1)
        final_shape = tuple(final_shape)
        result_tensor = ctx.build_op(
            "reshape", [result_tensor], final_shape, result_dtype
        )

    return NKIPyTensorRef(result_tensor)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    from nkipy.core.ops.binary import equal as equal_op, subtract as sub_op
    from nkipy.core.ops.indexing import where as where_op
    from nkipy.core.ops.reduce import sum as sum_op

    ctx = get_hlo_context()

    if isinstance(a, NKIPyTensorRef):
        a_bt = a.backend_tensor
    else:
        a_bt = a

    shape = a_bt.shape
    ndim = len(shape)

    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim

    row_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": axis1}
    )
    row_ref = NKIPyTensorRef(row_iota)

    col_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": axis2}
    )
    col_ref = NKIPyTensorRef(col_iota)

    if offset != 0:
        col_ref = sub_op(col_ref, offset)

    diag_mask = equal_op(row_ref, col_ref)

    masked = where_op(diag_mask, a, 0.0)

    axes_to_reduce = sorted([axis1, axis2], reverse=True)
    result = masked
    for ax in axes_to_reduce:
        result = sum_op(result, axis=ax)

    if dtype is not None:
        from nkipy.core.ops.transform import astype as astype_op

        result = astype_op(result, np.dtype(dtype))

    return result


def dot(x, y, out=None):
    ctx = get_hlo_context()

    result_dtype = find_common_type_hlo(x, y)

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor
    if isinstance(y, NKIPyTensorRef):
        y = y.backend_tensor

    assert len(x.shape) >= 1 and len(y.shape) >= 1, "dot requires at least 1D arrays"

    lhs_contracting_dims = [len(x.shape) - 1]
    rhs_contracting_dims = [max(0, len(y.shape) - 2)]

    assert x.shape[lhs_contracting_dims[0]] == y.shape[rhs_contracting_dims[0]], (
        f"shapes {x.shape} and {y.shape} not aligned"
    )

    lhs_batch_dims = []
    rhs_batch_dims = []

    result_shape = tuple(
        s for i, s in enumerate(x.shape) if i not in lhs_contracting_dims
    ) + tuple(s for i, s in enumerate(y.shape) if i not in rhs_contracting_dims)

    result_tensor = ctx.build_op(
        "dot",
        [x, y],
        result_shape,
        result_dtype,
        {
            "lhs_contracting_dimensions": lhs_contracting_dims,
            "rhs_contracting_dimensions": rhs_contracting_dims,
            "lhs_batch_dimensions": lhs_batch_dims,
            "rhs_batch_dimensions": rhs_batch_dims,
        },
    )

    return NKIPyTensorRef(result_tensor)


# =============================================================================
# Creation ops
# =============================================================================


def zeros(shape, dtype):
    ctx = get_hlo_context()

    if isinstance(shape, int):
        shape = (shape,)

    zero_tensor = as_hlo_tensor(ctx, 0.0, dtype)

    if shape:
        result_tensor = ctx.build_op(
            "broadcast", [zero_tensor], shape, dtype, {"broadcast_dimensions": []}
        )
    else:
        result_tensor = zero_tensor

    return NKIPyTensorRef(result_tensor)


def full(shape, fill_value, dtype):
    ctx = get_hlo_context()

    if isinstance(shape, int):
        shape = (shape,)

    fill_tensor = as_hlo_tensor(ctx, fill_value, dtype)

    if shape:
        result_tensor = ctx.build_op(
            "broadcast", [fill_tensor], shape, dtype, {"broadcast_dimensions": []}
        )
    else:
        result_tensor = fill_tensor

    return NKIPyTensorRef(result_tensor)


def constant(value, dtype=None):
    if isinstance(value, NKIPyTensorRef):
        if dtype is not None and value.dtype != np.dtype(dtype):
            from nkipy.core.ops.transform import astype as astype_op

            return astype_op(value, dtype)
        return value

    ctx = get_hlo_context()

    if dtype is not None:
        target_dtype = np.dtype(dtype)
    elif hasattr(value, "dtype"):
        target_dtype = np.dtype(value.dtype)
    elif isinstance(value, float):
        target_dtype = np.dtype(np.float32)
    elif isinstance(value, int):
        target_dtype = np.dtype(np.int32)
    elif isinstance(value, bool):
        target_dtype = np.dtype(np.bool_)
    else:
        target_dtype = np.dtype(np.asarray(value).dtype)

    if isinstance(value, (list, tuple)):
        value = np.asarray(value, dtype=target_dtype)

    hlo_tensor = as_hlo_tensor(ctx, value, target_dtype)
    return NKIPyTensorRef(hlo_tensor)


def zeros_like(x, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    zero_tensor = as_hlo_tensor(ctx, 0.0, result_dtype)

    if x_hlo.shape:
        result_tensor = ctx.build_op(
            "broadcast",
            [zero_tensor],
            x_hlo.shape,
            result_dtype,
            {"broadcast_dimensions": []},
        )
    else:
        result_tensor = zero_tensor

    # FIXME: Workaround to ensure x is referenced in the computation graph.
    zero_multiplier = as_hlo_tensor(ctx, 0.0, x_hlo.dtype)
    if x_hlo.shape:
        zero_multiplier = ctx.build_op(
            "broadcast",
            [zero_multiplier],
            x_hlo.shape,
            x_hlo.dtype,
            {"broadcast_dimensions": []},
        )

    x_times_zero = ctx.build_op(
        "multiply", [x_hlo, zero_multiplier], x_hlo.shape, x_hlo.dtype
    )

    if x_hlo.dtype != result_dtype:
        x_times_zero = ctx.build_op(
            "convert", [x_times_zero], x_hlo.shape, result_dtype
        )

    result_tensor = ctx.build_op(
        "add", [result_tensor, x_times_zero], x_hlo.shape, result_dtype
    )

    return NKIPyTensorRef(result_tensor)


def ones_like(x, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    one_tensor = as_hlo_tensor(ctx, 1.0, result_dtype)

    if x_hlo.shape:
        result_tensor = ctx.build_op(
            "broadcast",
            [one_tensor],
            x_hlo.shape,
            result_dtype,
            {"broadcast_dimensions": []},
        )
    else:
        result_tensor = one_tensor

    # FIXME: Workaround to ensure x is referenced in the computation graph
    zero_multiplier = as_hlo_tensor(ctx, 0.0, x_hlo.dtype)
    if x_hlo.shape:
        zero_multiplier = ctx.build_op(
            "broadcast",
            [zero_multiplier],
            x_hlo.shape,
            x_hlo.dtype,
            {"broadcast_dimensions": []},
        )

    x_times_zero = ctx.build_op(
        "multiply", [x_hlo, zero_multiplier], x_hlo.shape, x_hlo.dtype
    )

    if x_hlo.dtype != result_dtype:
        x_times_zero = ctx.build_op(
            "convert", [x_times_zero], x_hlo.shape, result_dtype
        )

    result_tensor = ctx.build_op(
        "add", [result_tensor, x_times_zero], x_hlo.shape, result_dtype
    )

    return NKIPyTensorRef(result_tensor)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    fill_tensor = as_hlo_tensor(ctx, fill_value, result_dtype)

    if x_hlo.shape:
        result_tensor = ctx.build_op(
            "broadcast",
            [fill_tensor],
            x_hlo.shape,
            result_dtype,
            {"broadcast_dimensions": []},
        )
    else:
        result_tensor = fill_tensor

    # FIXME: Workaround to ensure x is referenced in the computation graph
    zero_multiplier = as_hlo_tensor(ctx, 0.0, x_hlo.dtype)
    if x_hlo.shape:
        zero_multiplier = ctx.build_op(
            "broadcast",
            [zero_multiplier],
            x_hlo.shape,
            x_hlo.dtype,
            {"broadcast_dimensions": []},
        )

    x_times_zero = ctx.build_op(
        "multiply", [x_hlo, zero_multiplier], x_hlo.shape, x_hlo.dtype
    )

    if x_hlo.dtype != result_dtype:
        x_times_zero = ctx.build_op(
            "convert", [x_times_zero], x_hlo.shape, result_dtype
        )

    result_tensor = ctx.build_op(
        "add", [result_tensor, x_times_zero], x_hlo.shape, result_dtype
    )

    return NKIPyTensorRef(result_tensor)


def tril(x, k=0):
    from nkipy.core.ops.binary import greater_equal as ge_op, subtract as sub_op
    from nkipy.core.ops.indexing import where as where_op

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    shape = x_bt.shape
    ndim = len(shape)

    row_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 2}
    )
    row_ref = NKIPyTensorRef(row_iota)

    col_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 1}
    )
    col_ref = NKIPyTensorRef(col_iota)

    if k != 0:
        col_ref = sub_op(col_ref, k)
    mask = ge_op(row_ref, col_ref)

    return where_op(mask, x, 0.0)


def triu(x, k=0):
    from nkipy.core.ops.binary import less_equal as le_op, subtract as sub_op
    from nkipy.core.ops.indexing import where as where_op

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    shape = x_bt.shape
    ndim = len(shape)

    row_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 2}
    )
    row_ref = NKIPyTensorRef(row_iota)

    col_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 1}
    )
    col_ref = NKIPyTensorRef(col_iota)

    if k != 0:
        col_ref = sub_op(col_ref, k)
    mask = le_op(row_ref, col_ref)

    return where_op(mask, x, 0.0)


def diag(v, k=0):
    from nkipy.core.ops.binary import equal as equal_op, subtract as sub_op
    from nkipy.core.ops.indexing import take as take_op, where as where_op
    from nkipy.core.ops.reduce import sum as sum_op
    from nkipy.core.ops.transform import broadcast_to as bcast_op, reshape as reshape_op
    from nkipy.core.ops.unary import clip as clip_op

    ctx = get_hlo_context()

    if isinstance(v, NKIPyTensorRef):
        v_bt = v.backend_tensor
    else:
        v_bt = v

    ndim = len(v_bt.shape)

    if ndim == 1:
        n = v_bt.shape[0] + builtins.abs(k)
        shape_2d = (n, n)

        row_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 0}
        )
        row_ref = NKIPyTensorRef(row_iota)
        col_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 1}
        )
        col_ref = NKIPyTensorRef(col_iota)

        if k != 0:
            col_ref = sub_op(col_ref, k)

        diag_mask = equal_op(row_ref, col_ref)

        if k >= 0:
            idx_ref = row_ref
        else:
            idx_ref = NKIPyTensorRef(col_iota)

        idx_ref = clip_op(idx_ref, 0, v_bt.shape[0] - 1)

        v_gathered = take_op(v, idx_ref, axis=0)
        return where_op(diag_mask, v_gathered, 0.0)

    elif ndim == 2:
        rows, cols = v_bt.shape
        if k >= 0:
            diag_len = builtins_min(rows, cols - k)
        else:
            diag_len = builtins_min(rows + k, cols)

        if diag_len <= 0:
            return zeros((0,), v_bt.dtype)

        shape_2d = v_bt.shape
        row_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 0}
        )
        row_ref = NKIPyTensorRef(row_iota)
        col_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 1}
        )
        col_ref = NKIPyTensorRef(col_iota)

        if k != 0:
            col_ref = sub_op(col_ref, k)

        diag_mask = equal_op(row_ref, col_ref)

        masked = where_op(diag_mask, v, 0.0)

        if k >= 0:
            result = sum_op(masked, axis=1)
        else:
            result = sum_op(masked, axis=0)

        result_shape = result.shape
        if result_shape[0] != diag_len:
            from nkipy.core.ops.indexing import static_slice as static_slice_op

            result = static_slice_op(result, [0], [diag_len], [1], [])

        return result

    else:
        raise ValueError(f"Input must be 1-D or 2-D, got {ndim}-D")


# =============================================================================
# Transform ops
# =============================================================================


def reshape(x, newshape):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if isinstance(newshape, int):
        newshape = (newshape,)

    if -1 in newshape:
        total_size = int(np.prod(x.shape))
        known_size = int(np.prod([d for d in newshape if d != -1]))
        assert known_size > 0, "Cannot reshape to a size of 0"
        assert total_size % known_size == 0, (
            f"Cannot reshape array of size {total_size} into shape {newshape}"
        )
        newshape = tuple(total_size // known_size if d == -1 else d for d in newshape)

    if np.prod(x.shape) != np.prod(newshape):
        raise ValueError(
            f"Cannot reshape array of size {np.prod(x.shape)} into shape {newshape}"
        )

    result_tensor = ctx.build_op("reshape", [x], newshape, x.dtype)
    return NKIPyTensorRef(result_tensor)


def transpose(x, axes=None, out=None, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axes is None:
        axes = list(range(len(x.shape)))[::-1]

    result_shape = tuple(x.shape[i] for i in axes)

    result_tensor = ctx.build_op(
        "transpose", [x], result_shape, x.dtype, {"permutation": axes}
    )
    return NKIPyTensorRef(result_tensor)


def swapaxes(x, axis1, axis2):
    ndim = len(x.shape)
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
    if not (0 <= axis1 < ndim) or not (0 <= axis2 < ndim):
        raise np.AxisError(f"axis is out of bounds for array of dimension {ndim}")
    axes = list(range(ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    from nkipy.core.ops.transform import transpose as transpose_op

    return transpose_op(x, axes=axes)


def stack(arrays, axis=0, out=None, dtype=None):
    from nkipy.core.ops.transform import concatenate as concat_op, expand_dims as expand_op

    expanded = [expand_op(a, axis=axis) for a in arrays]
    return concat_op(expanded, axis=axis)


def expand_dims(x, axis):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    rank = len(x.shape)

    if isinstance(axis, (list, tuple)):
        final_rank = rank + len(axis)

        axes = []
        for ax in axis:
            if ax < 0:
                ax = final_rank + ax
            if ax < 0 or ax > final_rank - 1:
                raise ValueError(
                    f"axis {ax} is out of bounds for array of dimension {final_rank}"
                )
            axes.append(ax)

        if len(axes) != len(set(axes)):
            raise ValueError("repeated axis in expand_dims")

        axes = sorted(axes)

        new_shape = list(x.shape)
        for ax in axes:
            new_shape.insert(ax, 1)
        new_shape = tuple(new_shape)
    else:
        if axis < 0:
            axis = rank + axis + 1

        if axis < 0 or axis > rank:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {rank}"
            )

        new_shape = list(x.shape)
        new_shape.insert(axis, 1)
        new_shape = tuple(new_shape)

    result_tensor = ctx.build_op("reshape", [x], new_shape, x.dtype)
    return NKIPyTensorRef(result_tensor)


def concatenate(tensors, axis=0):
    ctx = get_hlo_context()

    hlo_tensors = []
    for t in tensors:
        if isinstance(t, NKIPyTensorRef):
            hlo_tensors.append(t.backend_tensor)
        elif isinstance(t, np.ndarray):
            from nkipy.core.ops.creation import constant as constant_op

            const_ref = constant_op(t)
            hlo_tensors.append(const_ref.backend_tensor)
        else:
            hlo_tensors.append(t)

    if not hlo_tensors:
        raise ValueError("Need at least one tensor to concatenate")

    if len(hlo_tensors) == 1:
        result_tensor = ctx.build_op(
            "copy", [hlo_tensors[0]], hlo_tensors[0].shape, hlo_tensors[0].dtype
        )
        return NKIPyTensorRef(result_tensor)

    ndim = len(hlo_tensors[0].shape)
    if axis < 0:
        axis = ndim + axis

    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    output_shape = list(hlo_tensors[0].shape)
    output_shape[axis] = builtins.sum(t.shape[axis] for t in hlo_tensors)
    output_shape = tuple(output_shape)

    dtype = hlo_tensors[0].dtype
    for t in hlo_tensors[1:]:
        dtype = np.result_type(dtype, t.dtype)

    result_tensor = ctx.build_op(
        "concatenate", hlo_tensors, output_shape, dtype, {"dimension": axis}
    )
    return NKIPyTensorRef(result_tensor)


def split(x, indices_or_sections, axis=0):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axis < 0:
        axis = len(x.shape) + axis

    if axis < 0 or axis >= len(x.shape):
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {len(x.shape)}"
        )

    axis_size = x.shape[axis]

    if isinstance(indices_or_sections, int):
        n_sections = indices_or_sections
        if n_sections <= 0:
            raise ValueError("Number of sections must be larger than 0")
        if axis_size % n_sections != 0:
            raise ValueError("Array split does not result in an equal division")

        section_size = axis_size // n_sections
        split_indices = [i * section_size for i in range(1, n_sections)]
    else:
        split_indices = list(indices_or_sections)

    split_points = [0] + split_indices + [axis_size]

    result_tensors = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]

        start_indices = [0] * len(x.shape)
        limit_indices = list(x.shape)
        strides = [1] * len(x.shape)

        start_indices[axis] = start_idx
        limit_indices[axis] = end_idx

        slice_shape = list(x.shape)
        slice_shape[axis] = end_idx - start_idx
        slice_shape = tuple(slice_shape)

        slice_tensor = ctx.build_op(
            "slice",
            [x],
            slice_shape,
            x.dtype,
            {
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            },
        )

        result_tensors.append(NKIPyTensorRef(slice_tensor))

    return result_tensors


def copy(x, out=None, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
    return NKIPyTensorRef(result_tensor)


def repeat(x, repeats, axis=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axis is None:
        flattened_shape = (int(np.prod(x.shape)),)
        x = ctx.build_op("reshape", [x], flattened_shape, x.dtype)
        axis = 0

    if axis < 0:
        axis = len(x.shape) + axis

    if not isinstance(repeats, (int, np.integer)):
        raise TypeError(
            f"Only compile-time-known integer repeats are supported, got {type(repeats).__name__}. "
            "Dynamic tensor repeats are not supported in tracing."
        )
    repeats = int(repeats)

    new_shape = list(x.shape)
    new_shape[axis] *= repeats
    new_shape = tuple(new_shape)

    broadcast_shape = list(x.shape)
    broadcast_shape.insert(axis + 1, repeats)
    broadcast_shape = tuple(broadcast_shape)

    broadcast_dims = [i if i <= axis else i + 1 for i in range(len(x.shape))]
    x_broadcast = ctx.build_op(
        "broadcast",
        [x],
        broadcast_shape,
        x.dtype,
        {"broadcast_dimensions": broadcast_dims},
    )

    result_tensor = ctx.build_op("reshape", [x_broadcast], new_shape, x.dtype)

    return NKIPyTensorRef(result_tensor)


def broadcast_to(x, shape, out=None, dtype=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if isinstance(shape, int):
        shape = (shape,)
    target_shape = tuple(shape)

    if x.shape == target_shape:
        result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
        return NKIPyTensorRef(result_tensor)

    result_tensor = broadcast_to_shape_hlo(ctx, x, target_shape)
    return NKIPyTensorRef(result_tensor)


def astype(x, dtype):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if x.dtype == dtype:
        result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
    else:
        result_tensor = ctx.build_op("convert", [x], x.shape, dtype)

    return NKIPyTensorRef(result_tensor)


def squeeze(x, axis=None):
    x_shape = x.shape
    ndim = len(x_shape)

    if axis is None:
        new_shape = tuple(s for s in x_shape if s != 1)
        if not new_shape:
            new_shape = ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axes = tuple(a if a >= 0 else ndim + a for a in axis)
        for a in axes:
            if x_shape[a] != 1:
                raise ValueError(
                    f"cannot select an axis to squeeze out which has size "
                    f"not equal to one, got shape[{a}] = {x_shape[a]}"
                )
        new_shape = tuple(s for i, s in enumerate(x_shape) if i not in axes)

    if new_shape == x_shape:
        return x
    from nkipy.core.ops.transform import reshape as reshape_op

    return reshape_op(x, new_shape)


def pad(x, pad_width, mode="constant", constant_values=0, **kwargs):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_shape = x.shape
        x_dtype = x.dtype
        x_bt = x.backend_tensor
    else:
        x_shape = x.shape
        x_dtype = x.dtype
        x_bt = x

    ndim = len(x_shape)

    pad_width_arr = np.asarray(pad_width)
    if pad_width_arr.ndim == 0:
        pad_width_arr = np.broadcast_to(pad_width_arr, (ndim, 2))
    elif pad_width_arr.ndim == 1:
        if len(pad_width_arr) == 2:
            pad_width_arr = np.broadcast_to(pad_width_arr, (ndim, 2))
        else:
            pad_width_arr = np.array([[p, p] for p in pad_width_arr])
            if len(pad_width_arr) != ndim:
                raise ValueError(
                    f"pad_width must have length {ndim} to match array dimensions, "
                    f"got {len(pad_width_arr)}"
                )
    if pad_width_arr.ndim == 2 and len(pad_width_arr) == 1:
        pad_width_arr = np.broadcast_to(pad_width_arr, (ndim, 2))
    pad_width_list = [(int(pad_width_arr[i, 0]), int(pad_width_arr[i, 1])) for i in range(ndim)]

    if mode == "constant":
        padding_config = [(low, high, 0) for low, high in pad_width_list]

        result_shape = tuple(
            s + low + high for s, (low, high) in zip(x_shape, pad_width_list)
        )

        pad_value_tensor = as_hlo_tensor(ctx, constant_values, x_dtype)

        result_tensor = ctx.build_op(
            "pad",
            [x_bt, pad_value_tensor],
            result_shape,
            x_dtype,
            {"padding_config": padding_config},
        )
        return NKIPyTensorRef(result_tensor)

    elif mode == "edge":
        from nkipy.core.ops.transform import (
            concatenate as concat_op,
            expand_dims as expand_op,
            repeat as repeat_op,
        )

        result = NKIPyTensorRef(x_bt) if not isinstance(x, NKIPyTensorRef) else x
        for dim in range(ndim):
            before, after = pad_width_list[dim]
            if before == 0 and after == 0:
                continue

            parts = []
            if before > 0:
                edge_slice = _slice_single(result, dim, 0)
                edge_expanded = expand_op(edge_slice, axis=dim)
                edge_repeated = repeat_op(edge_expanded, before, axis=dim)
                parts.append(edge_repeated)

            parts.append(result)

            if after > 0:
                last_idx = result.shape[dim] - 1
                edge_slice = _slice_single(result, dim, last_idx)
                edge_expanded = expand_op(edge_slice, axis=dim)
                edge_repeated = repeat_op(edge_expanded, after, axis=dim)
                parts.append(edge_repeated)

            result = concat_op(parts, axis=dim)

        return result

    else:
        raise NotImplementedError(
            f"Pad mode '{mode}' is not supported. Only 'constant' and 'edge' modes are available."
        )


def diff(a, n=1, axis=-1, prepend=None, append=None):
    from nkipy.core.ops.binary import subtract as sub_op

    ctx = get_hlo_context()

    ndim = len(a.shape)
    if axis < 0:
        axis += ndim

    result = a
    for _ in range(n):
        if isinstance(result, NKIPyTensorRef):
            r_bt = result.backend_tensor
        else:
            r_bt = result

        axis_size = r_bt.shape[axis]

        start1 = [0] * ndim
        limit1 = list(r_bt.shape)
        start1[axis] = 1
        shape1 = list(r_bt.shape)
        shape1[axis] = axis_size - 1

        t1 = ctx.build_op(
            "slice",
            [r_bt],
            tuple(shape1),
            r_bt.dtype,
            {"start_indices": start1, "limit_indices": limit1, "strides": [1] * ndim},
        )

        start0 = [0] * ndim
        limit0 = list(r_bt.shape)
        limit0[axis] = axis_size - 1
        shape0 = list(r_bt.shape)
        shape0[axis] = axis_size - 1

        t0 = ctx.build_op(
            "slice",
            [r_bt],
            tuple(shape0),
            r_bt.dtype,
            {"start_indices": start0, "limit_indices": limit0, "strides": [1] * ndim},
        )

        result = sub_op(NKIPyTensorRef(t1), NKIPyTensorRef(t0))

    return result


def flip(x, axis=None):
    from nkipy.core.ops.indexing import take as take_op

    ndim = len(x.shape)

    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis if axis >= 0 else axis + ndim]
    else:
        axes = [a if a >= 0 else a + ndim for a in axis]

    result = x
    for ax in axes:
        n = result.shape[ax]
        reversed_indices = np.arange(n - 1, -1, -1, dtype=np.int32)
        result = take_op(result, reversed_indices, axis=ax)

    return result


def tile(x, reps):
    from nkipy.core.ops.transform import (
        broadcast_to as bcast_op,
        copy as copy_op,
        reshape as reshape_op,
    )

    if isinstance(reps, int):
        reps = (reps,)
    reps = tuple(reps)

    x_shape = x.shape
    ndim = len(x_shape)

    if len(reps) < ndim:
        reps = (1,) * (ndim - len(reps)) + reps
    elif len(reps) > ndim:
        x = reshape_op(x, (1,) * (len(reps) - ndim) + x_shape)
        x_shape = x.shape
        ndim = len(x_shape)

    if all(r == 1 for r in reps):
        return copy_op(x)

    interleaved = []
    for r, s in zip(reps, x_shape):
        interleaved.append(1)
        interleaved.append(s)
    result = reshape_op(x, tuple(interleaved))

    bcast_shape = list(result.shape)
    for i, r in enumerate(reps):
        bcast_shape[i * 2] = r
    result = bcast_op(result, tuple(bcast_shape))

    final_shape = tuple(r * s for r, s in zip(reps, x_shape))
    return reshape_op(result, final_shape)


def roll(x, shift, axis=None):
    from nkipy.core.ops.transform import reshape as reshape_op

    x_shape = x.shape
    ndim = len(x_shape)

    if axis is None:
        total = int(np.prod(x_shape))
        flat = reshape_op(x, (total,))
        rolled = _roll_single_axis(flat, shift, 0)
        return reshape_op(rolled, x_shape)

    if isinstance(shift, (list, tuple)):
        if not isinstance(axis, (list, tuple)):
            raise ValueError("If shift is a tuple, axis must also be a tuple")
        result = x
        for s, a in zip(shift, axis):
            result = _roll_single_axis(result, s, a if a >= 0 else a + ndim)
        return result

    if axis < 0:
        axis += ndim
    return _roll_single_axis(x, shift, axis)


def _roll_single_axis(x, shift, axis):
    from nkipy.core.ops.transform import concatenate as concat_op

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    axis_size = x_bt.shape[axis]
    ndim = len(x_bt.shape)

    shift = shift % axis_size
    if shift == 0:
        return NKIPyTensorRef(x_bt) if not isinstance(x, NKIPyTensorRef) else x

    split_point = axis_size - shift

    start1 = [0] * ndim
    limit1 = list(x_bt.shape)
    start1[axis] = split_point
    shape1 = list(x_bt.shape)
    shape1[axis] = shift

    t1 = ctx.build_op(
        "slice",
        [x_bt],
        tuple(shape1),
        x_bt.dtype,
        {"start_indices": start1, "limit_indices": limit1, "strides": [1] * ndim},
    )

    start0 = [0] * ndim
    limit0 = list(x_bt.shape)
    limit0[axis] = split_point
    shape0 = list(x_bt.shape)
    shape0[axis] = split_point

    t0 = ctx.build_op(
        "slice",
        [x_bt],
        tuple(shape0),
        x_bt.dtype,
        {"start_indices": start0, "limit_indices": limit0, "strides": [1] * ndim},
    )

    return concat_op([NKIPyTensorRef(t1), NKIPyTensorRef(t0)], axis=axis)


def _slice_single(x, dim, index):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    ndim = len(x_bt.shape)
    start_indices = [0] * ndim
    limit_indices = list(x_bt.shape)
    strides_list = [1] * ndim

    start_indices[dim] = index
    limit_indices[dim] = index + 1

    slice_shape = list(x_bt.shape)
    slice_shape[dim] = 1

    sliced = ctx.build_op(
        "slice",
        [x_bt],
        tuple(slice_shape),
        x_bt.dtype,
        {
            "start_indices": start_indices,
            "limit_indices": limit_indices,
            "strides": strides_list,
        },
    )

    result_shape = tuple(s for i, s in enumerate(x_bt.shape) if i != dim)
    result = ctx.build_op("reshape", [sliced], result_shape, x_bt.dtype)
    return NKIPyTensorRef(result)


# =============================================================================
# Indexing ops
# =============================================================================


def where(condition, x, y):
    ctx = get_hlo_context()

    output_dtype = find_common_type_hlo(x, y)

    if isinstance(condition, NKIPyTensorRef):
        condition = condition.backend_tensor
    elif np.isscalar(condition):
        condition = as_hlo_tensor(ctx, bool(condition), np.bool_)
    elif isinstance(condition, np.ndarray):
        condition = as_hlo_tensor(ctx, condition.astype(bool), np.bool_)

    if hasattr(condition, "dtype") and condition.dtype != np.bool_:
        zero = as_hlo_tensor(ctx, 0, condition.dtype)
        if condition.shape:
            zero = ctx.build_op(
                "broadcast",
                [zero],
                condition.shape,
                condition.dtype,
                {"broadcast_dimensions": []},
            )
        condition = ctx.build_op(
            "compare",
            [condition, zero],
            condition.shape,
            np.bool_,
            {"comparison_direction": "NE"},
        )

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor
    elif np.isscalar(x):
        x = as_hlo_tensor(ctx, x, output_dtype)
    elif isinstance(x, np.ndarray):
        const_op = HLOOp(
            "constant",
            [],
            result_shape=x.shape,
            result_dtype=x.dtype,
            attributes={"value": x},
        )
        x = ctx.module.add_operation(const_op)

    if isinstance(y, NKIPyTensorRef):
        y = y.backend_tensor
    elif np.isscalar(y):
        y = as_hlo_tensor(ctx, y, output_dtype)
    elif isinstance(y, np.ndarray):
        const_op = HLOOp(
            "constant",
            [],
            result_shape=y.shape,
            result_dtype=y.dtype,
            attributes={"value": y},
        )
        y = ctx.module.add_operation(const_op)

    broadcast_shape = tuple(np.broadcast_shapes(condition.shape, x.shape, y.shape))

    if condition.shape != broadcast_shape:
        condition = broadcast_to_shape_hlo(ctx, condition, broadcast_shape)

    if x.shape != broadcast_shape:
        x = broadcast_to_shape_hlo(ctx, x, broadcast_shape)

    if y.shape != broadcast_shape:
        y = broadcast_to_shape_hlo(ctx, y, broadcast_shape)

    result_tensor = ctx.build_op(
        "select", [condition, x, y], broadcast_shape, output_dtype
    )

    return NKIPyTensorRef(result_tensor)


def take(x, indices, axis=None):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axis is None:
        flattened_shape = (int(np.prod(x.shape)),)
        x = ctx.build_op("reshape", [x], flattened_shape, x.dtype)
        axis = 0

    if axis < 0:
        axis = len(x.shape) + axis

    dtype = x.dtype

    if isinstance(indices, NKIPyTensorRef):
        indices_tensor = indices.backend_tensor
    elif np.isscalar(indices):
        if indices < 0:
            indices = x.shape[axis] + indices
        indices_tensor = as_hlo_tensor(ctx, int(indices), np.dtype(np.int32))
    elif isinstance(indices, (np.ndarray, list)):
        if isinstance(indices, list):
            indices_np = np.array(indices, dtype=np.int32)
        else:
            indices_np = indices.astype(np.int32)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=indices_np.shape,
            result_dtype=np.dtype(np.int32),
            attributes={"value": indices_np},
        )
        indices_tensor = ctx.module.add_operation(const_op)
    else:
        raise ValueError(
            "np.take only supports TensorRef, scalar, np.ndarray, or list as indices!"
        )

    indices_shape = indices_tensor.shape if hasattr(indices_tensor, "shape") else ()

    output_shape = []
    for i in range(len(x.shape)):
        if i == axis:
            output_shape.extend(indices_shape)
        else:
            output_shape.append(x.shape[i])
    output_shape = tuple(output_shape)

    offset_dims = []
    for i in range(len(output_shape)):
        if i < axis or i >= axis + len(indices_shape):
            offset_dims.append(i)

    collapsed_slice_dims = [axis]
    start_index_map = [axis]
    index_vector_dim = len(indices_shape)

    slice_sizes = list(x.shape)
    slice_sizes[axis] = 1

    result_tensor = ctx.build_op(
        "gather",
        [x, indices_tensor],
        output_shape,
        dtype,
        {
            "offset_dims": offset_dims,
            "collapsed_slice_dims": collapsed_slice_dims,
            "start_index_map": start_index_map,
            "index_vector_dim": index_vector_dim,
            "slice_sizes": slice_sizes,
            "indices_are_sorted": False,
        },
    )

    return NKIPyTensorRef(result_tensor)


def take_along_axis(x, indices, axis):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axis is None:
        flattened_shape = (int(np.prod(x.shape)),)
        x = ctx.build_op("reshape", [x], flattened_shape, x.dtype)
        axis = 0

    if axis < 0:
        axis = len(x.shape) + axis

    if isinstance(indices, NKIPyTensorRef):
        indices_tensor = indices.backend_tensor
    elif isinstance(indices, np.ndarray):
        indices_np = indices.astype(np.int32)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=indices_np.shape,
            result_dtype=np.dtype(np.int32),
            attributes={"value": indices_np},
        )
        indices_tensor = ctx.module.add_operation(const_op)
    else:
        raise ValueError(
            "take_along_axis only supports TensorRef or np.ndarray as indices!"
        )

    data_rank = len(x.shape)

    if indices_tensor.dtype != np.dtype(np.int32):
        indices_tensor = ctx.build_op(
            "convert", [indices_tensor], indices_tensor.shape, np.dtype(np.int32)
        )

    target_indices_shape = list(x.shape)
    target_indices_shape[axis] = (
        indices_tensor.shape[axis] if axis < len(indices_tensor.shape) else 1
    )
    target_indices_shape = tuple(target_indices_shape)

    if indices_tensor.shape != target_indices_shape:
        indices_tensor = broadcast_to_shape_hlo(
            ctx, indices_tensor, target_indices_shape
        )

    index_arrays = []
    for i in range(data_rank):
        if i == axis:
            index_arrays.append(indices_tensor)
        else:
            arange_shape = [1] * data_rank
            arange_shape[i] = x.shape[i]
            arange_shape = tuple(arange_shape)

            arange_vals = np.arange(x.shape[i], dtype=np.int32)
            const_op = HLOOp(
                "constant",
                [],
                result_shape=(x.shape[i],),
                result_dtype=np.dtype(np.int32),
                attributes={"value": arange_vals},
            )
            arange_tensor = ctx.module.add_operation(const_op)

            arange_tensor = ctx.build_op(
                "reshape", [arange_tensor], arange_shape, np.dtype(np.int32)
            )
            index_arrays.append(arange_tensor)

    broadcast_shape = target_indices_shape
    broadcasted_indices = []
    for idx_array in index_arrays:
        if idx_array.shape != broadcast_shape:
            broadcasted = broadcast_to_shape_hlo(ctx, idx_array, broadcast_shape)
            broadcasted_indices.append(broadcasted)
        else:
            broadcasted_indices.append(idx_array)

    reshaped_indices = []
    for idx in broadcasted_indices:
        new_shape = idx.shape + (1,)
        reshaped = ctx.build_op("reshape", [idx], new_shape, np.dtype(np.int32))
        reshaped_indices.append(reshaped)

    stacked_shape = broadcast_shape + (data_rank,)
    gather_indices = ctx.build_op(
        "concatenate",
        reshaped_indices,
        stacked_shape,
        np.dtype(np.int32),
        {"dimension": data_rank},
    )

    dnums = {
        "offset_dims": [],
        "collapsed_slice_dims": list(range(data_rank)),
        "start_index_map": list(range(data_rank)),
        "index_vector_dim": data_rank,
        "slice_sizes": [1] * data_rank,
        "indices_are_sorted": False,
    }

    result_tensor = ctx.build_op(
        "gather",
        [x, gather_indices],
        broadcast_shape,
        x.dtype,
        dnums,
    )

    return NKIPyTensorRef(result_tensor)


def scatter_along_axis(x, indices, values, axis):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    x_copy = ctx.build_op("copy", [x], x.shape, x.dtype)

    if axis < 0:
        axis = len(x.shape) + axis

    if isinstance(indices, NKIPyTensorRef):
        indices_tensor = indices.backend_tensor
    elif isinstance(indices, np.ndarray):
        indices_np = indices.astype(np.int32)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=indices_np.shape,
            result_dtype=np.dtype(np.int32),
            attributes={"value": indices_np},
        )
        indices_tensor = ctx.module.add_operation(const_op)
    else:
        raise ValueError("scatter_along_axis requires TensorRef or np.ndarray indices")

    if isinstance(values, NKIPyTensorRef):
        values_tensor = values.backend_tensor
    elif isinstance(values, np.ndarray):
        values_np = values.astype(x.dtype)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=values_np.shape,
            result_dtype=x.dtype,
            attributes={"value": values_np},
        )
        values_tensor = ctx.module.add_operation(const_op)
    else:
        values_tensor = as_hlo_tensor(ctx, values, x.dtype)

    update_window_dims = [i for i in range(len(x_copy.shape)) if i != axis]
    scattered_tensor = ctx.build_op(
        "scatter",
        [x_copy, indices_tensor, values_tensor],
        x_copy.shape,
        x.dtype,
        {
            "update_window_dims": update_window_dims,
            "inserted_window_dims": [axis],
            "scatter_dims_to_operand_dims": [axis],
            "index_vector_dim": len(indices_tensor.shape),
            "update_computation": "assign",
            "indices_are_sorted": False,
            "unique_indices": False,
        },
    )

    return NKIPyTensorRef(scattered_tensor)


def put_along_axis(x, indices, values, axis):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    x_shape = x.shape
    x_dtype = x.dtype
    x_copy = ctx.build_op("copy", [x], x_shape, x_dtype)

    if axis is None:
        axis = 0
        effective_shape = (int(np.prod(x_shape)),)
    else:
        if axis < 0:
            axis = len(x_shape) + axis
        effective_shape = x_shape

    if isinstance(indices, NKIPyTensorRef):
        indices_tensor = indices.backend_tensor
    elif isinstance(indices, np.ndarray):
        indices_np = indices.astype(np.int32)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=indices_np.shape,
            result_dtype=np.dtype(np.int32),
            attributes={"value": indices_np},
        )
        indices_tensor = ctx.module.add_operation(const_op)
    else:
        raise ValueError(
            "put_along_axis only supports TensorRef or np.ndarray as indices!"
        )

    if indices_tensor.dtype != np.dtype(np.int32):
        indices_tensor = ctx.build_op(
            "convert", [indices_tensor], indices_tensor.shape, np.dtype(np.int32)
        )

    idx_shape = indices_tensor.shape

    if np.isscalar(values):
        scalar_tensor = as_hlo_tensor(ctx, values, x_dtype)
        if idx_shape:
            values_tensor = ctx.build_op(
                "broadcast",
                [scalar_tensor],
                idx_shape,
                x_dtype,
                {"broadcast_dimensions": []},
            )
        else:
            values_tensor = scalar_tensor
    elif isinstance(values, NKIPyTensorRef):
        values_tensor = values.backend_tensor
    elif isinstance(values, np.ndarray):
        values_np = values.astype(x_dtype)
        const_op = HLOOp(
            "constant",
            [],
            result_shape=values_np.shape,
            result_dtype=x_dtype,
            attributes={"value": values_np},
        )
        values_tensor = ctx.module.add_operation(const_op)
    else:
        raise ValueError(
            "put_along_axis only supports scalar, TensorRef, or np.ndarray as values!"
        )

    if values_tensor.shape != idx_shape:
        values_tensor = ctx.build_op(
            "reshape", [values_tensor], idx_shape, values_tensor.dtype
        )

    ndim = len(effective_shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * effective_shape[d + 1]

    offset_np = np.zeros(idx_shape, dtype=np.int32)
    for d in range(ndim):
        if d == axis:
            continue
        coord = np.arange(idx_shape[d], dtype=np.int32)
        bcast = [1] * len(idx_shape)
        bcast[d] = idx_shape[d]
        offset_np = offset_np + coord.reshape(bcast) * strides[d]

    offset_const = ctx.build_op(
        "constant", [], idx_shape, np.dtype(np.int32), {"value": offset_np}
    )

    axis_stride_scalar = ctx.build_op(
        "constant",
        [],
        (),
        np.dtype(np.int32),
        {"value": np.int32(strides[axis])},
    )
    axis_stride = ctx.build_op(
        "broadcast",
        [axis_stride_scalar],
        idx_shape,
        np.dtype(np.int32),
        {"broadcast_dimensions": []},
    )
    scaled = ctx.build_op(
        "multiply",
        [indices_tensor, axis_stride],
        idx_shape,
        np.dtype(np.int32),
    )
    flat_indices = ctx.build_op(
        "add",
        [scaled, offset_const],
        idx_shape,
        np.dtype(np.int32),
    )

    flat_size = int(np.prod(effective_shape))
    num_elements = int(np.prod(idx_shape))

    x_flat = ctx.build_op("reshape", [x_copy], (flat_size,), x_dtype)
    flat_indices_1d = ctx.build_op(
        "reshape",
        [flat_indices],
        (num_elements,),
        np.dtype(np.int32),
    )
    flat_values_1d = ctx.build_op(
        "reshape",
        [values_tensor],
        (num_elements,),
        x_dtype,
    )

    scattered = ctx.build_op(
        "scatter",
        [x_flat, flat_indices_1d, flat_values_1d],
        (flat_size,),
        x_dtype,
        {
            "update_window_dims": [],
            "inserted_window_dims": [0],
            "scatter_dims_to_operand_dims": [0],
            "index_vector_dim": 1,
            "update_computation": "assign",
            "indices_are_sorted": False,
            "unique_indices": False,
        },
    )

    result_tensor = ctx.build_op("reshape", [scattered], x_shape, x_dtype)
    return NKIPyTensorRef(result_tensor)


def static_slice(
    x,
    start_indices: List[int],
    limit_indices: List[int],
    strides: List[int],
    squeeze_dims: List[int],
):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        backend_tensor = x.backend_tensor
        dtype = x.dtype
    else:
        backend_tensor = x
        dtype = x.dtype

    slice_shape = []
    for start, limit, stride in zip(start_indices, limit_indices, strides):
        size = (limit - start + stride - 1) // stride
        slice_shape.append(size)

    result_tensor = ctx.build_op(
        "slice",
        [backend_tensor],
        tuple(slice_shape),
        dtype,
        {
            "start_indices": start_indices,
            "limit_indices": limit_indices,
            "strides": strides,
        },
    )

    if squeeze_dims:
        output_shape = [s for i, s in enumerate(slice_shape) if i not in squeeze_dims]
        final_shape = tuple(output_shape) if output_shape else ()
        result_tensor = ctx.build_op("reshape", [result_tensor], final_shape, dtype)

    return NKIPyTensorRef(result_tensor)


def dynamic_update_slice(
    x,
    value,
    start_indices: List[int],
    update_shape: Tuple[int, ...],
):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_tensor = x.backend_tensor
        x_shape = x.shape
        x_dtype = x.dtype
    else:
        x_tensor = x
        x_shape = x.shape
        x_dtype = x.dtype

    if isinstance(value, NKIPyTensorRef):
        value_tensor = value.backend_tensor
    elif isinstance(value, (int, float)):
        value_array = np.full(update_shape, value, dtype=x_dtype)
        value_tensor = ctx.build_op(
            "constant", [], tuple(update_shape), x_dtype, {"value": value_array}
        )
    elif isinstance(value, np.ndarray):
        value_tensor = ctx.build_op(
            "constant", [], value.shape, value.dtype, {"value": value}
        )
    else:
        value_tensor = value

    if value_tensor.shape != update_shape:
        value_tensor = ctx.build_op(
            "reshape", [value_tensor], update_shape, value_tensor.dtype
        )

    start_index_tensors = []
    for start_idx in start_indices:
        scalar_tensor = ctx.build_op("constant", [], (), np.int32, {"value": start_idx})
        start_index_tensors.append(scalar_tensor)

    result_tensor = ctx.build_op(
        "dynamic-update-slice",
        [x_tensor, value_tensor] + start_index_tensors,
        x_shape,
        x_dtype,
        {},
    )

    return NKIPyTensorRef(result_tensor)


def scatter_strided(
    x,
    value,
    scatter_indices_per_dim: List[List[int]],
):
    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_tensor = x.backend_tensor
        x_shape = x.shape
        x_dtype = x.dtype
    else:
        x_tensor = x
        x_shape = x.shape
        x_dtype = x.dtype

    value_shape = tuple(len(indices) for indices in scatter_indices_per_dim)

    if isinstance(value, NKIPyTensorRef):
        value_tensor = value.backend_tensor
    elif isinstance(value, (int, float)):
        value_array = np.full(value_shape, value, dtype=x_dtype)
        value_tensor = ctx.build_op(
            "constant", [], value_shape, x_dtype, {"value": value_array}
        )
    elif isinstance(value, np.ndarray):
        value_tensor = ctx.build_op(
            "constant", [], value.shape, value.dtype, {"value": value}
        )
    else:
        value_tensor = value

    all_positions = list(itertools.product(*scatter_indices_per_dim))
    scatter_indices_array = np.array(all_positions, dtype=np.int32)

    indices_tensor = ctx.build_op(
        "constant",
        [],
        scatter_indices_array.shape,
        np.dtype(np.int32),
        {"value": scatter_indices_array},
    )

    flat_value_shape = (scatter_indices_array.shape[0],)
    flat_value = ctx.build_op(
        "reshape", [value_tensor], flat_value_shape, value_tensor.dtype
    )

    update_window_dims = []
    inserted_window_dims = list(range(len(x_shape)))
    scatter_dims_to_operand_dims = list(range(len(x_shape)))
    index_vector_dim = 1

    result_tensor = ctx.build_op(
        "scatter",
        [x_tensor, indices_tensor, flat_value],
        x_shape,
        x_dtype,
        {
            "update_window_dims": update_window_dims,
            "inserted_window_dims": inserted_window_dims,
            "scatter_dims_to_operand_dims": scatter_dims_to_operand_dims,
            "index_vector_dim": index_vector_dim,
            "update_computation": "assign",
            "indices_are_sorted": False,
            "unique_indices": True,
        },
    )

    return NKIPyTensorRef(result_tensor)


# =============================================================================
# NN ops
# =============================================================================


def topk(x, k, axis=0, is_ascend=False, out=None, dtype=None):
    ctx = get_hlo_context()

    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("the custom TopK op only supports last axis")

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    if axis < 0:
        axis = len(x.shape) + axis

    assert x.shape[axis] >= k, (
        f"k={k} must be <= size of axis {axis} which is {x.shape[axis]}"
    )

    output_shape = list(x.shape)
    output_shape[axis] = k
    output_shape = tuple(output_shape)

    input_for_topk = x
    if is_ascend:
        input_for_topk = ctx.build_op(
            "negate", [input_for_topk], input_for_topk.shape, input_for_topk.dtype
        )

    topk_output_shape = list(input_for_topk.shape)
    topk_output_shape[-1] = k
    topk_output_shape = tuple(topk_output_shape)

    topk_tuple = ctx.build_op(
        "topk",
        [input_for_topk],
        topk_output_shape,
        x.dtype,
        {"k": k, "largest": True, "is_tuple": True},
    )

    values_tensor = ctx.build_op(
        "get-tuple-element",
        [topk_tuple],
        topk_output_shape,
        x.dtype,
        {"tuple_index": 0},
    )

    indices_tensor = ctx.build_op(
        "get-tuple-element",
        [topk_tuple],
        topk_output_shape,
        np.dtype(np.uint32),
        {"tuple_index": 1},
    )

    if is_ascend:
        values_tensor = ctx.build_op(
            "negate", [values_tensor], topk_output_shape, x.dtype
        )

    return NKIPyTensorRef(values_tensor), NKIPyTensorRef(indices_tensor)


# =============================================================================
# Collective ops
# =============================================================================


def all_gather(data, all_gather_dim, replica_groups, **kwargs):
    ctx = get_hlo_context()

    rank = len(replica_groups[0])
    out_shape = list(data.shape)
    if out_shape:
        out_shape[all_gather_dim] *= rank

    result_tensor = ctx.build_op(
        "all-gather",
        [data.backend_tensor],
        tuple(out_shape),
        data.dtype,
        {
            "all_gather_dim": all_gather_dim,
            "replica_groups": replica_groups,
        },
    )
    return NKIPyTensorRef(result_tensor)


def all_reduce(data, replica_groups, reduce_op=np.add, **kwargs):
    ctx = get_hlo_context()

    reduce_op_map = {
        np.add: "add",
        np.multiply: "multiply",
        np.maximum: "maximum",
        np.minimum: "minimum",
    }
    reduce_op_str = reduce_op_map.get(reduce_op, "add")

    result_tensor = ctx.build_op(
        "all-reduce",
        [data.backend_tensor],
        data.shape,
        data.dtype,
        {
            "replica_groups": replica_groups,
            "reduce_op": reduce_op_str,
        },
    )
    return NKIPyTensorRef(result_tensor)


def reduce_scatter(data, reduce_scatter_dim: int, replica_groups, reduce_op=np.add, **kwargs):
    ctx = get_hlo_context()
    rank = len(replica_groups[0])
    out_shape = list(data.shape)
    if out_shape:
        out_shape[reduce_scatter_dim] //= rank

    reduce_op_map = {
        np.add: "add",
        np.multiply: "multiply",
        np.maximum: "maximum",
        np.minimum: "minimum",
    }
    reduce_op_str = reduce_op_map.get(reduce_op, "add")

    result_tensor = ctx.build_op(
        "reduce-scatter",
        [data.backend_tensor],
        tuple(out_shape),
        data.dtype,
        {
            "reduce_scatter_dim": reduce_scatter_dim,
            "replica_groups": replica_groups,
            "reduce_op": reduce_op_str,
        },
    )
    return NKIPyTensorRef(result_tensor)


def all_to_all(data, split_dimension: int, concat_dimension: int, replica_groups, **kwargs):
    ctx = get_hlo_context()
    result_tensor = ctx.build_op(
        "all-to-all",
        [data.backend_tensor],
        data.shape,
        data.dtype,
        {
            "split_dimension": split_dimension,
            "concat_dimension": concat_dimension,
            "replica_groups": replica_groups,
        },
    )
    return NKIPyTensorRef(result_tensor)


# =============================================================================
# Convolution ops
# =============================================================================


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out=None,
    dtype=None,
):
    from nkipy.core.ops.conv import _normalize_tuple_2d

    ctx = get_hlo_context()

    if isinstance(input, NKIPyTensorRef):
        input = input.backend_tensor
    if isinstance(weight, NKIPyTensorRef):
        weight = weight.backend_tensor

    stride = _normalize_tuple_2d(stride, "stride")
    dilation = _normalize_tuple_2d(dilation, "dilation")
    padding_tuple = _normalize_tuple_2d(padding, "padding")

    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    out_height = (
        in_height + 2 * padding_tuple[0] - dilation[0] * (kernel_height - 1) - 1
    ) // stride[0] + 1
    out_width = (
        in_width + 2 * padding_tuple[1] - dilation[1] * (kernel_width - 1) - 1
    ) // stride[1] + 1

    output_shape = (batch_size, out_channels, out_height, out_width)

    padding_config = [
        (padding_tuple[0], padding_tuple[0]),
        (padding_tuple[1], padding_tuple[1]),
    ]

    result_tensor = ctx.build_op(
        "convolution",
        [input, weight],
        output_shape,
        input.dtype,
        {
            "window_strides": list(stride),
            "padding": padding_config,
            "lhs_dilation": [1, 1],
            "rhs_dilation": list(dilation),
            "feature_group_count": groups,
            "batch_group_count": 1,
            "input_batch_dimension": 0,
            "input_feature_dimension": 1,
            "input_spatial_dimensions": [2, 3],
            "kernel_output_feature_dimension": 0,
            "kernel_input_feature_dimension": 1,
            "kernel_spatial_dimensions": [2, 3],
            "output_batch_dimension": 0,
            "output_feature_dimension": 1,
            "output_spatial_dimensions": [2, 3],
        },
    )

    if bias is not None:
        if isinstance(bias, NKIPyTensorRef):
            bias = bias.backend_tensor

        bias_reshaped = ctx.build_op(
            "reshape", [bias], (1, out_channels, 1, 1), bias.dtype
        )
        bias_broadcast = broadcast_to_shape_hlo(ctx, bias_reshaped, output_shape)
        result_tensor = ctx.build_op(
            "add", [result_tensor, bias_broadcast], output_shape, input.dtype
        )

    return NKIPyTensorRef(result_tensor)


def conv3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out=None,
    dtype=None,
):
    from nkipy.core.ops.conv import _normalize_tuple_3d

    ctx = get_hlo_context()

    if isinstance(input, NKIPyTensorRef):
        input = input.backend_tensor
    if isinstance(weight, NKIPyTensorRef):
        weight = weight.backend_tensor

    stride = _normalize_tuple_3d(stride, "stride")
    dilation = _normalize_tuple_3d(dilation, "dilation")
    padding_tuple = _normalize_tuple_3d(padding, "padding")

    batch_size, in_channels, in_depth, in_height, in_width = input.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape

    out_depth = (
        in_depth + 2 * padding_tuple[0] - dilation[0] * (kernel_depth - 1) - 1
    ) // stride[0] + 1
    out_height = (
        in_height + 2 * padding_tuple[1] - dilation[1] * (kernel_height - 1) - 1
    ) // stride[1] + 1
    out_width = (
        in_width + 2 * padding_tuple[2] - dilation[2] * (kernel_width - 1) - 1
    ) // stride[2] + 1

    output_shape = (batch_size, out_channels, out_depth, out_height, out_width)

    padding_config = [
        (padding_tuple[0], padding_tuple[0]),
        (padding_tuple[1], padding_tuple[1]),
        (padding_tuple[2], padding_tuple[2]),
    ]

    result_tensor = ctx.build_op(
        "convolution",
        [input, weight],
        output_shape,
        input.dtype,
        {
            "window_strides": list(stride),
            "padding": padding_config,
            "lhs_dilation": [1, 1, 1],
            "rhs_dilation": list(dilation),
            "feature_group_count": groups,
            "batch_group_count": 1,
            "input_batch_dimension": 0,
            "input_feature_dimension": 1,
            "input_spatial_dimensions": [2, 3, 4],
            "kernel_output_feature_dimension": 0,
            "kernel_input_feature_dimension": 1,
            "kernel_spatial_dimensions": [2, 3, 4],
            "output_batch_dimension": 0,
            "output_feature_dimension": 1,
            "output_spatial_dimensions": [2, 3, 4],
        },
    )

    if bias is not None:
        if isinstance(bias, NKIPyTensorRef):
            bias = bias.backend_tensor

        bias_reshaped = ctx.build_op(
            "reshape", [bias], (1, out_channels, 1, 1, 1), bias.dtype
        )
        bias_broadcast = broadcast_to_shape_hlo(ctx, bias_reshaped, output_shape)
        result_tensor = ctx.build_op(
            "add", [result_tensor, bias_broadcast], output_shape, input.dtype
        )

    return NKIPyTensorRef(result_tensor)
