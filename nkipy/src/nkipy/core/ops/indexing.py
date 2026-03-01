# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexing operations: where, take, take_along_axis, put_along_axis,
static_slice, dynamic_update_slice, scatter_strided
"""

import itertools
from typing import List, Tuple

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# where
# -----------------------------------------------------------------------------
where = Op("where")


@where.impl("hlo")
def _where_hlo(condition, x, y):
    """Select elements from x or y based on condition (HLO)."""
    from nkipy.core.backend.hlo import (
        HLOOp,
        as_hlo_tensor,
        broadcast_to_shape_hlo,
        find_common_type_hlo,
        get_hlo_context,
    )
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    output_dtype = find_common_type_hlo(x, y)

    # Convert inputs to HLO tensors
    if isinstance(condition, NKIPyTensorRef):
        condition = condition.backend_tensor
    elif np.isscalar(condition):
        condition = as_hlo_tensor(ctx, bool(condition), np.bool_)
    elif isinstance(condition, np.ndarray):
        condition = as_hlo_tensor(ctx, condition.astype(bool), np.bool_)

    # If other integer type, convert to bool first
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
        # Use output_dtype for scalar conversion to avoid losing precision
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
        # Use output_dtype for scalar conversion to avoid losing precision
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

    # Broadcast all three inputs to a common shape
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


# -----------------------------------------------------------------------------
# take
# -----------------------------------------------------------------------------
take = Op("take")


@take.impl("hlo")
def _take_hlo(x, indices, axis=None):
    """Take elements from an array along an axis (HLO)."""
    from nkipy.core.backend.hlo import HLOOp, as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Handle axis=None case - flatten the array first
    if axis is None:
        flattened_shape = (int(np.prod(x.shape)),)
        x = ctx.build_op("reshape", [x], flattened_shape, x.dtype)
        axis = 0

    # Normalize negative axis
    if axis < 0:
        axis = len(x.shape) + axis

    dtype = x.dtype

    # Convert indices to HLO tensor
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

    # Build output shape
    output_shape = []
    for i in range(len(x.shape)):
        if i == axis:
            output_shape.extend(indices_shape)
        else:
            output_shape.append(x.shape[i])
    output_shape = tuple(output_shape)

    # Configure gather dimension numbers
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


# -----------------------------------------------------------------------------
# take_along_axis
# -----------------------------------------------------------------------------
take_along_axis = Op("take_along_axis")


@take_along_axis.impl("hlo")
def _take_along_axis_hlo(x, indices, axis):
    """Take values from the input array by matching 1d index and data slices (HLO)."""
    from nkipy.core.backend.hlo import (
        HLOOp,
        broadcast_to_shape_hlo,
        get_hlo_context,
    )
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Handle axis=None case
    if axis is None:
        flattened_shape = (int(np.prod(x.shape)),)
        x = ctx.build_op("reshape", [x], flattened_shape, x.dtype)
        axis = 0

    # Normalize negative axis
    if axis < 0:
        axis = len(x.shape) + axis

    # Convert indices to HLO tensor
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

    # Convert indices to int32 if needed
    if indices_tensor.dtype != np.dtype(np.int32):
        indices_tensor = ctx.build_op(
            "convert", [indices_tensor], indices_tensor.shape, np.dtype(np.int32)
        )

    # Broadcast indices to match operand shape
    target_indices_shape = list(x.shape)
    target_indices_shape[axis] = (
        indices_tensor.shape[axis] if axis < len(indices_tensor.shape) else 1
    )
    target_indices_shape = tuple(target_indices_shape)

    if indices_tensor.shape != target_indices_shape:
        indices_tensor = broadcast_to_shape_hlo(
            ctx, indices_tensor, target_indices_shape
        )

    # Create index arrays for all dimensions
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

    # Broadcast all index arrays to the same shape
    broadcast_shape = target_indices_shape
    broadcasted_indices = []
    for idx_array in index_arrays:
        if idx_array.shape != broadcast_shape:
            broadcasted = broadcast_to_shape_hlo(ctx, idx_array, broadcast_shape)
            broadcasted_indices.append(broadcasted)
        else:
            broadcasted_indices.append(idx_array)

    # Stack indices along last dimension
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

    # Configure gather dimension numbers
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


# -----------------------------------------------------------------------------
# scatter_along_axis (internal) - Window-level scatter with 1D indices
# -----------------------------------------------------------------------------
# Used by _do_scatter_indexing (__setitem__ with tensor indices).
# NOT the same as np.put_along_axis: this scatters entire rows/columns
# (1D indices, update_window_dims covers non-axis dims), while numpy's
# put_along_axis does element-level scatter (indices same ndim as array).
scatter_along_axis = Op("scatter_along_axis")


@scatter_along_axis.impl("hlo")
def _scatter_along_axis_hlo(x, indices, values, axis):
    """Scatter whole slices along an axis using 1D indices.

    For a 2D array with axis=1 and indices [2, 0]:
      a[:, 2] = values[:, 0]
      a[:, 0] = values[:, 1]
    """
    from nkipy.core.backend.hlo import HLOOp, as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    x_copy = ctx.build_op("copy", [x], x.shape, x.dtype)

    if axis < 0:
        axis = len(x.shape) + axis

    # Convert indices to HLO tensor
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

    # Convert values to HLO tensor
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

    # Window-level scatter: update_window_dims covers all non-axis dims
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


# -----------------------------------------------------------------------------
# put_along_axis - numpy-compatible element-level scatter
# -----------------------------------------------------------------------------
put_along_axis = Op("put_along_axis")


@put_along_axis.impl("hlo")
def _put_along_axis_hlo(x, indices, values, axis):
    """Element-level scatter matching np.put_along_axis semantics.

    For each position (i, j, ...) in indices:
      arr[..., indices[i,j,...], ...] = values[i,j,...]
    where the index value replaces the coordinate along ``axis``.

    Lowered to a flat 1D scatter for Neuron compiler compatibility.
    """
    from nkipy.core.backend.hlo import HLOOp, as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    x_shape = x.shape
    x_dtype = x.dtype
    x_copy = ctx.build_op("copy", [x], x_shape, x_dtype)

    # Normalize axis
    if axis is None:
        axis = 0
        effective_shape = (int(np.prod(x_shape)),)
    else:
        if axis < 0:
            axis = len(x_shape) + axis
        effective_shape = x_shape

    # --- Convert indices to HLO tensor ---
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

    # Ensure int32 for arithmetic (indices may be uint32 from user code)
    if indices_tensor.dtype != np.dtype(np.int32):
        indices_tensor = ctx.build_op(
            "convert", [indices_tensor], indices_tensor.shape, np.dtype(np.int32)
        )

    idx_shape = indices_tensor.shape

    # --- Convert values to HLO tensor ---
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

    # --- Compute flat 1D scatter indices ---
    # Row-major strides of the effective shape.
    ndim = len(effective_shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * effective_shape[d + 1]

    # Static offset: for each position in idx_shape, the flat contribution
    # from all non-axis dimensions (known at trace time).
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

    # flat_indices = indices * stride[axis] + offset
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

    # --- Flatten and scatter ---
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

    # Reshape back to original x shape
    result_tensor = ctx.build_op("reshape", [scattered], x_shape, x_dtype)
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# static_slice - Backend-agnostic static slicing operation
# -----------------------------------------------------------------------------
static_slice = Op("static_slice")


@static_slice.impl("hlo")
def _static_slice_hlo(
    x,
    start_indices: List[int],
    limit_indices: List[int],
    strides: List[int],
    squeeze_dims: List[int],
):
    """Static slicing using HLO slice operation.

    Args:
        x: Input tensor (NKIPyTensorRef)
        start_indices: Start index for each dimension
        limit_indices: End index (exclusive) for each dimension
        strides: Step size for each dimension
        squeeze_dims: Dimensions to squeeze (remove) from output

    Returns:
        Sliced tensor
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        backend_tensor = x.backend_tensor
        dtype = x.dtype
    else:
        backend_tensor = x
        dtype = x.dtype

    # Calculate slice shape (before squeeze)
    slice_shape = []
    for start, limit, stride in zip(start_indices, limit_indices, strides):
        size = (limit - start + stride - 1) // stride
        slice_shape.append(size)

    # Build HLO slice operation
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

    # If we need to squeeze dimensions, reshape
    if squeeze_dims:
        output_shape = [s for i, s in enumerate(slice_shape) if i not in squeeze_dims]
        final_shape = tuple(output_shape) if output_shape else ()
        result_tensor = ctx.build_op("reshape", [result_tensor], final_shape, dtype)

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# dynamic_update_slice - Backend-agnostic contiguous slice assignment
# -----------------------------------------------------------------------------
dynamic_update_slice = Op("dynamic_update_slice")


@dynamic_update_slice.impl("hlo")
def _dynamic_update_slice_hlo(
    x,
    value,
    start_indices: List[int],
    update_shape: Tuple[int, ...],
):
    """Contiguous slice assignment using HLO dynamic-update-slice.

    Args:
        x: Input tensor to update (NKIPyTensorRef)
        value: Value tensor to insert (NKIPyTensorRef or scalar/array)
        start_indices: Start index for each dimension
        update_shape: Expected shape of the update region

    Returns:
        Updated tensor (new tensor, not in-place)
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_tensor = x.backend_tensor
        x_shape = x.shape
        x_dtype = x.dtype
    else:
        x_tensor = x
        x_shape = x.shape
        x_dtype = x.dtype

    # Convert value to tensor if needed
    if isinstance(value, NKIPyTensorRef):
        value_tensor = value.backend_tensor
    elif isinstance(value, (int, float)):
        # Scalar value - create constant and broadcast
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

    # Reshape value if needed to match update_shape
    if value_tensor.shape != update_shape:
        value_tensor = ctx.build_op(
            "reshape", [value_tensor], update_shape, value_tensor.dtype
        )

    # Create scalar constant tensors for start indices
    start_index_tensors = []
    for start_idx in start_indices:
        scalar_tensor = ctx.build_op("constant", [], (), np.int32, {"value": start_idx})
        start_index_tensors.append(scalar_tensor)

    # Build dynamic-update-slice operation
    result_tensor = ctx.build_op(
        "dynamic-update-slice",
        [x_tensor, value_tensor] + start_index_tensors,
        x_shape,
        x_dtype,
        {},
    )

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# scatter_strided - Backend-agnostic strided slice assignment
# -----------------------------------------------------------------------------
scatter_strided = Op("scatter_strided")


@scatter_strided.impl("hlo")
def _scatter_strided_hlo(
    x,
    value,
    scatter_indices_per_dim: List[List[int]],
):
    """Strided slice assignment using HLO scatter.

    For a[::2, ::2] = b, scatter values to strided positions.

    Args:
        x: Input tensor to update (NKIPyTensorRef)
        value: Value tensor to scatter (NKIPyTensorRef or scalar/array)
        scatter_indices_per_dim: List of index lists for each dimension

    Returns:
        Updated tensor (new tensor, not in-place)
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_tensor = x.backend_tensor
        x_shape = x.shape
        x_dtype = x.dtype
    else:
        x_tensor = x
        x_shape = x.shape
        x_dtype = x.dtype

    # Calculate expected value shape from scatter indices
    value_shape = tuple(len(indices) for indices in scatter_indices_per_dim)

    # Convert value to tensor if needed
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

    # Create meshgrid of all scatter positions
    all_positions = list(itertools.product(*scatter_indices_per_dim))
    scatter_indices_array = np.array(all_positions, dtype=np.int32)

    # Create HLO constant for scatter indices
    indices_tensor = ctx.build_op(
        "constant",
        [],
        scatter_indices_array.shape,
        np.dtype(np.int32),
        {"value": scatter_indices_array},
    )

    # Flatten the value tensor to match the number of scatter positions
    flat_value_shape = (scatter_indices_array.shape[0],)
    flat_value = ctx.build_op(
        "reshape", [value_tensor], flat_value_shape, value_tensor.dtype
    )

    # Configure scatter parameters
    update_window_dims = []
    inserted_window_dims = list(range(len(x_shape)))
    scatter_dims_to_operand_dims = list(range(len(x_shape)))
    index_vector_dim = 1

    # Build scatter operation
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
