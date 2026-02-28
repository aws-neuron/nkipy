# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape transformation operations:
reshape, transpose, expand_dims, concatenate, split, copy, repeat
"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# reshape
# -----------------------------------------------------------------------------
reshape = Op("reshape")


@reshape.impl("hlo")
def _reshape_hlo(x, newshape):
    """Reshape tensor to new shape (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Normalize shape (handle -1)
    if isinstance(newshape, int):
        newshape = (newshape,)

    # Handle -1 in newshape
    if -1 in newshape:
        total_size = int(np.prod(x.shape))
        known_size = int(np.prod([d for d in newshape if d != -1]))
        assert known_size > 0, "Cannot reshape to a size of 0"
        assert total_size % known_size == 0, (
            f"Cannot reshape array of size {total_size} into shape {newshape}"
        )
        newshape = tuple(total_size // known_size if d == -1 else d for d in newshape)

    # Verify total size matches
    if np.prod(x.shape) != np.prod(newshape):
        raise ValueError(
            f"Cannot reshape array of size {np.prod(x.shape)} into shape {newshape}"
        )

    result_tensor = ctx.build_op("reshape", [x], newshape, x.dtype)
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# transpose
# -----------------------------------------------------------------------------
transpose = Op("transpose")


@transpose.impl("hlo")
def _transpose_hlo(x, axes=None, out=None, dtype=None):
    """Transpose tensor (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Handle default axes (reverse all dimensions)
    if axes is None:
        axes = list(range(len(x.shape)))[::-1]

    # Calculate output shape
    result_shape = tuple(x.shape[i] for i in axes)

    result_tensor = ctx.build_op(
        "transpose", [x], result_shape, x.dtype, {"permutation": axes}
    )
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# expand_dims
# -----------------------------------------------------------------------------
expand_dims = Op("expand_dims")


@expand_dims.impl("hlo")
def _expand_dims_hlo(x, axis):
    """Expand dimensions of tensor (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    rank = len(x.shape)

    # Handle both single axis and list of axes
    if isinstance(axis, (list, tuple)):
        # Final rank after all dimensions are added
        final_rank = rank + len(axis)

        # Normalize negative axes relative to final rank
        axes = []
        for ax in axis:
            if ax < 0:
                ax = final_rank + ax
            if ax < 0 or ax > final_rank - 1:
                raise ValueError(
                    f"axis {ax} is out of bounds for array of dimension {final_rank}"
                )
            axes.append(ax)

        # Check for duplicate axes
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


# -----------------------------------------------------------------------------
# concatenate
# -----------------------------------------------------------------------------
concatenate = Op("concatenate")


@concatenate.impl("hlo")
def _concatenate_hlo(tensors, axis=0):
    """Concatenate tensors along axis (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    # Convert all tensors to HLOTensor
    hlo_tensors = []
    for t in tensors:
        if isinstance(t, NKIPyTensorRef):
            hlo_tensors.append(t.backend_tensor)
        elif isinstance(t, np.ndarray):
            # Convert concrete np.ndarray to HLO constant
            from nkipy.core.ops.creation import constant

            const_ref = constant(t)
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

    # Normalize negative axis
    ndim = len(hlo_tensors[0].shape)
    if axis < 0:
        axis = ndim + axis

    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    # Calculate output shape
    output_shape = list(hlo_tensors[0].shape)
    output_shape[axis] = sum(t.shape[axis] for t in hlo_tensors)
    output_shape = tuple(output_shape)

    # Get common dtype
    dtype = hlo_tensors[0].dtype
    for t in hlo_tensors[1:]:
        dtype = np.result_type(dtype, t.dtype)

    result_tensor = ctx.build_op(
        "concatenate", hlo_tensors, output_shape, dtype, {"dimension": axis}
    )
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# split
# -----------------------------------------------------------------------------
split = Op("split")


@split.impl("hlo")
def _split_hlo(x, indices_or_sections, axis=0):
    """Split tensor into multiple tensors (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Normalize negative axis
    if axis < 0:
        axis = len(x.shape) + axis

    if axis < 0 or axis >= len(x.shape):
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {len(x.shape)}"
        )

    axis_size = x.shape[axis]

    # Determine split points
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


# -----------------------------------------------------------------------------
# copy
# -----------------------------------------------------------------------------
copy = Op("copy")


@copy.impl("hlo")
def _copy_hlo(x, out=None, dtype=None):
    """Copy tensor (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# repeat
# -----------------------------------------------------------------------------
repeat = Op("repeat")


@repeat.impl("hlo")
def _repeat_hlo(x, repeats, axis=None):
    """Repeat elements of tensor (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
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

    if not isinstance(repeats, (int, np.integer)):
        raise TypeError(
            f"Only compile-time-known integer repeats are supported, got {type(repeats).__name__}. "
            "Dynamic tensor repeats are not supported in tracing."
        )
    repeats = int(repeats)

    # Calculate output shape
    new_shape = list(x.shape)
    new_shape[axis] *= repeats
    new_shape = tuple(new_shape)

    # Strategy: broadcast then reshape
    # 1. Build broadcast shape by inserting repeats dimension after axis
    broadcast_shape = list(x.shape)
    broadcast_shape.insert(axis + 1, repeats)
    broadcast_shape = tuple(broadcast_shape)

    # 2. Broadcast x to the expanded shape
    # broadcast_dims maps each dim of x to the corresponding dim in broadcast_shape
    # (skipping the newly inserted repeat dimension at axis+1)
    broadcast_dims = [i if i <= axis else i + 1 for i in range(len(x.shape))]
    x_broadcast = ctx.build_op(
        "broadcast",
        [x],
        broadcast_shape,
        x.dtype,
        {"broadcast_dimensions": broadcast_dims},
    )

    # 3. Reshape to merge axis and repeat dimensions
    result_tensor = ctx.build_op("reshape", [x_broadcast], new_shape, x.dtype)

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# broadcast_to
# -----------------------------------------------------------------------------
broadcast_to = Op("broadcast_to")


@broadcast_to.impl("hlo")
def _broadcast_to_hlo(x, shape, out=None, dtype=None):
    """Broadcast tensor to target shape (HLO)."""
    from nkipy.core.backend.hlo import broadcast_to_shape_hlo, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Convert shape to tuple if needed
    if isinstance(shape, int):
        shape = (shape,)
    target_shape = tuple(shape)

    # If shapes are already the same, just return a copy
    if x.shape == target_shape:
        result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
        return NKIPyTensorRef(result_tensor)

    result_tensor = broadcast_to_shape_hlo(ctx, x, target_shape)
    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# copyto
# -----------------------------------------------------------------------------
copyto = Op("copyto")

# Note: copyto for HLO is deprecated due to in-place semantics


# -----------------------------------------------------------------------------
# squeeze
# -----------------------------------------------------------------------------
squeeze = Op("squeeze")


@squeeze.impl("hlo")
def _squeeze_hlo(x, axis=None):
    """Remove size-1 dimensions from tensor."""
    x_shape = x.shape
    ndim = len(x_shape)

    if axis is None:
        # Remove all size-1 dimensions
        new_shape = tuple(s for s in x_shape if s != 1)
        if not new_shape:
            new_shape = ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        # Normalize negative axes
        axes = tuple(a if a >= 0 else ndim + a for a in axis)
        # Validate
        for a in axes:
            if x_shape[a] != 1:
                raise ValueError(
                    f"cannot select an axis to squeeze out which has size "
                    f"not equal to one, got shape[{a}] = {x_shape[a]}"
                )
        new_shape = tuple(s for i, s in enumerate(x_shape) if i not in axes)

    if new_shape == x_shape:
        return x
    return reshape(x, new_shape)


# -----------------------------------------------------------------------------
# astype
# -----------------------------------------------------------------------------
astype = Op("astype")


@astype.impl("hlo")
def _astype_hlo(x, dtype):
    """Convert tensor to specified dtype (HLO)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # If dtype is the same, just copy
    if x.dtype == dtype:
        result_tensor = ctx.build_op("copy", [x], x.shape, x.dtype)
    else:
        result_tensor = ctx.build_op("convert", [x], x.shape, dtype)

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# pad
# -----------------------------------------------------------------------------
pad = Op("pad")


@pad.impl("hlo")
def _pad_hlo(x, pad_width, mode="constant", constant_values=0, **kwargs):
    """Pad tensor with various modes.

    Supports:
    - mode='constant': Uses native HLO pad instruction
    - mode='edge': Composes from slice + concatenate
    """
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

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

    # Normalize pad_width to list of (before, after) tuples
    pad_width = np.asarray(pad_width)
    if pad_width.ndim == 0:
        pad_width = np.broadcast_to(pad_width, (ndim, 2))
    elif pad_width.ndim == 1:
        if len(pad_width) == 2:
            pad_width = np.broadcast_to(pad_width, (ndim, 2))
        else:
            pad_width = np.array([[p, p] for p in pad_width])
            if len(pad_width) != ndim:
                raise ValueError(
                    f"pad_width must have length {ndim} to match array dimensions, "
                    f"got {len(pad_width)}"
                )
    # Broadcast short 2D pad_width to all dims (e.g. np.pad(a_2d, ((1,2),)))
    if pad_width.ndim == 2 and len(pad_width) == 1:
        pad_width = np.broadcast_to(pad_width, (ndim, 2))
    pad_width = [(int(pad_width[i, 0]), int(pad_width[i, 1])) for i in range(ndim)]

    if mode == "constant":
        # Use native HLO pad instruction
        # Build padding_config: list of (low, high, interior) per dim
        padding_config = [(low, high, 0) for low, high in pad_width]

        # Calculate output shape
        result_shape = tuple(
            s + low + high for s, (low, high) in zip(x_shape, pad_width)
        )

        # Create padding value scalar
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
        # Compose from slice + concatenate for each dimension
        result = NKIPyTensorRef(x_bt) if not isinstance(x, NKIPyTensorRef) else x
        for dim in range(ndim):
            before, after = pad_width[dim]
            if before == 0 and after == 0:
                continue

            parts = []
            if before > 0:
                # Slice the first element along this dim and repeat
                edge_slice = _slice_single(result, dim, 0)
                edge_expanded = expand_dims(edge_slice, axis=dim)
                edge_repeated = repeat(edge_expanded, before, axis=dim)
                parts.append(edge_repeated)

            parts.append(result)

            if after > 0:
                # Slice the last element along this dim and repeat
                last_idx = result.shape[dim] - 1
                edge_slice = _slice_single(result, dim, last_idx)
                edge_expanded = expand_dims(edge_slice, axis=dim)
                edge_repeated = repeat(edge_expanded, after, axis=dim)
                parts.append(edge_repeated)

            result = concatenate(parts, axis=dim)

        return result

    else:
        raise NotImplementedError(
            f"Pad mode '{mode}' is not supported. Only 'constant' and 'edge' modes are available."
        )


def _slice_single(x, dim, index):
    """Slice a single element along a dimension, removing that dim."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

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

    # Remove the sliced dimension
    result_shape = tuple(s for i, s in enumerate(x_bt.shape) if i != dim)
    result = ctx.build_op("reshape", [sliced], result_shape, x_bt.dtype)
    return NKIPyTensorRef(result)
