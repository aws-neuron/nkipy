# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array creation operations: zeros, full, zeros_like, empty_like, full_like, ones_like"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# zeros
# -----------------------------------------------------------------------------
zeros = Op("zeros")


@zeros.impl("cpu")
def _zeros_cpu(shape, dtype):
    """Create a tensor filled with zeros (CPU)."""
    return np.zeros(shape, dtype=dtype)


@zeros.impl("hlo")
def _zeros_hlo(shape, dtype):
    """Create a tensor filled with zeros (HLO)."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    # Normalize shape to tuple
    if isinstance(shape, int):
        shape = (shape,)

    # Create a scalar zero constant
    zero_tensor = as_hlo_tensor(ctx, 0.0, dtype)

    # Broadcast to the target shape
    if shape:
        result_tensor = ctx.build_op(
            "broadcast", [zero_tensor], shape, dtype, {"broadcast_dimensions": []}
        )
    else:
        result_tensor = zero_tensor

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# zeros_like
# -----------------------------------------------------------------------------
zeros_like = Op("zeros_like")


@zeros_like.impl("cpu")
def _zeros_like_cpu(x, dtype=None):
    """Create a tensor of zeros with the same shape as x (CPU)."""
    return np.zeros_like(x, dtype=dtype)


@zeros_like.impl("hlo")
def _zeros_like_hlo(x, dtype=None):
    """Create a tensor of zeros with the same shape as x (HLO).

    Note: We need to reference the input tensor x in the computation graph
    even though we don't use its values, to ensure the HLO module parameter
    count matches the computation.
    """
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    # Create a scalar zero constant
    zero_tensor = as_hlo_tensor(ctx, 0.0, result_dtype)

    # Broadcast to the target shape
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
    # HLO requires all module parameters to be used in the computation
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


# -----------------------------------------------------------------------------
# ones_like
# -----------------------------------------------------------------------------
ones_like = Op("ones_like")


@ones_like.impl("cpu")
def _ones_like_cpu(x, dtype=None):
    """Create a tensor of ones with the same shape as x (CPU)."""
    return np.ones_like(x, dtype=dtype)


@ones_like.impl("hlo")
def _ones_like_hlo(x, dtype=None):
    """Create a tensor of ones with the same shape as x (HLO)."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    # Create a scalar one constant
    one_tensor = as_hlo_tensor(ctx, 1.0, result_dtype)

    # Broadcast to the target shape
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


# -----------------------------------------------------------------------------
# empty_like
# -----------------------------------------------------------------------------
empty_like = Op("empty_like")


@empty_like.impl("cpu")
def _empty_like_cpu(x, dtype=None):
    """Create an uninitialized tensor with the same shape as x (CPU)."""
    return np.empty_like(x, dtype=dtype)


@empty_like.impl("hlo")
def _empty_like_hlo(x, dtype=None):
    """Create an uninitialized tensor with the same shape as x (HLO).

    For HLO, same as zeros_like since we can't have uninitialized memory.
    """
    return _zeros_like_hlo(x, dtype=dtype)


# -----------------------------------------------------------------------------
# full_like
# -----------------------------------------------------------------------------
full_like = Op("full_like")


@full_like.impl("cpu")
def _full_like_cpu(x, fill_value, dtype=None):
    """Create a tensor filled with fill_value with the same shape as x (CPU)."""
    return np.full_like(x, fill_value, dtype=dtype)


@full_like.impl("hlo")
def _full_like_hlo(x, fill_value, dtype=None):
    """Create a tensor filled with fill_value with the same shape as x (HLO)."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_hlo = x.backend_tensor
    else:
        x_hlo = x

    result_dtype = dtype if dtype is not None else x_hlo.dtype

    # Create a scalar constant with the fill value
    fill_tensor = as_hlo_tensor(ctx, fill_value, result_dtype)

    # Broadcast to the target shape
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


# -----------------------------------------------------------------------------
# full
# -----------------------------------------------------------------------------
full = Op("full")


@full.impl("cpu")
def _full_cpu(shape, fill_value, dtype):
    """Create a tensor filled with a constant value (CPU)."""
    return np.full(shape, fill_value, dtype=dtype)


@full.impl("hlo")
def _full_hlo(shape, fill_value, dtype):
    """Create a tensor filled with a constant value (HLO)."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    # Normalize shape to tuple
    if isinstance(shape, int):
        shape = (shape,)

    # Create a scalar constant with the fill value
    fill_tensor = as_hlo_tensor(ctx, fill_value, dtype)

    # Broadcast to the target shape
    if shape:
        result_tensor = ctx.build_op(
            "broadcast", [fill_tensor], shape, dtype, {"broadcast_dimensions": []}
        )
    else:
        result_tensor = fill_tensor

    return NKIPyTensorRef(result_tensor)
