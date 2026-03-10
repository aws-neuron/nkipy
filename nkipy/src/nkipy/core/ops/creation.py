# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array creation operations: zeros, full, zeros_like, empty_like, full_like, ones_like"""

import builtins

import numpy as np

from nkipy.core.ops._registry import Op

builtins_min = builtins.min

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


# -----------------------------------------------------------------------------
# constant - promote compile-time numpy arrays to runtime tensors
# -----------------------------------------------------------------------------
constant = Op("constant")


@constant.impl("cpu")
def _constant_cpu(value, dtype=None):
    """Convert value to numpy array (CPU).

    When not tracing, this simply ensures the value is an ndarray.
    """
    return np.asarray(value, dtype=dtype)


@constant.impl("hlo")
def _constant_hlo(value, dtype=None):
    """Promote a numpy array or scalar to an HLO constant tensor.

    This is the primary API for promoting compile-time numpy arrays
    to runtime tensors during HLO tracing. It wraps the value as a
    single HLO constant op.

    Behavior:
    - NKIPyTensorRef: pass-through (idempotent), with optional dtype cast
    - np.ndarray: create HLO constant
    - scalar (int, float, bool): create scalar HLO constant
    - list/tuple: convert to np.ndarray first, then create HLO constant
    """
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    # Idempotent: if already a traced tensor, just handle dtype
    if isinstance(value, NKIPyTensorRef):
        if dtype is not None and value.dtype != np.dtype(dtype):
            from nkipy.core.ops.transform import astype

            return astype(value, dtype)
        return value

    ctx = get_hlo_context()

    # Determine target dtype
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

    # Convert lists/tuples to ndarray
    if isinstance(value, (list, tuple)):
        value = np.asarray(value, dtype=target_dtype)

    hlo_tensor = as_hlo_tensor(ctx, value, target_dtype)
    return NKIPyTensorRef(hlo_tensor)


# -----------------------------------------------------------------------------
# tril - lower triangle of an array
# -----------------------------------------------------------------------------
tril = Op("tril")


@tril.impl("hlo")
def _tril_hlo(x, k=0):
    """Lower triangle: where(row >= col - k, x, 0)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.ops.binary import greater_equal, subtract
    from nkipy.core.ops.indexing import where
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    shape = x_bt.shape
    ndim = len(shape)

    # Create iota for row indices (second-to-last dim)
    row_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 2}
    )
    row_ref = NKIPyTensorRef(row_iota)

    # Create iota for col indices (last dim)
    col_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 1}
    )
    col_ref = NKIPyTensorRef(col_iota)

    # Mask: row >= col - k  →  row + k >= col  →  row - (col - k) >= 0
    if k != 0:
        col_ref = subtract(col_ref, k)
    mask = greater_equal(row_ref, col_ref)

    return where(mask, x, 0.0)


# -----------------------------------------------------------------------------
# triu - upper triangle of an array
# -----------------------------------------------------------------------------
triu = Op("triu")


@triu.impl("hlo")
def _triu_hlo(x, k=0):
    """Upper triangle: where(row <= col - k, x, 0)."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.ops.binary import less_equal, subtract
    from nkipy.core.ops.indexing import where
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x_bt = x.backend_tensor
    else:
        x_bt = x

    shape = x_bt.shape
    ndim = len(shape)

    # Create iota for row indices (second-to-last dim)
    row_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 2}
    )
    row_ref = NKIPyTensorRef(row_iota)

    # Create iota for col indices (last dim)
    col_iota = ctx.build_op(
        "iota", [], shape, np.dtype(np.int32), {"iota_dimension": ndim - 1}
    )
    col_ref = NKIPyTensorRef(col_iota)

    # Mask: row <= col - k
    if k != 0:
        col_ref = subtract(col_ref, k)
    mask = less_equal(row_ref, col_ref)

    return where(mask, x, 0.0)


# -----------------------------------------------------------------------------
# diag - extract diagonal or construct diagonal matrix
# -----------------------------------------------------------------------------
diag = Op("diag")


@diag.impl("hlo")
def _diag_hlo(v, k=0):
    """Extract diagonal from 2D → 1D, or create diagonal matrix from 1D → 2D."""
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.ops.binary import equal, subtract
    from nkipy.core.ops.indexing import where
    from nkipy.core.ops.reduce import sum
    from nkipy.core.ops.transform import broadcast_to, reshape
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(v, NKIPyTensorRef):
        v_bt = v.backend_tensor
    else:
        v_bt = v

    ndim = len(v_bt.shape)

    if ndim == 1:
        # 1D → 2D: construct diagonal matrix
        n = v_bt.shape[0] + abs(k)
        shape_2d = (n, n)

        # Create row and col iota
        row_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 0}
        )
        row_ref = NKIPyTensorRef(row_iota)
        col_iota = ctx.build_op(
            "iota", [], shape_2d, np.dtype(np.int32), {"iota_dimension": 1}
        )
        col_ref = NKIPyTensorRef(col_iota)

        if k != 0:
            col_ref = subtract(col_ref, k)

        diag_mask = equal(row_ref, col_ref)

        # Broadcast v to (n, n): put v along the diagonal
        # v has length v_bt.shape[0], we need to gather it at the right positions
        # Use iota as gather index: for each (i, j) on the diagonal, take v[i] (or v[j-k])
        if k >= 0:
            # Diagonal starts at column k: row index gives position in v
            idx_ref = row_ref
        else:
            # Diagonal starts at row -k: col index (after offset) gives position in v
            idx_ref = col_ref

        # Use take to gather v values at idx positions, then mask
        from nkipy.core.ops.indexing import take

        v_gathered = take(v, idx_ref, axis=0)
        return where(diag_mask, v_gathered, 0.0)

    elif ndim == 2:
        # 2D → 1D: extract diagonal
        rows, cols = v_bt.shape
        if k >= 0:
            diag_len = builtins_min(rows, cols - k)
        else:
            diag_len = builtins_min(rows + k, cols)

        if diag_len <= 0:
            # Return empty-ish result — use shape (0,) but HLO needs >= 1
            # Just return a size-1 zero as a fallback
            return zeros((0,), v_bt.dtype)

        # Create indices for the diagonal
        diag_indices = np.arange(diag_len, dtype=np.int32)
        if k >= 0:
            row_indices = diag_indices
            col_indices = diag_indices + k
        else:
            row_indices = diag_indices - k
            col_indices = diag_indices

        # Create the 2D mask approach: iota == expected diagonal position
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
            col_ref = subtract(col_ref, k)

        diag_mask = equal(row_ref, col_ref)

        # Mask and sum along cols to extract diagonal
        masked = where(diag_mask, v, 0.0)

        # Sum along the appropriate axis to collapse to 1D
        if k >= 0:
            # Sum along columns, then slice to diag_len
            result = sum(masked, axis=1)
        else:
            # Sum along rows, then slice to diag_len
            result = sum(masked, axis=0)

        # Slice to the correct diagonal length
        from nkipy.core.ops.transform import astype as astype_op

        result_shape = result.shape
        if result_shape[0] != diag_len:
            from nkipy.core.ops.indexing import static_slice

            result = static_slice(
                result,
                [0],
                [diag_len],
                [1],
                [],
            )

        return result

    else:
        raise ValueError(f"Input must be 1-D or 2-D, got {ndim}-D")
