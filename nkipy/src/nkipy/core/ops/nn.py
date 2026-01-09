# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neural network operations: softmax, topk, rms_norm"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# softmax
# -----------------------------------------------------------------------------
softmax = Op("softmax")

# HLO implementation for softmax not implemented

# -----------------------------------------------------------------------------
# topk
# -----------------------------------------------------------------------------
topk = Op("topk")


@topk.impl("cpu")
def _topk_cpu(x, k, axis=0, is_ascend=False, out=None, dtype=None):
    """Top-k operation (CPU).

    Args:
        x: Input array
        k: Number of top elements to extract
        axis: Axis along which to find top k elements
        is_ascend: If True, find k smallest elements; if False, find k largest
        out: Unused (for API compatibility)
        dtype: Unused (for API compatibility)

    Returns:
        Tuple of (values, indices) arrays
    """
    # Normalize negative axis
    if axis < 0:
        axis = x.ndim + axis

    if is_ascend:
        # Find k smallest - use argpartition for efficiency
        # Use k-1 as partition index to avoid out-of-bounds when k equals array size
        indices = np.argpartition(x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        # Sort the k elements
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)
    else:
        # Find k largest - negate, partition, negate back
        # Use k-1 as partition index to avoid out-of-bounds when k equals array size
        indices = np.argpartition(-x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        # Sort the k elements in descending order
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(-values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)

    return values, indices.astype(np.uint32)


@topk.impl("hlo")
def _topk_hlo(x, k, axis=0, is_ascend=False, out=None, dtype=None):
    """Top-k operation for HLO backend.

    Args:
        x: Input tensor
        k: Number of top elements to extract
        axis: Axis along which to find top k elements
        is_ascend: If True, find k smallest elements; if False, find k largest
        out: Unused (for API compatibility)
        dtype: Unused (for API compatibility)

    Returns:
        Tuple of (values, indices) tensors
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("the custom TopK op only supports last axis")

    ctx = get_hlo_context()

    # Convert NKIPyTensorRef to HLOTensor if needed
    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Normalize negative axis
    if axis < 0:
        axis = len(x.shape) + axis

    # Validate k
    assert x.shape[axis] >= k, (
        f"k={k} must be <= size of axis {axis} which is {x.shape[axis]}"
    )

    output_shape = list(x.shape)
    output_shape[axis] = k
    output_shape = tuple(output_shape)

    input_for_topk = x
    # Build TopK operation
    # Note: HLO TopK always returns largest elements, so for smallest we need to negate
    if is_ascend:
        # For ascending (smallest k), negate the input
        input_for_topk = ctx.build_op(
            "negate", [input_for_topk], input_for_topk.shape, input_for_topk.dtype
        )

    # Create the TopK operation
    topk_output_shape = list(input_for_topk.shape)
    topk_output_shape[-1] = k
    topk_output_shape = tuple(topk_output_shape)

    # Build TopK - it returns a tuple of (values, indices)
    topk_tuple = ctx.build_op(
        "topk",
        [input_for_topk],
        topk_output_shape,
        x.dtype,
        {"k": k, "largest": True, "is_tuple": True},
    )

    # Extract values (tuple element 0)
    values_tensor = ctx.build_op(
        "get-tuple-element",
        [topk_tuple],
        topk_output_shape,
        x.dtype,
        {"tuple_index": 0},
    )

    # Extract indices (tuple element 1)
    indices_tensor = ctx.build_op(
        "get-tuple-element",
        [topk_tuple],
        topk_output_shape,
        np.dtype(np.uint32),
        {"tuple_index": 1},
    )

    # If we negated for ascending order, negate the values back
    if is_ascend:
        values_tensor = ctx.build_op(
            "negate", [values_tensor], topk_output_shape, x.dtype
        )

    return NKIPyTensorRef(values_tensor), NKIPyTensorRef(indices_tensor)


# -----------------------------------------------------------------------------
# rms_norm
# -----------------------------------------------------------------------------
rms_norm = Op("rms_norm")

# HLO implementation for rmsnorm not implemented
