# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed collective operations: all_gather, all_reduce, reduce_scatter, all_to_all"""

import numpy as np

from nkipy.core.backend.hlo import HLOTraceContext
from nkipy.core.ops._registry import Op
from nkipy.core.tensor import NKIPyTensorRef

# -----------------------------------------------------------------------------
# all_gather
# -----------------------------------------------------------------------------
all_gather = Op("all_gather")


@all_gather.impl("cpu")
def _all_gather_cpu(data, all_gather_dim, replica_groups, **kwargs):
    """CPU implementation of all_gather with duplicated data assumption.

    In CPU execution, we assume all ranks have identical data.
    all_gather collects data from all ranks and concatenates along the gather dimension.

    Shape: dim[all_gather_dim] *= world_size
    Data: Replicate along gather dimension (same data from all ranks)
    """
    world_size = len(replica_groups[0])
    tile_reps = tuple(
        world_size if i == all_gather_dim else 1 for i in range(data.ndim)
    )
    return np.tile(data, tile_reps)


@all_gather.impl("hlo")
def _all_gather_hlo(data, all_gather_dim, replica_groups, **kwargs):
    ctx = HLOTraceContext._global_ctx
    assert ctx is not None, "HLO context is not initialized"

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


# -----------------------------------------------------------------------------
# all_reduce
# -----------------------------------------------------------------------------
all_reduce = Op("all_reduce")


@all_reduce.impl("cpu")
def _all_reduce_cpu(data, replica_groups, reduce_op=np.add, **kwargs):
    """CPU implementation of all_reduce with duplicated data assumption.

    In CPU execution, we assume all ranks have identical data.
    all_reduce applies the reduction operation across all ranks.

    Shape: unchanged
    Data: For add, result = data * world_size (sum of identical values)
          For max/min, result = data (max/min of identical values)
          For multiply, result = data ** world_size
    """
    world_size = len(replica_groups[0])

    if reduce_op == np.add:
        return data * world_size
    elif reduce_op == np.maximum or reduce_op == np.minimum:
        return data.copy()
    elif reduce_op == np.multiply:
        return data**world_size
    else:
        # Default: return copy
        return data.copy()


@all_reduce.impl("hlo")
def _all_reduce_hlo(data, replica_groups, reduce_op=np.add, **kwargs):
    ctx = HLOTraceContext._global_ctx
    assert ctx is not None, "HLO context is not initialized"

    # Map reduce_op to string for HLO
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


# -----------------------------------------------------------------------------
# reduce_scatter
# -----------------------------------------------------------------------------
reduce_scatter = Op("reduce_scatter")


@reduce_scatter.impl("cpu")
def _reduce_scatter_cpu(
    data, reduce_scatter_dim: int, replica_groups, reduce_op=np.add, **kwargs
):
    """CPU implementation of reduce_scatter with duplicated data assumption.

    In CPU execution, we assume all ranks have identical data.
    reduce_scatter first reduces across ranks, then scatters the result.

    Shape: dim[reduce_scatter_dim] //= world_size
    Data: First reduce (as in all_reduce), then take 1/world_size slice (as rank 0)
    """
    world_size = len(replica_groups[0])

    # First apply reduction (as in all_reduce)
    if reduce_op == np.add:
        reduced = data * world_size
    elif reduce_op == np.maximum or reduce_op == np.minimum:
        reduced = data.copy()
    elif reduce_op == np.multiply:
        reduced = data**world_size
    else:
        reduced = data.copy()

    # Then scatter: take the first chunk (simulating rank 0)
    chunk_size = data.shape[reduce_scatter_dim] // world_size
    return np.take(reduced, range(chunk_size), axis=reduce_scatter_dim)


@reduce_scatter.impl("hlo")
def _reduce_scatter_hlo(
    data, reduce_scatter_dim: int, replica_groups, reduce_op=np.add, **kwargs
):
    ctx = HLOTraceContext._global_ctx
    assert ctx is not None, "HLO context is not initialized"
    rank = len(replica_groups[0])
    out_shape = list(data.shape)
    if out_shape:
        out_shape[reduce_scatter_dim] //= rank

    # Map reduce_op to string for HLO
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


# -----------------------------------------------------------------------------
# all_to_all
# -----------------------------------------------------------------------------
all_to_all = Op("all_to_all")


@all_to_all.impl("cpu")
def _all_to_all_cpu(
    data, split_dimension: int, concat_dimension: int, replica_groups, **kwargs
):
    """CPU implementation of all_to_all with duplicated data assumption.

    In CPU execution, we assume all ranks have identical data.
    all_to_all splits data along split_dimension and redistributes along
    concat_dimension.

    Shape: unchanged (splits and concats balance out)
    Data: With duplicated data, effectively rearranges chunks between dimensions.
          Since all ranks have the same data, the result is equivalent to:
          - Split into world_size chunks along split_dimension
          - Concatenate along concat_dimension
    """
    world_size = len(replica_groups[0])

    # Split along split_dimension
    chunks = np.split(data, world_size, axis=split_dimension)

    # Concatenate along concat_dimension
    # With duplicated data, each rank would send its chunk to all others
    # and receive chunks from all others. Since data is identical,
    # we just rearrange the chunks.
    result = np.concatenate(chunks, axis=concat_dimension)

    return result


@all_to_all.impl("hlo")
def _all_to_all_hlo(
    data, split_dimension: int, concat_dimension: int, replica_groups, **kwargs
):
    ctx = HLOTraceContext._global_ctx
    assert ctx is not None, "HLO context is not initialized"
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
