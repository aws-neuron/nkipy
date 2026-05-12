# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed collective operations: all_gather, all_reduce, reduce_scatter, all_to_all"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# all_gather
# -----------------------------------------------------------------------------
all_gather = Op("all_gather")


@all_gather.impl("cpu")
def _all_gather_cpu(data, all_gather_dim, replica_groups, **kwargs):
    """CPU implementation of all_gather with duplicated data assumption."""
    world_size = len(replica_groups[0])
    tile_reps = tuple(
        world_size if i == all_gather_dim else 1 for i in range(data.ndim)
    )
    return np.tile(data, tile_reps)


# -----------------------------------------------------------------------------
# all_reduce
# -----------------------------------------------------------------------------
all_reduce = Op("all_reduce")


@all_reduce.impl("cpu")
def _all_reduce_cpu(data, replica_groups, reduce_op=np.add, **kwargs):
    """CPU implementation of all_reduce with duplicated data assumption."""
    world_size = len(replica_groups[0])

    if reduce_op == np.add:
        return data * world_size
    elif reduce_op == np.maximum or reduce_op == np.minimum:
        return data.copy()
    elif reduce_op == np.multiply:
        return data**world_size
    else:
        return data.copy()


# -----------------------------------------------------------------------------
# reduce_scatter
# -----------------------------------------------------------------------------
reduce_scatter = Op("reduce_scatter")


@reduce_scatter.impl("cpu")
def _reduce_scatter_cpu(
    data, reduce_scatter_dim: int, replica_groups, reduce_op=np.add, **kwargs
):
    """CPU implementation of reduce_scatter with duplicated data assumption."""
    world_size = len(replica_groups[0])

    if reduce_op == np.add:
        reduced = data * world_size
    elif reduce_op == np.maximum or reduce_op == np.minimum:
        reduced = data.copy()
    elif reduce_op == np.multiply:
        reduced = data**world_size
    else:
        reduced = data.copy()

    chunk_size = data.shape[reduce_scatter_dim] // world_size
    return np.take(reduced, range(chunk_size), axis=reduce_scatter_dim)


# -----------------------------------------------------------------------------
# all_to_all
# -----------------------------------------------------------------------------
all_to_all = Op("all_to_all")


@all_to_all.impl("cpu")
def _all_to_all_cpu(
    data, split_dimension: int, concat_dimension: int, replica_groups, **kwargs
):
    """CPU implementation of all_to_all with duplicated data assumption."""
    world_size = len(replica_groups[0])

    chunks = np.split(data, world_size, axis=split_dimension)
    result = np.concatenate(chunks, axis=concat_dimension)

    return result
