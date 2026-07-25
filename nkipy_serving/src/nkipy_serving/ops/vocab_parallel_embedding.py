from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def vocab_range_from_global_vocab_size(
    global_vocab_size: int,
    rank: int,
    world_size: int,
) -> Sequence[int]:
    if int(world_size) <= 0:
        raise RuntimeError(f"world_size must be > 0, got {world_size}")
    if int(global_vocab_size) % int(world_size) != 0:
        raise RuntimeError(
            "global_vocab_size must be divisible by world_size for NKIPy vocab-parallel embedding. "
            f"Got global_vocab_size={global_vocab_size}, world_size={world_size}."
        )
    per_partition_vocab_size = int(global_vocab_size) // int(world_size)
    start = int(rank) * per_partition_vocab_size
    return start, start + per_partition_vocab_size


@dataclass(frozen=True)
class VocabParallelShard:
    vocab_start_index: int
    vocab_end_index: int
    local_vocab_size: int


def get_vocab_parallel_shard(
    *,
    vocab_size: int,
    rank: int,
    world_size: int,
) -> VocabParallelShard:
    start, end = vocab_range_from_global_vocab_size(
        int(vocab_size),
        int(rank),
        int(world_size),
    )
    return VocabParallelShard(
        vocab_start_index=int(start),
        vocab_end_index=int(end),
        local_vocab_size=int(end - start),
    )


def vocab_parallel_embedding_local_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
) -> np.ndarray:
    start = int(vocab_start_index)
    end = int(vocab_end_index)
    hidden_dtype = local_embeddings.dtype
    input_ids = input_ids.astype(np.int32)
    local_mask = (input_ids >= start).astype(np.int32) * (input_ids < end).astype(
        np.int32
    )
    safe_local_ids = ((input_ids - start) * local_mask).astype(np.int32)
    local_hidden = local_embeddings[safe_local_ids]
    local_hidden = local_hidden * np.expand_dims(
        local_mask.astype(hidden_dtype), axis=-1
    )
    return local_hidden.astype(hidden_dtype)


def vocab_parallel_embedding_local_dynamic_range_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    vocab_range: np.ndarray,
) -> np.ndarray:
    range_v = vocab_range.astype(np.int32).reshape(-1)
    start = range_v[0]
    end = range_v[1]
    hidden_dtype = local_embeddings.dtype
    input_ids = input_ids.astype(np.int32)
    local_mask = (input_ids >= start).astype(np.int32) * (input_ids < end).astype(
        np.int32
    )
    safe_local_ids = ((input_ids - start) * local_mask).astype(np.int32)
    local_hidden = local_embeddings[safe_local_ids]
    local_hidden = local_hidden * np.expand_dims(
        local_mask.astype(hidden_dtype), axis=-1
    )
    return local_hidden.astype(hidden_dtype)


def vocab_parallel_embedding_no_sp_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    local_hidden = vocab_parallel_embedding_local_fn(
        input_ids,
        local_embeddings,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
    )
    if int(tp_degree) <= 1:
        return local_hidden
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    return cc.all_reduce(
        local_hidden,
        replica_groups=_tp_groups,
        reduce_op=np.add,
    )


def vocab_parallel_embedding_no_sp_dynamic_range_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    vocab_range: np.ndarray,
    *,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    local_hidden = vocab_parallel_embedding_local_dynamic_range_fn(
        input_ids,
        local_embeddings,
        vocab_range,
    )
    if int(tp_degree) <= 1:
        return local_hidden
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    return cc.all_reduce(
        local_hidden,
        replica_groups=_tp_groups,
        reduce_op=np.add,
    )


def vocab_parallel_embedding_sp_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    local_hidden = vocab_parallel_embedding_local_fn(
        input_ids,
        local_embeddings,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
    )
    if int(tp_degree) <= 1:
        return local_hidden
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    return cc.reduce_scatter(
        local_hidden,
        replica_groups=_tp_groups,
        reduce_scatter_dim=0,
        reduce_op=np.add,
    )


def vocab_parallel_embedding_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    return vocab_parallel_embedding_no_sp_fn(
        input_ids,
        local_embeddings,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
