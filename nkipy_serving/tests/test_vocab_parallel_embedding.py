from __future__ import annotations

import sys
import types

import numpy as np

from nkipy_serving.ops.vocab_parallel_embedding import (
    get_vocab_parallel_shard,
    vocab_parallel_embedding_local_fn,
    vocab_parallel_embedding_no_sp_fn,
    vocab_parallel_embedding_sp_fn,
)


def _install_fake_collectives(
    monkeypatch, *, all_reduce=None, reduce_scatter=None
) -> None:
    cc_mod = types.ModuleType("nkipy.distributed.collectives")
    if all_reduce is not None:
        cc_mod.all_reduce = all_reduce
    if reduce_scatter is not None:
        cc_mod.reduce_scatter = reduce_scatter
    dist_mod = types.ModuleType("nkipy.distributed")
    dist_mod.collectives = cc_mod
    nkipy_mod = types.ModuleType("nkipy")
    nkipy_mod.distributed = dist_mod
    monkeypatch.setitem(sys.modules, "nkipy", nkipy_mod)
    monkeypatch.setitem(sys.modules, "nkipy.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "nkipy.distributed.collectives", cc_mod)


def _local_partial(
    input_ids: np.ndarray, local_embeddings: np.ndarray, *, start: int, end: int
) -> np.ndarray:
    mask = (input_ids >= int(start)).astype(np.int32) * (input_ids < int(end)).astype(
        np.int32
    )
    safe_ids = ((input_ids - int(start)) * mask).astype(np.int32)
    out = local_embeddings[safe_ids]
    return out * np.expand_dims(mask.astype(local_embeddings.dtype), axis=-1)


def test_vocab_parallel_embedding_no_sp_combines_local_vocab_shards(
    monkeypatch,
) -> None:
    input_ids = np.asarray([0, 3, 4, 7], dtype=np.int32)
    full_embeddings = np.arange(16, dtype=np.float32).reshape(8, 2)
    shard0 = get_vocab_parallel_shard(vocab_size=8, rank=0, world_size=2)
    shard1 = get_vocab_parallel_shard(vocab_size=8, rank=1, world_size=2)
    local0 = full_embeddings[shard0.vocab_start_index : shard0.vocab_end_index]
    local1 = full_embeddings[shard1.vocab_start_index : shard1.vocab_end_index]
    other_partial = _local_partial(
        input_ids,
        local1,
        start=shard1.vocab_start_index,
        end=shard1.vocab_end_index,
    )

    calls: list[str] = []

    def fake_all_reduce(x, replica_groups, reduce_op):
        calls.append("all_reduce")
        return x + other_partial

    _install_fake_collectives(monkeypatch, all_reduce=fake_all_reduce)

    out = vocab_parallel_embedding_no_sp_fn(
        input_ids,
        local0,
        vocab_start_index=shard0.vocab_start_index,
        vocab_end_index=shard0.vocab_end_index,
        tp_degree=2,
        tp_replica_groups=((0, 1),),
    )

    np.testing.assert_allclose(out, full_embeddings[input_ids])
    assert calls == ["all_reduce"]


def test_vocab_parallel_embedding_local_handles_batched_input_ids() -> None:
    input_ids = np.asarray([[4, 5], [6, 1]], dtype=np.int32)
    full_embeddings = np.arange(16, dtype=np.float32).reshape(8, 2)
    shard = get_vocab_parallel_shard(vocab_size=8, rank=1, world_size=2)
    local = full_embeddings[shard.vocab_start_index : shard.vocab_end_index]

    out = vocab_parallel_embedding_local_fn(
        input_ids,
        local,
        vocab_start_index=shard.vocab_start_index,
        vocab_end_index=shard.vocab_end_index,
    )

    expected = full_embeddings[input_ids]
    expected[1, 1] = 0.0
    np.testing.assert_allclose(out, expected)


def test_vocab_parallel_embedding_sp_reduce_scatters_summed_hidden(monkeypatch) -> None:
    input_ids = np.asarray([0, 3, 4, 7], dtype=np.int32)
    full_embeddings = np.arange(16, dtype=np.float32).reshape(8, 2)
    shard0 = get_vocab_parallel_shard(vocab_size=8, rank=0, world_size=2)
    shard1 = get_vocab_parallel_shard(vocab_size=8, rank=1, world_size=2)
    local0 = full_embeddings[shard0.vocab_start_index : shard0.vocab_end_index]
    local1 = full_embeddings[shard1.vocab_start_index : shard1.vocab_end_index]
    other_partial = _local_partial(
        input_ids,
        local1,
        start=shard1.vocab_start_index,
        end=shard1.vocab_end_index,
    )

    calls: list[str] = []

    def fake_reduce_scatter(x, replica_groups, reduce_scatter_dim, reduce_op):
        calls.append("reduce_scatter")
        assert int(reduce_scatter_dim) == 0
        combined = x + other_partial
        return combined[: combined.shape[0] // 2]

    _install_fake_collectives(monkeypatch, reduce_scatter=fake_reduce_scatter)

    out = vocab_parallel_embedding_sp_fn(
        input_ids,
        local0,
        vocab_start_index=shard0.vocab_start_index,
        vocab_end_index=shard0.vocab_end_index,
        tp_degree=2,
        tp_replica_groups=((0, 1),),
    )

    np.testing.assert_allclose(out, full_embeddings[input_ids][:2])
    assert calls == ["reduce_scatter"]


def test_vocab_parallel_embedding_shard_metadata_matches_even_split() -> None:
    shard = get_vocab_parallel_shard(vocab_size=32, rank=3, world_size=8)
    assert shard.vocab_start_index == 12
    assert shard.vocab_end_index == 16
    assert shard.local_vocab_size == 4
