"""Contract tests for the tree-based RadixCache."""

from __future__ import annotations

import numpy as np
import pytest

from nkipy_serving.mem_cache.common import PrefixCacheReq
from nkipy_serving.mem_cache.radix_cache import RadixCache


def _ids(result) -> list[int]:
    return result.device_indices.tolist()


def test_radix_cache_insert_match_split_and_payload_contracts() -> None:
    cache = RadixCache()
    assert cache.match_prefix([1, 2, 3]).host_hit_length == 0

    assert cache.insert([1, 2, 3, 4, 5], value=[10, 20, 30, 40, 50]) == 0
    assert cache.insert([1, 2, 9, 10], value=[10, 20, 90, 100]) == 2
    assert cache.total_size() == 7

    shared = cache.match_prefix([1, 2])
    assert shared.host_hit_length == 2
    assert _ids(shared) == [10, 20]
    assert _ids(cache.match_prefix([1, 2, 3, 4, 5])) == [10, 20, 30, 40, 50]
    assert _ids(cache.match_prefix([1, 2, 9, 10])) == [10, 20, 90, 100]

    split = cache.match_prefix([1, 2, 3, 999])
    assert split.host_hit_length == 3
    assert _ids(split) == [10, 20, 30]
    assert cache.total_size() == 7
    assert _ids(cache.match_prefix([1, 2, 3, 4, 5])) == [10, 20, 30, 40, 50]

    cache = RadixCache()
    cache.insert([1, 2, 3, 4], value=[10, 20, 30, 40], payload="leaf")
    assert cache.match_prefix([1, 2, 3, 4]).payload == "leaf"
    assert cache.match_prefix([1, 2]).payload == "leaf"


def test_radix_cache_page_alignment_and_edge_cap_contracts() -> None:
    cache = RadixCache(page_size=2)
    cache.insert([1, 2, 3, 4, 5], value=[10, 20, 30, 40, 50])
    assert cache.total_size() == 4
    assert cache.match_prefix([1, 2, 3, 4, 5]).host_hit_length == 4
    assert _ids(cache.match_prefix([1, 2, 3, 9])) == [10, 20]

    cache = RadixCache(page_size=2)
    seq = list(range(64))
    cache.insert(seq, value=list(range(100, 164)))
    assert cache.total_size() == 64
    assert _ids(cache.match_prefix(seq)) == list(range(100, 164))

    partial = cache.match_prefix([0, 1])
    cache.inc_lock_ref(partial.last_device_node)
    assert cache.protected_size() <= cache.max_edge_tokens
    cache.dec_lock_ref(partial.last_device_node)


def test_radix_cache_lock_and_evict_contracts() -> None:
    cache = RadixCache()
    cache.insert([1, 2], value=[10, 20])
    cache.insert([3, 4], value=[30, 40])
    older = cache.match_prefix([1, 2]).last_device_node
    newer = cache.match_prefix([3, 4]).last_device_node
    older.last_access_time = 1.0
    newer.last_access_time = 2.0

    freed = cache.evict(2)
    assert np.concatenate(freed).size == 2
    assert cache.match_prefix([1, 2]).host_hit_length == 0
    assert cache.match_prefix([3, 4]).host_hit_length == 2

    cache = RadixCache()
    cache.insert([1, 2], value=[10, 20])
    locked = cache.match_prefix([1, 2]).last_device_node
    cache.inc_lock_ref(locked)
    cache.insert([3, 4], value=[30, 40])
    assert cache.protected_size() == 2
    assert cache.evictable_size() == 2

    cache.evict(2)
    assert cache.match_prefix([1, 2]).host_hit_length == 2
    assert cache.match_prefix([3, 4]).host_hit_length == 0
    cache.dec_lock_ref(locked)
    assert cache.protected_size() == 0


def test_radix_cache_request_entrypoint_contracts() -> None:
    cache = RadixCache()
    finished = PrefixCacheReq(
        token_ids=[1, 2, 3, 4],
        kv_indices=np.array([100, 101, 102, 103], dtype=np.int32),
    )
    assert cache.cache_finished_req(finished) == 0

    duplicate = PrefixCacheReq(
        token_ids=[1, 2, 3, 4],
        kv_indices=np.array([200, 201, 202, 203], dtype=np.int32),
    )
    assert cache.cache_finished_req(duplicate) == 4
    assert _ids(cache.match_prefix([1, 2, 3, 4])) == [100, 101, 102, 103]

    unfinished = PrefixCacheReq(
        token_ids=[1, 2, 3, 4, 5],
        kv_indices=np.array([100, 101, 102, 103, 500], dtype=np.int32),
    )
    assert cache.cache_unfinished_req(unfinished) == 4
    assert _ids(cache.match_prefix([1, 2, 3, 4, 5])) == [100, 101, 102, 103, 500]


def test_radix_cache_disabled_reset_and_validation_contracts() -> None:
    with pytest.raises(RuntimeError):
        RadixCache(page_size=0)

    cache = RadixCache()
    cache.insert([1, 2, 3], value=[10, 20, 30])
    assert cache.total_size() == 3
    cache.reset()
    assert cache.total_size() == 0
    assert cache.match_prefix([1, 2, 3]).host_hit_length == 0

    disabled = RadixCache(disable=True)
    assert disabled.insert([1, 2, 3], value=[10, 20, 30]) == 0
    assert disabled.total_size() == 0
    assert _ids(disabled.match_prefix([1, 2, 3])) == []
    assert disabled.evictable_size() == 0
    assert disabled.protected_size() == 0
    assert disabled.inc_lock_ref(None) == 0
    assert disabled.dec_lock_ref(None) == 0
