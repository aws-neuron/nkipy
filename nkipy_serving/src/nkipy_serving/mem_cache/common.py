"""Common helpers for prefix-cache memory allocation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from nkipy_serving.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


def req_to_tokens(req: Any) -> list[int]:
    """Extract token ids from a request object (shared by all cache impls).

    Fast-path: when ``req.token_ids`` is already ``list[int]`` (e.g.
    from :class:`PrefixCacheReq`), no copy is made.
    """
    if hasattr(req, "token_ids"):
        ids = req.token_ids
        return ids if isinstance(ids, list) else [int(x) for x in ids]
    tokens: list[int] = []
    if hasattr(req, "origin_input_ids"):
        tokens.extend(int(x) for x in req.origin_input_ids)
    if hasattr(req, "output_ids"):
        tokens.extend(int(x) for x in req.output_ids)
    if tokens:
        return tokens
    raise RuntimeError("Request object missing token_ids/origin_input_ids+output_ids")


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    if allocator is None:
        out = np.arange(1, num_tokens + 1, dtype=np.int32)
        return (out, None) if backup_state else out

    evict_from_tree_cache(tree_cache, num_tokens)
    state = allocator.backup_state() if backup_state else None
    out_cache_loc = allocator.alloc(num_tokens)
    if out_cache_loc is None:
        msg = (
            f"Out of memory allocating {num_tokens} tokens. "
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if backup_state:
        return out_cache_loc, state
    return out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int) -> None:
    if tree_cache is None:
        return
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return
    if allocator.available_size() < num_tokens:
        tree_cache.evict(num_tokens)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return "allocator unavailable"

    available_size = allocator.available_size()
    evictable_size = tree_cache.evictable_size()
    return (
        f"Available tokens: {available_size + evictable_size} "
        f"(available_size={available_size} + evictable_size={evictable_size})"
    )


class PrefixCacheReq:
    """Small request container for prefix-cache hooks."""

    def __init__(
        self,
        token_ids: list[int],
        kv_indices: np.ndarray | None = None,
        req_id: str | None = None,
        cache_payload: object | None = None,
    ):
        self.token_ids = token_ids
        self.kv_indices = kv_indices
        self.req_id = req_id
        self.cache_payload = cache_payload
