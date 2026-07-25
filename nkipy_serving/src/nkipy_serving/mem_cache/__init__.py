"""Prefix-cache package for runtime hooks."""

from __future__ import annotations

from nkipy_serving.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from nkipy_serving.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from nkipy_serving.mem_cache.common import PrefixCacheReq
from nkipy_serving.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from nkipy_serving.mem_cache.radix_cache import RadixCache


def create_prefix_cache(
    cache_type: str,
    page_size: int = 1,
    disable: bool = False,
) -> BasePrefixCache | None:
    normalized = cache_type.strip().lower()
    if normalized in {"", "none", "disabled"}:
        return None
    if normalized == "radix":
        return RadixCache(page_size=page_size, disable=disable)
    raise RuntimeError(
        f"Unsupported prefix cache type: {cache_type}. Expected one of: none, radix"
    )


__all__ = [
    "BaseTokenToKVPoolAllocator",
    "BasePrefixCache",
    "KVCache",
    "MHATokenToKVPool",
    "MatchResult",
    "PagedTokenToKVPoolAllocator",
    "PrefixCacheReq",
    "RadixCache",
    "ReqToTokenPool",
    "TokenToKVPoolAllocator",
    "create_prefix_cache",
]
