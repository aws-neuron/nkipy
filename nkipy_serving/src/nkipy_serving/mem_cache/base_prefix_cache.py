"""Base interfaces for prefix-cache implementations."""

from __future__ import annotations

import abc
from typing import Any, NamedTuple

import numpy as np


class MatchResult(NamedTuple):
    device_indices: np.ndarray
    last_device_node: Any | None
    last_host_node: Any | None
    host_hit_length: int = 0
    payload: Any | None = None


class BasePrefixCache(abc.ABC):
    """Cache indexed by prompt-token prefixes."""

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def match_prefix(self, key: list[int], **kwargs) -> MatchResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def cache_finished_req(self, req: Any, **kwargs) -> int:
        """Cache a completed request.

        Returns the prefix length that was already in the cache (i.e. the
        number of leading tokens whose KV indices were NOT stored because
        the cache already holds values for them).  The caller must free
        the donated KV indices in ``[prefix_hit_length : returned_value]``
        to avoid leaking allocator capacity.
        """

    @abc.abstractmethod
    def cache_unfinished_req(self, req: Any, **kwargs) -> int:
        """Cache an in-progress (chunked) request.  Same return semantics
        as :meth:`cache_finished_req`."""

    @abc.abstractmethod
    def evict(self, num_tokens: int) -> list:
        """Evict at least *num_tokens* of unlocked cache entries.

        Returns a list of KV-index arrays removed from the cache.  The
        caller must free these back to the allocator.
        """

    @abc.abstractmethod
    def inc_lock_ref(self, node: Any) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: str | None = None) -> int:
        raise NotImplementedError()

    def evictable_size(self) -> int:
        return 0

    def protected_size(self) -> int:
        return 0

    def total_size(self) -> int:
        raise NotImplementedError()

    def pretty_print(self) -> str:
        raise NotImplementedError()

    def take_events(self) -> list[Any]:
        return []
