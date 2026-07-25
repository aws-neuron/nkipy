from __future__ import annotations

import abc

import numpy as np

from nkipy_serving.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    def __init__(
        self,
        size: int,
        page_size: int,
        kvcache: KVCache,
    ):
        if size <= 0:
            raise RuntimeError(f"size must be > 0, got {size}")
        if page_size <= 0:
            raise RuntimeError(f"page_size must be > 0, got {page_size}")
        self.size = int(size)
        self.page_size = int(page_size)
        self._kvcache = kvcache
        self.is_not_in_free_group = True
        self.free_group: list[np.ndarray] = []

    @abc.abstractmethod
    def available_size(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int) -> np.ndarray | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: np.ndarray):
        raise NotImplementedError()

    def get_kvcache(self) -> KVCache:
        return self._kvcache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Token-level allocator (`page_size=1`) with reserved 0 slot."""

    def __init__(
        self,
        size: int,
        kvcache: KVCache,
    ):
        super().__init__(size=size, page_size=1, kvcache=kvcache)
        self.clear()

    def clear(self):
        self.free_slots = list(range(self.size))
        self.free_group = []
        self.is_not_in_free_group = True

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> np.ndarray | None:
        if need_size <= 0:
            raise RuntimeError(f"need_size must be > 0, got {need_size}")
        if need_size > self.available_size():
            return None
        selected = self.free_slots[-need_size:]
        del self.free_slots[-need_size:]
        return np.array(selected, dtype=np.int32)

    def free(self, free_index: np.ndarray):
        free_index = np.asarray(free_index, dtype=np.int32).reshape((-1,))
        if free_index.size == 0:
            return
        if self.is_not_in_free_group:
            self.free_slots.extend(free_index.tolist())
        else:
            self.free_group.append(free_index)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Page-level allocator that returns contiguous token indices per page."""

    def __init__(
        self,
        size: int,
        page_size: int,
        kvcache: KVCache,
    ):
        super().__init__(size=size, page_size=page_size, kvcache=kvcache)
        if size % page_size != 0:
            raise RuntimeError(
                f"size must be divisible by page_size. size={size}, page_size={page_size}"
            )
        self.num_pages = size // page_size
        self.clear()

    def clear(self):
        # Keep page 0 reserved so token index 0 remains reserved.
        self.free_pages = list(range(1, self.num_pages))
        self.free_group = []
        self.is_not_in_free_group = True

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

    def alloc(self, need_size: int) -> np.ndarray | None:
        if need_size <= 0:
            raise RuntimeError(f"need_size must be > 0, got {need_size}")
        if need_size % self.page_size != 0:
            raise RuntimeError(
                "need_size must be page-aligned. "
                f"need_size={need_size}, page_size={self.page_size}"
            )
        need_pages = need_size // self.page_size
        if need_pages > len(self.free_pages):
            return None
        selected_pages = self.free_pages[-need_pages:]
        del self.free_pages[-need_pages:]
        selected_pages = np.array(selected_pages, dtype=np.int32)
        page_indices = selected_pages[:, None] * self.page_size + np.arange(
            self.page_size, dtype=np.int32
        )
        return page_indices.reshape((-1,))

    def free(self, free_index: np.ndarray):
        free_index = np.asarray(free_index, dtype=np.int32).reshape((-1,))
        if free_index.size == 0:
            return
        if free_index.size % self.page_size != 0:
            raise RuntimeError(
                "free index size must be page-aligned. "
                f"size={free_index.size}, page_size={self.page_size}"
            )
        pages = np.unique(free_index // self.page_size)
        if self.is_not_in_free_group:
            self.free_pages.extend(pages.tolist())
        else:
            self.free_group.append(pages)
