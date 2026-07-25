from __future__ import annotations

import abc

import numpy as np


class ReqToTokenPool:
    """Request-index to token-index table used by scheduler/runtime."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        dtype: np.dtype = np.int32,
    ):
        if size <= 0:
            raise RuntimeError(f"size must be > 0, got {size}")
        if max_context_len <= 0:
            raise RuntimeError(f"max_context_len must be > 0, got {max_context_len}")
        self.size = int(size)
        self.max_context_len = int(max_context_len)
        self.dtype = dtype
        self.req_to_token = np.zeros((size, max_context_len), dtype=dtype)
        self.free_slots = list(range(size))

    def write(self, indices, values) -> None:
        self.req_to_token[indices] = values

    def read(self, req_idx: int, length: int) -> np.ndarray:
        if length < 0 or length > self.max_context_len:
            raise RuntimeError(
                f"length out of range: length={length}, max_context_len={self.max_context_len}"
            )
        return self.req_to_token[req_idx, :length].copy()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int = 1) -> list[int] | None:
        if need_size <= 0:
            raise RuntimeError(f"need_size must be > 0, got {need_size}")
        if need_size > len(self.free_slots):
            return None
        selected = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return selected

    def free(self, free_index: int | list[int]) -> None:
        if isinstance(free_index, int):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self) -> None:
        self.free_slots = list(range(self.size))
        self.req_to_token.fill(0)


class KVCache(abc.ABC):
    """Base KV-cache interface aligned with sglang-jax memory-pool layout."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: np.dtype,
        layer_num: int,
    ):
        if size <= 0:
            raise RuntimeError(f"size must be > 0, got {size}")
        if page_size <= 0:
            raise RuntimeError(f"page_size must be > 0, got {page_size}")
        if layer_num <= 0:
            raise RuntimeError(f"layer_num must be > 0, got {layer_num}")
        self.size = int(size)
        self.page_size = int(page_size)
        self.dtype = dtype
        self.layer_num = int(layer_num)

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError()


class MHATokenToKVPool(KVCache):
    """Block-based KV-cache pool for MHA/GQA.

    Storage layout per layer: [2, num_blocks, num_kv_heads, block_size, head_dim]
    where axis 0 is K(0)/V(1).

    The ``size`` parameter is the total number of token slots (num_blocks * block_size).
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: np.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        self.head_num = int(head_num)
        self.head_dim = int(head_dim)
        self._block_size = int(page_size)
        if size % self._block_size != 0:
            # Round up to nearest block boundary.
            size = (
                (size + self._block_size - 1) // self._block_size
            ) * self._block_size
        self._num_blocks = size // self._block_size
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
        )
        self.clear()

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def clear(self) -> None:
        shape = (2, self._num_blocks, self.head_num, self._block_size, self.head_dim)
        self._kv_cache = [
            np.zeros(shape, dtype=self.dtype) for _ in range(self.layer_num)
        ]

    def get_kv_cache(self, layer_id: int) -> np.ndarray:
        """Return the block-based KV cache tensor for a layer.

        Shape: [2, num_blocks, num_kv_heads, block_size, head_dim]
        Attention backends write into this directly.
        """
        if layer_id < 0 or layer_id >= self.layer_num:
            raise RuntimeError(f"layer_id out of range: {layer_id}/{self.layer_num}")
        return self._kv_cache[layer_id]


class SchedulerKVPoolStub(KVCache):
    """Lightweight KV pool stub for the scheduler process.

    Provides size/block_size/num_blocks metadata without allocating
    per-layer numpy arrays.  The scheduler only needs these properties
    for token allocation bookkeeping — it never reads or writes KV data.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: np.dtype,
        layer_num: int,
    ):
        self._block_size = int(page_size)
        if size % self._block_size != 0:
            size = (
                (size + self._block_size - 1) // self._block_size
            ) * self._block_size
        self._num_blocks = size // self._block_size
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
        )

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def clear(self) -> None:
        # Scheduler-only stub owns no KV arrays; slot state lives in the allocator.
        return None
