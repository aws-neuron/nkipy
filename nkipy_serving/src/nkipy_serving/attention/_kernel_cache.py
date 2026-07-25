"""Attention kernel cache for compiled KV-update and attention kernels.

Used by NKI BlockSparse FlashAttention when running per-layer execution
(e.g. Qwen3 MoE). When attention is compiled inside a full-model graph
(e.g. Qwen3 Dense, GPT-OSS decode), this cache is not needed.
"""

from __future__ import annotations


class AttentionKernelCache:
    """Caches compiled NKI attention and KV-update kernels by shape key."""

    def __init__(self) -> None:
        self._kv_update_kernels: dict[tuple, object] = {}
        self._attention_kernels: dict[tuple, object] = {}

    def get_kv_update_kernel(self, key: tuple) -> object | None:
        return self._kv_update_kernels.get(key)

    def set_kv_update_kernel(self, key: tuple, kernel: object) -> None:
        self._kv_update_kernels[key] = kernel

    def get_attention_kernel(self, key: tuple) -> object | None:
        return self._attention_kernels.get(key)

    def set_attention_kernel(self, key: tuple, kernel: object) -> None:
        self._attention_kernels[key] = kernel
