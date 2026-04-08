# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy attention backend for vLLM."""

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


@register_backend(AttentionBackendEnum.CUSTOM)
class NKIPyAttentionBackend(AttentionBackend):
    """Stateless attention backend spec for NKIPy.

    The actual attention is handled inside NKIPy model code.
    This class only describes the KV cache layout.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return 2, num_blocks, num_kv_heads, block_size, head_size
