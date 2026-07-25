"""QKV graph variant setup and dispatch surface for DSV4 attention."""

from __future__ import annotations

from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.dispatcher import (
    _run_compressed_attention_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvResult,
    Dsv4CompressedAttentionQkvSetup,
    build_compressed_attention_qkv_setup,
)

__all__ = [
    "Dsv4CompressedAttentionQkvResult",
    "Dsv4CompressedAttentionQkvSetup",
    "_run_compressed_attention_qkv",
    "build_compressed_attention_qkv_setup",
]
