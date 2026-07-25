"""Attention backends for nkipy-serving."""

from nkipy_serving.attention.base import AttentionMetadata
from nkipy_serving.attention.vanilla import (
    vanilla_attention_core,
    vanilla_update_kv_cache,
)

__all__ = [
    "AttentionMetadata",
    "vanilla_attention_core",
    "vanilla_update_kv_cache",
]

# NKI backend is lazily imported to avoid requiring neuronxcc at import time.
# Import with:
#   from nkipy_serving.attention.nki_blocksparse_flash_attention import (
#       nki_update_kv_cache,
#       nki_blocksparse_attention,
#   )
