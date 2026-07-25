"""Qwen3 MoE model: config, weights, loader, and forward pass.

Subpackage re-exporting the public API that registry.py and tests depend on.
"""

from nkipy_serving.models.qwen3_moe.config import (
    Qwen3MoeModelConfig,
    Qwen3MoeWeights,
)
from nkipy_serving.models.qwen3_moe.eager_executor import Qwen3MoeEagerExecutor
from nkipy_serving.models.qwen3_moe.executor import Qwen3MoeExecutor
from nkipy_serving.models.qwen3_moe.graph_fns import (
    decode_layer_nki_no_sp_fn,
    embedding_fn,
    post_attn_decode_no_sp_fn,
    post_attn_fn,
    pre_attn_decode_no_sp_fn,
    pre_attn_fn,
    router_fn,
    router_moe_decode_no_sp_fn,
)
from nkipy_serving.models.qwen3_moe.weights import (
    get_qwen3_moe_kv_metadata,
    init_qwen3_moe_weights,
)

__all__ = [
    "Qwen3MoeModelConfig",
    "Qwen3MoeWeights",
    "Qwen3MoeExecutor",
    "Qwen3MoeEagerExecutor",
    "get_qwen3_moe_kv_metadata",
    "init_qwen3_moe_weights",
    "embedding_fn",
    "pre_attn_fn",
    "post_attn_fn",
    "router_fn",
    "pre_attn_decode_no_sp_fn",
    "post_attn_decode_no_sp_fn",
    "router_moe_decode_no_sp_fn",
    "decode_layer_nki_no_sp_fn",
]
