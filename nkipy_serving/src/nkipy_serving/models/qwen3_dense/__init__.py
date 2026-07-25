"""Qwen3-dense model: config, weights, loader, and forward pass.

Subpackage re-exporting the public API that registry.py and generated
kernel sources depend on.
"""

from nkipy_serving.models.qwen3_dense.config import Qwen3DenseModelConfig
from nkipy_serving.models.qwen3_dense.eager_executor import Qwen3DenseEagerExecutor
from nkipy_serving.models.qwen3_dense.executor import Qwen3DenseExecutor
from nkipy_serving.models.qwen3_dense.graph_fns import (
    cpu_attn_fn,
    embedding_fn,
    nki_attn_fn,
    post_attn_fn,
    pre_attn_fn,
    transformer_layer_nki_fn,
)
from nkipy_serving.models.qwen3_dense.weights import (
    Qwen3DenseLayerWeights,
    Qwen3DenseWeights,
    get_qwen3_dense_kv_metadata,
    init_qwen3_dense_weights,
)

__all__ = [
    "Qwen3DenseModelConfig",
    "Qwen3DenseLayerWeights",
    "Qwen3DenseWeights",
    "Qwen3DenseExecutor",
    "Qwen3DenseEagerExecutor",
    "get_qwen3_dense_kv_metadata",
    "init_qwen3_dense_weights",
    "embedding_fn",
    "pre_attn_fn",
    "post_attn_fn",
    "nki_attn_fn",
    "cpu_attn_fn",
    "transformer_layer_nki_fn",
]
