"""DeepSeek-V4 attention backend and metadata contracts."""

from nkipy_serving.attention.deepseek_v4.backend import (
    Dsv4SparseAttentionBackend,
)
from nkipy_serving.attention.deepseek_v4.kv_metadata import (
    HeterogeneousKVMetadata,
    KVCacheKind,
    LayerKVSpec,
    build_kv_metadata_from_ratios,
)
from nkipy_serving.attention.deepseek_v4.metadata import (
    SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
    SparseAttentionMetadata,
)
from nkipy_serving.attention.deepseek_v4.state import (
    Dsv4CompressorStateSpec,
    Dsv4DeviceCompressorState,
    Dsv4DeviceLayerState,
    Dsv4DeviceState,
    Dsv4KVFormat,
    allocate_dsv4_device_state,
    reset_dsv4_device_state,
)
from nkipy_serving.attention.deepseek_v4.types import (
    Dsv4AttentionMetadata,
    Dsv4DeviceAttentionInputs,
    Dsv4DpAttentionSuperstepMetadata,
    allocate_dsv4_device_attention_inputs,
    build_positions_per_token,
    build_req_id_per_token,
    dsv4_device_sparse_attention_kernel_inputs,
    run_dsv4_device_sparse_attention,
    run_dsv4_swa_global_slots,
    tensor_to_step_field_name,
)
from nkipy_serving.attention.deepseek_v4.vanilla import (
    dsv4_vanilla_attn_fn,
    dsv4_vanilla_sparse_attention_core,
    dsv4_vanilla_update_kv_cache,
)

__all__ = [
    "Dsv4AttentionMetadata",
    "Dsv4CompressorStateSpec",
    "Dsv4DpAttentionSuperstepMetadata",
    "Dsv4DeviceAttentionInputs",
    "Dsv4DeviceCompressorState",
    "Dsv4DeviceLayerState",
    "Dsv4DeviceState",
    "Dsv4KVFormat",
    "Dsv4SparseAttentionBackend",
    "HeterogeneousKVMetadata",
    "KVCacheKind",
    "LayerKVSpec",
    "SPARSE_INDEX_SPACE_GLOBAL_SLOTS",
    "SparseAttentionMetadata",
    "allocate_dsv4_device_attention_inputs",
    "allocate_dsv4_device_state",
    "reset_dsv4_device_state",
    "build_positions_per_token",
    "build_req_id_per_token",
    "build_kv_metadata_from_ratios",
    "dsv4_device_sparse_attention_kernel_inputs",
    "dsv4_vanilla_attn_fn",
    "dsv4_vanilla_sparse_attention_core",
    "dsv4_vanilla_update_kv_cache",
    "run_dsv4_device_sparse_attention",
    "run_dsv4_swa_global_slots",
    "tensor_to_step_field_name",
]
