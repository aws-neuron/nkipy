import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_QWEN3_DENSE_SPEC_KEY = "qwen3_dense_family"


@dataclass(frozen=True)
class ModelSpec:
    """Model specification with config builder, weight initializer, and executor factory."""

    build_config: Callable[..., Any]
    init_weights: Callable[[Any], Any]
    create_executor: Callable  # (model_config, kv_pool, runtime_config) → executor
    build_kv_metadata: Callable[[Any], tuple[int, int, int, np.dtype]]
    config_defaults: dict[str, Any] = field(default_factory=dict)


def _build_qwen3_dense_config(runtime_config: Any, tp_rank: int = 0, ep_rank: int = 0):
    from nkipy_serving.models.qwen3_dense import Qwen3DenseModelConfig

    if runtime_config.ep_degree > 1:
        raise RuntimeError(
            "ep_degree > 1 is only supported for MoE models. "
            f"Got ep_degree={runtime_config.ep_degree} for dense model '{runtime_config.model_id}'."
        )
    hf_model_id = runtime_config.hf_model_id
    if hf_model_id is None and str(runtime_config.model_id).startswith("Qwen/Qwen3-"):
        hf_model_id = runtime_config.model_id
    return Qwen3DenseModelConfig(
        vocab_size=runtime_config.prototype_vocab_size,
        hidden_size=runtime_config.prototype_hidden_size,
        seed=runtime_config.prototype_seed,
        hf_model_id=hf_model_id,
        hf_revision=runtime_config.hf_revision,
        hf_local_files_only=runtime_config.hf_local_files_only,
        hf_num_hidden_layers=runtime_config.hf_num_hidden_layers,
        nkipy_compiler_args=runtime_config.nkipy_compiler_args,
        kv_cache_block_size=runtime_config.kv_cache_block_size,
        attention_backend=runtime_config.attention_backend,
        tp_degree=runtime_config.tp_degree,
        tp_rank=tp_rank,
        tp_world_size=runtime_config.tp_degree,
    )


def _create_qwen3_dense_executor(model_config, kv_pool, runtime_config):
    from nkipy_serving.models.qwen3_dense import Qwen3DenseExecutor

    return Qwen3DenseExecutor(model_config, kv_pool, runtime_config)


def _init_qwen3_dense_weights(config):
    from nkipy_serving.models.qwen3_dense import init_qwen3_dense_weights

    return init_qwen3_dense_weights(config)


def _get_qwen3_dense_kv_metadata(config):
    from nkipy_serving.models.qwen3_dense import get_qwen3_dense_kv_metadata

    return get_qwen3_dense_kv_metadata(config)


def _build_gpt_oss_config(runtime_config: Any, tp_rank: int = 0, ep_rank: int = 0):
    from nkipy_serving.models.gpt_oss import GptOssModelConfig

    hf_model_id = runtime_config.hf_model_id
    model_id = str(runtime_config.model_id)
    if hf_model_id is None:
        if model_id == "gpt-oss":
            hf_model_id = "unsloth/gpt-oss-120b-BF16"
        elif model_id.startswith("unsloth/gpt-oss-"):
            hf_model_id = model_id
        elif model_id.startswith("openai/gpt-oss-"):
            raise RuntimeError(
                "model_id 'openai/gpt-oss-*' is not supported in nkipy-serving. "
                "Use the exact HF model id 'unsloth/gpt-oss-120b-BF16'."
            )
    if hf_model_id is None:
        raise RuntimeError(
            "gpt-oss requires a HuggingFace model id. "
            "Set hf_model_id or use model_id='unsloth/gpt-oss-*'."
        )
    return GptOssModelConfig(
        hf_model_id=hf_model_id,
        hf_revision=runtime_config.hf_revision,
        hf_local_files_only=runtime_config.hf_local_files_only,
        hf_num_hidden_layers=runtime_config.hf_num_hidden_layers,
        nkipy_compiler_args=runtime_config.nkipy_compiler_args,
        kv_cache_block_size=runtime_config.kv_cache_block_size,
        attention_backend=runtime_config.attention_backend,
        tp_degree=runtime_config.tp_degree,
        tp_rank=tp_rank,
        tp_world_size=runtime_config.tp_degree,
        ep_degree=runtime_config.ep_degree,
        ep_rank=ep_rank,
    )


def _create_gpt_oss_executor(model_config, kv_pool, runtime_config):
    from nkipy_serving.models.gpt_oss import GptOssExecutor

    return GptOssExecutor(model_config, kv_pool, runtime_config)


def _init_gpt_oss_weights(config):
    from nkipy_serving.models.gpt_oss import init_gpt_oss_weights

    return init_gpt_oss_weights(config)


def _get_gpt_oss_kv_metadata(config):
    from nkipy_serving.models.gpt_oss import get_gpt_oss_kv_metadata

    return get_gpt_oss_kv_metadata(config)


def _build_qwen3_moe_config(runtime_config: Any, tp_rank: int = 0, ep_rank: int = 0):
    from nkipy_serving.models.qwen3_moe import Qwen3MoeModelConfig

    hf_model_id = runtime_config.hf_model_id
    model_id = str(runtime_config.model_id)
    if hf_model_id is None and model_id.startswith("Qwen/Qwen3-"):
        hf_model_id = model_id
    if hf_model_id is None:
        raise RuntimeError(
            "qwen3-moe requires a HuggingFace model id. "
            "Set hf_model_id or use model_id='Qwen/Qwen3-*-A*-*'."
        )
    return Qwen3MoeModelConfig(
        hf_model_id=hf_model_id,
        hf_revision=runtime_config.hf_revision,
        hf_local_files_only=runtime_config.hf_local_files_only,
        hf_num_hidden_layers=runtime_config.hf_num_hidden_layers,
        nkipy_compiler_args=runtime_config.nkipy_compiler_args,
        kv_cache_block_size=runtime_config.kv_cache_block_size,
        attention_backend=runtime_config.attention_backend,
        tp_degree=runtime_config.tp_degree,
        tp_rank=tp_rank,
        tp_world_size=runtime_config.tp_degree,
        ep_degree=runtime_config.ep_degree,
        ep_rank=ep_rank,
    )


def _create_qwen3_moe_executor(model_config, kv_pool, runtime_config):
    from nkipy_serving.models.qwen3_moe import Qwen3MoeExecutor

    return Qwen3MoeExecutor(model_config, kv_pool, runtime_config)


def _init_qwen3_moe_weights(config):
    from nkipy_serving.models.qwen3_moe import init_qwen3_moe_weights

    return init_qwen3_moe_weights(config)


def _get_qwen3_moe_kv_metadata(config):
    from nkipy_serving.models.qwen3_moe import get_qwen3_moe_kv_metadata

    return get_qwen3_moe_kv_metadata(config)


def _is_qwen3_moe_model(model_id: str) -> bool:
    """Check if a Qwen3 model ID indicates MoE (has -A<size>B pattern for active params)."""
    return bool(re.search(r"-A\d+B", model_id))


def _build_deepseek_v4_config(runtime_config: Any, tp_rank: int = 0, ep_rank: int = 0):
    from nkipy_serving.config import DSV4_ATTENTION_BACKEND
    from nkipy_serving.models.deepseek_v4 import DeepseekV4ModelConfig

    if runtime_config.attention_backend != DSV4_ATTENTION_BACKEND:
        raise RuntimeError(
            "deepseek-v4 requires attention_backend="
            f"'{DSV4_ATTENTION_BACKEND}', got {runtime_config.attention_backend!r}"
        )

    hf_model_id = runtime_config.hf_model_id
    model_id = str(runtime_config.model_id)
    if hf_model_id is None and model_id.startswith("deepseek-ai/DeepSeek-V4"):
        hf_model_id = model_id
    if hf_model_id is None:
        raise RuntimeError(
            "deepseek-v4 requires a HuggingFace model id. "
            "Set hf_model_id or use model_id='deepseek-ai/DeepSeek-V4-*'."
        )
    # For V4 the caller-passed `ep_rank` is the TP-row index (`rank //
    # tp_degree`). V4 splits that into (replica, row_in_replica); the row also
    # identifies the attention-DP lane. Map:
    #     row              = ep_rank (as passed in)
    #     replica          = row // ep_degree
    #     row_in_replica   = row %  ep_degree     ← this is what V4 calls ep_rank
    #     attention lane   = row
    #     request_lane_rank = row                 ← needed by weights.py rank reconstruct
    row = ep_rank
    replica_degree = runtime_config.replica_degree
    ep_degree = runtime_config.ep_degree
    row_in_replica = row % ep_degree if ep_degree else row
    attention_dp_degree = runtime_config.attention_dp_degree
    # Caller is expected to pass 0..(tp*ep*replica-1) / tp_degree as ep_rank,
    # i.e. 0..(replica*ep)-1. Bounds-check at V4 sharding.
    rows_total = replica_degree * ep_degree
    if row >= rows_total:
        raise RuntimeError(
            f"V4 row {row} out of range (replica_degree={replica_degree}, "
            f"ep_degree={ep_degree}, expected row < {rows_total})"
        )
    return DeepseekV4ModelConfig(
        hf_model_id=hf_model_id,
        hf_revision=runtime_config.hf_revision,
        hf_local_files_only=runtime_config.hf_local_files_only,
        hf_num_hidden_layers=runtime_config.hf_num_hidden_layers,
        nkipy_compiler_args=runtime_config.nkipy_compiler_args,
        kv_cache_block_size=runtime_config.kv_cache_block_size,
        attention_backend=runtime_config.attention_backend,
        dsv4_prepared_weight_dir=runtime_config.dsv4_prepared_weight_dir,
        dsv4_prepared_weight_local_dir=runtime_config.dsv4_prepared_weight_local_dir,
        tp_degree=runtime_config.tp_degree,
        tp_rank=tp_rank,
        tp_world_size=runtime_config.tp_degree,
        ep_degree=ep_degree,
        ep_rank=row_in_replica,
        replica_degree=replica_degree,
        attention_dp_degree=attention_dp_degree,
        attention_tp_degree=runtime_config.attention_tp_degree,
        moe_tp_degree=runtime_config.moe_tp_degree,
        # Identifies the attention-DP lane this rank owns.
        request_lane_rank=row,
        request_lane_world_size=rows_total,
        dsv4_disable_mtp=runtime_config.dsv4_disable_mtp,
    )


def _create_deepseek_v4_executor(model_config, kv_pool, runtime_config):
    from nkipy_serving.models.deepseek_v4 import DeepseekV4Executor

    return DeepseekV4Executor(model_config, kv_pool, runtime_config)


def _init_deepseek_v4_weights(config):
    from nkipy_serving.models.deepseek_v4 import init_deepseek_v4_weights

    return init_deepseek_v4_weights(config)


def _get_deepseek_v4_kv_metadata(config):
    from nkipy_serving.models.deepseek_v4 import get_deepseek_v4_kv_metadata

    return get_deepseek_v4_kv_metadata(config)


_MODEL_SPECS: dict[str, ModelSpec] = {
    _QWEN3_DENSE_SPEC_KEY: ModelSpec(
        build_config=_build_qwen3_dense_config,
        init_weights=_init_qwen3_dense_weights,
        create_executor=_create_qwen3_dense_executor,
        build_kv_metadata=_get_qwen3_dense_kv_metadata,
        config_defaults={
            "execution_backend": "nkipy",
            "attention_backend": "NKIBlockSparseFlashAttention",
            "paged_attn_impl": "nki_blocksparse_flash_attention",
        },
    ),
    "gpt-oss": ModelSpec(
        build_config=_build_gpt_oss_config,
        init_weights=_init_gpt_oss_weights,
        create_executor=_create_gpt_oss_executor,
        build_kv_metadata=_get_gpt_oss_kv_metadata,
        config_defaults={
            "execution_backend": "nkipy",
            "attention_backend": "NKIBlockSparseFlashAttention",
            "paged_attn_impl": "nki_blocksparse_flash_attention",
        },
    ),
    "qwen3-moe": ModelSpec(
        build_config=_build_qwen3_moe_config,
        init_weights=_init_qwen3_moe_weights,
        create_executor=_create_qwen3_moe_executor,
        build_kv_metadata=_get_qwen3_moe_kv_metadata,
        config_defaults={
            "execution_backend": "nkipy",
            "attention_backend": "NKIBlockSparseFlashAttention",
            "paged_attn_impl": "nki_blocksparse_flash_attention",
        },
    ),
    "deepseek-v4": ModelSpec(
        build_config=_build_deepseek_v4_config,
        init_weights=_init_deepseek_v4_weights,
        create_executor=_create_deepseek_v4_executor,
        build_kv_metadata=_get_deepseek_v4_kv_metadata,
        config_defaults={
            "execution_backend": "nkipy",
            "attention_backend": "Dsv4SparseAttention",
            "paged_attn_impl": "dsv4_sparse_attention",
            # Primary layout: 128 workers as 16 rows by 8 TP columns across
            # 2 replicas. Each TP row is one attention-DP lane and one MoE EP
            # row. Override at launch for dev-size configs such as R1.
            "tp_degree": 8,
            "ep_degree": 8,
            "replica_degree": 2,
            "attention_dp_degree": 16,
            "attention_tp_degree": 1,
            "moe_tp_degree": 1,
            "kv_cache_block_size": 32,
            "dsv4_disable_mtp": True,
        },
    ),
}


def get_model_config_defaults(model_id: str) -> dict[str, Any]:
    """Best-effort lookup of model config defaults. Returns {} for unknown models."""
    spec = _MODEL_SPECS.get(model_id)
    if spec is None:
        if model_id.startswith("Qwen/Qwen3-"):
            if _is_qwen3_moe_model(model_id):
                spec = _MODEL_SPECS.get("qwen3-moe")
            else:
                spec = _MODEL_SPECS.get(_QWEN3_DENSE_SPEC_KEY)
        elif model_id.startswith("unsloth/gpt-oss-"):
            spec = _MODEL_SPECS.get("gpt-oss")
        elif model_id.startswith("deepseek-ai/DeepSeek-V4"):
            spec = _MODEL_SPECS.get("deepseek-v4")
    if spec is None:
        return {}
    return dict(spec.config_defaults)


def resolve_model_spec(model_id: str) -> ModelSpec:
    spec = _MODEL_SPECS.get(model_id)
    if spec is None:
        if model_id.startswith("Qwen/Qwen3-"):
            if _is_qwen3_moe_model(model_id):
                spec = _MODEL_SPECS["qwen3-moe"]
            else:
                spec = _MODEL_SPECS[_QWEN3_DENSE_SPEC_KEY]
        elif model_id.startswith("openai/gpt-oss-"):
            raise RuntimeError(
                "model_id 'openai/gpt-oss-*' is not supported in nkipy-serving. "
                "Use the exact HF model id 'unsloth/gpt-oss-120b-BF16'."
            )
        elif model_id.startswith("unsloth/gpt-oss-"):
            spec = _MODEL_SPECS["gpt-oss"]
        elif model_id.startswith("deepseek-ai/DeepSeek-V4"):
            spec = _MODEL_SPECS["deepseek-v4"]
        else:
            raise RuntimeError(
                f"Unsupported model_id: {model_id}. Supported models: "
                "HuggingFace Qwen3 dense checkpoints (Qwen/Qwen3-* without -A...), "
                "the direct key 'qwen3-moe', HuggingFace Qwen3 MoE checkpoints "
                "(Qwen/Qwen3-*-A*-*), the direct key 'gpt-oss', GPT-OSS "
                "checkpoints (unsloth/gpt-oss-*), the direct key 'deepseek-v4', "
                "and DeepSeek-V4 checkpoints (deepseek-ai/DeepSeek-V4-*)."
            )
    return spec


def build_model_config(
    model_id: str, runtime_config: Any, tp_rank: int = 0, ep_rank: int = 0
) -> Any:
    return resolve_model_spec(model_id).build_config(
        runtime_config, tp_rank=tp_rank, ep_rank=ep_rank
    )
