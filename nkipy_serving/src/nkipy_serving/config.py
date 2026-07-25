import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

DEFAULT_ATTENTION_BACKEND = "NKIBlockSparseFlashAttention"
DEFAULT_PAGED_ATTN_IMPL = "nki_blocksparse_flash_attention"
DSV4_ATTENTION_BACKEND = "Dsv4SparseAttention"
DSV4_PAGED_ATTN_IMPL = "dsv4_sparse_attention"
ALLOWED_ATTENTION_BACKEND_IMPL = {
    "VanillaPagedAttention": "vanilla_paged_attention_kv_cache",
    "NKIBlockSparseFlashAttention": "nki_blocksparse_flash_attention",
    DSV4_ATTENTION_BACKEND: DSV4_PAGED_ATTN_IMPL,
}
ALLOWED_TOKENIZER_BACKENDS = ("hf",)
ALLOWED_EXECUTION_BACKENDS = ("numpy", "nkipy")
ALLOWED_DECODE_GRAPH_SCOPES = ("embed_layers",)
DEFAULT_REQUEST_BUCKETS = (1, 2, 4, 8, 16, 32)
DEFAULT_TOKEN_BUCKETS = (32, 128, 1024, 4096)
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_DTYPE = "bf16"
DEFAULT_TOKENIZER_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_TOKENIZER_LOCAL_FILES_ONLY = True
DEFAULT_ATTENTION_BACKEND_VERSION = "latest"
DEFAULT_MOE_KERNEL_VERSION = "v1"
DEFAULT_COMPILE_OPTIONS_HASH = "dev"
DEFAULT_PROTOTYPE_VOCAB_SIZE = 256
DEFAULT_PROTOTYPE_HIDDEN_SIZE = 64
DEFAULT_PROTOTYPE_SEED = 0
DEFAULT_HF_LOCAL_FILES_ONLY = True
DEFAULT_HF_NUM_HIDDEN_LAYERS: int | None = None
DEFAULT_NKIPY_COMPILER_ARGS = ""
DEFAULT_TP_DEGREE = 1
DEFAULT_EP_DEGREE = 1
DEFAULT_REPLICA_DEGREE = 1
DEFAULT_ATTENTION_DP_DEGREE = 1
DEFAULT_ATTENTION_TP_DEGREE = 1
DEFAULT_MOE_TP_DEGREE = 1
DEFAULT_DEVICE_OFFSET = 0
DEFAULT_KV_CACHE_BLOCK_SIZE = 32
DEFAULT_DECODE_GRAPH_SCOPE = "embed_layers"
DEFAULT_DENSE_LOCAL_TOPK = 1
DEFAULT_CHUNKED_PREFILL_SIZE = 4096
DEFAULT_ENABLE_MIXED_CHUNK = False
DEFAULT_PREFIX_CACHE_ENABLED = False
DEFAULT_PREFIX_CACHE_TYPE = "radix"
DEFAULT_PREFIX_CACHE_PAGE_SIZE = 32
DEFAULT_KV_POOL_SIZE = 16384
DEFAULT_MAX_CONTEXT_LEN = 4096
DEFAULT_REQUEST_TIMEOUT_S = 600  # 10 minutes; 0 = disabled


@dataclass(frozen=True)
class RuntimeConfig:
    attention_backend: str = DEFAULT_ATTENTION_BACKEND
    paged_attn_impl: str = DEFAULT_PAGED_ATTN_IMPL
    tokenizer_backend: str = "hf"
    tokenizer_model_id: str = DEFAULT_TOKENIZER_MODEL_ID
    tokenizer_revision: str | None = None
    tokenizer_local_files_only: bool = DEFAULT_TOKENIZER_LOCAL_FILES_ONLY
    execution_backend: str = "nkipy"
    nkipy_compiler_args: str = DEFAULT_NKIPY_COMPILER_ARGS
    nkipy_build_dir: str = "/tmp/build"
    model_id: str = DEFAULT_MODEL_ID
    model_dtype: str = DEFAULT_MODEL_DTYPE
    attention_backend_version: str = DEFAULT_ATTENTION_BACKEND_VERSION
    moe_kernel_version: str = DEFAULT_MOE_KERNEL_VERSION
    compile_options_hash: str = DEFAULT_COMPILE_OPTIONS_HASH
    prototype_vocab_size: int = DEFAULT_PROTOTYPE_VOCAB_SIZE
    prototype_hidden_size: int = DEFAULT_PROTOTYPE_HIDDEN_SIZE
    prototype_seed: int = DEFAULT_PROTOTYPE_SEED
    hf_model_id: str | None = None
    hf_revision: str | None = None
    hf_local_files_only: bool = DEFAULT_HF_LOCAL_FILES_ONLY
    hf_num_hidden_layers: int | None = DEFAULT_HF_NUM_HIDDEN_LAYERS
    precompile_catalog_file: str | None = None
    request_buckets: tuple[int, ...] = DEFAULT_REQUEST_BUCKETS
    token_buckets: tuple[int, ...] = DEFAULT_TOKEN_BUCKETS
    chunked_prefill_size: int = DEFAULT_CHUNKED_PREFILL_SIZE
    enable_mixed_chunk: bool = DEFAULT_ENABLE_MIXED_CHUNK
    # DSV4 currently serves the target model only. MTP remains disabled until
    # the device-resident draft loop and rollback/snapshot state are complete.
    dsv4_disable_mtp: bool = False
    # Prepared per-rank DSV4 weight cache. These paths flow through the
    # DeepSeek-V4 model metadata so rank-local loaders can avoid the slower
    # snapshot conversion path without depending on process-global env state.
    dsv4_prepared_weight_dir: str | None = None
    dsv4_prepared_weight_local_dir: str | None = None
    dsv4_prepared_weight_prestage: bool = False
    dsv4_prepared_weight_prestage_workers: int = 8
    # Explicit DSV4 mutable state sequence length in token slots. DSV4 serving
    # configs must set this to cover the served context/token bucket capacity.
    dsv4_state_size: int = 0
    # Optional resource guard for the heaviest product fusion:
    # DP-attention post/pre + router + blockwise prefill MoE in one NEFF.
    # 0 keeps the fusion unlimited; positive values fall back to the lighter
    # post/pre + router fusion when padded prefill rows exceed the cap.
    dsv4_product_prefill_moe_blockwise_fusion_max_rows: int = 0
    # Optional resource/validity guard for the lighter product fusion:
    # DP-attention post/pre + router dispatch in one NEFF. 0 keeps this
    # dispatch concat unlimited; positive values fall back to split MoE when
    # padded prefill rows exceed the cap.
    dsv4_product_prefill_moe_dispatch_fusion_max_rows: int = 0
    # Optional resource/validity guard for product DP-attention all-reduce
    # fused with mHC post/pre. 0 keeps this concat unlimited; positive values
    # split DP all-reduce from post/pre when padded prefill rows exceed the cap.
    dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows: int = 0
    # Reserved config knob for future layer-level NEFF fusion. Only "none" is
    # accepted until runtime support lands.
    dsv4_layer_fusion: str = "none"
    # Execute synthetic warmup forwards before marking workers ready. This must
    # stay enabled in serving configs so product kernels and first-touch runtime
    # allocations are completed before any request can reach the scheduler.
    dsv4_warmup_execute_forwards: bool = True

    tp_degree: int = DEFAULT_TP_DEGREE
    ep_degree: int = DEFAULT_EP_DEGREE
    # `replica_degree > 1` expands total_workers to tp*ep*replica for models
    # (e.g. DeepSeek-V4) that hold multiple full-model replicas. Keep at 1 for
    # Qwen3 / GPT-OSS so `total_workers = tp*ep` stays unchanged.
    replica_degree: int = DEFAULT_REPLICA_DEGREE
    # DP-attention lanes for V4-style hybrid DP. Defaults to 1; V4 sets to 16.
    attention_dp_degree: int = DEFAULT_ATTENTION_DP_DEGREE
    # Per-lane attention TP degree; currently always 1 on V4.
    attention_tp_degree: int = DEFAULT_ATTENTION_TP_DEGREE
    # TP-of-experts degree. 1 on V4 (experts are EP-only sharded).
    moe_tp_degree: int = DEFAULT_MOE_TP_DEGREE
    device_offset: int = DEFAULT_DEVICE_OFFSET
    kv_cache_block_size: int = DEFAULT_KV_CACHE_BLOCK_SIZE
    decode_graph_scope: str = DEFAULT_DECODE_GRAPH_SCOPE
    dense_local_topk: int = DEFAULT_DENSE_LOCAL_TOPK
    prefix_cache_enabled: bool = DEFAULT_PREFIX_CACHE_ENABLED
    prefix_cache_type: str = DEFAULT_PREFIX_CACHE_TYPE
    prefix_cache_page_size: int = DEFAULT_PREFIX_CACHE_PAGE_SIZE
    kv_pool_size: int = DEFAULT_KV_POOL_SIZE
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S
    overlap_schedule: bool = True

    @property
    def total_workers(self) -> int:
        return self.tp_degree * self.ep_degree * self.replica_degree

    @property
    def max_requests(self) -> int:
        return max(self.request_buckets)

    def _config_hash_fields(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_dtype": self.model_dtype,
            "attention_backend": self.attention_backend,
            "attention_backend_version": self.attention_backend_version,
            "paged_attn_impl": self.paged_attn_impl,
            "moe_kernel_version": self.moe_kernel_version,
            "nkipy_compiler_args": self.nkipy_compiler_args,
            "compile_options_hash": self.compile_options_hash,
            "tp_degree": self.tp_degree,
            "ep_degree": self.ep_degree,
            "replica_degree": self.replica_degree,
            "attention_dp_degree": self.attention_dp_degree,
            "attention_tp_degree": self.attention_tp_degree,
            "moe_tp_degree": self.moe_tp_degree,
            "dsv4_disable_mtp": self.dsv4_disable_mtp,
            "dsv4_state_size": self.dsv4_state_size,
            "dsv4_product_prefill_moe_blockwise_fusion_max_rows": (
                self.dsv4_product_prefill_moe_blockwise_fusion_max_rows
            ),
            "dsv4_product_prefill_moe_dispatch_fusion_max_rows": (
                self.dsv4_product_prefill_moe_dispatch_fusion_max_rows
            ),
            "dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows": (
                self.dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows
            ),
            "dsv4_warmup_execute_forwards": self.dsv4_warmup_execute_forwards,
            "kv_cache_block_size": self.kv_cache_block_size,
            "max_context_len": self.max_context_len,
            "dense_local_topk": self.dense_local_topk,
            "prototype_vocab_size": self.prototype_vocab_size,
            "prototype_hidden_size": self.prototype_hidden_size,
            "prototype_seed": self.prototype_seed,
            "hf_model_id": self.hf_model_id,
            "hf_num_hidden_layers": self.hf_num_hidden_layers,
        }

    def compute_config_hash(self) -> str:
        fields = self._config_hash_fields()
        raw = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def config_build_dir(self) -> str:
        build_dir = Path(self.nkipy_build_dir) / self.compute_config_hash()
        manifest_path = build_dir / "config.json"
        if not manifest_path.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
            self.write_config_manifest(manifest_path)
        return str(build_dir)

    def write_config_manifest(self, path: str | Path) -> None:
        fields = self._config_hash_fields()
        fields["config_hash"] = self.compute_config_hash()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid bool value: {value}")


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"Invalid int value: {value}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int value: {value}") from exc


def _parse_bucket_value(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        parsed = tuple(int(x) for x in value)
        return parsed or default
    if isinstance(value, str):
        parsed = tuple(int(x.strip()) for x in value.split(",") if x.strip())
        return parsed or default
    raise RuntimeError(f"Invalid bucket value type: {type(value)}")


def _load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise RuntimeError(f"Config file does not exist: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise RuntimeError("Only JSON config files are supported in bootstrap phase")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config JSON root must be an object: {config_path}")
    return data


def _derive_model_default_max_context_len(
    model_id: str, data: Mapping[str, Any]
) -> int:
    """Best-effort default max context length when the user does not specify one.

    Match upstream serving behavior by using the model config when we can resolve
    a real HF checkpoint. GPT-OSS stays pinned to 4096 by default to avoid
    unexpectedly compiling/loading 128K surfaces.
    """
    if model_id == "gpt-oss" or model_id.startswith("unsloth/gpt-oss-"):
        return DEFAULT_MAX_CONTEXT_LEN

    hf_model_id = _parse_optional_str(data.get("hf_model_id"))
    if hf_model_id is None:
        if model_id.startswith("Qwen/Qwen3-"):
            hf_model_id = model_id
        elif "/" in model_id:
            hf_model_id = model_id

    if hf_model_id is None:
        return DEFAULT_MAX_CONTEXT_LEN

    try:
        from nkipy_serving.configs.model_config import ModelConfig

        model_config = ModelConfig(
            model_path=hf_model_id,
            revision=_parse_optional_str(data.get("hf_revision")),
            context_length=None,
            model_override_args="{}",
            dtype=str(data.get("model_dtype", DEFAULT_MODEL_DTYPE)),
            local_files_only=_parse_bool(
                data.get("hf_local_files_only"),
                DEFAULT_HF_LOCAL_FILES_ONLY,
            ),
        )
        return int(model_config.context_len)
    except (ImportError, OSError, RuntimeError, ValueError):
        # Keep bootstrap resilient when the HF config is unavailable locally.
        return DEFAULT_MAX_CONTEXT_LEN


def _raise_if_unknown_config_fields(data: Mapping[str, Any]) -> None:
    allowed = {field.name for field in fields(RuntimeConfig)}
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise RuntimeError(f"Unknown runtime config field(s): {joined}")


def _runtime_config_from_mapping(data: Mapping[str, Any]) -> RuntimeConfig:
    _raise_if_unknown_config_fields(data)
    request_buckets = _parse_bucket_value(
        data.get("request_buckets"), DEFAULT_REQUEST_BUCKETS
    )
    token_buckets = _parse_bucket_value(
        data.get("token_buckets"), DEFAULT_TOKEN_BUCKETS
    )
    dsv4_disable_mtp = _parse_bool(data.get("dsv4_disable_mtp"), default=False)
    dsv4_prepared_weight_dir = _parse_optional_str(data.get("dsv4_prepared_weight_dir"))
    dsv4_prepared_weight_local_dir = _parse_optional_str(
        data.get("dsv4_prepared_weight_local_dir")
    )
    dsv4_prepared_weight_prestage = _parse_bool(
        data.get("dsv4_prepared_weight_prestage"),
        default=False,
    )
    dsv4_prepared_weight_prestage_workers = int(
        data.get("dsv4_prepared_weight_prestage_workers", 8)
    )
    dsv4_state_size = int(data.get("dsv4_state_size", 0))
    dsv4_product_prefill_moe_blockwise_fusion_max_rows = int(
        data.get("dsv4_product_prefill_moe_blockwise_fusion_max_rows", 0)
    )
    dsv4_product_prefill_moe_dispatch_fusion_max_rows = int(
        data.get("dsv4_product_prefill_moe_dispatch_fusion_max_rows", 0)
    )
    dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows = int(
        data.get(
            "dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows",
            0,
        )
    )
    dsv4_layer_fusion = str(data.get("dsv4_layer_fusion", "none")).strip().lower()
    dsv4_warmup_execute_forwards = _parse_bool(
        data.get("dsv4_warmup_execute_forwards"),
        default=True,
    )

    tp_degree = int(data.get("tp_degree", DEFAULT_TP_DEGREE))
    ep_degree = int(data.get("ep_degree", DEFAULT_EP_DEGREE))
    replica_degree = int(data.get("replica_degree", DEFAULT_REPLICA_DEGREE))
    attention_dp_degree = int(
        data.get("attention_dp_degree", DEFAULT_ATTENTION_DP_DEGREE)
    )
    attention_tp_degree = int(
        data.get("attention_tp_degree", DEFAULT_ATTENTION_TP_DEGREE)
    )
    moe_tp_degree = int(data.get("moe_tp_degree", DEFAULT_MOE_TP_DEGREE))
    device_offset = int(data.get("device_offset", DEFAULT_DEVICE_OFFSET))

    return RuntimeConfig(
        attention_backend=str(data.get("attention_backend", DEFAULT_ATTENTION_BACKEND)),
        paged_attn_impl=str(data.get("paged_attn_impl", DEFAULT_PAGED_ATTN_IMPL)),
        tokenizer_backend=str(data.get("tokenizer_backend", "hf")),
        tokenizer_model_id=str(
            data.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)
        ),
        tokenizer_revision=_parse_optional_str(data.get("tokenizer_revision")),
        tokenizer_local_files_only=_parse_bool(
            data.get("tokenizer_local_files_only"),
            DEFAULT_TOKENIZER_LOCAL_FILES_ONLY,
        ),
        execution_backend=str(data.get("execution_backend", "nkipy")),
        nkipy_compiler_args=str(
            data.get("nkipy_compiler_args", DEFAULT_NKIPY_COMPILER_ARGS)
        ),
        nkipy_build_dir=str(data.get("nkipy_build_dir", "/tmp/build")),
        model_id=str(data.get("model_id", DEFAULT_MODEL_ID)),
        model_dtype=str(data.get("model_dtype", DEFAULT_MODEL_DTYPE)),
        attention_backend_version=str(
            data.get("attention_backend_version", DEFAULT_ATTENTION_BACKEND_VERSION)
        ),
        moe_kernel_version=str(
            data.get("moe_kernel_version", DEFAULT_MOE_KERNEL_VERSION)
        ),
        compile_options_hash=str(
            data.get("compile_options_hash", DEFAULT_COMPILE_OPTIONS_HASH)
        ),
        prototype_vocab_size=int(
            data.get("prototype_vocab_size", DEFAULT_PROTOTYPE_VOCAB_SIZE)
        ),
        prototype_hidden_size=int(
            data.get("prototype_hidden_size", DEFAULT_PROTOTYPE_HIDDEN_SIZE)
        ),
        prototype_seed=int(data.get("prototype_seed", DEFAULT_PROTOTYPE_SEED)),
        hf_model_id=_parse_optional_str(data.get("hf_model_id")),
        hf_revision=_parse_optional_str(data.get("hf_revision")),
        hf_local_files_only=_parse_bool(
            data.get("hf_local_files_only"),
            DEFAULT_HF_LOCAL_FILES_ONLY,
        ),
        hf_num_hidden_layers=_parse_optional_int(data.get("hf_num_hidden_layers")),
        precompile_catalog_file=_parse_optional_str(
            data.get("precompile_catalog_file")
        ),
        request_buckets=request_buckets,
        token_buckets=token_buckets,
        chunked_prefill_size=int(
            data.get("chunked_prefill_size", DEFAULT_CHUNKED_PREFILL_SIZE)
        ),
        enable_mixed_chunk=_parse_bool(
            data.get("enable_mixed_chunk"), DEFAULT_ENABLE_MIXED_CHUNK
        ),
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        replica_degree=replica_degree,
        attention_dp_degree=attention_dp_degree,
        attention_tp_degree=attention_tp_degree,
        moe_tp_degree=moe_tp_degree,
        dsv4_disable_mtp=dsv4_disable_mtp,
        dsv4_prepared_weight_dir=dsv4_prepared_weight_dir,
        dsv4_prepared_weight_local_dir=dsv4_prepared_weight_local_dir,
        dsv4_prepared_weight_prestage=dsv4_prepared_weight_prestage,
        dsv4_prepared_weight_prestage_workers=dsv4_prepared_weight_prestage_workers,
        dsv4_state_size=dsv4_state_size,
        dsv4_product_prefill_moe_blockwise_fusion_max_rows=(
            dsv4_product_prefill_moe_blockwise_fusion_max_rows
        ),
        dsv4_product_prefill_moe_dispatch_fusion_max_rows=(
            dsv4_product_prefill_moe_dispatch_fusion_max_rows
        ),
        dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows=(
            dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows
        ),
        dsv4_layer_fusion=dsv4_layer_fusion,
        dsv4_warmup_execute_forwards=dsv4_warmup_execute_forwards,
        device_offset=device_offset,
        kv_cache_block_size=int(
            data.get("kv_cache_block_size", DEFAULT_KV_CACHE_BLOCK_SIZE)
        ),
        decode_graph_scope=str(
            data.get("decode_graph_scope", DEFAULT_DECODE_GRAPH_SCOPE)
        ),
        dense_local_topk=int(data.get("dense_local_topk", DEFAULT_DENSE_LOCAL_TOPK)),
        prefix_cache_enabled=_parse_bool(
            data.get("prefix_cache_enabled"),
            DEFAULT_PREFIX_CACHE_ENABLED,
        ),
        prefix_cache_type=str(data.get("prefix_cache_type", DEFAULT_PREFIX_CACHE_TYPE)),
        prefix_cache_page_size=int(
            data.get("prefix_cache_page_size", DEFAULT_PREFIX_CACHE_PAGE_SIZE)
        ),
        kv_pool_size=int(data.get("kv_pool_size", DEFAULT_KV_POOL_SIZE)),
        max_context_len=int(data.get("max_context_len", DEFAULT_MAX_CONTEXT_LEN)),
        request_timeout_s=int(data.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)),
        overlap_schedule=_parse_bool(
            data.get("overlap_schedule"),
            default=True,
        ),
    )


def load_runtime_config(
    config_path: str | None = None, overrides: Mapping[str, Any] | None = None
) -> RuntimeConfig:
    base = _load_config_file(config_path or os.getenv("NKIPY_SERVING_CONFIG_FILE"))
    # Config field → env var name. Convention: NKIPY_SERVING_ + UPPER(field).
    # Only fields that deviate from the convention are listed explicitly.
    _ENV_FIELD_NAMES = [
        "attention_backend",
        "paged_attn_impl",
        "tokenizer_backend",
        "tokenizer_model_id",
        "tokenizer_revision",
        "tokenizer_local_files_only",
        "execution_backend",
        "model_id",
        "model_dtype",
        "attention_backend_version",
        "moe_kernel_version",
        "compile_options_hash",
        "prototype_vocab_size",
        "prototype_hidden_size",
        "prototype_seed",
        "hf_model_id",
        "hf_revision",
        "hf_local_files_only",
        "hf_num_hidden_layers",
        "precompile_catalog_file",
        "request_buckets",
        "token_buckets",
        "chunked_prefill_size",
        "enable_mixed_chunk",
        "tp_degree",
        "ep_degree",
        "replica_degree",
        "attention_dp_degree",
        "attention_tp_degree",
        "moe_tp_degree",
        "dsv4_disable_mtp",
        "dsv4_prepared_weight_dir",
        "dsv4_prepared_weight_local_dir",
        "dsv4_prepared_weight_prestage",
        "dsv4_prepared_weight_prestage_workers",
        "dsv4_state_size",
        "dsv4_product_prefill_moe_blockwise_fusion_max_rows",
        "dsv4_product_prefill_moe_dispatch_fusion_max_rows",
        "dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows",
        "dsv4_layer_fusion",
        "dsv4_warmup_execute_forwards",
        "device_offset",
        "kv_cache_block_size",
        "decode_graph_scope",
        "dense_local_topk",
        "prefix_cache_enabled",
        "prefix_cache_type",
        "prefix_cache_page_size",
        "kv_pool_size",
        "max_context_len",
        "request_timeout_s",
        "overlap_schedule",
    ]
    _ENV_OVERRIDES_MAP = {
        "nkipy_compiler_args": "NKIPY_SERVING_COMPILER_ARGS",
        "nkipy_build_dir": "NKIPY_SERVING_BUILD_DIR",
    }
    env_overrides: dict[str, str | None] = {
        name: os.getenv(_ENV_OVERRIDES_MAP.get(name, f"NKIPY_SERVING_{name.upper()}"))
        for name in _ENV_FIELD_NAMES
    }
    env_overrides.update(
        {name: os.getenv(env_var) for name, env_var in _ENV_OVERRIDES_MAP.items()}
    )
    # Resolve model_id first (highest precedence wins) to look up model defaults
    model_id = str(
        (overrides or {}).get("model_id")
        or os.getenv("NKIPY_SERVING_MODEL_ID")
        or base.get("model_id")
        or DEFAULT_MODEL_ID
    )

    from nkipy_serving.models.registry import get_model_config_defaults

    model_defaults = get_model_config_defaults(model_id)

    # Precedence: model defaults < config file < env < explicit overrides
    merged = dict(model_defaults)
    merged.update(base)
    merged.update({k: v for k, v in env_overrides.items() if v is not None})
    if overrides:
        merged.update(dict(overrides))
    if (
        model_id.startswith("Qwen/Qwen3-")
        and "tokenizer_model_id" not in merged
        and "NKIPY_SERVING_TOKENIZER_MODEL_ID" not in os.environ
    ):
        merged["tokenizer_model_id"] = model_id

    if merged.get("max_context_len") is None:
        merged["max_context_len"] = _derive_model_default_max_context_len(
            model_id, merged
        )

    return _runtime_config_from_mapping(merged)


def _is_supported_hf_model_id(model_id: str, hf_model_id: str) -> bool:
    if hf_model_id.startswith("Qwen/Qwen3-"):
        return True
    if hf_model_id.startswith("unsloth/gpt-oss-"):
        return True
    if hf_model_id.startswith("deepseek-ai/DeepSeek-V4"):
        return True

    model_allows_local_snapshot = (
        model_id.startswith("Qwen/Qwen3-")
        or model_id == "qwen3-moe"
        or model_id == "gpt-oss"
        or model_id.startswith("unsloth/gpt-oss-")
        or model_id.startswith("deepseek-ai/DeepSeek-V4")
        or model_id == "deepseek-v4"
    )
    if not model_allows_local_snapshot:
        return False

    path_like = hf_model_id.startswith(("/", ".", "~"))
    if path_like:
        return True
    return Path(hf_model_id).expanduser().exists()


def _is_deepseek_v4_model_id(model_id: str) -> bool:
    return model_id == "deepseek-v4" or model_id.startswith("deepseek-ai/DeepSeek-V4")


def validate_runtime_config(config: RuntimeConfig) -> None:
    required_impl = ALLOWED_ATTENTION_BACKEND_IMPL.get(config.attention_backend)
    if required_impl is None:
        raise RuntimeError(
            "Invalid attention backend: "
            f"{config.attention_backend}. Allowed: {tuple(ALLOWED_ATTENTION_BACKEND_IMPL)}"
        )
    if config.paged_attn_impl != required_impl:
        raise RuntimeError(
            "Invalid paged attention impl for selected backend: "
            f"backend={config.attention_backend}, "
            f"paged_attn_impl={config.paged_attn_impl}, required={required_impl}"
        )
    is_dsv4_model = _is_deepseek_v4_model_id(config.model_id)
    if is_dsv4_model and config.attention_backend != DSV4_ATTENTION_BACKEND:
        raise RuntimeError(
            "DeepSeek-V4 requires attention_backend="
            f"'{DSV4_ATTENTION_BACKEND}', got {config.attention_backend!r}"
        )
    if is_dsv4_model and not config.dsv4_disable_mtp:
        raise RuntimeError(
            "DeepSeek-V4 serving requires dsv4_disable_mtp=true. "
            "Device-resident MTP is not implemented in the product runtime."
        )
    if config.attention_backend == DSV4_ATTENTION_BACKEND and not is_dsv4_model:
        raise RuntimeError(
            f"{DSV4_ATTENTION_BACKEND} is only supported for DeepSeek-V4 models. "
            f"Got model_id={config.model_id!r}"
        )
    if config.tokenizer_backend not in ALLOWED_TOKENIZER_BACKENDS:
        raise RuntimeError(
            "Invalid tokenizer backend: "
            f"{config.tokenizer_backend}. Allowed: {ALLOWED_TOKENIZER_BACKENDS}"
        )
    if config.execution_backend not in ALLOWED_EXECUTION_BACKENDS:
        raise RuntimeError(
            "Invalid execution backend: "
            f"{config.execution_backend}. Allowed: {ALLOWED_EXECUTION_BACKENDS}"
        )
    if config.decode_graph_scope not in ALLOWED_DECODE_GRAPH_SCOPES:
        raise RuntimeError(
            "Invalid decode_graph_scope: "
            f"{config.decode_graph_scope}. Allowed: {ALLOWED_DECODE_GRAPH_SCOPES}"
        )
    if config.dense_local_topk <= 0:
        raise RuntimeError(
            f"dense_local_topk must be > 0, got {config.dense_local_topk}"
        )
    if config.dsv4_state_size < 0:
        raise RuntimeError(
            f"dsv4_state_size must be >= 0, got {config.dsv4_state_size}"
        )
    if (
        config.dsv4_prepared_weight_local_dir is not None
        and config.dsv4_prepared_weight_dir is None
    ):
        raise RuntimeError(
            "dsv4_prepared_weight_local_dir requires dsv4_prepared_weight_dir"
        )
    if config.dsv4_prepared_weight_prestage and (
        config.dsv4_prepared_weight_dir is None
        or config.dsv4_prepared_weight_local_dir is None
    ):
        raise RuntimeError(
            "dsv4_prepared_weight_prestage requires both "
            "dsv4_prepared_weight_dir and dsv4_prepared_weight_local_dir"
        )
    if config.dsv4_prepared_weight_dir is not None:
        prepared_root = Path(config.dsv4_prepared_weight_dir).expanduser()
        if not prepared_root.exists():
            raise RuntimeError(
                "dsv4_prepared_weight_dir does not exist: "
                f"{config.dsv4_prepared_weight_dir}"
            )
    if config.dsv4_prepared_weight_prestage_workers <= 0:
        raise RuntimeError(
            "dsv4_prepared_weight_prestage_workers must be > 0, "
            f"got {config.dsv4_prepared_weight_prestage_workers}"
        )

    for field_name, value in (
        ("model_id", config.model_id),
        ("model_dtype", config.model_dtype),
        ("tokenizer_model_id", config.tokenizer_model_id),
        ("attention_backend_version", config.attention_backend_version),
        ("moe_kernel_version", config.moe_kernel_version),
        ("compile_options_hash", config.compile_options_hash),
    ):
        if not value.strip():
            raise RuntimeError(f"{field_name} must be non-empty")
    for field_name, value in (
        ("prototype_vocab_size", config.prototype_vocab_size),
        ("prototype_hidden_size", config.prototype_hidden_size),
    ):
        if value <= 0:
            raise RuntimeError(f"{field_name} must be > 0, got {value}")
    if config.prototype_seed < 0:
        raise RuntimeError(f"prototype_seed must be >= 0, got {config.prototype_seed}")
    if config.hf_model_id is not None and not _is_supported_hf_model_id(
        config.model_id,
        config.hf_model_id,
    ):
        raise RuntimeError(
            "hf_model_id is currently restricted to Qwen3, GPT-OSS, or "
            "DeepSeek-V4 checkpoints/local snapshots. "
            f"Got: {config.hf_model_id}"
        )
    if config.hf_num_hidden_layers is not None and config.hf_num_hidden_layers <= 0:
        raise RuntimeError(
            "hf_num_hidden_layers must be > 0 when set, "
            f"got {config.hf_num_hidden_layers}"
        )

    bucket_fields: list[tuple[str, tuple[int, ...] | None, bool]] = [
        ("request_buckets", config.request_buckets, False),
        ("token_buckets", config.token_buckets, False),
    ]
    for bucket_name, buckets, allow_empty in bucket_fields:
        if buckets is None:
            continue
        if not buckets and not allow_empty:
            raise RuntimeError(f"{bucket_name} must not be empty")
        if any(v <= 0 for v in buckets):
            raise RuntimeError(f"{bucket_name} must be positive: {buckets}")
        if tuple(sorted(buckets)) != tuple(buckets):
            raise RuntimeError(f"{bucket_name} must be sorted ascending: {buckets}")
    if config.chunked_prefill_size == 0:
        raise RuntimeError(
            "chunked_prefill_size cannot be 0. Use -1 to disable chunked prefill."
        )
    if config.enable_mixed_chunk and config.chunked_prefill_size <= 0:
        raise RuntimeError(
            "enable_mixed_chunk requires chunked_prefill_size > 0. "
            f"Got chunked_prefill_size={config.chunked_prefill_size}"
        )

    for degree_name, degree_val in [
        ("tp_degree", config.tp_degree),
        ("ep_degree", config.ep_degree),
        ("replica_degree", config.replica_degree),
        ("attention_dp_degree", config.attention_dp_degree),
        ("attention_tp_degree", config.attention_tp_degree),
        ("moe_tp_degree", config.moe_tp_degree),
    ]:
        if degree_val <= 0:
            raise RuntimeError(f"{degree_name} must be > 0, got {degree_val}")
        if degree_val > 1 and config.execution_backend != "nkipy":
            raise RuntimeError(
                f"{degree_name} > 1 is only supported on execution_backend=nkipy. "
                f"Got execution_backend={config.execution_backend}"
            )
    # V4-style lane config sanity: if attention_dp_degree > 1, it must equal
    # the number of TP rows (ep_degree * replica_degree).
    if config.attention_dp_degree > 1:
        expected_lanes = config.ep_degree * config.replica_degree
        if config.attention_dp_degree != expected_lanes:
            raise RuntimeError(
                "attention_dp_degree must equal ep_degree * replica_degree "
                f"(expected {expected_lanes}, got {config.attention_dp_degree})."
            )
    if config.device_offset < 0:
        raise RuntimeError(f"device_offset must be >= 0, got {config.device_offset}")
    if config.kv_cache_block_size <= 0:
        raise RuntimeError(
            f"kv_cache_block_size must be > 0, got {config.kv_cache_block_size}"
        )

    if config.prefix_cache_page_size <= 0:
        raise RuntimeError(
            f"prefix_cache_page_size must be > 0, got {config.prefix_cache_page_size}"
        )
    for field_name, value in (
        ("kv_pool_size", config.kv_pool_size),
        ("max_context_len", config.max_context_len),
    ):
        if value <= 0:
            raise RuntimeError(f"{field_name} must be > 0, got {value}")
    if is_dsv4_model:
        if config.dsv4_state_size <= 0:
            raise RuntimeError("DeepSeek-V4 serving requires dsv4_state_size > 0")
        max_token_bucket = max(config.token_buckets)
        if config.dsv4_state_size < config.max_context_len:
            raise RuntimeError(
                "dsv4_state_size must cover max_context_len: "
                f"dsv4_state_size={config.dsv4_state_size}, "
                f"max_context_len={config.max_context_len}"
            )
        if config.dsv4_state_size < max_token_bucket:
            raise RuntimeError(
                "dsv4_state_size must cover the largest token bucket: "
                f"dsv4_state_size={config.dsv4_state_size}, "
                f"max_token_bucket={max_token_bucket}"
            )
        if config.dsv4_state_size % 128 != 0:
            raise RuntimeError(
                "dsv4_state_size must be divisible by 128 for DeepSeek-V4 "
                f"compressed-state allocation, got {config.dsv4_state_size}"
            )
    if config.request_timeout_s < 0:
        raise RuntimeError(
            f"request_timeout_s must be >= 0, got {config.request_timeout_s}"
        )
    if config.dsv4_product_prefill_moe_blockwise_fusion_max_rows < 0:
        raise RuntimeError(
            "dsv4_product_prefill_moe_blockwise_fusion_max_rows must be >= 0, "
            f"got {config.dsv4_product_prefill_moe_blockwise_fusion_max_rows}"
        )
    if config.dsv4_product_prefill_moe_dispatch_fusion_max_rows < 0:
        raise RuntimeError(
            "dsv4_product_prefill_moe_dispatch_fusion_max_rows must be >= 0, "
            f"got {config.dsv4_product_prefill_moe_dispatch_fusion_max_rows}"
        )
    if config.dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows < 0:
        raise RuntimeError(
            "dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows must be "
            ">= 0, got "
            f"{config.dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows}"
        )
    if config.dsv4_layer_fusion != "none":
        raise RuntimeError(
            "dsv4_layer_fusion only supports 'none' until layer-level NEFF "
            f"fusion is implemented, got {config.dsv4_layer_fusion!r}"
        )
    if not config.dsv4_warmup_execute_forwards:
        raise RuntimeError(
            "dsv4_warmup_execute_forwards=false is no longer supported. "
            "Startup must execute synthetic first-touch forwards before readiness."
        )

    if (
        config.attention_backend
        in {"NKIBlockSparseFlashAttention", DSV4_ATTENTION_BACKEND}
        and config.execution_backend != "nkipy"
    ):
        raise RuntimeError(
            f"{config.attention_backend} requires execution_backend='nkipy'. "
            f"Got execution_backend={config.execution_backend}"
        )
    if config.prefix_cache_enabled and config.prefix_cache_type != "radix":
        raise RuntimeError(
            "prefix_cache_type must be 'radix' when enabled, "
            f"got {config.prefix_cache_type}"
        )


def configure_runtime_environment(runtime_config: RuntimeConfig) -> None:
    """Set environment variables required for multi-worker (TP > 1 or EP > 1)."""
    # Neuron's NKI tracer logs every kernel call by default, which can dominate
    # startup logs for product warmup. Keep user overrides possible.
    os.environ.setdefault("LOG_NKI_KERNEL_CALL", "0")
    if runtime_config.total_workers <= 1:
        return
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:62182")
