import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkipy_serving.batching.contracts import ForwardMode
from nkipy_serving.config import RuntimeConfig
from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings


@dataclass(frozen=True)
class VariantKey:
    model_id: str
    dtype: str
    forward_mode: str
    request_bucket: int
    token_bucket: int
    attention_backend: str
    attention_backend_version: str
    moe_kernel_version: str
    compile_options_hash: str
    tp_degree: int
    ep_degree: int = 1


@dataclass(frozen=True)
class CompiledVariant:
    key: VariantKey
    paged_attn_impl: str
    artifact_path: str | None = None


@dataclass(frozen=True)
class PrecompileCatalog:
    variants: tuple[CompiledVariant, ...]


def _variant_key_from_mapping(data: dict[str, Any]) -> VariantKey:
    try:
        return VariantKey(
            model_id=str(data["model_id"]),
            dtype=str(data["dtype"]),
            forward_mode=str(data.get("forward_mode", ForwardMode.EXTEND.value)),
            request_bucket=int(data["request_bucket"]),
            token_bucket=int(data["token_bucket"]),
            attention_backend=str(data["attention_backend"]),
            attention_backend_version=str(data["attention_backend_version"]),
            moe_kernel_version=str(data["moe_kernel_version"]),
            compile_options_hash=str(data["compile_options_hash"]),
            tp_degree=int(data.get("tp_degree", 1)),
            ep_degree=int(data.get("ep_degree", 1)),
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required variant key field: {exc}") from exc


def _variant_from_mapping(data: dict[str, Any]) -> CompiledVariant:
    key = _variant_key_from_mapping(data)
    if "paged_attn_impl" not in data:
        raise RuntimeError("Missing required catalog field: paged_attn_impl")
    artifact_path = data.get("artifact_path")
    if artifact_path is not None:
        artifact_path = str(artifact_path).strip() or None
    return CompiledVariant(
        key=key,
        paged_attn_impl=str(data["paged_attn_impl"]),
        artifact_path=artifact_path,
    )


def _make_variant_key(
    config: RuntimeConfig,
    forward_mode: str,
    request_bucket: int,
    token_bucket: int,
) -> VariantKey:
    """Build a VariantKey from runtime config and per-variant parameters."""
    return VariantKey(
        model_id=config.model_id,
        dtype=config.model_dtype,
        forward_mode=forward_mode,
        request_bucket=request_bucket,
        token_bucket=token_bucket,
        attention_backend=config.attention_backend,
        attention_backend_version=config.attention_backend_version,
        moe_kernel_version=config.moe_kernel_version,
        compile_options_hash=config.compile_options_hash,
        tp_degree=config.tp_degree,
        ep_degree=config.ep_degree,
    )


def _default_catalog(config: RuntimeConfig) -> PrecompileCatalog:
    paddings = build_precompile_paddings(config)
    variants: list[CompiledVariant] = []
    for tb in paddings.token_paddings:
        key = _make_variant_key(
            config, ForwardMode.EXTEND.value, paddings.max_padded_batch_size, tb
        )
        variants.append(
            CompiledVariant(key=key, paged_attn_impl=config.paged_attn_impl)
        )
    for bs in paddings.bs_paddings:
        key = _make_variant_key(config, ForwardMode.DECODE.value, bs, bs)
        variants.append(
            CompiledVariant(key=key, paged_attn_impl=config.paged_attn_impl)
        )
    return PrecompileCatalog(variants=tuple(variants))


def load_precompile_catalog(config: RuntimeConfig) -> PrecompileCatalog:
    if not config.precompile_catalog_file:
        return _default_catalog(config)

    path = Path(config.precompile_catalog_file)
    if not path.exists():
        raise RuntimeError(f"Precompile catalog file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        variant_items = payload.get("variants")
    elif isinstance(payload, list):
        variant_items = payload
    else:
        raise RuntimeError("Precompile catalog JSON must be object/list")

    if not isinstance(variant_items, list):
        raise RuntimeError("Precompile catalog must contain a 'variants' list")

    variants = tuple(_variant_from_mapping(item) for item in variant_items)
    return PrecompileCatalog(variants=variants)


def _make_expected_keys(config: RuntimeConfig) -> set[VariantKey]:
    paddings = build_precompile_paddings(config)
    expected: set[VariantKey] = set()
    for tb in paddings.token_paddings:
        expected.add(
            _make_variant_key(
                config, ForwardMode.EXTEND.value, paddings.max_padded_batch_size, tb
            )
        )
    for bs in paddings.bs_paddings:
        expected.add(_make_variant_key(config, ForwardMode.DECODE.value, bs, bs))
    return expected


def validate_precompile_catalog(
    catalog: PrecompileCatalog, config: RuntimeConfig
) -> None:
    if not catalog.variants:
        raise RuntimeError("Precompile catalog has no variants")

    paddings = build_precompile_paddings(config)
    seen: set[VariantKey] = set()
    for variant in catalog.variants:
        key = variant.key
        if key in seen:
            raise RuntimeError(f"Duplicate variant key in catalog: {repr(key)}")
        seen.add(key)

        if key.request_bucket not in paddings.bs_paddings:
            raise RuntimeError(
                f"Variant request bucket not allowed by runtime config: {repr(key)}"
            )
        if key.forward_mode == ForwardMode.EXTEND.value:
            if key.request_bucket != paddings.max_padded_batch_size:
                raise RuntimeError(
                    "Extend variant request bucket must match max padded batch size: "
                    f"{repr(key)}, expected_request_bucket={paddings.max_padded_batch_size}"
                )
            if key.token_bucket not in paddings.token_paddings:
                raise RuntimeError(
                    f"Extend variant token bucket not allowed: {repr(key)}"
                )
        elif key.forward_mode == ForwardMode.DECODE.value:
            if key.token_bucket != key.request_bucket:
                raise RuntimeError(
                    "Decode variant requires token_bucket == request_bucket: "
                    f"{repr(key)}"
                )
            if key.token_bucket not in paddings.bs_paddings:
                raise RuntimeError(
                    f"Decode variant token bucket not allowed: {repr(key)}"
                )
        else:
            raise RuntimeError(
                "Variant forward mode must be one of {'extend','decode'}: "
                f"{repr(key)}"
            )
        if variant.paged_attn_impl != config.paged_attn_impl:
            raise RuntimeError(
                "Variant paged attention impl mismatch: "
                f"variant={variant.paged_attn_impl}, expected={config.paged_attn_impl}, "
                f"key={repr(key)}"
            )
        if key.tp_degree != config.tp_degree:
            raise RuntimeError(
                "Variant tp_degree mismatch: "
                f"variant={key.tp_degree}, expected={config.tp_degree}, "
                f"key={repr(key)}"
            )

    expected_keys = _make_expected_keys(config)
    missing_keys = expected_keys - seen
    if missing_keys:
        missing = ", ".join(
            repr(key)
            for key in sorted(
                missing_keys,
                key=lambda k: (k.forward_mode, k.request_bucket, k.token_bucket),
            )
        )
        raise RuntimeError(f"Precompile catalog missing required variants: {missing}")
