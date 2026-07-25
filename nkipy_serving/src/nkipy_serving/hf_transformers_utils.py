"""Lightweight Hugging Face helpers without transformers/jax dependency."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import snapshot_download

from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer
from nkipy_serving.utils.common_utils import is_remote_url, lru_cache_frozenset

_UNSET = object()


def check_gguf_file(path: str) -> bool:
    return path.strip().lower().endswith(".gguf")


def download_from_hf(
    model_path: str,
    allow_patterns: list[str] | None | object = _UNSET,
    local_files_only: bool = True,
    revision: str | None = None,
) -> str:
    if os.path.exists(model_path):
        return model_path
    if is_remote_url(model_path):
        raise RuntimeError(
            f"Remote URLs are not supported in this runtime: {model_path}"
        )
    if allow_patterns is _UNSET:
        allow_patterns = ["*.json", "*.safetensors", "*.model", "*.tiktoken"]
    snapshot_path = snapshot_download(
        repo_id=model_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
    )
    return str(snapshot_path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def get_hf_text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"config must be mapping, got {type(config)}")

    if isinstance(config.get("text_config"), Mapping):
        return config["text_config"]
    if isinstance(config.get("language_config"), Mapping):
        return config["language_config"]

    thinker = config.get("thinker_config")
    if isinstance(thinker, Mapping):
        if isinstance(thinker.get("text_config"), Mapping):
            return thinker["text_config"]
        return thinker

    return config


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict[str, Any] | None = None,
    local_files_only: bool = True,
    **kwargs,
) -> dict[str, Any]:
    _ = trust_remote_code
    _ = kwargs

    if check_gguf_file(model):
        raise RuntimeError("GGUF configs are not supported in this runtime")

    model_path = Path(
        download_from_hf(
            model_path=model,
            local_files_only=local_files_only,
            revision=revision,
            allow_patterns=["*.json"],
        )
    )
    config_path = model_path / "config.json"
    config = _load_json_if_exists(config_path)
    if config is None:
        raise RuntimeError(f"Missing config.json in {model_path}")

    text_config = get_hf_text_config(config)
    if isinstance(text_config, Mapping):
        for key, val in text_config.items():
            if key not in config and val is not None:
                config[key] = val

    if model_override_args:
        config.update(dict(model_override_args))

    return dict(config)


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    local_files_only: bool = True,
    **kwargs,
) -> dict[str, Any] | None:
    _ = trust_remote_code
    _ = kwargs
    model_path = Path(
        download_from_hf(
            model_path=model,
            local_files_only=local_files_only,
            revision=revision,
            allow_patterns=["generation_config.json"],
        )
    )
    return _load_json_if_exists(model_path / "generation_config.json")


CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def _cfg_get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def get_context_length(config: Mapping[str, Any]) -> int:
    rope_scaling = _cfg_get(config, "rope_scaling", None)
    rope_scaling_factor = 1.0
    if isinstance(rope_scaling, Mapping):
        rope_scaling_factor = float(_cfg_get(rope_scaling, "factor", 1.0))
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1.0
        if _cfg_get(rope_scaling, "rope_type", None) == "llama3":
            rope_scaling_factor = 1.0

    for key in CONTEXT_LENGTH_KEYS:
        val = _cfg_get(config, key, None)
        if val is not None:
            return int(float(val) * rope_scaling_factor)
    return 2048


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    sub_dir: str = "",
    local_files_only: bool = True,
    **kwargs,
) -> HfTokenizer:
    _ = args
    _ = trust_remote_code
    _ = kwargs

    if tokenizer_mode not in {"auto", "slow", "fast"}:
        raise RuntimeError(f"Unsupported tokenizer_mode: {tokenizer_mode}")

    tokenizer_path = tokenizer_name
    if tokenizer_name.endswith("tokenizer.json"):
        tokenizer_path = str(Path(tokenizer_name).parent)
    if sub_dir:
        tokenizer_path = str(Path(tokenizer_path) / sub_dir)

    return HfTokenizer(
        model_id=tokenizer_path,
        revision=tokenizer_revision,
        local_files_only=local_files_only,
    )


def attach_additional_stop_token_ids(tokenizer: Any) -> None:
    if tokenizer is None:
        return
    if not hasattr(tokenizer, "additional_stop_token_ids"):
        tokenizer.additional_stop_token_ids = []


def get_tokenizer_from_processor(processor: Any) -> Any:
    if hasattr(processor, "encode") and hasattr(processor, "decode"):
        return processor
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    raise RuntimeError(f"Unsupported processor object: {type(processor)}")


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    local_files_only: bool = True,
    **kwargs,
) -> Any:
    # Multimodal processors are out of scope for this prototype.
    return get_tokenizer(
        tokenizer_name,
        *args,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        local_files_only=local_files_only,
        **kwargs,
    )
