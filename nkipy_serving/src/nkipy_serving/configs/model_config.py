"""Model metadata config helper (no-JAX variant)."""

from __future__ import annotations

import json
import logging
from enum import Enum, IntEnum, auto
from typing import Any, Mapping

from nkipy_serving.configs.quantization_config import QuantizationConfig
from nkipy_serving.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_generation_config,
    get_hf_text_config,
)

logger = logging.getLogger(__name__)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


class ModelImpl(str, Enum):
    AUTO = "auto"
    SGLANG = "sglang"
    TRANSFORMERS = "transformers"


class MoEBackend(str, Enum):
    EPMOE = "epmoe"
    FUSED = "fused"
    AUTO = "auto"


def _cfg_get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return cfg[key] if key in cfg else default


def _as_int(cfg: Mapping[str, Any], key: str, default: int | None = None) -> int:
    raw = _cfg_get(cfg, key, default)
    if raw is None:
        raise RuntimeError(f"Missing required config key: {key}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Config key {key} must be integer-like, got {raw!r}"
        ) from exc


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: str | None = None,
        context_length: int | None = None,
        model_override_args: str | dict[str, Any] = "{}",
        dtype: str = "bf16",
        quantization: str | None = None,
        quantization_config_path: str | None = None,
        model_layer_nums: int | None = None,
        model_impl: str | ModelImpl = ModelImpl.AUTO,
        moe_backend: str | MoEBackend = MoEBackend.AUTO,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.dtype = dtype
        self.quantization = quantization
        self.model_impl = ModelImpl(model_impl)
        self.moe_backend = MoEBackend(moe_backend)
        self.local_files_only = local_files_only

        if isinstance(model_override_args, str):
            self.model_override_args = json.loads(model_override_args)
        else:
            self.model_override_args = dict(model_override_args)

        self.quantization_config = QuantizationConfig.from_path(
            quantization_config_path
        )

        self.hf_config = get_config(
            model=model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            local_files_only=local_files_only,
        )
        self.hf_generation_config = get_generation_config(
            model=model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
        )

        self.hf_text_config = get_hf_text_config(self.hf_config)
        if not isinstance(self.hf_text_config, Mapping):
            raise RuntimeError("hf_text_config must be a dict-like object")

        derived_context_len = get_context_length(self.hf_text_config)
        self.context_len = (
            int(context_length) if context_length is not None else derived_context_len
        )
        if self.context_len <= 0:
            raise RuntimeError(f"context_len must be positive, got {self.context_len}")

        self.hidden_size = _as_int(self.hf_text_config, "hidden_size")
        self.num_attention_heads = _as_int(self.hf_text_config, "num_attention_heads")
        self.num_key_value_heads = _as_int(
            self.hf_text_config,
            "num_key_value_heads",
            default=self.num_attention_heads,
        )
        self.num_hidden_layers = _as_int(self.hf_text_config, "num_hidden_layers")
        self.vocab_size = _as_int(self.hf_text_config, "vocab_size")

        if model_layer_nums is not None:
            if model_layer_nums <= 0:
                raise RuntimeError(
                    f"model_layer_nums must be > 0, got {model_layer_nums}"
                )
            self.num_hidden_layers = min(self.num_hidden_layers, int(model_layer_nums))

        self.head_dim = int(
            _cfg_get(
                self.hf_text_config,
                "head_dim",
                self.hidden_size // self.num_attention_heads,
            )
        )
        if self.head_dim <= 0:
            raise RuntimeError(f"Invalid head_dim derived from config: {self.head_dim}")

        self.sliding_window = _cfg_get(self.hf_text_config, "sliding_window", None)
        self.attention_arch = AttentionArch.MHA
        self.hf_eos_token_id = self.get_hf_eos_token_id()

    def get_hf_eos_token_id(self) -> int | list[int] | None:
        for cfg in (self.hf_generation_config, self.hf_text_config, self.hf_config):
            if isinstance(cfg, Mapping) and "eos_token_id" in cfg:
                return cfg["eos_token_id"]
        return None

    def get_total_num_kv_heads(self) -> int:
        return self.num_key_value_heads
