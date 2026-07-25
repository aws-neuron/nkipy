"""Quantization config helpers (string/dict based, no JAX dtypes)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantizationRule:
    module_path: str
    weight_dtype: str | None = None
    activation_dtype: str | None = None


@dataclass
class QuantizationConfig:
    linear_rules: list[QuantizationRule] = field(default_factory=list)
    moe_weight_dtype: str | None = None
    moe_activation_dtype: str | None = None
    is_static_checkpoint: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationConfig":
        linear_rules_raw = data.get("linear_rules", [])
        if not isinstance(linear_rules_raw, list):
            raise RuntimeError("quantization linear_rules must be a list")

        linear_rules: list[QuantizationRule] = []
        for idx, rule in enumerate(linear_rules_raw):
            if not isinstance(rule, dict):
                raise RuntimeError(f"linear_rules[{idx}] must be an object")
            module_path = str(rule.get("module_path", "")).strip()
            if not module_path:
                raise RuntimeError(f"linear_rules[{idx}].module_path must be non-empty")
            linear_rules.append(
                QuantizationRule(
                    module_path=module_path,
                    weight_dtype=(
                        str(rule["weight_dtype"])
                        if rule.get("weight_dtype") is not None
                        else None
                    ),
                    activation_dtype=(
                        str(rule["activation_dtype"])
                        if rule.get("activation_dtype") is not None
                        else None
                    ),
                )
            )

        return cls(
            linear_rules=linear_rules,
            moe_weight_dtype=(
                str(data["moe_weight_dtype"])
                if data.get("moe_weight_dtype") is not None
                else None
            ),
            moe_activation_dtype=(
                str(data["moe_activation_dtype"])
                if data.get("moe_activation_dtype") is not None
                else None
            ),
            is_static_checkpoint=bool(data.get("is_static_checkpoint", False)),
        )

    @classmethod
    def from_path(cls, path: str | None) -> "QuantizationConfig | None":
        if not path:
            return None
        config_path = Path(path)
        if not config_path.exists():
            raise RuntimeError(
                f"Quantization config file does not exist: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(f"Quantization config must be an object: {config_path}")
        return cls.from_dict(data)
