"""Shared GPT-OSS / Harmony helpers used by chat and completions handlers."""

from __future__ import annotations


def is_gpt_oss_model(model_name: str) -> bool:
    return "gpt-oss" in str(model_name).lower()


def apply_harmony_stop(stop: str | list[str] | None) -> str | list[str]:
    """Ensure the Harmony end-of-generation marker ``<|return|>`` is present."""
    marker = "<|return|>"
    if stop is None:
        return marker
    if isinstance(stop, str):
        return [stop, marker] if stop != marker else stop
    return list(stop) + [marker] if marker not in stop else stop
