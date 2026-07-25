"""Shared diagnostic helpers for DeepSeek-V4 runtime modules."""

from __future__ import annotations

from nkipy_serving.runtime.diagnostics import (
    current_rank as current_rank,
)
from nkipy_serving.runtime.diagnostics import (
    env_flag,
    env_rank_filter_allows,
)


def rank_trace_allowed(rank: int) -> bool:
    return env_rank_filter_allows(
        rank,
        "NKIPY_SERVING_DSV4_RANK_TRACE_FILTER",
        "NKIPY_SERVING_DSV4_WARMUP_TRACE_RANKS",
    )


def warmup_trace_enabled() -> bool:
    return env_flag("NKIPY_SERVING_DSV4_WARMUP_TRACE")


def stage_profile_enabled() -> bool:
    return env_flag("NKIPY_SERVING_DSV4_STAGE_PROFILE")
