"""Shared DSV4 graph-function types and compile-time graph options."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from nkipy_serving.models.deepseek_v4.diagnostics import (
    current_rank,
    rank_trace_allowed,
    warmup_trace_enabled,
)

Dsv4GraphFns = dict[str, Callable[..., Any]]
_TRN_FP8_E4M3FN_COMPILER_FLAG = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
_TRN_FP8_E4M3FN_COMPILER_ARG = (
    f"--internal-hlo2tensorizer-options='{_TRN_FP8_E4M3FN_COMPILER_FLAG}'"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dsv4SampledForwardOptions:
    """Runtime knobs for DSV4 graph-function execution."""

    index_construction_max_c_len: int = 0


def _with_dsv4_fp8_compiler_arg(compiler_args: str) -> str:
    """Ensure Trn2 accepts rescaled OCP E4M3FN tensors as Neuron E4M3."""
    args = str(compiler_args or "").strip()
    if _TRN_FP8_E4M3FN_COMPILER_FLAG in args:
        return args
    return (
        f"{args} {_TRN_FP8_E4M3FN_COMPILER_ARG}".strip()
        if args
        else _TRN_FP8_E4M3FN_COMPILER_ARG
    )


def _sampled_warmup_trace(message: str) -> None:
    if not warmup_trace_enabled():
        return
    rank = current_rank()
    if not rank_trace_allowed(rank):
        return
    logger.info("DSV4 sampled forward rank=%d %s", rank, message)
    if os.getenv("NKIPY_SERVING_DSV4_WARMUP_TRACE_FILE"):
        try:
            with open("/tmp/_dsv4_warmup_trace.log", "a") as trace_file:
                trace_file.write(f"rank={rank} sampled {message}\n")
                trace_file.flush()
        except Exception:
            pass


__all__ = [
    "Dsv4GraphFns",
    "Dsv4SampledForwardOptions",
    "_sampled_warmup_trace",
    "_with_dsv4_fp8_compiler_arg",
]
