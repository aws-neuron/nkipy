"""Batched sampling metadata (numpy-only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nkipy_serving.sampling.params import DEFAULT_SAMPLING_SEED, SamplingParams
from nkipy_serving.utils import get_bool_env_var


@dataclass
class SamplingBatchInfo:
    temperatures: np.ndarray
    top_ps: np.ndarray
    top_ks: np.ndarray
    min_ps: np.ndarray
    vocab_size: int
    is_all_greedy: bool = False
    need_top_p_sampling: bool = False
    need_top_k_sampling: bool = False
    need_min_p_sampling: bool = False
    sampling_seeds: np.ndarray | None = None

    @classmethod
    def from_sampling_params(
        cls,
        params: list[SamplingParams],
        vocab_size: int,
    ) -> "SamplingBatchInfo":
        if not params:
            raise RuntimeError("SamplingBatchInfo requires at least one SamplingParams")

        temperatures = np.asarray([p.temperature for p in params], dtype=np.float32)
        top_ps = np.asarray([p.top_p for p in params], dtype=np.float32)
        top_ks = np.asarray([p.top_k for p in params], dtype=np.int32)
        min_ps = np.asarray([p.min_p for p in params], dtype=np.float32)

        seeds: np.ndarray | None = None
        if any(p.sampling_seed is not None for p in params) or get_bool_env_var(
            "SGLANG_ENABLE_DETERMINISTIC_SAMPLING"
        ):
            seeds = np.asarray(
                [
                    p.sampling_seed
                    if p.sampling_seed is not None
                    else DEFAULT_SAMPLING_SEED
                    for p in params
                ],
                dtype=np.int32,
            )

        return cls(
            temperatures=temperatures.reshape(-1, 1),
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=bool(np.all(top_ks == 1)),
            need_top_p_sampling=bool(np.any(top_ps < 1.0)),
            need_top_k_sampling=bool(np.any(top_ks < (1 << 30))),
            need_min_p_sampling=bool(np.any(min_ps > 0.0)),
            sampling_seeds=seeds,
        )

    @classmethod
    def generate_for_precompile(
        cls, bs: int, vocab_size: int = 32000
    ) -> "SamplingBatchInfo":
        params = [
            SamplingParams(
                temperature=0.6,
                top_p=0.9,
                top_k=30,
                min_p=0.6,
                sampling_seed=(
                    DEFAULT_SAMPLING_SEED
                    if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING")
                    else None
                ),
            )
            for _ in range(bs)
        ]
        return cls.from_sampling_params(params=params, vocab_size=vocab_size)
