"""Shared DSV4 NEFF-runtime state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import (
    Dsv4GraphFns,
    Dsv4SampledForwardOptions,
    _with_dsv4_fp8_compiler_arg,
)
from nkipy_serving.sampling.logits_processor import LogitsProcessor


class Dsv4RuntimeStateMixin:
    """Shared orchestration state for the DSV4 NEFF runtime.

    Owns the DSV4 graph-function table, runtime options, build-dir,
    attention backend, device state, and optional blockwise-MoE state.
    All fields are required non-None except ``blockwise_moe_state``.
    """

    def _init_runtime_state(
        self,
        runtime_surface: Any,
        *,
        graph: Dsv4GraphFns,
        options: Dsv4SampledForwardOptions,
        build_dir: str | None,
        attention_backend: Any,
        device_state: Any,
        blockwise_moe_state: Any | None = None,
        logits_processor: LogitsProcessor | None = None,
        final_norm_dev: Any | None = None,
        lm_head_dev: Any | None = None,
        embed_tp_sharded: bool = False,
        embed_vocab_offset: int = 0,
        embed_vocab_end: int = 0,
        embed_tp_degree: int = 1,
        embed_tp_replica_groups: tuple[tuple[int, ...], ...] = (),
        max_requests_per_step: int = 1,
        compiler_args: str = "",
        product_prefill_moe_blockwise_fusion_max_rows: int = 0,
        product_prefill_moe_dispatch_fusion_max_rows: int = 0,
        product_prefill_dp_attention_post_pre_fusion_max_rows: int = 0,
    ) -> None:
        if attention_backend is None:
            raise RuntimeError("attention_backend is required")
        if device_state is None:
            raise RuntimeError("device_state is required")
        self.runtime_surface = runtime_surface
        self.graph = graph
        self.options = options
        self.build_dir = build_dir
        self.attention_backend = attention_backend
        self.device_state = device_state
        self.blockwise_moe_state = blockwise_moe_state
        self.logits_processor = logits_processor
        self.final_norm_dev = final_norm_dev
        self.lm_head_dev = lm_head_dev
        self.embed_tp_sharded = bool(embed_tp_sharded)
        self.embed_vocab_offset = int(embed_vocab_offset)
        self.embed_vocab_end = int(embed_vocab_end)
        self.embed_tp_degree = int(embed_tp_degree)
        self.embed_tp_replica_groups = tuple(
            tuple(int(r) for r in group) for group in embed_tp_replica_groups
        )
        self.max_requests_per_step = int(max_requests_per_step)
        self.product_prefill_moe_blockwise_fusion_max_rows = int(
            product_prefill_moe_blockwise_fusion_max_rows
        )
        self.product_prefill_moe_dispatch_fusion_max_rows = int(
            product_prefill_moe_dispatch_fusion_max_rows
        )
        self.product_prefill_dp_attention_post_pre_fusion_max_rows = int(
            product_prefill_dp_attention_post_pre_fusion_max_rows
        )
        self.compiler_args = _with_dsv4_fp8_compiler_arg(compiler_args)

    @staticmethod
    def _base_metadata(metadata: Any | None) -> Any | None:
        if metadata is None:
            return None
        return getattr(metadata, "base", metadata)

    @staticmethod
    def _batch_size_from_input(input_ids: np.ndarray) -> int:
        ids = np.asarray(input_ids)
        if ids.ndim >= 2:
            return int(ids.shape[0])
        return 1

    def _dp_attention_replica_groups(self) -> tuple[tuple[int, ...], ...]:
        state = getattr(self, "blockwise_moe_state", None)
        groups = getattr(state, "ep_replica_groups", ()) if state is not None else ()
        return tuple(tuple(int(rank) for rank in group) for group in groups)

    def _dp_attention_lane_context(self, metadata: Any | None) -> Any | None:
        dp = getattr(metadata, "dp_superstep", None) if metadata is not None else None
        if dp is None or not hasattr(metadata, "slice_for_dp_lane"):
            return None
        groups = self._dp_attention_replica_groups()
        if not groups or all(len(group) <= 1 for group in groups):
            return None
        lane = int(getattr(self.runtime_surface.v4, "attention_lane", -1))
        token_start, token_end = dp.token_range(lane)
        batch_start, batch_end = dp.batch_range(lane)
        return SimpleNamespace(
            lane=lane,
            token_start=token_start,
            token_end=token_end,
            batch_start=batch_start,
            batch_end=batch_end,
            batch_size=int(batch_end - batch_start),
            total_batch_size=int(dp.batch_size),
            replica_groups=groups,
        )

    def _prepare_dp_attention_lane_metadata(
        self,
        metadata: Any | None,
        ctx: Any | None,
    ) -> Any | None:
        if ctx is None or int(getattr(ctx, "batch_size", 0)) <= 0:
            return None
        if metadata is None or not hasattr(metadata, "slice_for_dp_lane"):
            return None
        lane_metadata = metadata.slice_for_dp_lane(int(ctx.lane))
        return self.attention_backend.prepare(lane_metadata)

    def _is_decode_step(self, metadata: Any | None, start_pos: int) -> bool:
        # Decode heuristic: attention backend metadata carries forward_mode when
        # present; otherwise decode = (start_pos > 0).
        if metadata is not None and hasattr(metadata, "base"):
            base_md = getattr(metadata, "base", metadata)
            fm = getattr(base_md, "forward_mode", None)
            if fm is not None:
                from nkipy_serving.attention.base import FORWARD_MODE_DECODE

                return int(fm) == int(FORWARD_MODE_DECODE)
        return int(start_pos) > 0

    @property
    def has_compressed_layers(self) -> bool:
        return any(
            int(getattr(block.attn, "compress_ratio", 0)) > 0
            for block in self.runtime_surface.blocks
        )
