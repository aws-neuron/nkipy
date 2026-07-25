"""Runtime-owned DSV4 component bundle and installation helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nkipy_serving.models.deepseek_v4.graph_types import (
    Dsv4SampledForwardOptions,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.state import (
    Dsv4RuntimeStateMixin,
)
from nkipy_serving.sampling.logits_processor import LogitsProcessor


@dataclass(frozen=True)
class Dsv4RuntimeComponents:
    runtime_surface: Any
    graph: dict[str, Any]
    options: Dsv4SampledForwardOptions
    build_dir: str | None
    attention_backend: Any
    device_state: Any
    blockwise_moe_state: Any | None
    logits_processor: LogitsProcessor
    final_norm_dev: Any
    lm_head_dev: Any
    embed_tp_sharded: bool
    embed_vocab_offset: int
    embed_vocab_end: int
    embed_tp_degree: int
    embed_tp_replica_groups: tuple[tuple[int, ...], ...]
    max_requests_per_step: int
    compiler_args: str
    product_prefill_moe_blockwise_fusion_max_rows: int
    product_prefill_moe_dispatch_fusion_max_rows: int
    product_prefill_dp_attention_post_pre_fusion_max_rows: int


def init_dsv4_runtime_components(
    runtime: Dsv4RuntimeStateMixin,
    components: Dsv4RuntimeComponents,
) -> None:
    """Install assembled runtime components onto an existing runtime object."""

    runtime._init_runtime_state(
        components.runtime_surface,
        graph=components.graph,
        options=components.options,
        build_dir=components.build_dir,
        attention_backend=components.attention_backend,
        device_state=components.device_state,
        blockwise_moe_state=components.blockwise_moe_state,
        logits_processor=components.logits_processor,
        final_norm_dev=components.final_norm_dev,
        lm_head_dev=components.lm_head_dev,
        embed_tp_sharded=components.embed_tp_sharded,
        embed_vocab_offset=components.embed_vocab_offset,
        embed_vocab_end=components.embed_vocab_end,
        embed_tp_degree=components.embed_tp_degree,
        embed_tp_replica_groups=components.embed_tp_replica_groups,
        max_requests_per_step=components.max_requests_per_step,
        compiler_args=components.compiler_args,
        product_prefill_moe_blockwise_fusion_max_rows=(
            components.product_prefill_moe_blockwise_fusion_max_rows
        ),
        product_prefill_moe_dispatch_fusion_max_rows=(
            components.product_prefill_moe_dispatch_fusion_max_rows
        ),
        product_prefill_dp_attention_post_pre_fusion_max_rows=(
            components.product_prefill_dp_attention_post_pre_fusion_max_rows
        ),
    )
