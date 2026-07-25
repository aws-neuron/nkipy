"""NEFF-backed DSV4 runtime mixin.

This module owns runtime orchestration for the NEFF-backed path:
bucket selection, NEFF cache lookup, warmup sequencing, lane slicing, and
staged execution. Pure graph math stays in ``neff_graphs`` and
state-mutation kernels stay under ``ops/deepseek_v4``.

Runtime layering:

* ``DeepseekV4Executor`` in ``executor.py`` is the ModelRunner-facing facade.
* ``Dsv4RuntimeStateMixin`` in ``neff_runtime.state`` owns shared runtime
  state and generic orchestration helpers.
* ``Dsv4NeffRuntimeMixin`` mixes in the NEFF-backed execution stages.
* Product bundles below keep the final executor inheritance list readable
  while preserving the concrete mixin MRO order.
"""

from __future__ import annotations

import logging
from typing import Any

from nkipy_serving.models.deepseek_v4.neff_runtime.buffers import (
    Dsv4ProductBuffersMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.components import (
    Dsv4RuntimeComponents,
    init_dsv4_runtime_components,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.forward import (
    Dsv4ProductForwardMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.infra import (
    Dsv4ProductAliasMixin,
    Dsv4ProductEmbeddingMixin,
    Dsv4ProductFrequencyMixin,
    Dsv4ProductHeadMixin,
    Dsv4ProductMetadataMixin,
    Dsv4ProductProfileMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    Dsv4ProductManifestMixin,
    Dsv4ProductWarmupMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.moe.dispatch import (
    Dsv4ProductMoeDispatchMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.moe.dp_attention_moe_fused import (
    Dsv4ProductDpAttentionMoeFusedMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.moe.shared_expert import (
    Dsv4ProductSharedExpertMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.base import (
    Dsv4ProductQkvBaseMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_allkv_runtime import (
    Dsv4ProductQkvIndexerAllKvRuntimeMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_kernels import (
    Dsv4ProductQkvIndexerKernelsMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_precompile import (
    Dsv4ProductQkvIndexerPrecompileMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_runtime import (
    Dsv4ProductQkvIndexerRuntimeMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.token_topk import (
    Dsv4ProductQkvTokenTopkMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.token_topk_runtime import (
    Dsv4ProductQkvTokenTopkRuntimeMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.manager import (
    Dsv4ProductBucketManagerMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.attention_graph import (
    Dsv4ProductAttentionGraphMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.attention_out import (
    Dsv4ProductAttentionOutMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.attention_runtime import (
    Dsv4ProductAttentionRuntimeMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.compressor import (
    Dsv4ProductCompressorMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.dp_attention import (
    Dsv4ProductDpAttentionMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.state import (
    Dsv4RuntimeStateMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.support_warmup import (
    Dsv4ProductSupportKernelWarmupMixin,
)

logger = logging.getLogger(__name__)


class Dsv4ProductQkvIndexerMixin(
    Dsv4ProductQkvIndexerRuntimeMixin,
    Dsv4ProductQkvIndexerPrecompileMixin,
    Dsv4ProductQkvIndexerAllKvRuntimeMixin,
    Dsv4ProductQkvTokenTopkRuntimeMixin,
    Dsv4ProductQkvIndexerKernelsMixin,
    Dsv4ProductQkvTokenTopkMixin,
):
    """Composed product QKV/indexer mixin."""


class Dsv4ProductDpAttentionMoeMixin(
    Dsv4ProductDpAttentionMoeFusedMixin,
    Dsv4ProductMoeDispatchMixin,
    Dsv4ProductDpAttentionMixin,
):
    """Composed DP-attention and MoE product mixin."""


class Dsv4ProductAttentionCoreMixin(
    Dsv4ProductAttentionGraphMixin,
    Dsv4ProductAttentionOutMixin,
    Dsv4ProductAttentionRuntimeMixin,
):
    """NEFF-backed attention graph/runtime entrypoints."""


class Dsv4ProductResourceMixin(
    Dsv4ProductBucketManagerMixin,
    Dsv4ProductBuffersMixin,
):
    """NEFF-backed runtime bucket and scratch-buffer resources."""


class Dsv4ProductAttentionComputeMixin(
    Dsv4ProductCompressorMixin,
    Dsv4ProductDpAttentionMoeMixin,
):
    """NEFF-backed compressed-attention and DP-attention/MoE kernels."""


class Dsv4ProductModelCoreMixin(
    Dsv4ProductEmbeddingMixin,
    Dsv4ProductFrequencyMixin,
    Dsv4ProductForwardMixin,
    Dsv4ProductHeadMixin,
    Dsv4ProductManifestMixin,
    Dsv4ProductMetadataMixin,
    Dsv4ProductProfileMixin,
):
    """NEFF-backed model-level execution, metadata, and lifecycle helpers."""


class Dsv4ProductQkvSharedMixin(
    Dsv4ProductQkvBaseMixin,
    Dsv4ProductQkvIndexerMixin,
    Dsv4ProductSharedExpertMixin,
    Dsv4ProductSupportKernelWarmupMixin,
    Dsv4ProductWarmupMixin,
):
    """NEFF-backed QKV/indexer, shared-expert, support-kernel, and warmup helpers."""


class Dsv4NeffRuntimeMixin(
    Dsv4ProductAliasMixin,
    Dsv4ProductAttentionCoreMixin,
    Dsv4ProductResourceMixin,
    Dsv4ProductAttentionComputeMixin,
    Dsv4ProductModelCoreMixin,
    Dsv4ProductQkvSharedMixin,
    Dsv4RuntimeStateMixin,
):
    """NEFF-backed DSV4 runtime.

    NEFF execution uses coarser graph functions and raw DeviceKernel
    handles where side-effect-free math can be fused.
    """

    def _init_neff_runtime(
        self,
        components: Dsv4RuntimeComponents,
    ) -> None:
        init_dsv4_runtime_components(self, components)
        self._product_buckets: dict[int, Dsv4ProductBucket] = {}
        self._product_freq_tables: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._product_manifest_sealed = False
        self._product_manifest_snapshot: dict[str, Any] = {}
        self._product_attention_out_dp_flat_warmup_rows: dict[
            int,
            set[tuple[Any, ...]],
        ] = {}
        self._init_product_stage_profile_writer()
