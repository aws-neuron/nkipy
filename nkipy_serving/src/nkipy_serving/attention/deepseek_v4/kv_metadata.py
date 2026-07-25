"""Heterogeneous KV cache metadata for DeepSeek-V4.

This module owns metadata schema and per-layer spec derivation. Runtime
allocation still uses the serving KV pool; the DSV4 device state owns the
additional heterogeneous SWA/compressor/indexer buffers.

V4 cache kinds:
  - "full"    : uncompressed attention KV (layers 0, 1, 42).
  - "c128a"   : compressed at stride 128.
  - "c4a"     : compressed at stride 4 + indexer KV + sparse top-512.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KVCacheKind(str, Enum):
    FULL = "full"
    C4A = "c4a"
    C128A = "c128a"


# Ratios as stored in HF config.
_RATIO_TO_KIND = {
    0: KVCacheKind.FULL,
    4: KVCacheKind.C4A,
    128: KVCacheKind.C128A,
}


@dataclass(frozen=True)
class LayerKVSpec:
    layer_id: int
    kind: KVCacheKind
    compress_ratio: int  # 1 for FULL, 4 or 128 for compressed kinds.
    has_sliding_window: bool  # c4a layers carry sliding-window raw attention KV.
    has_indexer: bool  # c4a layers also carry indexer KV.


@dataclass(frozen=True)
class HeterogeneousKVMetadata:
    """Per-model KV metadata; built once per worker from `DeepseekV4Weights`."""

    layer_specs: tuple[LayerKVSpec, ...]

    def layers_of_kind(self, kind: KVCacheKind) -> tuple[int, ...]:
        return tuple(s.layer_id for s in self.layer_specs if s.kind == kind)

    @property
    def num_layers(self) -> int:
        return len(self.layer_specs)


def build_kv_metadata_from_ratios(
    compress_ratios: tuple[int, ...],
    sliding_window: int,
) -> HeterogeneousKVMetadata:
    """Derive per-layer KV specs from `compress_ratios`.

    c4a layers carry sliding-window raw attention KV in addition to the
    compressed/indexer state. `sliding_window` comes from the HF config and
    is informational at this stage.
    """
    specs: list[LayerKVSpec] = []
    for layer_id, ratio in enumerate(compress_ratios):
        kind = _RATIO_TO_KIND.get(int(ratio))
        if kind is None:
            raise RuntimeError(
                f"Unsupported compress_ratio={ratio} at layer {layer_id}. "
                "Expected 0, 4, or 128."
            )
        compress = 1 if kind == KVCacheKind.FULL else int(ratio)
        is_c4a = kind == KVCacheKind.C4A
        specs.append(
            LayerKVSpec(
                layer_id=layer_id,
                kind=kind,
                compress_ratio=compress,
                has_sliding_window=is_c4a and sliding_window > 0,
                has_indexer=is_c4a,
            )
        )
    return HeterogeneousKVMetadata(layer_specs=tuple(specs))
