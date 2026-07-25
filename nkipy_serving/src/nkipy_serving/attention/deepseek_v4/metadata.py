"""Scheduler interface for DSV4 sparse attention.

``SparseAttentionMetadata`` carries the per-(token, layer-type) fields that
the sparse-attention kernels consume:

- ``topk_indices [total_tokens, K_max]`` int32, -1 = invalid.
- ``topk_lens    [total_tokens]`` int32, true length per query (for
  diagnostics; kernel relies on the -1 sentinels).
- ``index_space`` declares what the integers index. The DSV4 product backend
  uses ``"global_slots"``: indices are already flat KV-cache slots.
- ``attention_sink [h]`` fp32 — per-layer weight; passed into the kernel
  directly by the layer call site, not carried in metadata.
- Per-layer ``compress_ratio`` (1 = SWA-only, 4 = C4A, 128 = C128A) is
  the field the builder uses to decide which top-k set to union for each
  token. Same taxonomy as the vLLM DSV4 PR's ``_LAYER_TYPE_*``.

Design note — why a separate struct instead of extending
``AttentionMetadata``:

- ``AttentionMetadata`` is consumed by multiple attention backends
  (Qwen, GPT-OSS, the paged-block-sparse backend). Adding DSV4-specific
  fields there would bleed V4 concerns across models.
- Lane-aware scheduling already threads ``attention_lane`` on
  ``ForwardBatch``; this struct piggybacks on that, so the only
  scheduler-level addition is "carry one extra dataclass alongside".

Builder responsibilities:

1. **SWA indices**: for each query token ``i``, pick the last
   ``min(seq_len_i, window_size)`` KV slots. Produces
   ``indices_swa [total_tokens, window_size]``.
2. **Compressed indices** (C4A, C128A): for each query token with
   ``compress_ratio != 1``, select the top-k compressed KV positions via
   the indexer (C4A) or dense gating (C128A). Produces
   ``indices_compressed [total_tokens, index_topk]``.
3. **Union + pad**: concatenate per-token along the K axis, pad to
   ``K_max = window_size + index_topk`` with ``-1``.
4. Populate ``SparseAttentionMetadata(topk_indices, topk_lens, ...)`` and pass
   it into the DSV4 attention backend used by the executor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPARSE_INDEX_SPACE_GLOBAL_SLOTS = "global_slots"
_VALID_INDEX_SPACES = {SPARSE_INDEX_SPACE_GLOBAL_SLOTS}


@dataclass(frozen=True)
class SparseAttentionMetadata:
    """Per-step sparse attention metadata.

    ``topk_indices`` is the contract with the kernel. All other fields are
    either bookkeeping or shape guarantees.
    """

    # Core sparse selection. Shape ``[total_tokens, K_max]``.
    topk_indices: np.ndarray
    # Optional true-length vector, ``[total_tokens]``. -1 sentinels already
    # encode invalids; this is for observability and for the compressor
    # builder to know how many positions it actually selected.
    topk_lens: np.ndarray
    # Per-layer compress ratio (1 = SWA-only, 4 = C4A, 128 = C128A). None =
    # model hasn't set it yet (e.g. tests).
    compress_ratio: int = 1
    # Absolute bounds - helpful for kernel shape validation.
    num_kv_positions: int = 0
    window_size: int = 0
    index_topk: int = 0
    # ``global_slots`` is the product contract.
    index_space: str = SPARSE_INDEX_SPACE_GLOBAL_SLOTS

    def __post_init__(self) -> None:
        if self.topk_indices.ndim != 2:
            raise ValueError(
                f"topk_indices must be 2D, got shape {self.topk_indices.shape}"
            )
        if self.topk_indices.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"topk_indices must be int32 or int64, got {self.topk_indices.dtype}"
            )
        if self.topk_lens.shape != (self.topk_indices.shape[0],):
            raise ValueError(
                f"topk_lens must be [{self.topk_indices.shape[0]}], "
                f"got {self.topk_lens.shape}"
            )
        if self.index_space not in _VALID_INDEX_SPACES:
            raise ValueError(
                f"index_space must be one of {sorted(_VALID_INDEX_SPACES)}, "
                f"got {self.index_space!r}"
            )

    @property
    def total_tokens(self) -> int:
        return int(self.topk_indices.shape[0])

    @property
    def k_max(self) -> int:
        return int(self.topk_indices.shape[1])
