"""Attention metadata — flat dataclass for traceability (no ABC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Forward mode constants (int for traceability).
FORWARD_MODE_EXTEND = 0
FORWARD_MODE_DECODE = 1


@dataclass(frozen=True)
class AttentionMetadata:
    """Flat metadata consumed by attention kernels.

    All fields are plain ints or numpy arrays so the struct can be
    plumbed through nkipy tracing without opaque Python objects.
    """

    forward_mode: int  # 0=EXTEND, 1=DECODE
    seq_lens: np.ndarray  # [batch_size]
    slot_mapping: np.ndarray  # [total_tokens] → cache slot indices
    block_tables: np.ndarray  # [batch_size, max_blocks] → block IDs
    query_start_loc: np.ndarray  # [batch_size + 1] cumulative query offsets
    total_tokens: int
    batch_size: int
    max_seq_len: int
    num_kv_heads: int
    head_dim: int
    block_size: int
