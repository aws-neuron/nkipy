from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ForwardMode(str, Enum):
    EXTEND = "extend"
    DECODE = "decode"


@dataclass(frozen=True)
class ForwardBatch:
    """Batch contract for a single forward step.

    Contains flattened token data and page-table indirection for
    the attention backend.
    """

    forward_mode: ForwardMode
    batch_size: int
    input_ids: np.ndarray  # [total_tokens]
    positions: np.ndarray  # [total_tokens]
    seq_lens: np.ndarray  # [batch_size]
    slot_mapping: np.ndarray  # [total_tokens] → cache slot indices
    block_tables: np.ndarray  # [batch_size, max_blocks] → block IDs
    query_start_loc: np.ndarray  # [batch_size + 1] cumulative query offsets
    sample_mask: (
        np.ndarray
    )  # [batch_size] whether this request should emit a sampled token this step
    requested_topk: int = 1  # distributed nkipy local candidate width (1 = top-1 only)
    token_bucket: int = 0  # padded token dimension (0 = no padding)
    real_total_tokens: int = 0  # actual token count before padding (0 = no padding)
    use_full_sampler: bool = (
        False  # use all-gather + full-vocab sampler instead of greedy local top-k
    )
    needs_logprobs: bool = False  # any request in batch wants logprobs
    logprobs_k: int = 0  # max logprobs top-k across requests (0 = disabled)
    temperatures: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # [batch_size]
    top_ks: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [batch_size]
    top_ps: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # [batch_size]
    min_ps: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # [batch_size]
    uniform_u: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # [batch_size]
    # Stable request-owned state index. For DSV4 this is the scheduler
    # req_pool_idx, not the transient batch row index.
    state_owner_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [batch_size]
    # DP-attention lane assignment. -1 means single-lane; DSV4 sets this to
    # ScheduleBatch.attention_lane (0..attention_dp_degree-1). Workers whose
    # row matches this lane run the real forward; other-lane workers no-op.
    attention_lane: int = -1
    # DSV4 DP-attention superstep metadata. When true, lane_* arrays describe
    # the replica-local global token/request layout used to gather lane-local
    # attention outputs before MoE/FFN and scatter back afterward.
    dp_attention_superstep: bool = False
    dp_attention_num_lanes: int = 1
    dp_attention_lane_token_counts: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [num_lanes]
    dp_attention_lane_batch_sizes: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [num_lanes]
    dp_attention_lane_token_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [num_lanes + 1]
    dp_attention_lane_batch_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )  # [num_lanes + 1]

    def __post_init__(self) -> None:
        bs = int(self.batch_size)
        _defaults: dict[str, np.ndarray] = {
            "temperatures": np.ones((bs,), dtype=np.float32),
            "top_ks": np.ones((bs,), dtype=np.int32),
            "top_ps": np.ones((bs,), dtype=np.float32),
            "min_ps": np.zeros((bs,), dtype=np.float32),
            "uniform_u": np.zeros((bs,), dtype=np.float32),
            "state_owner_ids": np.arange(bs, dtype=np.int32),
        }
        for name, default in _defaults.items():
            if int(getattr(self, name).shape[0]) == 0:
                object.__setattr__(self, name, default)
        lanes = max(1, int(self.dp_attention_num_lanes))
        object.__setattr__(self, "dp_attention_num_lanes", lanes)
        dp_defaults: dict[str, np.ndarray] = {
            "dp_attention_lane_token_counts": np.zeros((lanes,), dtype=np.int32),
            "dp_attention_lane_batch_sizes": np.zeros((lanes,), dtype=np.int32),
            "dp_attention_lane_token_offsets": np.zeros((lanes + 1,), dtype=np.int32),
            "dp_attention_lane_batch_offsets": np.zeros((lanes + 1,), dtype=np.int32),
        }
        for name, default in dp_defaults.items():
            arr = np.asarray(getattr(self, name))
            if int(arr.shape[0]) == 0:
                object.__setattr__(self, name, default)
            else:
                object.__setattr__(self, name, arr.astype(np.int32, copy=False))
        if bool(self.dp_attention_superstep):
            if self.attention_lane != -1:
                raise ValueError(
                    "DP-attention supersteps must not name a single attention_lane"
                )
            for name in (
                "dp_attention_lane_token_counts",
                "dp_attention_lane_batch_sizes",
            ):
                if tuple(getattr(self, name).shape) != (lanes,):
                    raise ValueError(
                        f"{name} must have shape ({lanes},), got "
                        f"{getattr(self, name).shape}"
                    )
            for name in (
                "dp_attention_lane_token_offsets",
                "dp_attention_lane_batch_offsets",
            ):
                if tuple(getattr(self, name).shape) != (lanes + 1,):
                    raise ValueError(
                        f"{name} must have shape ({lanes + 1},), got "
                        f"{getattr(self, name).shape}"
                    )

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def max_seq_len(self) -> int:
        return int(np.max(self.seq_lens)) if self.batch_size > 0 else 0
