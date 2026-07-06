"""
Layout: three-way I/P/F dim classification for SBUF tiles.

  I (iteration): loop indices, bare-int DMA, not in SBUF tile
  P (partition): SBUF dim-0, computes in parallel
  F (free):      SBUF dim-1, contiguous per partition

Whole dims are assigned to groups; dims are never split or reordered within
a group (each group keeps ascending dim order). A P-extent larger than the
hardware partition count (128) is legal here — downstream tiling
(compute_tile_sizes) chunks the P group to fit; default_layout's scoring
merely stops rewarding P-extent beyond 128.

Layouts are per-segment decisions, not per-value graph properties: every
segment boundary round-trips through HBM, which is layout-agnostic
(row-major contiguous), so a consumer segment can load a value in any layout
regardless of how the producer stored it. Each emitter picks the layout for
its own tile loop — matmul's is fixed by the tensor engine
(get_matmul_layouts), reduce classifies its axes against default_layout of
its input, elementwise uses canonical row-major. A global cross-value solver
becomes meaningful only when values stay SBUF-resident across segments
(explicitly deferred; see SBUF_FUSION_PLAN.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nkigen_lite.tensor_ir.passes.hardware import TRN2

PARTITION_MAX = TRN2.partition_max


@dataclass(frozen=True)
class Layout:
    """Layout assignment for a tensor: three-way I/P/F classification.

    i_dims — iteration dims (bare-int loop indices, not in SBUF tile)
    p_dims — partition dims (SBUF dim-0; extent may exceed 128, tiling
             chunks it to the hardware partition count downstream)
    f_dims — free dims (SBUF dim-1, contiguous per partition)

    Invariant: dims within each group are always sorted ascending (row-major
    canonical form). Two Layouts with the same I/P/F group membership are equal
    regardless of the order in which dims were originally specified.
    """

    i_dims: tuple[int, ...]
    p_dims: tuple[int, ...]
    f_dims: tuple[int, ...]

    def __post_init__(self):
        # Normalize: sort dims within each group to enforce canonical form.
        object.__setattr__(self, "i_dims", tuple(sorted(self.i_dims)))
        object.__setattr__(self, "p_dims", tuple(sorted(self.p_dims)))
        object.__setattr__(self, "f_dims", tuple(sorted(self.f_dims)))
        # Validate: no dim appears in multiple groups.
        all_dims = self.i_dims + self.p_dims + self.f_dims
        if len(all_dims) != len(set(all_dims)):
            raise ValueError(
                f"Layout has overlapping groups: I={self.i_dims}, P={self.p_dims}, F={self.f_dims}"
            )

    def p_extent(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[d] for d in self.p_dims) if self.p_dims else 1

    def f_extent(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[d] for d in self.f_dims) if self.f_dims else 1

    def is_valid(self, shape: tuple[int, ...]) -> bool:
        all_dims = self.i_dims + self.p_dims + self.f_dims
        if len(all_dims) != len(shape):
            return False
        if set(all_dims) != set(range(len(shape))):
            return False
        return True


def get_matmul_layouts(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
) -> tuple[Layout, Layout, Layout]:
    """The tensor engine's fixed layouts for matmul operands.

    A[..., M, K] @ B[..., K, N] → C[..., M, N]
    Stationary A: I=batch, P={K}, F={M}
    Moving B:     I=batch, P={K}, F={N}
    Output C:     I=batch, P={M}, F={N}

    This is the hard constraint a matmul anchor imposes on its segment's tile
    loop (emit_matmul embodies it in its tiling; epilogue fusion consumers
    must accept the C layout).
    """
    a_rank = len(a_shape)
    b_rank = len(b_shape)
    c_rank = len(c_shape)

    a_m_idx = a_rank - 2
    a_k_idx = a_rank - 1
    b_k_idx = b_rank - 2
    b_n_idx = b_rank - 1
    c_m_idx = c_rank - 2
    c_n_idx = c_rank - 1

    a_batch = tuple(range(a_rank - 2))
    b_batch = tuple(range(b_rank - 2))
    c_batch = tuple(range(c_rank - 2))

    a_layout = Layout(i_dims=a_batch, p_dims=(a_k_idx,), f_dims=(a_m_idx,))
    b_layout = Layout(i_dims=b_batch, p_dims=(b_k_idx,), f_dims=(b_n_idx,))
    c_layout = Layout(i_dims=c_batch, p_dims=(c_m_idx,), f_dims=(c_n_idx,))

    return a_layout, b_layout, c_layout


def default_layout(shape: tuple[int, ...]) -> Layout:
    """Pick a layout for a tensor using contiguous I|P|F splits.

    Scoring: (1 + K / f_extent) / utilization.
    Lower is better. Prefers layouts that maximize both partition
    utilization and f-extent (amortizing the per-iteration overhead K).
    """
    rank = len(shape)
    if rank == 0:
        return Layout(i_dims=(), p_dims=(), f_dims=())
    if rank == 1:
        return Layout(i_dims=(), p_dims=(), f_dims=(0,))
    if rank == 2:
        return Layout(i_dims=(), p_dims=(0,), f_dims=(1,))

    K = 1024  # per-iteration overhead in element-equivalents

    def _score(layout: Layout) -> float:
        p_ext = layout.p_extent(shape) if layout.p_dims else 1
        f_ext = layout.f_extent(shape) if layout.f_dims else 1
        util = min(p_ext, PARTITION_MAX) / PARTITION_MAX
        return (1.0 + K / f_ext) / util

    best_layout = Layout(i_dims=(), p_dims=(0,), f_dims=tuple(range(1, rank)))
    best_score = _score(best_layout)

    # Enumerate contiguous splits: dims [0:i_end) = I, [i_end:f_start) = P, [f_start:rank) = F
    for i_end in range(rank):
        for f_start in range(i_end + 1, rank):
            layout = Layout(
                i_dims=tuple(range(i_end)),
                p_dims=tuple(range(i_end, f_start)),
                f_dims=tuple(range(f_start, rank)),
            )
            s = _score(layout)
            if s < best_score:
                best_score = s
                best_layout = layout

    return best_layout
