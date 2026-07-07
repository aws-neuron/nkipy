"""Direct lowering of indexed data-movement ops (gather_along_axis,
scatter_rows, gather_rows) to NKI IR.

All three use the hardware indirect DMA (``dma_copy_indirect``) and/or the
per-partition ``nisa.gather``:

  - gather_along_axis: ``out[p, i] = data[p, idx[p, i]]`` via ``nisa.gather``,
    tiled by partition chunk.
  - scatter_rows: ``out = base.copy(); out[idx[r]] = updates[r]`` via an
    indirect DMA store.
  - gather_rows: ``out[r] = src[idx[r]]`` via an indirect DMA load, with a
    partition-packed fast path for wide-but-few rows (MoE expert weights).

``emit_gather_along_axis`` / ``emit_scatter_rows`` / ``emit_gather_rows`` are
the single implementations, emitting into an existing Builder with
pre-allocated HBM buffers.
"""

from __future__ import annotations

from nkigen_lite.core import Value
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
)
from nkigen_lite.nki_ir import ir as nki_ir

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    max_free_elems,
)


def emit_gather_along_axis(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower gather_along_axis via the hardware per-partition gather.

    ``out[p, i] = data[p, idx[p, i]]``.  Each partition chunk (up to
    PARTITION_MAX rows) loads its data and index rows into SBUF, runs
    ``nisa.gather``, and stores the gathered row back to HBM.  The free
    dims of data and idx differ (F_data vs F_idx); the gather dst matches
    the idx shape.
    """
    data_val, idx_val = op.inputs[0], op.inputs[1]
    out_val = op.results[0]
    P, F_data = data_val.type.shape
    F_idx = idx_val.type.shape[1]
    data_hbm = hbm_map[data_val.name]
    idx_hbm = hbm_map[idx_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    for p_i in range(ceildiv(P, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, P - p_off)

        data_tile = nb.dma_copy(
            nb.alloc((p_size, F_data), vdtype, MemorySpace.SBUF),
            data_hbm, (DimSlice(p_off, p_size), DimSlice(0, F_data)),
        )
        idx_tile = nb.dma_copy(
            nb.alloc((p_size, F_idx), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(p_off, p_size), DimSlice(0, F_idx)),
        )
        out_tile = nb.gather(
            nb.alloc((p_size, F_idx), vdtype, MemorySpace.SBUF),
            data_tile, idx_tile,
        )
        nb.dma_copy(out_hbm, out_tile, (DimSlice(p_off, p_size), DimSlice(0, F_idx)))


def emit_scatter_rows(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower scatter_rows: ``out = base.copy(); out[idx[r], :] = updates[r, :]``.

    First copy ``base`` HBM -> result HBM (tiled by N rows, the unchanged
    backdrop), then scatter the M update rows into the result via the indirect
    DMA store (``dma_copy_indirect``), tiled by M update rows.  The index tile
    is (m_size, 1) U32: 1-D SBUF index tiles are rejected by the hardware.
    """
    base_val, idx_val, upd_val = op.inputs[0], op.inputs[1], op.inputs[2]
    out_val = op.results[0]
    N, W = base_val.type.shape
    M = upd_val.type.shape[0]
    base_hbm = hbm_map[base_val.name]
    idx_hbm = hbm_map[idx_val.name]
    upd_hbm = hbm_map[upd_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    # Tile the row width so wide rows never need a (P, W) SBUF tile.
    w_tile = min(W, max_free_elems(vdtype))

    # Backdrop: copy base -> result, tiled over N rows and W columns.
    for p_i in range(ceildiv(N, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, N - p_off)
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            tile = nb.dma_copy(
                nb.alloc((p_size, w_size), vdtype, MemorySpace.SBUF),
                base_hbm, (DimSlice(p_off, p_size), DimSlice(w_off, w_size)),
            )
            nb.dma_copy(
                out_hbm, tile, (DimSlice(p_off, p_size), DimSlice(w_off, w_size))
            )

    # Scatter the M update rows, tiled over M and W.  dma_copy_indirect
    # addresses whole rows of the result HBM tensor via the per-row index;
    # row_width keeps the full row stride while a column window is written.
    for m_i in range(ceildiv(M, PARTITION_MAX)):
        m_off = m_i * PARTITION_MAX
        m_size = min(PARTITION_MAX, M - m_off)
        idx_tile = nb.dma_copy(
            nb.alloc((m_size, 1), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(m_off, m_size), DimSlice(0, 1)),
        )
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            upd_tile = nb.dma_copy(
                nb.alloc((m_size, w_size), vdtype, MemorySpace.SBUF),
                upd_hbm, (DimSlice(m_off, m_size), DimSlice(w_off, w_size)),
            )
            nb.dma_copy_indirect(
                out_hbm, upd_tile, idx_tile, row_width=W, free_offset=w_off,
            )


def _emit_gather_rows_packed(
    nb: Builder, src_hbm, idx_hbm, out_hbm, N: int, W: int, M: int,
    vdtype, idtype,
) -> None:
    """Gather wide rows with the partition packed.

    Views the (N, W) table as (N*128, W/128): row ``i`` of the original table
    becomes the 128 consecutive sub-rows ``i*128 + lane``. Per output row, the
    dynamic index is fanned across the partition (stride-0 load), scaled by
    128, and offset by a per-lane iota; one indirect DMA then fetches the whole
    row as a (128, W/128) tile. 9 ops per row vs ``2 + 3*ceil(W/w_tile)`` for
    the column-window path (~212 for an MoE expert weight).
    """
    sub_w = W // PARTITION_MAX
    src_sub = nb.view(src_hbm, (N * PARTITION_MAX, sub_w))
    out_sub = nb.view(out_hbm, (M * PARTITION_MAX, sub_w))

    # Per-lane offset (p) and the scale constant, hoisted across rows.
    lane = nb.iota(
        nb.alloc((PARTITION_MAX, 1), idtype, MemorySpace.SBUF),
        pattern=[[0, 1]], offset=0, channel_multiplier=1,
    )
    scale = nb.constant(
        float(PARTITION_MAX), (PARTITION_MAX, 1), idtype, MemorySpace.SBUF)

    for r in range(M):
        # idx[r] fanned across all 128 lanes, then sub_idx = idx*128 + lane.
        idx_rep = nb.dma_copy(
            nb.alloc((PARTITION_MAX, 1), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(r, PARTITION_MAX, stride=0), DimSlice(0, 1)),
        )
        scaled = nb.tensor_scalar_arith(
            nb.alloc((PARTITION_MAX, 1), idtype, MemorySpace.SBUF),
            idx_rep, scale, nki_ir.NisaArithOp.MULTIPLY,
        )
        sub_idx = nb.tensor_tensor_arith(
            nb.alloc((PARTITION_MAX, 1), idtype, MemorySpace.SBUF),
            scaled, lane, nki_ir.NisaArithOp.ADD,
        )
        row_tile = nb.dma_copy_indirect(
            nb.alloc((PARTITION_MAX, sub_w), vdtype, MemorySpace.SBUF),
            src_sub, sub_idx,
        )
        nb.dma_copy(
            out_sub, row_tile,
            (DimSlice(r * PARTITION_MAX, PARTITION_MAX), DimSlice(0, sub_w)),
        )


def emit_gather_rows(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower gather_rows: ``out[r, :] = src[idx[r], :]`` via the indirect DMA
    load (``dma_copy_indirect``), gathering whole rows from the (N, W) src HBM
    tensor into the (M, W) result.  Tiled over M gathered rows.  The index tile
    is (m_size, 1) U32.  Avoids materializing the full (N, W) table on chip, so
    it scales to tall tables (e.g. embedding (128256, 2048))."""
    src_val, idx_val = op.inputs[0], op.inputs[1]
    out_val = op.results[0]
    N, W = src_val.type.shape
    M = out_val.type.shape[0]
    src_hbm = hbm_map[src_val.name]
    idx_hbm = hbm_map[idx_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    # Fast path: wide rows, few of them (the MoE expert-weight gather is M=1,
    # W=786432). The generic path below windows each row into ceil(W/w_tile)
    # column DMAs that use ONE of the 128 partition lanes each (~200+ ops per
    # expert weight). A gathered row is contiguous in HBM, so instead view the
    # table as (N*128, W/128), expand the dynamic index to the row's 128
    # sub-rows (idx*128 + lane, via iota), and fetch the whole row with a
    # single partition-packed indirect DMA (~12 ops). Only when it actually
    # wins: packed emits ~9 ops per output row while the generic path amortizes
    # its windows over 128-row chunks, so tall gathers keep the generic path.
    w_tile = min(W, max_free_elems(vdtype))
    n_windows = ceildiv(W, w_tile)
    packed_cost = 6 + 9 * M
    generic_cost = ceildiv(M, PARTITION_MAX) * (2 + 3 * n_windows)
    if (
        W % PARTITION_MAX == 0
        and W // PARTITION_MAX <= max_free_elems(vdtype)
        and packed_cost < generic_cost
    ):
        _emit_gather_rows_packed(
            nb, src_hbm, idx_hbm, out_hbm, N, W, M, vdtype, idtype)
        return

    # Tile the row width: a single (m_size, W) tile would overflow SBUF for
    # wide rows (e.g. an MoE expert's flattened weight, W=786432). Gather a
    # column window [w_off, w_off+w_size) per DMA, addressing the full row
    # stride W via row_width so the index still selects whole source rows.
    for m_i in range(ceildiv(M, PARTITION_MAX)):
        m_off = m_i * PARTITION_MAX
        m_size = min(PARTITION_MAX, M - m_off)
        idx_tile = nb.dma_copy(
            nb.alloc((m_size, 1), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(m_off, m_size), DimSlice(0, 1)),
        )
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            # Indirect load: gather a column window of m_size rows of src.
            out_tile = nb.dma_copy_indirect(
                nb.alloc((m_size, w_size), vdtype, MemorySpace.SBUF),
                src_hbm, idx_tile, row_width=W, free_offset=w_off,
            )
            nb.dma_copy(
                out_hbm, out_tile, (DimSlice(m_off, m_size), DimSlice(w_off, w_size))
            )
