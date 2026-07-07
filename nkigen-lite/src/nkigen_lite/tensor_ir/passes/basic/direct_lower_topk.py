"""Direct lowering of topk to NKI IR via the hardware scan.

``topk`` lowers to the canonical hardware scan (``max8`` + ``match_replace8``):
each fold reads the next 8 largest values and masks them to -inf in place,
which also yields their indices. ``ceil(k/8)`` folds cover any ``k``. Rows are
independent, so the partition is tiled at PARTITION_MAX and each <=128-row
chunk runs the full scan. When the row width exceeds the max8 free-dim limit
the row is split into chunks, each scanned for its local top-k, then the chunk
winners are merged with a second scan.

Assembling the per-fold 8-wide results into one wide tile needs sub-tile column
writes that nki_ir lacks, so the folds round-trip through HBM scratch
(``_first_cols`` / ``_overlay_columns``).

``emit_topk`` is the single implementation, emitting into an existing Builder
with pre-allocated HBM buffers.
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

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import ceildiv


# max8 / match_replace8 read at most this many free elements per call (a real
# hardware limit the MLIR verifier enforces).  Wider rows are tiled.
TOPK_FREE_MAX = 16384


def _aligned(k: int) -> int:
    """Fold-aligned width ceil(k/8)*8 (the column count _topk_scan returns)."""
    return ((k + 7) // 8) * 8


def _topk_scan(nb: Builder, data: Value, P: int, k: int, vdtype, idtype):
    """Run the max8 + match_replace8 scan over a resident (P, W>=8) SBUF tile.

    Returns ``(vals_sbuf (P, kp), idx_sbuf (P, kp))`` where ``kp = ceil(k/8)*8``
    (the fold-aligned width); callers slice the first ``k`` columns.  Indices
    are positions within ``data``.  Assembling the per-fold 8-wide results into
    one wide SBUF tile needs sub-tile column writes, which nki_ir lacks, so the
    folds round-trip through an HBM scratch buffer.
    """
    n_fold = (k + 7) // 8
    kp = n_fold * 8
    if n_fold == 1:
        val8 = nb.max8(nb.alloc((P, 8), vdtype, MemorySpace.SBUF), data)
        idx8 = nb.alloc((P, 8), idtype, MemorySpace.SBUF)
        _, idx8 = nb.match_replace8(data, idx8, data, val8, float("-inf"))
        return val8, idx8
    val_scratch = nb.alloc((P, kp), vdtype, MemorySpace.HBM)
    idx_scratch = nb.alloc((P, kp), idtype, MemorySpace.HBM)
    for fold in range(n_fold):
        val8 = nb.max8(nb.alloc((P, 8), vdtype, MemorySpace.SBUF), data)
        idx8 = nb.alloc((P, 8), idtype, MemorySpace.SBUF)
        data, idx8 = nb.match_replace8(data, idx8, data, val8, float("-inf"))
        col = DimSlice(fold * 8, 8)
        nb.dma_copy(val_scratch, val8, (DimSlice(0, P), col))
        nb.dma_copy(idx_scratch, idx8, (DimSlice(0, P), col))
    vals = nb.dma_copy(nb.alloc((P, kp), vdtype, MemorySpace.SBUF),
                       val_scratch, (DimSlice(0, P), DimSlice(0, kp)))
    idxs = nb.dma_copy(nb.alloc((P, kp), idtype, MemorySpace.SBUF),
                       idx_scratch, (DimSlice(0, P), DimSlice(0, kp)))
    return vals, idxs


def _load_topk_data(nb: Builder, src_hbm, p_off: int, p_size: int,
                    off: int, f: int, vdtype):
    """Load src_hbm[p_off:p_off+p_size, off:off+f] into a (p_size, max(f,8))
    SBUF tile, padding the tail with -inf so max8's >=8 free-dim requirement
    always holds."""
    width = max(f, 8)
    if width == f:
        return nb.dma_copy(nb.alloc((p_size, width), vdtype, MemorySpace.SBUF),
                           src_hbm, (DimSlice(p_off, p_size), DimSlice(off, f)))
    padded = nb.memset(nb.alloc((p_size, width), vdtype, MemorySpace.SBUF), float("-inf"))
    loaded = nb.dma_copy(nb.alloc((p_size, f), vdtype, MemorySpace.SBUF),
                         src_hbm, (DimSlice(p_off, p_size), DimSlice(off, f)))
    return _overlay_columns(nb, padded, loaded, f)


def emit_topk(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower topk via the canonical hardware scan (max8 + match_replace8).

    Rows are independent, so the partition is tiled at PARTITION_MAX and each
    chunk of <=128 rows runs the full scan (a taller source previously
    allocated an over-wide (P, .) tile). Per chunk: the source (p, F) tile is
    loaded into SBUF; each fold reads the next 8 largest (max8) and masks
    them to -inf in place (match_replace8), which also yields their indices.
    ceil(k/8) folds cover any k.

    When F exceeds the max8 free-dim limit, the row is split into chunks of
    <= TOPK_FREE_MAX: each chunk is scanned for its local top-k (indices
    rebased to global), then the chunk winners (n_chunks * k candidates) are
    merged with a second scan, and a gather maps the merged positions back to
    the global indices.
    """
    src_val = op.inputs[0]
    val_out, idx_out = op.results[0], op.results[1]
    P, F = src_val.type.shape
    k = op.attrs["k"]
    src_hbm = hbm_map[src_val.name]
    val_hbm = hbm_map[val_out.name]
    idx_hbm = hbm_map[idx_out.name]
    vdtype = val_out.type.dtype
    idtype = idx_out.type.dtype

    for p_i in range(ceildiv(P, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, P - p_off)
        _emit_topk_rows(nb, src_hbm, val_hbm, idx_hbm, p_off, p_size, F, k,
                        vdtype, idtype)


def _emit_topk_rows(nb: Builder, src_hbm, val_hbm, idx_hbm, p_off: int,
                    p_size: int, F: int, k: int, vdtype, idtype) -> None:
    """Emit the topk scan for one partition chunk of <=128 rows."""
    out_rows = DimSlice(p_off, p_size)

    if F <= TOPK_FREE_MAX:
        data = _load_topk_data(nb, src_hbm, p_off, p_size, 0, F, vdtype)
        vals, idxs = _topk_scan(nb, data, p_size, k, vdtype, idtype)
        v_store = vals if k == _aligned(k) else _first_cols(nb, vals, k)
        i_store = idxs if k == _aligned(k) else _first_cols(nb, idxs, k)
        nb.dma_copy(val_hbm, v_store, (out_rows, DimSlice(0, k)))
        nb.dma_copy(idx_hbm, i_store, (out_rows, DimSlice(0, k)))
        return

    n_chunks = ceildiv(F, TOPK_FREE_MAX)
    cand = n_chunks * k
    if cand > TOPK_FREE_MAX:
        raise NotImplementedError(
            f"topk: F={F}, k={k} needs {cand} merge candidates, exceeds "
            f"{TOPK_FREE_MAX}; a multi-level merge is not implemented"
        )

    # Per-chunk local top-k, with indices rebased to global, gathered into
    # candidate buffers (p_size, cand).
    cand_vals_hbm = nb.alloc((p_size, cand), vdtype, MemorySpace.HBM)
    cand_idx_hbm = nb.alloc((p_size, cand), idtype, MemorySpace.HBM)
    for c in range(n_chunks):
        off = c * TOPK_FREE_MAX
        fc = min(TOPK_FREE_MAX, F - off)
        data = _load_topk_data(nb, src_hbm, p_off, p_size, off, fc, vdtype)
        vals, idxs = _topk_scan(nb, data, p_size, k, vdtype, idtype)
        if off != 0:
            # Rebase local indices to global: idx += off.
            off_tile = nb.constant(float(off), (p_size, 1), idtype, MemorySpace.SBUF)
            idxs = nb.tensor_scalar_arith(
                nb.alloc(idxs.type.shape, idtype, MemorySpace.SBUF),
                idxs, off_tile, nki_ir.NisaArithOp.ADD,
            )
        col = DimSlice(c * k, k)
        v_store = vals if k == _aligned(k) else _first_cols(nb, vals, k)
        i_store = idxs if k == _aligned(k) else _first_cols(nb, idxs, k)
        nb.dma_copy(cand_vals_hbm, v_store, (DimSlice(0, p_size), col))
        nb.dma_copy(cand_idx_hbm, i_store, (DimSlice(0, p_size), col))

    # Merge: scan the candidate values for the global top-k, then gather the
    # corresponding global indices by the merged positions.
    cand_data = _load_topk_data(nb, cand_vals_hbm, 0, p_size, 0, cand, vdtype)
    width = max(cand, 8)
    cand_idx_sbuf = nb.memset(
        nb.alloc((p_size, width), idtype, MemorySpace.SBUF), 0.0)
    cand_idx_sbuf = _overlay_columns(
        nb,
        cand_idx_sbuf,
        nb.dma_copy(nb.alloc((p_size, cand), idtype, MemorySpace.SBUF),
                    cand_idx_hbm, (DimSlice(0, p_size), DimSlice(0, cand))),
        cand,
    )
    mvals, mpos = _topk_scan(nb, cand_data, p_size, k, vdtype, idtype)
    # gather: global_idx[p, i] = cand_idx_sbuf[p, mpos[p, i]]
    gidx = nb.gather(
        nb.alloc(mpos.type.shape, idtype, MemorySpace.SBUF),
        cand_idx_sbuf, mpos,
    )
    v_store = mvals if k == _aligned(k) else _first_cols(nb, mvals, k)
    i_store = gidx if k == _aligned(k) else _first_cols(nb, gidx, k)
    nb.dma_copy(val_hbm, v_store, (out_rows, DimSlice(0, k)))
    nb.dma_copy(idx_hbm, i_store, (out_rows, DimSlice(0, k)))


def _first_cols(nb: Builder, tile: Value, keep: int) -> Value:
    """Return a (P, keep) SBUF tile holding the first ``keep`` columns of
    ``tile``, via an HBM scratch round-trip (nki_ir has no SBUF sub-view).

    The scratch is sized to the tile's full width: callers pass fold-aligned
    ``kp = ceil(k/8)*8``-wide tiles, so ``keep`` can exceed 8 (e.g. k=12 on a
    16-wide tile) — a hard-coded 8-wide scratch would leave columns 8..keep-1
    reading past the buffer.
    """
    P, W = tile.type.shape
    scratch = nb.alloc((P, W), tile.type.dtype, MemorySpace.HBM)
    nb.dma_copy(scratch, tile, (DimSlice(0, P), DimSlice(0, W)))
    return nb.dma_copy(
        nb.alloc((P, keep), tile.type.dtype, MemorySpace.SBUF),
        scratch, (DimSlice(0, P), DimSlice(0, keep)),
    )


def _overlay_columns(nb: Builder, base: Value, cols: Value, n: int) -> Value:
    """Write the first ``n`` columns of ``cols`` over ``base`` (P, W>=n) via an
    HBM scratch round-trip, returning the merged SBUF tile."""
    P, W = base.type.shape
    scratch = nb.alloc((P, W), base.type.dtype, MemorySpace.HBM)
    nb.dma_copy(scratch, base, (DimSlice(0, P), DimSlice(0, W)))
    nb.dma_copy(scratch, cols, (DimSlice(0, P), DimSlice(0, n)))
    return nb.dma_copy(
        nb.alloc((P, W), base.type.dtype, MemorySpace.SBUF),
        scratch, (DimSlice(0, P), DimSlice(0, W)),
    )
