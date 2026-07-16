"""Direct lowering of iota from tensor IR to NKI IR.

``iota`` produces an index ramp along ``dim``, broadcast over the other axes.
``emit_iota`` is the single implementation, emitting into an existing Builder
with pre-allocated HBM buffers.
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import Value
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    PARTITION_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    max_free_elems,
    unravel,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Allocator


def emit_iota(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    """Lower iota: an index ramp along ``dim``, broadcast over other axes.

    Tiled with a canonical row-major layout (last dim = free, penultimate =
    partition, earlier = batch).  ``nisa.iota`` produces, per SBUF tile,
    ``offset + p * channel_multiplier + f * step``.  We pick those so the
    value equals the global index along ``dim``:

      - dim is the free axis:      step = 1,  channel_multiplier = 0, offset = f_off
      - dim is the partition axis: step = 0,  channel_multiplier = 1, offset = p_off
      - dim is a batch axis:       constant per tile = batch index on that axis
    """
    out_val = op.results[0]
    dim = op.attrs["dim"]
    dst_hbm = hbm_map[out_val.name]
    dtype = out_val.type.dtype
    shape = out_val.type.shape
    rank = len(shape)

    tile_p = min(shape[-2], PARTITION_MAX) if rank >= 2 else 1
    f_extent = shape[-1] if rank >= 2 else shape[0]
    # Cap the free extent to the per-partition SBUF budget: a vocab-wide iota
    # (e.g. (1, 128256)) would otherwise allocate an oversized tile.
    tile_f = min(f_extent, max_free_elems(dtype))
    p_extent = shape[-2] if rank >= 2 else 1
    batch_dims = list(shape[:-2]) if rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    f_axis = rank - 1
    p_axis = rank - 2  # only meaningful when rank >= 2

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)
            for f_i in range(ceildiv(f_extent, tile_f)):
                f_off = f_i * tile_f
                f_size = min(tile_f, f_extent - f_off)

                if dim == f_axis:
                    pattern, ch_mul, offset = [[1, f_size]], 0, f_off
                elif rank >= 2 and dim == p_axis:
                    pattern, ch_mul, offset = [[0, f_size]], 1, p_off
                else:
                    # batch axis: every element in this tile shares the index
                    pattern, ch_mul, offset = [[0, f_size]], 0, int(batch_idx[dim])

                tile = alloc.sbuf((p_size, f_size), dtype)
                tile = nb.iota(tile, pattern=pattern, offset=offset, channel_multiplier=ch_mul)

                dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
                if rank >= 2:
                    dst_slices.append(DimSlice(p_off, p_size))
                dst_slices.append(DimSlice(f_off, f_size))
                nb.dma_copy(dst_hbm, tile, dst_slices)
