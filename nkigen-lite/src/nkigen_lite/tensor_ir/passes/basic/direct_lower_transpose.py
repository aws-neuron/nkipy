"""Direct lowering of transpose from tensor IR to NKI IR.

Supports arbitrary permutations on any-rank tensors (rank >= 2). Two strategies:

  - DMA transpose: batch dims are reordered via DMA slice remapping. When the
    permutation swaps the last two dims (P↔F), dma_transpose handles the
    on-chip swap. When no P↔F swap is needed, a plain DMA copy suffices.

  - Tensor engine transpose: same batch-dim remapping, but for the P↔F swap
    uses the matmul trick: stat[K, M].T @ I[K, N] materializes the transpose.
    Only needed when the permutation swaps the last two dims.

For any permutation perm, the output shape is [in_shape[perm[i]] for i in range(rank)].
The key observation: on NeuronCore, only the last two dims are "on-chip" (P and F).
Batch dim reordering is just DMA slice coordinate remapping (reading from different
positions in HBM). The only operation that requires on-chip work is swapping P↔F.

Optimization: adjacent dims that stay consecutive under the permutation are merged
into a single axis before tiling (_collapse_perm). This dramatically reduces the
number of batch iterations for cases like (Co, Ci, *K) -> (Co, *K, Ci) where the
spatial dims K form a single contiguous run in the output.
"""

from __future__ import annotations

import math

from nkigen_lite.core import DType, Graph
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    flat_range_to_src_chunks,
    row_major_strides,
    unravel,
)


def _needs_pf_swap(perm: tuple[int, ...]) -> bool:
    """Check if the permutation swaps the relative order of the last two source dims.

    If the source dim that maps to output[-2] has a higher index than the one
    that maps to output[-1], the on-chip tile needs a P↔F transpose.
    """
    rank = len(perm)
    return perm[rank - 2] > perm[rank - 1]


def _collapse_perm(
    in_shape: tuple[int, ...], perm: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], list[list[int]], list[list[int]]]:
    """Merge runs of consecutive source dims that appear adjacent in perm.

    Returns (collapsed_in_shape, collapsed_perm, groups, src_order) where:
      - groups[k] = original source dim indices at collapsed output position k
      - src_order[j] = group of original dims merged into collapsed source dim j
      - collapsed_in[j] = product of original dim sizes for src_order[j]
      - collapsed_perm maps collapsed output positions to collapsed source positions
    """
    groups: list[list[int]] = [[perm[0]]]
    for j in range(1, len(perm)):
        if perm[j] == perm[j - 1] + 1:
            groups[-1].append(perm[j])
        else:
            groups.append([perm[j]])
    src_order = sorted(groups, key=lambda g: g[0])
    collapsed_in = tuple(math.prod(in_shape[a] for a in g) for g in src_order)
    pos = {tuple(g): i for i, g in enumerate(src_order)}
    collapsed_perm = tuple(pos[tuple(g)] for g in groups)
    return collapsed_in, collapsed_perm, groups, src_order


def _tile_iter(in_shape, perm, groups, c_out, c_rank, tile_p, tile_f):
    """Yield (src_slices, dst_slices, p_covered, f_covered) for each sub-tile.

    Handles axis-collapse: iterates over the collapsed output shape and expands
    each tile's coordinates back to original-rank DimSlices using
    flat_range_to_src_chunks for the P and F groups.
    """
    rank = len(in_shape)

    c_batch_dims = list(c_out[:-2])
    n_batch = math.prod(c_batch_dims) if c_batch_dims else 1
    n_p_tiles = ceildiv(c_out[-2], tile_p)
    n_f_tiles = ceildiv(c_out[-1], tile_f)

    p_group = groups[c_rank - 2]
    f_group = groups[c_rank - 1]
    p_sub_shape = tuple(in_shape[d] for d in p_group)
    f_sub_shape = tuple(in_shape[d] for d in f_group)
    p_sub_strides = row_major_strides(p_sub_shape)
    f_sub_strides = row_major_strides(f_sub_shape)

    # Output dim ranges for each group (maps collapsed output pos -> original output dims)
    out_dim_starts: list[int] = []
    pos = 0
    for g in groups:
        out_dim_starts.append(pos)
        pos += len(g)
    p_out_start = out_dim_starts[c_rank - 2]
    f_out_start = out_dim_starts[c_rank - 1]

    for batch_flat in range(n_batch):
        # Expand batch flat index to per-collapsed-batch-dim coords
        batch_coords: tuple[int, ...] = ()
        if c_batch_dims:
            remaining = batch_flat
            coords = []
            for d in reversed(c_batch_dims):
                coords.append(remaining % d)
                remaining //= d
            batch_coords = tuple(reversed(coords))

        # Build batch portion of src and dst slices
        batch_src: dict[int, DimSlice] = {}
        batch_dst: dict[int, DimSlice] = {}
        for k, coord in enumerate(batch_coords):
            group = groups[k]
            sub_shape = tuple(in_shape[d] for d in group)
            indices = unravel(coord, list(sub_shape))
            for i, d in enumerate(group):
                batch_src[d] = DimSlice(indices[i], 1)
            o_start = out_dim_starts[k]
            for i in range(len(group)):
                batch_dst[o_start + i] = DimSlice(indices[i], 1)

        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, c_out[-2] - p_off)
            p_chunks = flat_range_to_src_chunks(
                p_off, p_size, p_sub_shape, p_sub_strides
            )

            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, c_out[-1] - f_off)
                f_chunks = flat_range_to_src_chunks(
                    f_off, f_size, f_sub_shape, f_sub_strides
                )

                for p_slices, p_covered in p_chunks:
                    for f_slices, f_covered in f_chunks:
                        src_slices = [None] * rank
                        dst_slices = [None] * rank
                        for d, ds in batch_src.items():
                            src_slices[d] = ds
                        for d, ds in batch_dst.items():
                            dst_slices[d] = ds
                        for i, d in enumerate(p_group):
                            src_slices[d] = p_slices[i]
                        for i, d in enumerate(f_group):
                            src_slices[d] = f_slices[i]
                        for i in range(len(p_group)):
                            dst_slices[p_out_start + i] = p_slices[i]
                        for i in range(len(f_group)):
                            dst_slices[f_out_start + i] = f_slices[i]
                        yield (
                            tuple(src_slices),
                            tuple(dst_slices),
                            p_covered,
                            f_covered,
                        )


def _can_use_passthrough_partition(
    c_in: tuple[int, ...], c_perm: tuple[int, ...], dtype: DType
) -> bool:
    """Check if the collapsed perm qualifies for the passthrough-partition path.

    Requirements:
      - Collapsed rank == 3 with perm == (0, 2, 1): leading dim is passthrough
        and only the last two dims are swapped.
      - I * J (the product of the two transposed dims) fits in SBUF free dim.
    """
    if len(c_perm) != 3 or c_perm != (0, 2, 1):
        return False
    _B, I, J = c_in
    free_elts = I * J
    # SBUF free-dim budget: need two tiles (src + dst) each of width I*J or J*I.
    # Conservative limit: 48 KiB per tile for f32 -> 12288 elements.
    bytes_per_elem = {DType.F32: 4, DType.F16: 2, DType.BF16: 2}.get(dtype, 4)
    max_free = 49152 // bytes_per_elem
    return free_elts <= max_free


def _lower_transpose_passthrough_partition(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...],
    c_in: tuple[int, ...],
    groups: list[list[int]],
    dtype: DType,
) -> Graph:
    """Fast transpose for (B, I, J) -> (B, J, I) using access-pattern remapping.

    Folds the leading passthrough dim B into the partition. Loads tiles of
    (P_tile, I*J) from HBM, uses access_pattern + tensor_copy to transpose
    I and J on-chip, then stores (P_tile, J*I) to HBM. This matches what
    the neuronx-cc compiler generates via its DVE transpose kernel.

    Produces O(B/128) instructions instead of O(B * ceil(I/128) * ceil(J/128)).
    """
    out_shape = tuple(in_shape[p] for p in perm)
    B, I, J = c_in

    b = Builder("transpose_dma")
    # Declare HBM tensors at original rank for correct emit_to_kb mapping
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    # We view both HBM tensors as 2D: (B, I*J) and (B, J*I)
    # by providing DimSlices that flatten the inner dims.
    in_strides = row_major_strides(in_shape)
    out_strides = row_major_strides(out_shape)

    for p0 in range(0, B, PARTITION_MAX):
        p_size = min(PARTITION_MAX, B - p0)

        # Load (p_size, I*J) from x_hbm
        # The source region is p_size rows at the partition boundary of in_shape.
        # Since B is a prefix-product of in_shape (from collapse), p0..p0+p_size
        # maps to a contiguous region.
        src_chunks = flat_range_to_src_chunks(
            p0 * I * J, p_size * I * J, in_shape, in_strides
        )
        (src_slices, _), = src_chunks

        tile = b.dma_copy(
            b.alloc((p_size, I * J), dtype, MemorySpace.SBUF),
            x_hbm, src_slices,
        )

        # On-chip transpose via access_pattern + tensor_copy.
        # Source tile layout: (P, I*J) where each partition row stores
        # I groups of J contiguous elements:
        #   flat: i0_j0, i0_j1, ..., i0_{J-1}, i1_j0, ..., i_{I-1}_{J-1}
        #
        # We want output per row in (J outer, I inner) order:
        #   j0_i0, j0_i1, ..., j0_{I-1}, j1_i0, ..., j_{J-1}_{I-1}
        #
        # Read via AP with transposed addressing, write to a fresh 2D tile.
        # The AP reads (p_size, J*I) elements from tile by walking J positions
        # at stride 1, then I positions at stride J:
        #   [[I*J, p_size], [1, J], [J, I]]
        src_view = b.access_pattern(
            tile, [[I * J, p_size], [1, J], [J, I]]
        )

        # Allocate destination with matching shape and tensor_copy into it.
        dst_tile = b.alloc((p_size, J, I), dtype, MemorySpace.SBUF)
        b.tensor_copy(dst_tile, src_view)
        b.dealloc(tile)

        # Store (p_size, J*I) to y_hbm
        dst_chunks = flat_range_to_src_chunks(
            p0 * J * I, p_size * J * I, out_shape, out_strides
        )
        (dst_slices, _), = dst_chunks
        b.dma_copy(y_hbm, dst_tile, dst_slices)
        b.dealloc(dst_tile)

    b.set_outputs({"y": y_hbm})
    return b.graph


def lower_transpose_dma(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via DMA engine.

    Args:
        in_shape: Input tensor shape, rank >= 2.
        perm: Permutation of axes. None defaults to swapping last two dims.
        dtype: Element type.

    Batch dim reordering is handled by reading from remapped HBM coordinates.
    P↔F swap (when needed) uses dma_transpose on-chip. Both P and F tiles
    are capped at 128 since after transposing, either could be a partition dim.
    """
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if perm is None:
        perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"invalid permutation: {perm}")

    c_in, c_perm, groups, src_order = _collapse_perm(in_shape, perm)
    c_out = tuple(c_in[p] for p in c_perm)
    c_rank = len(c_out)
    out_shape = tuple(in_shape[p] for p in perm)

    # Fast path: when collapsed to (B, I, J) perm (0,2,1) and I*J fits in SBUF,
    # fold B into partition and transpose via access-pattern remapping.
    if _can_use_passthrough_partition(c_in, c_perm, dtype):
        return _lower_transpose_passthrough_partition(
            in_shape, perm, c_in, groups, dtype
        )

    b = Builder("transpose_dma")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    if c_rank < 2:
        # Identity permutation after collapse — fall back to uncollapsed.
        groups = [[d] for d in perm]
        c_out = out_shape
        c_rank = rank
        c_perm = perm

    swap_pf = _needs_pf_swap(c_perm)
    tile_p = min(c_out[-2], PARTITION_MAX)
    tile_f = min(c_out[-1], PARTITION_MAX) if swap_pf else c_out[-1]

    for src_slices, dst_slices, p_cov, f_cov in _tile_iter(
        in_shape, perm, groups, c_out, c_rank, tile_p, tile_f
    ):
        if swap_pf:
            tile = b.dma_copy(
                b.alloc((f_cov, p_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            transposed = b.transpose(tile, (1, 0))
            b.dealloc(tile)
            b.dma_copy(y_hbm, transposed, dst_slices)
            b.dealloc(transposed)
        else:
            tile = b.dma_copy(
                b.alloc((p_cov, f_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            b.dma_copy(y_hbm, tile, dst_slices)
            b.dealloc(tile)

    b.set_outputs({"y": y_hbm})
    return b.graph


def lower_transpose_te(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via tensor engine (A.T @ I trick).

    Args:
        in_shape: Input tensor shape, rank >= 2.
        perm: Permutation of axes. None defaults to swapping last two dims.
        dtype: Element type.

    Batch dim reordering is DMA slice remapping (same as DMA strategy). For
    the P↔F swap, uses matmul:
      stat[K=f_size, M=p_size].T @ I[K=f_size, N=f_size] -> dst[p_size, f_size]

    The loaded source tile is (f_size, p_size) — used as stationary with K=f_size,
    M=p_size. The identity I is (f_size, f_size). The result stat.T @ I is
    (p_size, f_size) = the transposed tile.

    Constraints: K=f_size <= 128, M=p_size <= 128, N=f_size <= 512.
    So both tile_p and tile_f are capped at 128.

    When no P↔F swap is needed, falls back to plain DMA copy.

    Requires an identity matrix as HBM input ("eye").
    """
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if perm is None:
        perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"invalid permutation: {perm}")

    c_in, c_perm, groups, src_order = _collapse_perm(in_shape, perm)
    c_out = tuple(c_in[p] for p in c_perm)
    c_rank = len(c_out)
    out_shape = tuple(in_shape[p] for p in perm)

    if c_rank < 2:
        groups = [[d] for d in perm]
        c_out = out_shape
        c_rank = rank
        c_perm = perm

    swap_pf = _needs_pf_swap(c_perm)
    tile_p = min(c_out[-2], PARTITION_MAX)
    tile_f = min(c_out[-1], PARTITION_MAX) if swap_pf else c_out[-1]
    eye_size = tile_f if swap_pf else 0

    bld = Builder("transpose_te")
    x_hbm = bld.add_input("x", in_shape, dtype)
    if swap_pf:
        y_hbm = bld.add_input("y", out_shape, DType.F32)
        eye_hbm = bld.add_input("eye", (eye_size, eye_size), dtype)
    else:
        y_hbm = bld.add_input("y", out_shape, dtype)

    for src_slices, dst_slices, p_cov, f_cov in _tile_iter(
        in_shape, perm, groups, c_out, c_rank, tile_p, tile_f
    ):
        if swap_pf:
            stat = bld.dma_copy(
                bld.alloc((f_cov, p_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            eye_tile = bld.dma_copy(
                bld.alloc((f_cov, f_cov), dtype, MemorySpace.SBUF),
                eye_hbm,
                (DimSlice(0, f_cov), DimSlice(0, f_cov)),
            )

            psum = bld.alloc((p_cov, f_cov), DType.F32, MemorySpace.PSUM)
            bld.matmul(psum, stat, eye_tile, accumulate=False)
            bld.dealloc(stat)
            bld.dealloc(eye_tile)

            out_sbuf = bld.tensor_copy(
                bld.alloc((p_cov, f_cov), DType.F32, MemorySpace.SBUF), psum
            )
            bld.dealloc(psum)
            bld.dma_copy(y_hbm, out_sbuf, dst_slices)
            bld.dealloc(out_sbuf)
        else:
            tile = bld.dma_copy(
                bld.alloc((p_cov, f_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            bld.dma_copy(y_hbm, tile, dst_slices)
            bld.dealloc(tile)

    bld.set_outputs({"y": y_hbm})
    return bld.graph


def emit_transpose(
    nb: Builder,
    x_hbm,
    y_hbm,
    in_shape: tuple[int, ...],
    perm: tuple[int, ...],
    dtype: DType = DType.F32,
) -> None:
    """Emit transpose tiling into an existing Builder (DMA strategy)."""
    rank = len(in_shape)

    c_in, c_perm, groups, src_order = _collapse_perm(in_shape, perm)
    c_out = tuple(c_in[p] for p in c_perm)
    c_rank = len(c_out)

    # Fast path: passthrough-partition (same logic as lower_transpose_dma)
    if _can_use_passthrough_partition(c_in, c_perm, dtype):
        out_shape = tuple(in_shape[p] for p in perm)
        B, I, J = c_in
        in_strides = row_major_strides(in_shape)
        out_strides = row_major_strides(out_shape)
        for p0 in range(0, B, PARTITION_MAX):
            p_size = min(PARTITION_MAX, B - p0)
            src_chunks = flat_range_to_src_chunks(
                p0 * I * J, p_size * I * J, in_shape, in_strides)
            (src_slices, _), = src_chunks
            tile = nb.dma_copy(
                nb.alloc((p_size, I * J), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            src_view = nb.access_pattern(
                tile, [[I * J, p_size], [1, J], [J, I]]
            )
            dst_tile = nb.alloc((p_size, J, I), dtype, MemorySpace.SBUF)
            nb.tensor_copy(dst_tile, src_view)
            dst_chunks = flat_range_to_src_chunks(
                p0 * J * I, p_size * J * I, out_shape, out_strides)
            (dst_slices, _), = dst_chunks
            nb.dma_copy(y_hbm, dst_tile, dst_slices)
        return

    if c_rank < 2:
        groups = [[d] for d in perm]
        c_out = tuple(in_shape[p] for p in perm)
        c_rank = rank
        c_perm = perm

    swap_pf = _needs_pf_swap(c_perm)
    tile_p = min(c_out[-2], PARTITION_MAX)
    tile_f = min(c_out[-1], PARTITION_MAX) if swap_pf else c_out[-1]

    for src_slices, dst_slices, p_cov, f_cov in _tile_iter(
        in_shape, perm, groups, c_out, c_rank, tile_p, tile_f
    ):
        if swap_pf:
            tile = nb.dma_copy(
                nb.alloc((f_cov, p_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            transposed = nb.transpose(tile, (1, 0))
            nb.dma_copy(y_hbm, transposed, dst_slices)
        else:
            tile = nb.dma_copy(
                nb.alloc((p_cov, f_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            nb.dma_copy(y_hbm, tile, dst_slices)
