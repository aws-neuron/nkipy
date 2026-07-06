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

``emit_transpose`` / ``emit_transpose_te`` are the single implementations; the
``lower_transpose_*`` entry points are thin standalone-graph wrappers over them.
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
from nkigen_lite.nki_ir.insert_deallocs import insert_deallocs

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

    # Fast path: (B, I, J) -> (B, J, I) with the leading collapsed dim folded
    # onto the partition; the I/J swap happens on-chip via access-pattern
    # remapping + tensor_copy (matches the neuronx-cc DVE transpose kernel).
    # Produces O(B/128) instructions instead of O(B * ceil(I/128) * ceil(J/128)).
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
            # Source tile row layout is I groups of J contiguous elements; the
            # AP walks J at stride 1 then I at stride J, i.e. (J outer, I inner).
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


def emit_transpose_te(
    nb: Builder,
    x_hbm,
    y_hbm,
    eye_hbm,
    in_shape: tuple[int, ...],
    perm: tuple[int, ...],
    dtype: DType = DType.F32,
) -> None:
    """Emit transpose tiling into an existing Builder (tensor engine strategy).

    Batch dim reordering is DMA slice remapping (same as the DMA strategy). For
    the P↔F swap, uses matmul:
      stat[K=f_size, M=p_size].T @ I[K=f_size, N=f_size] -> dst[p_size, f_size]

    The loaded source tile is (f_size, p_size) — used as stationary with
    K=f_size, M=p_size. The identity I is (f_size, f_size). The result
    stat.T @ I is (p_size, f_size) = the transposed tile.

    Constraints: K=f_size <= 128, M=p_size <= 128, N=f_size <= 512.
    So both tile_p and tile_f are capped at 128.

    ``eye_hbm`` must be an HBM identity of at least (tile_f, tile_f) when the
    permutation swaps P↔F; it is unused (may be None) otherwise, in which case
    this degenerates to a plain DMA copy.
    """
    rank = len(in_shape)

    c_in, c_perm, groups, src_order = _collapse_perm(in_shape, perm)
    c_out = tuple(c_in[p] for p in c_perm)
    c_rank = len(c_out)

    if c_rank < 2:
        groups = [[d] for d in perm]
        c_out = tuple(in_shape[p] for p in perm)
        c_rank = rank
        c_perm = perm

    swap_pf = _needs_pf_swap(c_perm)
    tile_p = min(c_out[-2], PARTITION_MAX)
    tile_f = min(c_out[-1], PARTITION_MAX) if swap_pf else c_out[-1]

    if swap_pf and eye_hbm is None:
        raise ValueError("emit_transpose_te: P↔F swap requires an eye_hbm identity")

    for src_slices, dst_slices, p_cov, f_cov in _tile_iter(
        in_shape, perm, groups, c_out, c_rank, tile_p, tile_f
    ):
        if swap_pf:
            stat = nb.dma_copy(
                nb.alloc((f_cov, p_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            eye_tile = nb.dma_copy(
                nb.alloc((f_cov, f_cov), dtype, MemorySpace.SBUF),
                eye_hbm,
                (DimSlice(0, f_cov), DimSlice(0, f_cov)),
            )

            psum = nb.alloc((p_cov, f_cov), DType.F32, MemorySpace.PSUM)
            nb.matmul(psum, stat, eye_tile, accumulate=False)

            out_sbuf = nb.tensor_copy(
                nb.alloc((p_cov, f_cov), DType.F32, MemorySpace.SBUF), psum
            )
            nb.dma_copy(y_hbm, out_sbuf, dst_slices)
        else:
            tile = nb.dma_copy(
                nb.alloc((p_cov, f_cov), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            nb.dma_copy(y_hbm, tile, dst_slices)


# ---------------------------------------------------------------------------
# Standalone graph wrappers
# ---------------------------------------------------------------------------


def _check_perm(
    in_shape: tuple[int, ...], perm: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Validate rank/perm, defaulting to a last-two-dims swap."""
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if perm is None:
        perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"invalid permutation: {perm}")
    return perm


def lower_transpose_dma(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via DMA engine into a standalone graph.

    Thin wrapper over ``emit_transpose``: HBM inputs ``x``/``y``, one emit call.
    """
    perm = _check_perm(in_shape, perm)
    out_shape = tuple(in_shape[p] for p in perm)

    b = Builder("transpose_dma")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)
    emit_transpose(b, x_hbm, y_hbm, in_shape, perm, dtype)
    b.set_outputs({"y": y_hbm})
    insert_deallocs(b.graph)
    return b.graph


def lower_transpose_te(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via tensor engine into a standalone graph.

    Thin wrapper over ``emit_transpose_te``. When the permutation swaps P↔F the
    graph takes an identity matrix as an extra HBM input ("eye") and the output
    buffer is F32 (PSUM result dtype); otherwise no eye input is declared and
    the emission degenerates to a plain DMA copy.
    """
    perm = _check_perm(in_shape, perm)
    out_shape = tuple(in_shape[p] for p in perm)

    c_in, c_perm, _groups, _src_order = _collapse_perm(in_shape, perm)
    c_out = tuple(c_in[p] for p in c_perm)
    if len(c_perm) < 2:
        c_perm = perm
        c_out = out_shape
    swap_pf = _needs_pf_swap(c_perm)
    eye_size = min(c_out[-1], PARTITION_MAX) if swap_pf else 0

    bld = Builder("transpose_te")
    x_hbm = bld.add_input("x", in_shape, dtype)
    if swap_pf:
        y_hbm = bld.add_input("y", out_shape, DType.F32)
        eye_hbm = bld.add_input("eye", (eye_size, eye_size), dtype)
    else:
        y_hbm = bld.add_input("y", out_shape, dtype)
        eye_hbm = None
    emit_transpose_te(bld, x_hbm, y_hbm, eye_hbm, in_shape, perm, dtype)
    bld.set_outputs({"y": y_hbm})
    insert_deallocs(bld.graph)
    return bld.graph
