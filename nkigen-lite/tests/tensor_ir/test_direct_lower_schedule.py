"""Equivalence tests for TileSchedule against the loop bodies it replaces.

TileSchedule replaced three free functions (``iter_pf_tiles``, ``build_slices``,
``clamped_extent``) plus the hand-rolled ``loop_dims`` + ``_nested`` recursion in
reduce. Those are gone from the library; the reference implementations below are
inlined here so this file remains a standalone regression spec — if a future
change to TileSchedule diverges from the original iteration/slicing, these fail.
"""

from __future__ import annotations

import pytest
from nkigen_lite.core import DType
from nkigen_lite.nki_ir.ir import PARTITION_MAX, DimSlice
from nkigen_lite.tensor_ir.passes.basic.direct_lower_schedule import TileSchedule
from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    max_free_elems,
)

# --- reference implementations (the pre-refactor free functions) -------------


def iter_pf_tiles(P, F, dtype):
    cap = max_free_elems(dtype)
    for p_i in range(ceildiv(P, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, P - p_off)
        for f_i in range(ceildiv(F, cap)):
            f_off = f_i * cap
            f_size = min(cap, F - f_off)
            yield p_off, p_size, f_off, f_size


def build_slices(shape, tile_sizes, indices):
    slices = []
    for d in range(len(shape)):
        ts = tile_sizes.get(d, shape[d])
        if ts >= shape[d]:
            slices.append(DimSlice(0, shape[d]))
        else:
            off = indices.get(d, 0) * ts
            slices.append(DimSlice(off, min(ts, shape[d] - off)))
    return slices


def clamped_extent(dims, shape, tile_sizes, indices):
    result = 1
    for d in dims:
        ts = tile_sizes[d]
        if ts >= shape[d]:
            result *= shape[d]
        else:
            result *= min(ts, shape[d] - indices.get(d, 0) * ts)
    return result


PF_SHAPES = [
    (1, 1), (1, 500), (128, 512), (129, 513), (256, 4096),
    (300, 8), (1, 131072), (127, 1), (1024, 1024),
]
DTYPES = [DType.F32, DType.BF16, DType.I8]


@pytest.mark.parametrize("P,F", PF_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pf_tiles_matches_iter_pf_tiles(P, F, dtype):
    expected = list(iter_pf_tiles(P, F, dtype))
    got = list(TileSchedule.pf(P, F, dtype).pf_tiles())
    assert got == expected


@pytest.mark.parametrize("P,F", PF_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pf_schedule_slices_match(P, F, dtype):
    """Iterating the schedule and reading slices matches the flat tuples."""
    sched = TileSchedule.pf(P, F, dtype)
    for idx, tup in zip(sched, iter_pf_tiles(P, F, dtype)):
        ps, fs = idx.slices(sched.shape, sched.tile_sizes)
        assert (ps.offset, ps.size, fs.offset, fs.size) == tup


# N-D nested schedule (reduce-style): compare against build_slices / clamped_extent
# driven by the same hand-rolled loop_dims recursion they use today.
ND_CASES = [
    # (shape, tile_sizes)
    ((10, 30, 20), {0: PARTITION_MAX, 1: 1, 2: 20}),
    ((4, 8, 128), {0: 1, 1: min(8, PARTITION_MAX), 2: 128}),
    ((256, 131072), {0: min(256, PARTITION_MAX), 1: 1}),
    ((3, 100, 8), {0: 1, 1: PARTITION_MAX, 2: 8}),
    ((5,), {0: PARTITION_MAX}),
]


def _reference_indices(shape, tile_sizes):
    """Replicate reduce's loop_dims + _nested index enumeration."""
    loop_dims = [
        (d, shape[d], tile_sizes[d])
        for d in sorted(tile_sizes)
        if tile_sizes[d] < shape[d]
    ]
    out = []

    def _rec(depth, indices):
        if depth >= len(loop_dims):
            out.append(dict(indices))
            return
        d, extent, ts = loop_dims[depth]
        for i in range((extent + ts - 1) // ts):
            _rec(depth + 1, {**indices, d: i})

    _rec(0, {})
    return out


@pytest.mark.parametrize("shape,tile_sizes", ND_CASES)
def test_nd_schedule_indices_match(shape, tile_sizes):
    ref = _reference_indices(shape, tile_sizes)
    got = [idx.indices for idx in TileSchedule(shape, tile_sizes)]
    assert got == ref


@pytest.mark.parametrize("shape,tile_sizes", ND_CASES)
def test_nd_slices_and_extent_match(shape, tile_sizes):
    dims = tuple(range(len(shape)))
    for indices, idx in zip(
        _reference_indices(shape, tile_sizes), TileSchedule(shape, tile_sizes)
    ):
        assert idx.slices(shape, tile_sizes) == build_slices(shape, tile_sizes, indices)
        assert idx.extent(dims, shape, tile_sizes) == clamped_extent(
            dims, shape, tile_sizes, indices
        )


def test_tileindex_slices_against_different_tile_sizes():
    """A single index can address input- and output-tiled buffers (reduce)."""
    shape = (10, 30, 20)
    in_ts = {0: PARTITION_MAX, 1: 1, 2: 20}
    out_ts = {0: PARTITION_MAX, 1: 1, 2: 1}
    for indices, idx in zip(
        _reference_indices(shape, in_ts), TileSchedule(shape, in_ts)
    ):
        out_shape = (10, 30, 1)
        assert idx.slices(out_shape, out_ts) == build_slices(out_shape, out_ts, indices)
