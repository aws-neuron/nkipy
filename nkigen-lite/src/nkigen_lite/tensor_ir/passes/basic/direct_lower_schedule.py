"""Tile schedule: tiling expressed as a data object, independent of alloc/codegen.

Every direct-lowering loop-nest is the same three-part object spelled
differently: per-dim tile sizes, an iteration over ``ceildiv(extent, tile)`` per
tiled dim, and a clamped slice + extent at each index. This module promotes that
shape to a single type so tiling policy lives in one place and emitters stop
hand-rolling ``ceildiv``/``range``/``min``.

Design boundary: a ``TileSchedule`` knows about *shapes, tile sizes, and
indices* only. It never touches the Builder, an op, or a dtype-as-buffer — that
keeps it cleanly separable from allocation (``Allocator``) and codegen (the
emitter body). Tile shape is a scheduling output, so alloc stays fused to the
schedule via the ``TileIndex`` handed to the body; only the artificial
alloc<->codegen coupling is removed.

A ``TileSchedule`` owns one ``tile_sizes`` map that drives iteration. A
``TileIndex`` carries just the per-dim tile ordinals, so the body can build
slices/extents against a *different* tile-size map sharing the same indices
(e.g. reduce's input vs. output tiling) — matching the free-function contract
these replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from nkigen_lite.core import DType
from nkigen_lite.nki_ir.ir import (
    PARTITION_MAX,
    DimSlice,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    max_free_elems,
)


@dataclass(frozen=True)
class TileIndex:
    """The tile ordinal per tiled dim for one iteration of a schedule.

    ``indices`` maps ``dim -> tile ordinal``; a dim absent from the map is
    untiled (full extent). ``slices`` and ``extent`` take an explicit
    ``tile_sizes`` so a single index can address buffers tiled differently
    (input vs. output of a reduce).
    """

    indices: dict[int, int]

    def slices(
        self, shape: tuple[int, ...], tile_sizes: dict[int, int]
    ) -> list[DimSlice]:
        """One ``DimSlice`` per dim of ``shape``, clamped on the boundary."""
        slices = []
        for d in range(len(shape)):
            ts = tile_sizes.get(d, shape[d])
            if ts >= shape[d]:
                slices.append(DimSlice(0, shape[d]))
            else:
                off = self.indices.get(d, 0) * ts
                slices.append(DimSlice(off, min(ts, shape[d] - off)))
        return slices

    def extent(
        self,
        dims: tuple[int, ...],
        shape: tuple[int, ...],
        tile_sizes: dict[int, int],
    ) -> int:
        """Product of the per-dim tile extents over ``dims``, clamped."""
        result = 1
        for d in dims:
            ts = tile_sizes[d]
            if ts >= shape[d]:
                result *= shape[d]
            else:
                off = self.indices.get(d, 0) * ts
                result *= min(ts, shape[d] - off)
        return result


@dataclass(frozen=True)
class TileSchedule:
    """An iteration space: a shape plus per-dim tile sizes.

    Iterating yields one ``TileIndex`` per tile, nesting dims in ascending
    order (a dim whose tile covers its full extent contributes no loop level).
    This generalizes the ``loop_dims`` + ``_nested`` recursion that reduce
    reinvented privately.
    """

    shape: tuple[int, ...]
    tile_sizes: dict[int, int]

    def __iter__(self) -> Iterator[TileIndex]:
        loop_dims = [
            (d, self.shape[d], self.tile_sizes[d])
            for d in sorted(self.tile_sizes)
            if self.tile_sizes[d] < self.shape[d]
        ]

        def _rec(depth: int, indices: dict[int, int]) -> Iterator[TileIndex]:
            if depth >= len(loop_dims):
                yield TileIndex(indices)
                return
            d, extent, ts = loop_dims[depth]
            for i in range(ceildiv(extent, ts)):
                yield from _rec(depth + 1, {**indices, d: i})

        yield from _rec(0, {})

    # -- policy constructors: the single place a tile budget is chosen --------

    @classmethod
    def pf(cls, P: int, F: int, dtype: DType) -> "TileSchedule":
        """The data-movement 2D ``(P, F)`` schedule: partition at 128 lanes,
        free dim at ``max_free_elems(dtype)`` so each tile fits one partition's
        SBUF budget. This is the single place the data-movement tile budget is
        chosen (formerly the free function ``iter_pf_tiles``)."""
        return cls((P, F), {0: PARTITION_MAX, 1: max_free_elems(dtype)})

    @classmethod
    def free_pow2(cls, P: int, F: int, free_max: int = 512) -> "TileSchedule":
        """The elementwise 2D ``(P, F)`` schedule: partition at 128 lanes, free
        dim at the largest power of two ``<= min(F, free_max)``.

        Distinct from ``pf``: elementwise compute wants a power-of-two free
        width capped at the tensor/vector engine's comfortable 512, not the
        dtype-derived data-movement budget ``max_free_elems``. This is the
        single place that policy lives (formerly ``_free_tile``)."""
        cap = min(F, free_max)
        t = 1
        while t * 2 <= cap:
            t *= 2
        return cls((P, F), {0: PARTITION_MAX, 1: t})

    def pf_tiles(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(p_off, p_size, f_off, f_size)`` for a 2D schedule.

        Convenience for the many data-movement loops that want flat offsets
        rather than a ``TileIndex``; equivalent to iterating and reading the
        two ``DimSlice``s. Only valid on a rank-2 schedule."""
        assert len(self.shape) == 2, "pf_tiles requires a 2D schedule"
        for idx in self:
            ps, fs = idx.slices(self.shape, self.tile_sizes)
            yield ps.offset, ps.size, fs.offset, fs.size
