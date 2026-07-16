"""Scratch: the allocation seam between tiling and codegen.

Every direct-lowering emitter allocates two kinds of buffer inline: SBUF working
tiles (shape derived from the tile schedule) and HBM scratch (staging round-trips
for transpose/topk/collective/broadcast). Today those are raw ``nb.alloc(...)``
calls scattered across ~11 HBM sites and buried as the first argument of ~57
``dma_copy`` loads, mingling *what memory to allocate* with *what op to emit*.

``Scratch`` collects both behind one named surface. It wraps a Builder and
exposes only allocation — never a compute op — so it stays a distinct layer from
codegen. One instance is threaded per lowered graph, which makes every HBM
scratch buffer flow through a single method (``hbm``): the one place to add size
tracking, tagging, or a reuse arena when scratch lifetime becomes an OOM
concern. Deallocation stays the existing post-pass (``insert_deallocs``);
``Scratch`` governs allocation only.

The tile *shape* is still a scheduling output (see ``TileSchedule``), so alloc
stays fused to tiling — the caller passes the shape it computed from the tile
index. ``Scratch`` removes only the artificial coupling of alloc to the
``dma_copy`` call, not the intrinsic tiling→alloc one.
"""

from __future__ import annotations

from nkigen_lite.core import DType, Value
from nkigen_lite.nki_ir.ir import Builder, DimSlice, MemorySpace


class Scratch:
    """Allocation surface for one lowered graph, wrapping a Builder."""

    def __init__(self, nb: Builder) -> None:
        self._nb = nb

    def sbuf(self, shape: tuple[int, ...], dtype: DType) -> Value:
        """Allocate an uninitialized SBUF working tile."""
        return self._nb.alloc(shape, dtype, MemorySpace.SBUF)

    def hbm(self, shape: tuple[int, ...], dtype: DType) -> Value:
        """Allocate an HBM scratch buffer (staging round-trips).

        The single choke point for scratch HBM — audit here for OOM.
        """
        return self._nb.alloc(shape, dtype, MemorySpace.HBM)

    def load(
        self,
        src: Value,
        slices: tuple[DimSlice | int | Value, ...],
        shape: tuple[int, ...],
        dtype: DType,
    ) -> Value:
        """Allocate an SBUF tile and DMA-load ``src[slices]`` into it.

        Collapses the pervasive ``dma_copy(alloc(shape, dtype, SBUF), src,
        slices)`` idiom. Returns the loaded tile.
        """
        return self._nb.dma_copy(self.sbuf(shape, dtype), src, slices)
