"""Unit tests for the Scratch allocation surface.

Scratch is a thin wrapper over Builder; these pin the contract the emitters
rely on: sbuf/hbm allocate in the right memory space, and load() emits an
alloc + a single DMA load returning the loaded tile.
"""

from __future__ import annotations

from nkigen_lite.core import DType
from nkigen_lite.nki_ir.ir import Builder, DimSlice, MemorySpace
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Scratch


def test_sbuf_allocates_in_sbuf():
    nb = Builder("t")
    s = Scratch(nb)
    t = s.sbuf((4, 8), DType.F32)
    assert t.type.memory == MemorySpace.SBUF
    assert t.type.shape == (4, 8)
    assert t.type.dtype == DType.F32


def test_hbm_allocates_in_hbm():
    nb = Builder("t")
    s = Scratch(nb)
    t = s.hbm((16, 16), DType.BF16)
    assert t.type.memory == MemorySpace.HBM
    assert t.type.shape == (16, 16)


def test_load_emits_alloc_and_dma_and_returns_tile():
    nb = Builder("t")
    s = Scratch(nb)
    src = nb.add_input("x", (128, 512), DType.F32)
    n_ops_before = len(nb.graph.ops)
    tile = s.load(src, (DimSlice(0, 64), DimSlice(0, 256)), (64, 256), DType.F32)
    # An alloc + a dma_copy were emitted.
    new_ops = [op.opcode for op in nb.graph.ops[n_ops_before:]]
    assert new_ops == ["alloc", "dma_copy"]
    # The returned value is the loaded SBUF tile of the requested shape.
    assert tile.type.memory == MemorySpace.SBUF
    assert tile.type.shape == (64, 256)


def test_load_matches_hand_written_alloc_plus_dma():
    """load() must be behaviourally identical to alloc + dma_copy."""
    nb_a = Builder("a")
    src_a = nb_a.add_input("x", (32, 32), DType.F32)
    manual = nb_a.dma_copy(
        nb_a.alloc((8, 16), DType.F32, MemorySpace.SBUF),
        src_a, (DimSlice(0, 8), DimSlice(0, 16)),
    )

    nb_b = Builder("b")
    src_b = nb_b.add_input("x", (32, 32), DType.F32)
    via = Scratch(nb_b).load(
        src_b, (DimSlice(0, 8), DimSlice(0, 16)), (8, 16), DType.F32
    )

    assert manual.type.shape == via.type.shape
    assert manual.type.memory == via.type.memory
    assert [op.opcode for op in nb_a.graph.ops] == [op.opcode for op in nb_b.graph.ops]
