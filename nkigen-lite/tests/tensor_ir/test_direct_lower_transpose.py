"""Tests for direct transpose lowering (tensor IR -> NKI IR).

Verifies both DMA transpose and tensor engine transpose on real Trainium
hardware with arbitrary permutations: batch-only reorders, P↔F swaps,
and complex multi-axis permutations.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_transpose import (
    lower_transpose_dma,
    lower_transpose_te,
)

import nki.compiler.kernel_builder as nb

pytestmark = pytest.mark.hw


def _check_dma(in_shape, perm=None, atol=1e-3):
    """Lower transpose via DMA and verify on real Trainium hardware."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(in_shape).astype(np.float32)
    if perm is None:
        perm = tuple(range(len(in_shape) - 2)) + (len(in_shape) - 1, len(in_shape) - 2)
    ref = np.transpose(x, perm)

    graph = lower_transpose_dma(in_shape, perm=perm)
    kernel_fn = build_kb_kernel(graph)
    hw_out = {"y": np.zeros(ref.shape, dtype=np.float32)}
    nb.compile_and_execute(
        kernel_fn,
        inputs={"x": x},
        outputs=hw_out,
        compile_opts=nb.CompileOptions(target="trn2"),
    )
    np.testing.assert_allclose(
        hw_out["y"], ref, atol=atol, rtol=atol,
        err_msg="HW mismatch (DMA transpose)",
    )


def _check_te(in_shape, perm=None, atol=1e-3):
    """Lower transpose via tensor engine and verify on real Trainium hardware."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(in_shape).astype(np.float32)
    if perm is None:
        perm = tuple(range(len(in_shape) - 2)) + (len(in_shape) - 1, len(in_shape) - 2)
    ref = np.transpose(x, perm).astype(np.float32)

    graph = lower_transpose_te(in_shape, perm=perm)
    kernel_fn = build_kb_kernel(graph)

    # Check if TE path needs identity matrix
    rank = len(in_shape)
    swap_pf = perm[rank - 2] > perm[rank - 1]
    out_shape = tuple(in_shape[p] for p in perm)

    if swap_pf:
        tile_f = min(out_shape[-1], 128)
        eye = np.eye(tile_f, dtype=np.float32)
        hw_out = {"y": np.zeros(ref.shape, dtype=np.float32)}
        nb.compile_and_execute(
            kernel_fn,
            inputs={"x": x, "eye": eye},
            outputs=hw_out,
            compile_opts=nb.CompileOptions(target="trn2"),
        )
    else:
        hw_out = {"y": np.zeros(ref.shape, dtype=np.float32)}
        nb.compile_and_execute(
            kernel_fn,
            inputs={"x": x},
            outputs=hw_out,
            compile_opts=nb.CompileOptions(target="trn2"),
        )
    np.testing.assert_allclose(
        hw_out["y"], ref, atol=atol, rtol=atol,
        err_msg="HW mismatch (TE transpose)",
    )


class TestDmaSwapLastTwo:
    """DMA transpose: swap last two dims (P↔F)."""

    def test_rank2(self):
        _check_dma((128, 64))

    def test_rank2_remainder(self):
        _check_dma((200, 300))

    def test_rank2_large(self):
        _check_dma((512, 512))

    def test_rank3(self):
        _check_dma((4, 128, 64))

    def test_rank4(self):
        _check_dma((2, 3, 64, 128))


class TestDmaBatchReorder:
    """DMA transpose: permutations that only reorder batch dims (no P↔F swap)."""

    def test_swap_batch_rank3(self):
        _check_dma((3, 64, 128), perm=(0, 1, 2))  # identity (sanity)

    def test_move_batch_to_p(self):
        # (0,2,1,3): perm[-2]=1, perm[-1]=3, 1<3 -> no swap
        _check_dma((2, 3, 4, 64), perm=(0, 2, 1, 3))

    def test_swap_first_two_batch(self):
        # (1,0,2,3): swaps batch dims, perm[-2]=2, perm[-1]=3, 2<3 -> no swap
        _check_dma((3, 5, 64, 128), perm=(1, 0, 2, 3))

    def test_reverse_batch(self):
        # (2,1,0,3): reverse batch, perm[-2]=0, perm[-1]=3, 0<3 -> no swap
        _check_dma((2, 3, 4, 64), perm=(2, 1, 0, 3))

    def test_complex_no_swap(self):
        # (3,1,0,2): perm[-2]=0, perm[-1]=2, 0<2 -> no swap
        _check_dma((2, 3, 4, 64), perm=(3, 1, 0, 2))


class TestDmaArbitrary:
    """DMA transpose: complex permutations with P↔F swap."""

    def test_rank3_rotate(self):
        # (1,2,0): perm[-2]=2, perm[-1]=0, 2>0 -> P↔F swap
        _check_dma((3, 64, 128), perm=(1, 2, 0))

    def test_rank4_complex(self):
        # (2,0,3,1): perm[-2]=3, perm[-1]=1, 3>1 -> P↔F swap
        _check_dma((2, 3, 64, 128), perm=(2, 0, 3, 1))

    def test_rank4_pf_swap_with_batch(self):
        # (0,1,3,2): standard P↔F swap with batch
        _check_dma((2, 3, 64, 128), perm=(0, 1, 3, 2))

    def test_rank3_pf_swap(self):
        # (0,2,1): perm[-2]=2, perm[-1]=1, 2>1 -> P↔F swap
        _check_dma((4, 64, 128), perm=(0, 2, 1))

    def test_rank4_remainder(self):
        _check_dma((2, 3, 100, 200), perm=(1, 0, 3, 2))


class TestTeSwapLastTwo:
    """Tensor engine transpose: swap last two dims via matmul."""

    def test_rank2(self):
        _check_te((64, 128))

    def test_rank2_square(self):
        _check_te((128, 128))

    def test_rank2_remainder(self):
        _check_te((200, 100))

    def test_rank2_large(self):
        _check_te((256, 256))

    def test_rank3(self):
        _check_te((4, 128, 64))

    def test_rank4(self):
        _check_te((2, 3, 64, 128))


class TestTeBatchReorder:
    """Tensor engine: batch-only reorder (no matmul needed, DMA fallback)."""

    def test_swap_batch(self):
        _check_te((2, 3, 4, 64), perm=(0, 2, 1, 3))

    def test_swap_first_two(self):
        _check_te((3, 5, 64, 128), perm=(1, 0, 2, 3))


class TestTeArbitrary:
    """Tensor engine: complex permutations with P↔F swap via matmul."""

    def test_rank3_rotate(self):
        _check_te((3, 64, 128), perm=(1, 2, 0))

    def test_rank4_complex(self):
        _check_te((2, 3, 64, 128), perm=(2, 0, 3, 1))

    def test_rank4_pf_swap(self):
        _check_te((2, 3, 64, 128), perm=(0, 1, 3, 2))

    def test_rank4_remainder(self):
        _check_te((2, 3, 100, 100), perm=(1, 0, 3, 2))


class TestDmaCollapse:
    """DMA transpose: axis-collapse optimization (merged contiguous dim runs)."""

    def test_qwen_like(self):
        """Multi-dim spatial merge: (Co, Ci, *K) -> (Co, *K, Ci)."""
        _check_dma((4, 3, 2, 16, 16), perm=(0, 2, 3, 4, 1))

    def test_boundary_straddle(self):
        """Merged axis where PARTITION_MAX tile straddles original dim boundaries."""
        _check_dma((4, 3, 5, 7), perm=(0, 2, 3, 1))

    def test_all_reorder_with_merge(self):
        """All dims reordered, adjacent pair merges."""
        _check_dma((6, 8, 4), perm=(1, 2, 0))

    def test_batch_merge_no_swap(self):
        """Adjacent batch dims merge, no P↔F swap."""
        _check_dma((3, 5, 64, 128), perm=(1, 0, 2, 3))

    def test_large_spatial_merge(self):
        """Large merged spatial exceeding PARTITION_MAX."""
        _check_dma((4, 3, 4, 8, 16), perm=(0, 2, 3, 4, 1))


class TestTeCollapse:
    """Tensor engine: axis-collapse optimization."""

    def test_qwen_like(self):
        _check_te((4, 3, 2, 16, 16), perm=(0, 2, 3, 4, 1))

    def test_boundary_straddle(self):
        _check_te((4, 3, 5, 7), perm=(0, 2, 3, 1))

    def test_all_reorder_with_merge(self):
        _check_te((6, 8, 4), perm=(1, 2, 0))
