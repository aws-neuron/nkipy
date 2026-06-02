"""Tests for direct broadcast lowering (tensor IR -> NKI IR).

Verifies correctness on real Trainium hardware across I-dim, P-dim, and F-dim
broadcast strategies with various shapes and remainder tiles.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_broadcast import lower_broadcast

import nki.compiler.kernel_builder as nb

pytestmark = pytest.mark.hw


def _check(in_shape, out_shape, broadcast_axis, atol=1e-3):
    """Lower broadcast and verify on real Trainium hardware."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(in_shape).astype(np.float32)
    ref = np.broadcast_to(x, out_shape).copy()

    graph = lower_broadcast(in_shape, out_shape, broadcast_axis)
    kernel_fn = build_kb_kernel(graph)
    hw_out = {"y": np.zeros(out_shape, dtype=np.float32)}
    nb.compile_and_execute(
        kernel_fn,
        inputs={"x": x},
        outputs=hw_out,
        compile_opts=nb.CompileOptions(target="trn2"),
    )
    np.testing.assert_allclose(
        hw_out["y"], ref, atol=atol, rtol=atol,
        err_msg="HW mismatch",
    )


class TestIdimBroadcast:
    """I-dim (batch) broadcast: loop over output batch."""

    def test_basic(self):
        _check((1, 64, 128), (4, 64, 128), broadcast_axis=0)

    def test_large_batch(self):
        _check((1, 128, 128), (8, 128, 128), broadcast_axis=0)

    def test_remainder(self):
        _check((1, 100, 200), (3, 100, 200), broadcast_axis=0)

    def test_rank4_middle(self):
        _check((2, 1, 64, 64), (2, 4, 64, 64), broadcast_axis=1)


class TestPdimBroadcast:
    """P-dim (partition) broadcast: tensor engine ones.T @ src."""

    def test_basic(self):
        _check((1, 128), (64, 128), broadcast_axis=0)

    def test_full_partition(self):
        _check((1, 128), (128, 128), broadcast_axis=0)

    def test_remainder_p(self):
        _check((1, 200), (100, 200), broadcast_axis=0)

    def test_large_f(self):
        _check((1, 512), (128, 512), broadcast_axis=0)

    def test_f_tiled(self):
        _check((1, 700), (64, 700), broadcast_axis=0)

    def test_batched(self):
        _check((2, 1, 128), (2, 64, 128), broadcast_axis=1)

    def test_batched_remainder(self):
        _check((3, 1, 200), (3, 100, 200), broadcast_axis=1)


class TestFdimBroadcast:
    """F-dim (free) broadcast: vector engine tensor_scalar_arith."""

    def test_basic(self):
        _check((64, 1), (64, 128), broadcast_axis=1)

    def test_full_tile(self):
        _check((128, 1), (128, 512), broadcast_axis=1)

    def test_remainder_f(self):
        _check((128, 1), (128, 300), broadcast_axis=1)

    def test_remainder_p(self):
        _check((100, 1), (100, 128), broadcast_axis=1)

    def test_both_remainder(self):
        _check((100, 1), (100, 300), broadcast_axis=1)

    def test_large(self):
        _check((128, 1), (128, 1024), broadcast_axis=1)

    def test_batched(self):
        _check((3, 128, 1), (3, 128, 256), broadcast_axis=2)

    def test_batched_remainder(self):
        _check((2, 100, 1), (2, 100, 300), broadcast_axis=2)
