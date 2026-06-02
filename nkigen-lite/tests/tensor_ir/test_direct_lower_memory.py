"""Tests for direct_lower_memory (reshape, slice, concat).

Verifies via numpy interpreter then real Trainium hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_memory import (
    lower_reshape,
    lower_slice,
    lower_concat,
)

try:
    import nki.compiler.kernel_builder as nb_kb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

pytestmark = pytest.mark.hw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(nki_graph, inputs, expected, atol=1e-5):
    """Verify NKI IR graph: interpreter gate then real HW execution."""
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")

    # Interpreter gate
    interp = nki_run(nki_graph, inputs)
    np.testing.assert_allclose(
        interp["y"], expected, atol=atol, rtol=atol,
        err_msg="Interpreter mismatch (must pass before HW)",
    )

    # Real hardware execution
    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_inputs = {k: v for k, v in inputs.items() if k != "y"}
    hw_outputs = {"y": np.zeros_like(expected)}
    nb_kb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    np.testing.assert_allclose(
        hw_outputs["y"], expected, atol=atol, rtol=atol,
        err_msg="HW mismatch",
    )


# ---------------------------------------------------------------------------
# Reshape tests
# ---------------------------------------------------------------------------


def _reshape_inputs(in_shape, out_shape, x):
    """Build inputs dict for reshape, including scratch if needed."""
    from math import prod as _prod
    inputs = {"x": x, "y": np.zeros(out_shape, dtype=np.float32)}
    if in_shape[-1] != out_shape[-1]:
        total = _prod(in_shape)
        in_f = in_shape[-1]
        inputs["scratch"] = np.zeros((total // in_f, in_f), dtype=np.float32)
    return inputs


class TestReshape:
    def test_flatten(self):
        """(4, 128, 64) -> (512, 64)"""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 128, 64)).astype(np.float32)
        expected = x.reshape(512, 64)
        graph = lower_reshape((4, 128, 64), (512, 64))
        _check(graph, _reshape_inputs((4, 128, 64), (512, 64), x), expected)

    def test_unflatten(self):
        """(512, 64) -> (4, 128, 64)"""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((512, 64)).astype(np.float32)
        expected = x.reshape(4, 128, 64)
        graph = lower_reshape((512, 64), (4, 128, 64))
        _check(graph, _reshape_inputs((512, 64), (4, 128, 64), x), expected)

    def test_merge_last_two(self):
        """(4, 8, 32) -> (4, 256)"""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, 8, 32)).astype(np.float32)
        expected = x.reshape(4, 256)
        graph = lower_reshape((4, 8, 32), (4, 256))
        _check(graph, _reshape_inputs((4, 8, 32), (4, 256), x), expected)

    def test_split_last(self):
        """(128, 256) -> (128, 4, 64)"""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((128, 256)).astype(np.float32)
        expected = x.reshape(128, 4, 64)
        graph = lower_reshape((128, 256), (128, 4, 64))
        _check(graph, _reshape_inputs((128, 256), (128, 4, 64), x), expected)

    def test_same_shape(self):
        """No-op reshape (128, 64) -> (128, 64)"""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((128, 64)).astype(np.float32)
        expected = x.copy()
        graph = lower_reshape((128, 64), (128, 64))
        _check(graph, _reshape_inputs((128, 64), (128, 64), x), expected)

    def test_large_p(self):
        """P > 128: (300, 64) -> (300, 64) identity with P-tiling."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((300, 64)).astype(np.float32)
        expected = x.copy()
        graph = lower_reshape((300, 64), (300, 64))
        _check(graph, _reshape_inputs((300, 64), (300, 64), x), expected)

    def test_column_to_row(self):
        """(256, 1) -> (1, 256): column vector to row vector."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal((256, 1)).astype(np.float32)
        expected = x.reshape(1, 256)
        graph = lower_reshape((256, 1), (1, 256))
        _check(graph, _reshape_inputs((256, 1), (1, 256), x), expected)

    def test_row_to_column(self):
        """(1, 256) -> (256, 1): row vector to column vector."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((1, 256)).astype(np.float32)
        expected = x.reshape(256, 1)
        graph = lower_reshape((1, 256), (256, 1))
        _check(graph, _reshape_inputs((1, 256), (256, 1), x), expected)


# ---------------------------------------------------------------------------
# Slice tests
# ---------------------------------------------------------------------------


class TestSlice:
    def test_basic_rank2(self):
        """Slice middle of a 2D tensor."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((128, 256)).astype(np.float32)
        expected = x[32:96, 64:192]
        graph = lower_slice((128, 256), starts=(32, 64), stops=(96, 192))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)

    def test_first_half(self):
        """First half along P-dim."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((256, 128)).astype(np.float32)
        expected = x[:128, :]
        graph = lower_slice((256, 128), starts=(0, 0), stops=(128, 128))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)

    def test_second_half(self):
        """Second half along P-dim."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((256, 128)).astype(np.float32)
        expected = x[128:, :]
        graph = lower_slice((256, 128), starts=(128, 0), stops=(256, 128))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)

    def test_f_dim_slice(self):
        """Slice along F-dim only."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((128, 512)).astype(np.float32)
        expected = x[:, 128:384]
        graph = lower_slice((128, 512), starts=(0, 128), stops=(128, 384))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)

    def test_rank3(self):
        """Slice in rank-3 tensor (batch + P + F)."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((4, 128, 64)).astype(np.float32)
        expected = x[1:3, 32:96, :32]
        graph = lower_slice((4, 128, 64), starts=(1, 32, 0), stops=(3, 96, 32))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)

    def test_single_element(self):
        """Extract a single row."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((128, 64)).astype(np.float32)
        expected = x[42:43, :]
        graph = lower_slice((128, 64), starts=(42, 0), stops=(43, 64))
        _check(graph, {"x": x, "y": np.zeros_like(expected)}, expected)


# ---------------------------------------------------------------------------
# Concat tests
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_p_dim(self):
        """Concat along P-dim (axis=-2)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 128)).astype(np.float32)
        b_arr = rng.standard_normal((64, 128)).astype(np.float32)
        expected = np.concatenate([a, b_arr], axis=0)
        graph = lower_concat([(64, 128), (64, 128)], axis=0)
        _check(graph, {"x0": a, "x1": b_arr, "y": np.zeros_like(expected)}, expected)

    def test_concat_f_dim(self):
        """Concat along F-dim (axis=-1)."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((128, 64)).astype(np.float32)
        b_arr = rng.standard_normal((128, 128)).astype(np.float32)
        expected = np.concatenate([a, b_arr], axis=1)
        graph = lower_concat([(128, 64), (128, 128)], axis=1)
        _check(graph, {"x0": a, "x1": b_arr, "y": np.zeros_like(expected)}, expected)

    def test_concat_batch_dim(self):
        """Concat along batch dim (axis=0 in rank-3)."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((2, 64, 32)).astype(np.float32)
        b_arr = rng.standard_normal((3, 64, 32)).astype(np.float32)
        expected = np.concatenate([a, b_arr], axis=0)
        graph = lower_concat([(2, 64, 32), (3, 64, 32)], axis=0)
        _check(graph, {"x0": a, "x1": b_arr, "y": np.zeros_like(expected)}, expected)

    def test_concat_three_inputs(self):
        """Concat three tensors."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal((128, 32)).astype(np.float32)
        b_arr = rng.standard_normal((128, 64)).astype(np.float32)
        c = rng.standard_normal((128, 32)).astype(np.float32)
        expected = np.concatenate([a, b_arr, c], axis=1)
        graph = lower_concat([(128, 32), (128, 64), (128, 32)], axis=1)
        _check(graph, {"x0": a, "x1": b_arr, "x2": c, "y": np.zeros_like(expected)}, expected)

    def test_concat_large_p(self):
        """Concat with P > 128."""
        rng = np.random.default_rng(4)
        a = rng.standard_normal((200, 64)).astype(np.float32)
        b_arr = rng.standard_normal((100, 64)).astype(np.float32)
        expected = np.concatenate([a, b_arr], axis=0)
        graph = lower_concat([(200, 64), (100, 64)], axis=0)
        _check(graph, {"x0": a, "x1": b_arr, "y": np.zeros_like(expected)}, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-q"])
