"""Tests for direct matmul lowering (tensor IR -> NKI IR).

Verifies correctness on real Trainium hardware across a range of shapes:
single-tile, multi-tile, remainder tiles, batched, and broadcast batches.
"""

from __future__ import annotations

import numpy as np
import ml_dtypes
import pytest

from nkigen_lite.core import DType
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_matmul import lower_matmul

import nki.compiler.kernel_builder as nb

pytestmark = pytest.mark.hw


def _check(a_shape, b_shape, dtype=DType.F32, atol=1e-3):
    """Lower matmul and verify on real Trainium hardware."""
    rng = np.random.default_rng(42)
    np_dtype = ml_dtypes.bfloat16 if dtype == DType.BF16 else np.float32
    a = rng.standard_normal(a_shape).astype(np_dtype)
    b = rng.standard_normal(b_shape).astype(np_dtype)
    ref = a.astype(np.float32) @ b.astype(np.float32)

    graph = lower_matmul(a_shape, b_shape, dtype=dtype)
    kernel_fn = build_kb_kernel(graph)
    hw_out = {"c": np.zeros(ref.shape, dtype=np.float32)}
    nb.compile_and_execute(
        kernel_fn,
        inputs={"a": a, "b": b},
        outputs=hw_out,
        compile_opts=nb.CompileOptions(target="trn2"),
    )
    np.testing.assert_allclose(
        hw_out["c"], ref, atol=atol, rtol=atol,
        err_msg="HW mismatch",
    )


class TestRank2:
    """Basic 2D matmul A[M,K] @ B[K,N] -> C[M,N]."""

    def test_single_tile(self):
        _check((64, 64), (64, 64))

    def test_exact_tile(self):
        _check((128, 128), (128, 128))

    def test_m_tiled(self):
        _check((256, 128), (128, 128))

    def test_k_tiled(self):
        _check((128, 256), (256, 128))

    def test_n_tiled(self):
        _check((128, 128), (128, 512))

    def test_n_large(self):
        _check((128, 128), (128, 1024))

    def test_all_tiled(self):
        _check((256, 256), (256, 256))

    def test_all_remainder(self):
        _check((200, 300), (300, 400))

    def test_deep_k(self):
        _check((128, 512), (512, 128))

    def test_small(self):
        _check((32, 64), (64, 48))

    def test_m_remainder(self):
        _check((300, 128), (128, 128))

    def test_k_remainder(self):
        _check((128, 100), (100, 128))

    def test_n_remainder(self):
        _check((128, 128), (128, 300))

    def test_large(self):
        _check((512, 256), (256, 512))


class TestBatched:
    """Batched matmul with matching batch dims."""

    def test_single_batch(self):
        _check((2, 128, 128), (2, 128, 128))

    def test_multi_batch(self):
        _check((4, 64, 64), (4, 64, 64))

    def test_batch_remainder(self):
        _check((3, 200, 100), (3, 100, 150))

    def test_multi_dim_batch(self):
        _check((2, 3, 64, 64), (2, 3, 64, 64))


class TestBroadcast:
    """Batched matmul with broadcast batch dims."""

    def test_b_broadcast(self):
        _check((4, 128, 64), (1, 64, 128))

    def test_a_broadcast(self):
        _check((1, 128, 64), (4, 64, 128))

    def test_multi_dim_broadcast(self):
        _check((2, 1, 64, 64), (1, 3, 64, 64))


class TestBF16:
    """BF16 input matmul (output always FP32 from PSUM)."""

    def test_exact_tile(self):
        _check((128, 128), (128, 128), dtype=DType.BF16)

    def test_all_remainder(self):
        _check((200, 300), (300, 400), dtype=DType.BF16)

    def test_large(self):
        _check((256, 512), (512, 256), dtype=DType.BF16)

    def test_batched(self):
        _check((2, 128, 64), (2, 64, 128), dtype=DType.BF16)

    def test_broadcast(self):
        _check((4, 64, 64), (1, 64, 64), dtype=DType.BF16)
