"""Tests for nki_ir example kernels (elementwise add, matmul, softmax).

Verifies correctness via the numpy interpreter across a range of shapes,
tile sizes, and boundary conditions. Catches regressions in tiling logic,
remainder-tile handling, and PSUM accumulation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax

from nkigen_lite.nki_ir import run
from nkigen_lite.nki_ir.examples import lower_elementwise_add, lower_matmul, lower_softmax


def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)


class TestElementwiseAdd:
    """C = A + B, tiled over P and F dimensions."""

    @pytest.mark.parametrize("M,N", [
        (128, 512),       # single P-tile, single F-tile
        (256, 512),       # multiple P-tiles, single F-tile
        (128, 1024),      # single P-tile, multiple F-tiles
        (256, 1024),      # multiple P and F tiles
        (200, 700),       # P-remainder (200 % 128 = 72), F-remainder (700 % 512 = 188)
        (128, 100),       # small F (single tile, no remainder)
        (1, 512),         # single partition
        (128, 1),         # single free element
        (300, 300),       # both remainder
        (64, 256),        # P < tile_p (partial first tile)
    ])
    def test_shapes(self, M, N):
        graph = lower_elementwise_add(M, N)
        np.random.seed(42)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(out["c"], a + b, rtol=1e-5)

    @pytest.mark.parametrize("tile_p,tile_f", [
        (128, 512),   # default
        (64, 256),    # smaller tiles
        (128, 128),   # small F tile
        (32, 1024),   # small P tile, large F tile
        (128, 64),    # very small F tile
    ])
    def test_tile_sizes(self, tile_p, tile_f):
        M, N = 256, 1024
        graph = lower_elementwise_add(M, N, tile_p=tile_p, tile_f=tile_f)
        np.random.seed(7)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(out["c"], a + b, rtol=1e-5)

    def test_exact_tile_boundary(self):
        """Shape is exact multiple of tile size — no remainder handling needed."""
        M, N = 256, 1024
        graph = lower_elementwise_add(M, N, tile_p=128, tile_f=512)
        np.random.seed(11)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(out["c"], a + b, rtol=1e-5)

    def test_single_tile(self):
        """Entire tensor fits in one tile."""
        M, N = 64, 256
        graph = lower_elementwise_add(M, N, tile_p=128, tile_f=512)
        np.random.seed(13)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(out["c"], a + b, rtol=1e-5)

    def test_large_shape(self):
        """Stress test with many tiles."""
        M, N = 1024, 2048
        graph = lower_elementwise_add(M, N, tile_p=128, tile_f=512)
        np.random.seed(17)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(out["c"], a + b, rtol=1e-5)


class TestMatmul:
    """C[M,N] = A[M,K] @ B[K,N], tiled with K-accumulation in PSUM."""

    @pytest.mark.parametrize("M,K,N", [
        (128, 128, 128),   # single tile per dim
        (256, 128, 128),   # M tiled
        (128, 256, 128),   # K tiled (accumulation)
        (128, 128, 256),   # N tiled
        (256, 256, 256),   # all dims tiled
        (200, 300, 400),   # all remainders
        (128, 512, 128),   # deep K accumulation (4 tiles)
        (64, 64, 64),      # everything fits single tile
        (300, 128, 300),   # M and N remainder, K exact
        (128, 100, 128),   # K remainder only
    ])
    def test_shapes(self, M, K, N):
        graph = lower_matmul(M, K, N)
        np.random.seed(42)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        expected = a @ b
        np.testing.assert_allclose(out["c"], expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("tile_m,tile_k,tile_n", [
        (128, 128, 128),  # default
        (64, 64, 64),     # smaller tiles
        (128, 64, 128),   # smaller K tile (more accumulation steps)
        (64, 128, 256),   # asymmetric
        (128, 128, 64),   # small N tile
    ])
    def test_tile_sizes(self, tile_m, tile_k, tile_n):
        M, K, N = 256, 256, 256
        graph = lower_matmul(M, K, N, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
        np.random.seed(7)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        expected = a @ b
        np.testing.assert_allclose(out["c"], expected, rtol=1e-4, atol=1e-4)

    def test_k_accumulation_correctness(self):
        """Verify K-tiling accumulates partial products correctly.

        With K=512 and tile_k=128, the matmul runs 4 K-iterations and
        accumulates in PSUM. Result must match single un-tiled matmul.
        """
        M, K, N = 128, 512, 128
        graph = lower_matmul(M, K, N, tile_k=128)
        np.random.seed(19)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        expected = a @ b
        np.testing.assert_allclose(out["c"], expected, rtol=1e-4, atol=1e-4)

    def test_non_square(self):
        """Highly non-square: tall-skinny × skinny-wide."""
        M, K, N = 512, 32, 1024
        graph = lower_matmul(M, K, N)
        np.random.seed(23)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        expected = a @ b
        np.testing.assert_allclose(out["c"], expected, rtol=1e-4, atol=1e-4)

    def test_single_k_tile(self):
        """K fits in one tile — no accumulation loop."""
        M, K, N = 256, 64, 256
        graph = lower_matmul(M, K, N, tile_k=128)
        np.random.seed(29)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        out = run(graph, {"a": a, "b": b, "c": c})
        expected = a @ b
        np.testing.assert_allclose(out["c"], expected, rtol=1e-4, atol=1e-4)


class TestSoftmax:
    """softmax(x, axis=1): row-wise softmax over free dimension."""

    @pytest.mark.parametrize("M,N", [
        (128, 128),    # single P-tile, moderate N
        (128, 256),    # single P-tile, larger N
        (128, 512),    # single P-tile, max PSUM free
        (256, 128),    # multiple P-tiles
        (256, 256),    # multiple P-tiles, moderate N
        (200, 300),    # P-remainder
        (64, 128),     # P < tile_p
        (300, 256),    # P-remainder, moderate N
        (512, 128),    # many P-tiles
        (1, 128),      # single row
    ])
    def test_shapes(self, M, N):
        graph = lower_softmax(M, N)
        np.random.seed(42)
        x = np.random.randn(M, N).astype(np.float32)
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        expected = scipy_softmax(x, axis=1)
        np.testing.assert_allclose(out["y"], expected, rtol=1e-5)

    @pytest.mark.parametrize("tile_p", [128, 64, 32])
    def test_tile_sizes(self, tile_p):
        M, N = 256, 256
        graph = lower_softmax(M, N, tile_p=tile_p)
        np.random.seed(7)
        x = np.random.randn(M, N).astype(np.float32)
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        expected = scipy_softmax(x, axis=1)
        np.testing.assert_allclose(out["y"], expected, rtol=1e-5)

    def test_numerical_stability(self):
        """Large input values should not overflow due to max-subtraction."""
        M, N = 128, 256
        graph = lower_softmax(M, N)
        np.random.seed(31)
        x = np.random.randn(M, N).astype(np.float32) * 100  # large values
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        expected = scipy_softmax(x, axis=1)
        np.testing.assert_allclose(out["y"], expected, rtol=1e-4)

    def test_uniform_input(self):
        """Uniform input → uniform output (1/N per element)."""
        M, N = 128, 128
        graph = lower_softmax(M, N)
        x = np.ones((M, N), dtype=np.float32)
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        expected = np.full((M, N), 1.0 / N, dtype=np.float32)
        np.testing.assert_allclose(out["y"], expected, rtol=1e-5)

    def test_one_hot_input(self):
        """One large value per row → output should be near-one-hot."""
        M, N = 128, 128
        graph = lower_softmax(M, N)
        x = np.full((M, N), -100.0, dtype=np.float32)
        for i in range(M):
            x[i, i % N] = 100.0
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        for i in range(M):
            assert out["y"][i, i % N] > 0.99

    def test_row_sums_to_one(self):
        """Each row of softmax output must sum to 1."""
        M, N = 200, 300
        graph = lower_softmax(M, N)
        np.random.seed(37)
        x = np.random.randn(M, N).astype(np.float32)
        y = np.zeros_like(x)
        out = run(graph, {"x": x, "y": y})
        row_sums = out["y"].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(M), rtol=1e-5)


class TestElementwiseAddHW:
    @pytest.mark.parametrize("M,N", [
        (256, 512),
        (200, 700),
        (128, 1024),
    ])
    def test_shapes_hw(self, compile_and_run, M, N):
        graph = lower_elementwise_add(M, N)
        np.random.seed(42)
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        compile_and_run(graph, {"a": a, "b": b}, {"c": c})
        np.testing.assert_allclose(c, a + b, rtol=1e-5)


class TestMatmulHW:
    @pytest.mark.parametrize("M,K,N", [
        (256, 256, 256),
        (200, 300, 400),
        (128, 512, 128),
    ])
    def test_shapes_hw(self, compile_and_run, M, K, N):
        graph = lower_matmul(M, K, N)
        np.random.seed(42)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        c = np.zeros((M, N), dtype=np.float32)
        compile_and_run(graph, {"a": a, "b": b}, {"c": c})
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)


class TestSoftmaxHW:
    @pytest.mark.parametrize("M,N", [
        (256, 256),
        (200, 300),
        (512, 128),
    ])
    def test_shapes_hw(self, compile_and_run, M, N):
        graph = lower_softmax(M, N)
        np.random.seed(42)
        x = np.random.randn(M, N).astype(np.float32)
        y = np.zeros_like(x)
        compile_and_run(graph, {"x": x}, {"y": y})
        expected = scipy_softmax(x, axis=1)
        np.testing.assert_allclose(y, expected, rtol=1e-4)
