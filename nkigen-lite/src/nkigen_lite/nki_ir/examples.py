"""Tiled kernel builders for nki_ir.

These generate NKI IR graphs using fori_loop / sequential_loop for
structured iteration.  The graphs can be executed directly (the interpreter
runs the loops), or unrolled into flat op sequences via `unroll_tile_loops`
before NISA lowering.

Tile indexing uses ``Builder.ts(tile_i, size, total)`` which mirrors
Kernel Builder's ``nb.ts(tile_i, size)``.  The ``total`` parameter
enables remainder-tile clamping: ``min(size, total - offset)`` when
the loop index is a concrete ``int`` (after unrolling).
"""

from __future__ import annotations

import math

from nkigen_lite.core import DType, Graph
from nkigen_lite.nki_ir.ir import (
    Builder,
    MemorySpace,
    NisaActivationOp,
    NisaArithOp,
    NisaReduceOp,
)


def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)


def lower_elementwise_add(
    M: int,
    N: int,
    dtype: DType = DType.F32,
    tile_p: int = 128,
    tile_f: int = 512,
) -> Graph:
    """Tile a 2D elementwise add: C = A + B.

    Tiles over partition dim (M, step tile_p) and free dim (N, step tile_f).
    Each tile: load A chunk, load B chunk, add in SBUF, store to C.
    Handles remainder tiles when M or N are not divisible by tile sizes.

    Uses fori_loop for full tiles and a static remainder body so that
    boundary DMA slices are clamped to valid extents.
    """
    b = Builder("tiled_add")
    a_hbm = b.add_input("a", (M, N), dtype)
    b_hbm = b.add_input("b", (M, N), dtype)
    c_hbm = b.add_input("c", (M, N), dtype)

    n_full_m = M // tile_p
    has_rem_m = (M % tile_p) != 0
    n_full_n = N // tile_f
    has_rem_n = (N % tile_f) != 0

    def _emit_add_tile(b, m_slice, n_slice):
        a_tile = b.dma_copy(b.alloc((m_slice.size, n_slice.size), dtype, MemorySpace.SBUF), a_hbm, (m_slice, n_slice))
        b_tile = b.dma_copy(b.alloc((m_slice.size, n_slice.size), dtype, MemorySpace.SBUF), b_hbm, (m_slice, n_slice))
        c_tile = b.tensor_tensor_arith(b.alloc((m_slice.size, n_slice.size), dtype, MemorySpace.SBUF), a_tile, b_tile, NisaArithOp.ADD)
        b.dma_copy(c_hbm, c_tile, (m_slice, n_slice))

    def _emit_n_loop(b, m_slice):
        if n_full_n > 0:
            def n_body(b, n_idx):
                n = b.ts(n_idx, tile_f, N)
                _emit_add_tile(b, m_slice, n)
            b.fori_loop("n_loop", n_full_n, 1, n_body)
        if has_rem_n:
            n_rem = b.ts(n_full_n, tile_f, N)
            _emit_add_tile(b, m_slice, n_rem)

    if n_full_m > 0:
        def m_body(b, m_idx):
            m = b.ts(m_idx, tile_p, M)
            _emit_n_loop(b, m)
        b.fori_loop("m_loop", n_full_m, 1, m_body)

    if has_rem_m:
        m_rem = b.ts(n_full_m, tile_p, M)
        _emit_n_loop(b, m_rem)

    b.set_outputs({"c": c_hbm})
    return b.graph


def lower_matmul(
    M: int,
    K: int,
    N: int,
    dtype: DType = DType.F32,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> Graph:
    """Tile matmul C[M,N] = A[M,K] @ B[K,N].

    Triple-nested tiling over M (partition of output), N (free of output),
    and K (contraction). For each (m, n) output tile, accumulates partial
    products across K tiles in PSUM via fori_loop.

    matmul computes stationary[K,M].T @ moving[K,N]:
      - A[m:, k:] is loaded as [tm, tk], transposed to stationary [tk, tm]
      - B[k:, n:] is loaded directly as moving [tk, tn]
      - Result [tm, tn] accumulates in PSUM (always FP32)

    Outer m/n loops are parallel (each tile writes to disjoint output region).
    Inner k loop accumulates into PSUM via matmul(accumulate=True).
    Remainder tiles are emitted as static bodies with clamped DMA extents.
    """
    b = Builder("tiled_matmul")
    a_hbm = b.add_input("a", (M, K), dtype)
    b_hbm = b.add_input("b", (K, N), dtype)
    c_hbm = b.add_input("c", (M, N), DType.F32)

    n_full_m = M // tile_m
    has_rem_m = (M % tile_m) != 0
    n_full_n = N // tile_n
    has_rem_n = (N % tile_n) != 0
    n_full_k = K // tile_k
    has_rem_k = (K % tile_k) != 0

    def _emit_matmul_tile(b, m_slice, n_slice):
        acc = b.alloc((m_slice.size, n_slice.size), DType.F32, MemorySpace.PSUM)
        acc = b.memset(acc, 0.0)

        if n_full_k > 0:
            def k_body(b, k_idx):
                k = b.ts(k_idx, tile_k, K)
                a_tile = b.dma_copy(b.alloc((m_slice.size, k.size), dtype, MemorySpace.SBUF), a_hbm, (m_slice, k))
                a_stat = b.transpose(a_tile, (1, 0))
                b_mov = b.dma_copy(b.alloc((k.size, n_slice.size), dtype, MemorySpace.SBUF), b_hbm, (k, n_slice))
                b.matmul(acc, a_stat, b_mov, accumulate=True)
            b.fori_loop("k_loop", n_full_k, 1, k_body)

        if has_rem_k:
            k_rem = b.ts(n_full_k, tile_k, K)
            a_tile = b.dma_copy(b.alloc((m_slice.size, k_rem.size), dtype, MemorySpace.SBUF), a_hbm, (m_slice, k_rem))
            a_stat = b.transpose(a_tile, (1, 0))
            b_mov = b.dma_copy(b.alloc((k_rem.size, n_slice.size), dtype, MemorySpace.SBUF), b_hbm, (k_rem, n_slice))
            b.matmul(acc, a_stat, b_mov, accumulate=True)

        c_sbuf = b.tensor_copy(b.alloc((m_slice.size, n_slice.size), DType.F32, MemorySpace.SBUF), acc)
        b.dma_copy(c_hbm, c_sbuf, (m_slice, n_slice))

    def _emit_n_loop(b, m_slice):
        if n_full_n > 0:
            def n_body(b, n_idx):
                n = b.ts(n_idx, tile_n, N)
                _emit_matmul_tile(b, m_slice, n)
            b.fori_loop("n_loop", n_full_n, 1, n_body)
        if has_rem_n:
            n_rem = b.ts(n_full_n, tile_n, N)
            _emit_matmul_tile(b, m_slice, n_rem)

    if n_full_m > 0:
        def m_body(b, m_idx):
            m = b.ts(m_idx, tile_m, M)
            _emit_n_loop(b, m)
        b.fori_loop("m_loop", n_full_m, 1, m_body)

    if has_rem_m:
        m_rem = b.ts(n_full_m, tile_m, M)
        _emit_n_loop(b, m_rem)

    b.set_outputs({"c": c_hbm})
    return b.graph


def lower_softmax(
    M: int,
    N: int,
    dtype: DType = DType.F32,
    tile_p: int = 128,
) -> Graph:
    """Tile softmax along axis=1 (free dim).

    Tiles over partition dim (M, step tile_p). Each partition-tile loads
    the full row (N elements), computes softmax, and stores back.
    Remainder tiles are emitted as a static body with clamped P-extent.

    Requires N <= PSUM_FREE_MAX (512 on gen2/gen3) so each row fits in
    one free-dim tile. For larger N, use online (flash) softmax.

    Uses fori_loop for full tiles. Body uses NISA ops (tensor_reduce_arith,
    activation, tensor_scalar_arith).
    """
    b = Builder("tiled_softmax")
    x_hbm = b.add_input("x", (M, N), dtype)
    y_hbm = b.add_input("y", (M, N), dtype)

    n_full_p = M // tile_p
    has_rem_p = (M % tile_p) != 0

    def _emit_softmax_tile(b, p_slice):
        f = b.full(N)
        x = b.dma_copy(b.alloc((p_slice.size, N), dtype, MemorySpace.SBUF), x_hbm, (p_slice, f))

        x_max = b.tensor_reduce_arith(b.alloc((p_slice.size, 1), dtype, MemorySpace.SBUF), x, NisaReduceOp.MAX, num_r_dim=1)
        neg_max = b.neg(b.alloc((p_slice.size, 1), dtype, MemorySpace.SBUF), x_max)
        x_exp = b.activation(b.alloc((p_slice.size, N), dtype, MemorySpace.SBUF), x, NisaActivationOp.EXP, bias=neg_max)

        x_sum = b.tensor_reduce_arith(b.alloc((p_slice.size, 1), dtype, MemorySpace.SBUF), x_exp, NisaReduceOp.ADD, num_r_dim=1)
        inv_sum = b.activation(b.alloc((p_slice.size, 1), dtype, MemorySpace.SBUF), x_sum, NisaActivationOp.RECIPROCAL)
        probs = b.tensor_scalar_arith(b.alloc((p_slice.size, N), dtype, MemorySpace.SBUF), x_exp, inv_sum, NisaArithOp.MULTIPLY)

        b.dma_copy(y_hbm, probs, (p_slice, f))

    if n_full_p > 0:
        def p_body(b, p_idx):
            p = b.ts(p_idx, tile_p, M)
            _emit_softmax_tile(b, p)
        b.fori_loop("p_loop", n_full_p, 1, p_body)

    if has_rem_p:
        p_rem = b.ts(n_full_p, tile_p, M)
        _emit_softmax_tile(b, p_rem)

    b.set_outputs({"y": y_hbm})
    return b.graph
