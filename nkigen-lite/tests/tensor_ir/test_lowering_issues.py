"""Regression tests for issues catalogued in nkigen_lite/docs/LOWERING_ISSUES.md.

Tests are intentionally minimal — they check the symptom described in
the doc, not full correctness. Add a richer test elsewhere when fixing.
"""
from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder, run as tensor_run
from nkigen_lite.tensor_ir.passes.lower_to_nki import lower_to_nki
from nkigen_lite.nki_ir import run as nki_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lower_and_run(build_fn, inputs, out_shapes, atol=1e-3, rtol=1e-3):
    """Build a tensor_ir graph, lower, run both interpreters, and compare.

    Raises whatever the lowering / run / compare step raises so xfail can
    catch on the documented exception type.
    """
    b = Builder("t")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)
    nki_graph = lower_to_nki(b.graph)
    nki_inputs = dict(inputs)
    for name, shape in out_shapes.items():
        nki_inputs[f"{name}_out"] = np.zeros(shape, dtype=np.float32)
    actual = nki_run(nki_graph, nki_inputs)
    for k in ref:
        np.testing.assert_allclose(actual[k], ref[k], atol=atol, rtol=rtol)


# bug-1 (M > partition_max): fixed by using out_ts.sizes[m_dim_in_out] for
# the M slice in _emit_matmul. Verified directly:

def test_bug1_matmul_m_greater_than_partition_max():
    """M=256 > partition_max=128 — must tile M and produce correct result."""
    np.random.seed(0)
    M, K, N = 256, 64, 64
    A = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)

    def build(b):
        a = b.add_input("a", (M, K))
        w = b.add_input("w", (K, N))
        b.set_outputs({"out": b.matmul(a, w)})

    _lower_and_run(build, {"a": A, "w": W}, {"out": (M, N)})


# bug-2 (stride-0 broadcast load): fixed; covered by
# test_notebook_patterns.TestRmsnormPatterns::test_rank2_with_d_above_sbuf_split.


# bug-3: _propagate rewrites priority to ELEMENTWISE
# This was a bug in the old layout_analysis.py constraint system.
# The replacement layout_solver.py uses direct layout propagation via
# _adapt_layout (no priority system) so this class of bug doesn't apply.


# ---------------------------------------------------------------------------
# bug-5: for_loop is silently dropped
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="for_loop not supported by basic direct_lower strategy")
def test_bug5_for_loop_silently_dropped():
    """A for_loop over add(accum, x) for 4 iterations should yield 4*x."""
    x_np = np.random.randn(8, 64).astype(np.float32)

    def build(b):
        x = b.add_input("x", (8, 64))
        init = b.constant(0.0, (8, 64), DType.F32)

        def body(b2, idx, accum):
            return [b2.add(accum, x)]

        out = b.for_loop(4, [init], body)[0]
        b.set_outputs({"out": out})

    b = Builder("t")
    build(b)
    ref = tensor_run(b.graph, {"x": x_np})
    nki_graph = lower_to_nki(b.graph)
    nki_inputs = {"x": x_np, "out_out": np.zeros((8, 64), dtype=np.float32)}
    actual = nki_run(nki_graph, nki_inputs)
    for k in ref:
        np.testing.assert_allclose(actual[k], ref[k], atol=1e-3, rtol=1e-3)


# gap-1 (split-after-matmul): fixed by SliceAfterMatmulPattern in
# nkigen_lite/tensor_ir/passes/decompose.py. Covered by
# test_notebook_patterns.TestFeedforwardPatterns::test_swiglu_fused_gate_up_split.

# gap-2 (rank-3+ matmul LHS, rank-2 RHS): fixed by flatten_to_2d pass.
# Covered by test_notebook_patterns.TestRmsnormPatterns::test_rank3_input_rank1_weight
# (which exercises the rank-3 elementwise path) and the feedforward pattern
# (which exercises rank-3 LHS @ rank-2 RHS via flatten + 2D matmul).

# gap-3 (rank-N elementwise + rank-2 operand): partially fixed.
# - rank-3 + rank-1 weight: works (covered by rmsnorm test).
# - rank-4 with rank-2 broadcast across multiple leading axes (RoPE): not yet
#   handled — requires structured replication that simple flatten can't do.
# See also perf-1 below: a rank-4 RoPE q/k mix no longer unrolls the sequence
# axis (segmenter splits on a second distinct collapsed extent).


# smell-4 (late `Graph` import in lower_to_nki): fixed; the function now
# uses the module-level `Graph` import via the `_verify` helper.


# ---------------------------------------------------------------------------
# perf-1: mixed-collapse elementwise segment unrolled a packed axis
# ---------------------------------------------------------------------------
# RoPE applies the same cos/sin to a q tensor (1, S, Hq, D) and a k tensor
# (1, S, Hk, D) with Hq != Hk. The layout solver classifies both with the same
# (p_dims, f_dims), so the segmenter used to merge their elementwise ops into one
# group. The group then collapsed to two different partition extents
# (prod(1,S,Hq) vs prod(1,S,Hk)), the fast leading-dims-onto-partition path
# bailed, and the generic fallback put the size-1-ish axis on the partition and
# unrolled S one element at a time — an S-fold op blowup (8351 nki ops for S=128
# RoPE). The segmenter now splits on a second distinct collapsed extent so each
# group stays cleanly collapsible.

def test_perf1_mixed_collapse_elementwise_no_blowup():
    """q-path (Hq=8) and k-path (Hk=1) elementwise ops must not share a segment
    and unroll the sequence axis. Checks correctness and a bounded op count."""
    np.random.seed(0)
    S, Hq, Hk, D = 64, 8, 1, 16
    xq = np.random.randn(1, S, Hq, D).astype(np.float32)
    xk = np.random.randn(1, S, Hk, D).astype(np.float32)
    cq = np.random.randn(1, S, Hq, D).astype(np.float32)
    ck = np.random.randn(1, S, Hk, D).astype(np.float32)

    def build(b):
        q = b.add_input("xq", (1, S, Hq, D))
        k = b.add_input("xk", (1, S, Hk, D))
        wq = b.add_input("cq", (1, S, Hq, D))
        wk = b.add_input("ck", (1, S, Hk, D))
        # Interleave the q/k chains the way attention does, so the trace order
        # puts both shapes' ops adjacent (the merge condition).
        b.set_outputs({"oq": b.mul(q, wq), "ok": b.mul(k, wk)})

    _lower_and_run(
        build,
        {"xq": xq, "xk": xk, "cq": cq, "ck": ck},
        {"oq": (1, S, Hq, D), "ok": (1, S, Hk, D)},
    )

    # Op-count guard on the precise symptom: before the fix the merged segment
    # put the size-1-ish axis on the partition and unrolled S=64, emitting one
    # compute op per (sequence, ...) element (128 tensor_tensor_arith). A clean
    # collapse tiles each mul over the packed partition — a handful of ops.
    b = Builder("t")
    build(b)
    nki_graph = lower_to_nki(b.graph)
    n_compute = sum(1 for o in nki_graph.ops if o.opcode == "tensor_tensor_arith")
    assert n_compute <= 8, (
        f"mixed-collapse elementwise unrolled to {n_compute} compute ops"
    )


# ---------------------------------------------------------------------------
# perf-2: concat/slice on a non-last axis unrolled the leading (sequence) axis
# ---------------------------------------------------------------------------
# The KV-cache update `cache[:, :seq] = x` lowers to a concat on axis 1 of a
# (1, max_seq_len, n_kv, head_dim) tensor, and its read-back to a slice on the
# same axis. Both used to tile shape[-2] (= n_kv, often 1) as the partition and
# unroll prod(shape[:-2]) = max_seq_len leading rows one DMA at a time — e.g.
# (1, 4096, 1, 128) emitted ~12k ops per concat/slice, ~44% of the whole 30B
# layer. The fast path now folds the operated axis onto the partition (an
# (L, A, T) view), packing 128 lanes. The full-tensor HBM output copy got the
# same rank>=3 collapse.

def test_perf2_kv_cache_concat_slice_no_seq_unroll():
    """concat/slice on a non-last axis of a tall (1, S, 1, D) tensor must fold
    S onto the partition, not unroll it. Checks correctness and op count."""
    np.random.seed(0)
    S, NEW, D = 512, 8, 64  # S tall enough that a per-row unroll would dominate
    old = np.random.randn(1, S - NEW, 1, D).astype(np.float32)
    new = np.random.randn(1, NEW, 1, D).astype(np.float32)
    full = np.random.randn(1, S, 1, D).astype(np.float32)

    # concat (cache write): [new, old] along axis 1 -> (1, S, 1, D)
    def build_concat(b):
        n = b.add_input("new", (1, NEW, 1, D))
        o = b.add_input("old", (1, S - NEW, 1, D))
        b.set_outputs({"y": b.concat([n, o], axis=1)})

    _lower_and_run(build_concat, {"new": new, "old": old}, {"y": (1, S, 1, D)})

    # slice (cache read): x[:, :S-NEW] along axis 1
    def build_slice(b):
        x = b.add_input("x", (1, S, 1, D))
        b.set_outputs({"y": b.slice(x, (0, 0, 0, 0), (1, S - NEW, 1, D))})

    _lower_and_run(build_slice, {"x": full}, {"y": (1, S - NEW, 1, D)})

    # Op-count guard: a per-sequence-row unroll would emit O(S) DMAs (~1000+ for
    # concat's two inputs). The collapsed path tiles S at 128 → a few dozen.
    for name, build in [("concat", build_concat), ("slice", build_slice)]:
        b = Builder("t")
        build(b)
        n_dma = sum(1 for o in lower_to_nki(b.graph).ops if o.opcode == "dma_copy")
        assert n_dma < 64, f"{name} unrolled the sequence axis: {n_dma} DMAs"


# REVIEW_basic_lowering.md item 6: where must be inf-safe. The old
# cond*x + (1-cond)*y form produced NaN whenever the unselected branch was
# inf (0 * inf = NaN) — exactly the masked-attention where(mask, scores, -inf)
# pattern. Now lowered via copy_predicated, which never evaluates arithmetic
# on the unselected branch.

def test_where_inf_safe():
    np.random.seed(0)
    S = 64
    mask = (np.random.uniform(size=(S, S)) > 0.5).astype(np.float32)
    scores = np.random.randn(S, S).astype(np.float32)

    def build(b):
        m = b.add_input("mask", (S, S))
        s = b.add_input("scores", (S, S))
        ninf = b.constant(float("-inf"), (S, S), DType.F32)
        b.set_outputs({"r": b.where(m, s, ninf)})

    b = Builder("t")
    build(b)
    nki_graph = lower_to_nki(b.graph)
    out = nki_run(nki_graph, {
        "mask": mask, "scores": scores,
        "r_out": np.zeros((S, S), dtype=np.float32),
    })["r"]
    assert not np.isnan(out).any(), "where produced NaN on an inf branch"
    np.testing.assert_array_equal(out, np.where(mask > 0, scores, -np.inf))


# REVIEW_basic_lowering.md item 1: the transpose passthrough-partition fast
# path crashed ('too many values to unpack') when the collapsed leading dim
# is not 128-aligned: (2,100,8,64) perm (0,1,3,2) collapses B=200 from (2,100)
# and the second 128-row tile straddles the boundary at row 100.

def test_transpose_collapsed_leading_dim_unaligned():
    np.random.seed(0)
    x = np.random.randn(2, 100, 8, 64).astype(np.float32)

    def build(b):
        t = b.add_input("x", (2, 100, 8, 64))
        b.set_outputs({"y": b.transpose(t, (0, 1, 3, 2))})

    _lower_and_run(build, {"x": x}, {"y": (2, 100, 64, 8)})


# REVIEW_basic_lowering.md item 3: P-reduce with a wide F emitted tiles over
# hardware limits (gpsimd: SBUF per-partition overflow at full-F tiles;
# matmul strategy: PSUM accumulator wider than PSUM_FREE_MAX). Both now tile
# the innermost F. Checked through the standalone wrappers since lower_graph
# always uses gpsimd.

def test_p_reduce_wide_f_legal_tiles():
    from nkigen_lite.tensor_ir.passes.layout_solver import solve_graph
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_reduce import (
        lower_p_reduce_gpsimd,
        lower_p_reduce_matmul,
    )

    np.random.seed(0)
    for fn, shape in [(lower_p_reduce_gpsimd, (256, 131072)),
                      (lower_p_reduce_matmul, (256, 1024))]:
        b = Builder("t")
        x = b.add_input("x", shape, DType.F32)
        y = b.reduce(x, axis=(0,), kind="sum", keepdims=True)
        b.set_outputs({"y": y})
        layouts = solve_graph(b.graph)
        g = fn(b.graph, layouts)
        errs = g.verify()
        assert not errs, f"{fn.__name__} emitted illegal tiles: {errs[:2]}"
        xv = np.random.randn(*shape).astype(np.float32)
        ref = tensor_run(b.graph, {"x": xv})["y"]
        out = nki_run(g, {"x": xv, "y_out": np.zeros((1, shape[1]), np.float32)})["y"]
        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-4)


# REVIEW_basic_lowering.md items 2, 4, 7:
#  - item 2: compare/bitwise ops lacked free-dim broadcast (greater(x, y[:, :1])
#    raised 'shapes must match'); now materialized via a ones-multiply.
#  - item 4: topk's _first_cols used a hard-coded 8-wide HBM scratch, reading
#    out of bounds for k in 9..15 (silently — slice bounds are unchecked).
#  - item 7: topk didn't tile the partition, so P > 128 emitted illegal tiles.

def test_compare_broadcast():
    np.random.seed(0)
    for sa, sb in [((128, 64), (128, 1)),   # F-broadcast
                   ((128, 64), (1, 64)),    # P-broadcast
                   ((128, 64), (1, 1))]:    # both
        x = np.random.randn(*sa).astype(np.float32)
        y = np.random.randn(*sb).astype(np.float32)

        def build(b):
            tx = b.add_input("x", sa)
            ty = b.add_input("y", sb)
            b.set_outputs({"z": b.greater(tx, ty)})

        _lower_and_run(build, {"x": x, "y": y}, {"z": sa})


@pytest.mark.parametrize("P,F,k", [
    (4, 100, 12),    # item 4: k in 9..15 (fold-aligned width 16 > old scratch)
    (4, 100, 20),    # k > 16
    (200, 64, 8),    # item 7: P > PARTITION_MAX
    (2, 20000, 12),  # chunked-F merge path with unaligned k
])
def test_topk_k_and_p_coverage(P, F, k):
    np.random.seed(0)
    x = np.random.randn(P, F).astype(np.float32)

    def build(b):
        t = b.add_input("x", (P, F))
        vals, idxs = b.topk(t, k)
        b.set_outputs({"v": vals, "i": idxs})

    _lower_and_run(build, {"x": x}, {"v": (P, k), "i": (P, k)}, atol=1e-6)
