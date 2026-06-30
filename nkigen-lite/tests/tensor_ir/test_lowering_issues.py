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
