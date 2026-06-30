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


# smell-4 (late `Graph` import in lower_to_nki): fixed; the function now
# uses the module-level `Graph` import via the `_verify` helper.
