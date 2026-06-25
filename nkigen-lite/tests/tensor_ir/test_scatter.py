"""Tests for the scatter_rows op and its lowering to nki_ir.

``scatter_rows`` is the row-granular runtime scatter primitive that
``scatter_along_axis`` and ``put_along_axis`` normalize onto.  It lowers to the
indirect-DMA store (``dma_copy_indirect``):

    out = base.copy(); out[idx[r], :] = updates[r, :]

Coverage at three levels:
  1. tensor_ir numpy interpreter (golden model).
  2. nki_ir numpy interpreter gate (lowering correctness, no HW).
  3. real Trainium hardware execution.

Run interpreter tests only:
    pytest nkigen-lite/tests/tensor_ir/test_scatter.py -m "not hw"
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder, run as tensor_run
from nkigen_lite.tensor_ir.passes.lower_to_nki import lower_to_nki
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

try:
    import nki.compiler.kernel_builder as nb_kb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False


# (N, W, M, dup) — N rows in base, W wide, M scattered rows, dup=duplicate idx
CASES = [
    (8, 4, 3, False),       # tiny
    (16, 8, 4, False),
    (16, 8, 4, True),       # duplicate indices (last-write-wins)
    (300, 16, 5, False),    # N > PARTITION_MAX (128): base-copy tiling
    (64, 8, 200, False),    # M > PARTITION_MAX: scatter tiling
]


def _inputs(N, W, M, dup, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((N, W)).astype(np.float32)
    updates = rng.standard_normal((M, W)).astype(np.float32)
    if dup or M > N:
        # force collisions (or unavoidable when M > N)
        idx = rng.integers(0, max(1, N // 4 if dup else N), size=(M, 1)).astype(np.uint32)
    else:
        idx = rng.choice(N, size=M, replace=False).reshape(M, 1).astype(np.uint32)
    return base, idx, updates


def _expected(base, idx, updates):
    out = base.copy()
    flat = idx.reshape(-1)
    for r in range(updates.shape[0]):
        out[int(flat[r])] = updates[r]
    return out


def _build(b, N, W, M):
    base = b.add_input("base", (N, W), DType.F32)
    idx = b.add_input("idx", (M, 1), DType.U32)
    upd = b.add_input("upd", (M, W), DType.F32)
    b.set_outputs({"out": b.scatter_rows(base, idx, upd)})


@pytest.mark.parametrize("N,W,M,dup", CASES)
def test_scatter_rows_interp(N, W, M, dup):
    base, idx, upd = _inputs(N, W, M, dup, seed=N * 100 + M)
    b = Builder("t")
    _build(b, N, W, M)
    result = tensor_run(b.graph, {"base": base, "idx": idx, "upd": upd})
    np.testing.assert_array_equal(result["out"], _expected(base, idx, upd))


@pytest.mark.parametrize("N,W,M,dup", CASES)
def test_scatter_rows_lowered_interp(N, W, M, dup):
    base, idx, upd = _inputs(N, W, M, dup, seed=N * 100 + M + 1)
    b = Builder("t")
    _build(b, N, W, M)
    ref = tensor_run(b.graph, {"base": base, "idx": idx, "upd": upd})

    nki_graph = lower_to_nki(b.graph)
    nki_inputs = {
        "base": base, "idx": idx, "upd": upd,
        "out_out": np.zeros((N, W), dtype=np.float32),
    }
    nki_result = nki_run(nki_graph, nki_inputs)
    np.testing.assert_array_equal(nki_result["out"], ref["out"])


@pytest.mark.hw
@pytest.mark.parametrize("N,W,M,dup", CASES)
def test_scatter_rows_hw(N, W, M, dup):
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")
    base, idx, upd = _inputs(N, W, M, dup, seed=N * 100 + M + 2)
    b = Builder("t")
    _build(b, N, W, M)
    ref = tensor_run(b.graph, {"base": base, "idx": idx, "upd": upd})

    nki_graph = lower_to_nki(b.graph)
    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_inputs = {"base": base, "idx": idx, "upd": upd}
    hw_outputs = {"out_out": np.zeros((N, W), dtype=np.float32)}
    nb_kb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    has_dups = dup or M > N
    # Within a single scatter chunk (M <= 128) the hardware applies writes in
    # order, so last-write-wins is deterministic and matches the interpreter.
    # Across chunks (M > 128) the order of colliding writes is not guaranteed,
    # so for that case only the untouched rows and the written-row set are
    # well-defined.
    if has_dups and M > 128:
        flat = idx.reshape(-1)
        written = set(int(x) for x in flat)
        for r in range(N):
            if r not in written:
                np.testing.assert_allclose(hw_outputs["out_out"][r], base[r], atol=1e-5)
    else:
        np.testing.assert_allclose(hw_outputs["out_out"], ref["out"], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# gather_rows: out[r, :] = src[idx[r], :]  (indirect-DMA load)
# ---------------------------------------------------------------------------

GATHER_CASES = [
    (16, 8, 4),       # tiny
    (300, 16, 5),     # N > PARTITION_MAX (tall table, e.g. embedding)
    (64, 8, 200),     # M > PARTITION_MAX (gather tiling)
]


def _gather_inputs(N, W, M, seed):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((N, W)).astype(np.float32)
    idx = rng.integers(0, N, size=(M, 1)).astype(np.uint32)
    return src, idx


def _build_gather(b, N, W, M):
    src = b.add_input("src", (N, W), DType.F32)
    idx = b.add_input("idx", (M, 1), DType.U32)
    b.set_outputs({"out": b.gather_rows(src, idx)})


@pytest.mark.parametrize("N,W,M", GATHER_CASES)
def test_gather_rows_interp(N, W, M):
    src, idx = _gather_inputs(N, W, M, seed=N * 7 + M)
    b = Builder("t")
    _build_gather(b, N, W, M)
    result = tensor_run(b.graph, {"src": src, "idx": idx})
    np.testing.assert_array_equal(result["out"], src[idx.reshape(-1)])


@pytest.mark.parametrize("N,W,M", GATHER_CASES)
def test_gather_rows_lowered_interp(N, W, M):
    src, idx = _gather_inputs(N, W, M, seed=N * 7 + M + 1)
    b = Builder("t")
    _build_gather(b, N, W, M)
    ref = tensor_run(b.graph, {"src": src, "idx": idx})
    nki_graph = lower_to_nki(b.graph)
    nki_inputs = {"src": src, "idx": idx, "out_out": np.zeros((M, W), dtype=np.float32)}
    np.testing.assert_array_equal(nki_run(nki_graph, nki_inputs)["out"], ref["out"])


@pytest.mark.hw
@pytest.mark.parametrize("N,W,M", GATHER_CASES)
def test_gather_rows_hw(N, W, M):
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")
    src, idx = _gather_inputs(N, W, M, seed=N * 7 + M + 2)
    b = Builder("t")
    _build_gather(b, N, W, M)
    ref = tensor_run(b.graph, {"src": src, "idx": idx})
    nki_graph = lower_to_nki(b.graph)
    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_outputs = {"out_out": np.zeros((M, W), dtype=np.float32)}
    nb_kb.compile_and_execute(
        kernel_fn, inputs={"src": src, "idx": idx}, outputs=hw_outputs, compile_opts=opts,
    )
    np.testing.assert_allclose(hw_outputs["out_out"], ref["out"], atol=1e-5, rtol=1e-5)
