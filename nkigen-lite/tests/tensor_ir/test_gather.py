"""Tests for the gather_along_axis op and its lowering to nki_ir.

``gather_along_axis`` is the 2-D per-partition runtime gather primitive that
``np.take_along_axis`` and dynamic ``np.take`` normalize onto.  It lowers to
the hardware ``nisa.gather`` instruction.

Coverage at three levels:
  1. tensor_ir numpy interpreter (golden model).
  2. nki_ir numpy interpreter gate (lowering correctness, no HW).
  3. real Trainium hardware execution.

Run interpreter tests only:
    pytest nkigen-lite/tests/tensor_ir/test_gather.py -m "not hw"
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


# ---------------------------------------------------------------------------
# Cases: (P, F_data, F_idx)
# ---------------------------------------------------------------------------

CASES = [
    (2, 3, 3),       # tiny, F_idx == F_data
    (2, 3, 1),       # single gathered column
    (4, 16, 8),      # F_idx < F_data
    (8, 8, 16),      # F_idx > F_data (repeats allowed)
    (128, 64, 64),   # full partition tile
    (300, 64, 64),   # P > PARTITION_MAX (128) -> partition tiling
]


def _random_inputs(P, F_data, F_idx, seed):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((P, F_data)).astype(np.float32)
    idx = rng.integers(0, F_data, size=(P, F_idx)).astype(np.uint32)
    return data, idx


def _build(b, P, F_data, F_idx):
    data = b.add_input("data", (P, F_data), DType.F32)
    idx = b.add_input("idx", (P, F_idx), DType.U32)
    b.set_outputs({"out": b.gather_along_axis(data, idx)})


# ---------------------------------------------------------------------------
# tensor_ir interpreter (golden model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("P,F_data,F_idx", CASES)
def test_gather_along_axis_interp(P, F_data, F_idx):
    data, idx = _random_inputs(P, F_data, F_idx, seed=P * 100 + F_idx)
    b = Builder("t")
    _build(b, P, F_data, F_idx)
    result = tensor_run(b.graph, {"data": data, "idx": idx})
    expected = np.take_along_axis(data, idx.astype(np.intp), axis=1)
    np.testing.assert_array_equal(result["out"], expected)


# ---------------------------------------------------------------------------
# nki_ir interpreter gate (lowering correctness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("P,F_data,F_idx", CASES)
def test_gather_along_axis_lowered_interp(P, F_data, F_idx):
    data, idx = _random_inputs(P, F_data, F_idx, seed=P * 100 + F_idx + 1)
    b = Builder("t")
    _build(b, P, F_data, F_idx)
    ref = tensor_run(b.graph, {"data": data, "idx": idx})

    nki_graph = lower_to_nki(b.graph)
    nki_inputs = {
        "data": data,
        "idx": idx,
        "out_out": np.zeros((P, F_idx), dtype=np.float32),
    }
    nki_result = nki_run(nki_graph, nki_inputs)
    np.testing.assert_array_equal(nki_result["out"], ref["out"])


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@pytest.mark.hw
@pytest.mark.parametrize("P,F_data,F_idx", CASES)
def test_gather_along_axis_hw(P, F_data, F_idx):
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")
    data, idx = _random_inputs(P, F_data, F_idx, seed=P * 100 + F_idx + 2)
    b = Builder("t")
    _build(b, P, F_data, F_idx)
    ref = tensor_run(b.graph, {"data": data, "idx": idx})

    nki_graph = lower_to_nki(b.graph)
    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_inputs = {"data": data, "idx": idx}
    hw_outputs = {"out_out": np.zeros((P, F_idx), dtype=np.float32)}
    nb_kb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    np.testing.assert_allclose(hw_outputs["out_out"], ref["out"], atol=1e-5, rtol=1e-5)
