"""Shape coverage tests for tensor_ir → nki_ir lowering.

Systematically exercises:
  a) Tiling — shapes that exceed partition_max (128) or SBUF budget
  b) Imperfect loop nests — dimensions not evenly divisible by tile size
  c) Different input ranks — rank-2, rank-3, rank-4

Each pattern (softmax, rmsnorm, attention, feedforward, rope) is tested
with multiple shape configurations to stress the tiling and lowering
pipeline.
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
    import nki.compiler.kernel_builder as nb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False


@pytest.fixture
def compile_and_run():
    if not HAS_NKI:
        pytest.skip("nki not installed")
    opts = nb.CompileOptions(target="trn2")

    def _run(graph, inputs, outputs):
        kernel_fn = build_kb_kernel(graph)
        nb.compile_and_execute(
            kernel_fn, inputs=inputs, outputs=outputs, compile_opts=opts,
        )
        return outputs

    return _run


def _reshape_inputs_for_nki(nki_graph, inputs, out_shapes):
    """Reshape numpy arrays to match the nki_graph's (possibly flattened) interface."""
    nki_inputs = {}
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            nki_inputs[name] = arr.reshape(expected)
        else:
            nki_inputs[name] = arr
    for name, shape in out_shapes.items():
        key = f"{name}_out"
        expected = graph_input_shapes.get(key)
        if expected is not None and expected != shape:
            nki_inputs[key] = np.zeros(expected, dtype=np.float32)
        else:
            nki_inputs[key] = np.zeros(shape, dtype=np.float32)
    return nki_inputs


def _lower_and_check(build_fn, inputs, out_shapes, atol=1e-3, rtol=1e-3):
    b = Builder("t")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)
    nki_graph = lower_to_nki(b.graph)
    # Step 1: interpreter sanity check
    nki_inputs = _reshape_inputs_for_nki(nki_graph, inputs, out_shapes)
    actual = nki_run(nki_graph, nki_inputs)
    for k in ref:
        actual_k = actual[k]
        ref_k = ref[k]
        if actual_k.shape != ref_k.shape:
            actual_k = actual_k.reshape(ref_k.shape)
        np.testing.assert_allclose(actual_k, ref_k, atol=atol, rtol=rtol)
    # Step 2: real HW execution
    if not HAS_NKI:
        raise RuntimeError("HW execution requested but nki not available")
    opts = nb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    hw_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            hw_inputs[name] = arr.reshape(expected)
        else:
            hw_inputs[name] = arr
    hw_outputs = {}
    for n, sh in out_shapes.items():
        key = f"{n}_out"
        expected = graph_input_shapes.get(key, sh)
        hw_outputs[key] = np.zeros(expected, dtype=np.float32)
    nb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    for k in ref:
        hw_k = hw_outputs[f"{k}_out"]
        ref_k = ref[k]
        if hw_k.shape != ref_k.shape:
            hw_k = hw_k.reshape(ref_k.shape)
        np.testing.assert_allclose(hw_k, ref_k, atol=atol, rtol=rtol)


def _lower_and_check_hw(compile_and_run, build_fn, inputs, out_shapes, atol=1e-3, rtol=1e-3):
    b = Builder("t")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)
    nki_graph = lower_to_nki(b.graph)
    # Step 1: verify interpreter produces correct results
    interp_inputs = _reshape_inputs_for_nki(nki_graph, inputs, out_shapes)
    interp_result = nki_run(nki_graph, interp_inputs)
    for k in ref:
        actual_k = interp_result[k]
        ref_k = ref[k]
        if actual_k.shape != ref_k.shape:
            actual_k = actual_k.reshape(ref_k.shape)
        np.testing.assert_allclose(
            actual_k, ref_k, atol=atol, rtol=rtol,
            err_msg=f"Interpreter mismatch on '{k}' (must pass before HW)",
        )
    # Step 2: run on real HW
    nki_inputs = _reshape_inputs_for_nki(nki_graph, inputs, out_shapes)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    nki_outputs = {}
    for n, sh in out_shapes.items():
        key = f"{n}_out"
        expected = graph_input_shapes.get(key, sh)
        nki_outputs[key] = np.zeros(expected, dtype=np.float32)
    hw_inputs = {k: v for k, v in nki_inputs.items() if k not in nki_outputs}
    hw_result = compile_and_run(nki_graph, hw_inputs, nki_outputs)
    for k in ref:
        hw_k = hw_result[f"{k}_out"]
        ref_k = ref[k]
        if hw_k.shape != ref_k.shape:
            hw_k = hw_k.reshape(ref_k.shape)
        np.testing.assert_allclose(hw_k, ref_k, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Softmax — shape coverage
# ---------------------------------------------------------------------------

class TestSoftmaxShapes:
    """Softmax with various shapes exercising tiling and rank combinations."""

    def _build_softmax(self, b, shape):
        x = b.add_input("x", shape)
        m = b.reduce(x, axis=-1, keepdims=True, kind="max")
        e = b.exp(b.sub(x, m))
        s = b.reduce(e, axis=-1, keepdims=True, kind="sum")
        out = b.mul(e, b.reciprocal(s))
        b.set_outputs({"y": out})

    def test_rank2_single_tile(self):
        """(64, 128) — fits in one tile, no loops."""
        np.random.seed(0)
        shape = (64, 128)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank2_p_tiled(self):
        """(256, 128) — P-axis needs 2 tiles (256/128)."""
        np.random.seed(1)
        shape = (256, 128)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank2_p_imperfect(self):
        """(200, 128) — P-axis imperfect: 200/128 = 1 full + 1 partial (72)."""
        np.random.seed(2)
        shape = (200, 128)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank2_f_tiled(self):
        """(128, 1024) — F-axis needs tiling (128*1024*4 > SBUF budget)."""
        np.random.seed(3)
        shape = (128, 1024)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank2_both_tiled_imperfect(self):
        """(300, 768) — both P and F tiled, P imperfect (300/128)."""
        np.random.seed(4)
        shape = (300, 768)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank3(self):
        """(4, 64, 256) — rank-3 with flatten_to_2d."""
        np.random.seed(5)
        shape = (4, 64, 256)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank3_imperfect(self):
        """(3, 50, 128) — rank-3, flatten gives (150, 128), P imperfect."""
        np.random.seed(6)
        shape = (3, 50, 128)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    def test_rank4(self):
        """(2, 4, 32, 64) — rank-4 (B, H, S, D) with batch loops."""
        np.random.seed(7)
        shape = (2, 4, 32, 64)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check(lambda b: self._build_softmax(b, shape), {"x": x}, {"y": shape})

    @pytest.mark.hw
    def test_rank2_p_imperfect_hw(self, compile_and_run):
        """(200, 128) on HW — imperfect P tiling."""
        np.random.seed(2)
        shape = (200, 128)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check_hw(compile_and_run, lambda b: self._build_softmax(b, shape),
                            {"x": x}, {"y": shape})

    @pytest.mark.hw
    def test_rank2_both_tiled_imperfect_hw(self, compile_and_run):
        """(300, 768) on HW — both axes tiled, P imperfect."""
        np.random.seed(4)
        shape = (300, 768)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check_hw(compile_and_run, lambda b: self._build_softmax(b, shape),
                            {"x": x}, {"y": shape})

    @pytest.mark.hw
    def test_rank3_hw(self, compile_and_run):
        """(4, 64, 256) on HW — rank-3."""
        np.random.seed(5)
        shape = (4, 64, 256)
        x = np.random.randn(*shape).astype(np.float32)
        _lower_and_check_hw(compile_and_run, lambda b: self._build_softmax(b, shape),
                            {"x": x}, {"y": shape})


# ---------------------------------------------------------------------------
# RMSNorm — shape coverage
# ---------------------------------------------------------------------------

class TestRmsnormShapes:
    """RMSNorm with different ranks and tiling configurations."""

    def _build_rmsnorm(self, b, x_shape, w_shape):
        xv = b.add_input("x", x_shape)
        wv = b.add_input("w", w_shape)
        xs = b.mul(xv, xv)
        mean_sq = b.reduce(xs, axis=-1, keepdims=True, kind="mean")
        eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
        rstd = b.rsqrt(b.add(mean_sq, eps))
        out = b.mul(b.mul(xv, rstd), wv)
        b.set_outputs({"y": out})

    def test_rank2_small(self):
        """(64, 128) — single tile, rank-2."""
        np.random.seed(0)
        S, D = 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (S, D), (D,)),
                         {"x": x, "w": w}, {"y": (S, D)})

    def test_rank2_p_tiled(self):
        """(256, 256) — P-axis tiled (256/128=2)."""
        np.random.seed(1)
        S, D = 256, 256
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (S, D), (D,)),
                         {"x": x, "w": w}, {"y": (S, D)})

    def test_rank2_p_imperfect(self):
        """(200, 256) — P imperfect (200/128 = 1 full + 72 partial)."""
        np.random.seed(2)
        S, D = 200, 256
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (S, D), (D,)),
                         {"x": x, "w": w}, {"y": (S, D)})

    def test_rank2_f_tiled(self):
        """(128, 768) — F-axis tiled (exceeds SBUF budget)."""
        np.random.seed(3)
        S, D = 128, 768
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(1, D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (S, D), (1, D)),
                         {"x": x, "w": w}, {"y": (S, D)})

    def test_rank3_bsd(self):
        """(2, 32, 256) — rank-3 (B, S, D) with rank-1 weight."""
        np.random.seed(4)
        B, S, D = 2, 32, 256
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (B, S, D), (D,)),
                         {"x": x, "w": w}, {"y": (B, S, D)})

    def test_rank3_imperfect(self):
        """(3, 50, 128) — rank-3, flattened P dim imperfect (150/128)."""
        np.random.seed(5)
        B, S, D = 3, 50, 128
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (B, S, D), (D,)),
                         {"x": x, "w": w}, {"y": (B, S, D)})

    def test_rank3_large_d(self):
        """(2, 16, 512) — rank-3 with F-tiling on D."""
        np.random.seed(6)
        B, S, D = 2, 16, 512
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check(lambda b: self._build_rmsnorm(b, (B, S, D), (D,)),
                         {"x": x, "w": w}, {"y": (B, S, D)})

    @pytest.mark.hw
    def test_rank2_p_imperfect_hw(self, compile_and_run):
        """(200, 256) on HW."""
        np.random.seed(2)
        S, D = 200, 256
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_rmsnorm(b, (S, D), (D,)),
                            {"x": x, "w": w}, {"y": (S, D)})

    @pytest.mark.hw
    def test_rank3_bsd_hw(self, compile_and_run):
        """(2, 32, 256) rank-3 on HW."""
        np.random.seed(4)
        B, S, D = 2, 32, 256
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_rmsnorm(b, (B, S, D), (D,)),
                            {"x": x, "w": w}, {"y": (B, S, D)})


# ---------------------------------------------------------------------------
# Attention — shape coverage
# ---------------------------------------------------------------------------

class TestAttentionShapes:
    """Scaled dot-product attention with various shapes."""

    def _build_attention(self, b, B, H, S, D):
        q = b.add_input("q", (B, H, S, D))
        k = b.add_input("k", (B, H, S, D))
        v = b.add_input("v", (B, H, S, D))
        kt = b.transpose(k, (0, 1, 3, 2))
        scores = b.matmul(q, kt)
        scale = b.constant(1.0 / (D ** 0.5), scores.type.shape, DType.F32)
        scaled = b.mul(scores, scale)
        s_max = b.reduce(scaled, axis=-1, keepdims=True, kind="max")
        s_exp = b.exp(b.sub(scaled, s_max))
        s_sum = b.reduce(s_exp, axis=-1, keepdims=True, kind="sum")
        weights = b.mul(s_exp, b.reciprocal(s_sum))
        out = b.matmul(weights, v)
        b.set_outputs({"out": out})

    def test_small_single_head(self):
        """(1, 1, 32, 16) — single batch, single head, no batch loops."""
        np.random.seed(0)
        B, H, S, D = 1, 1, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check(lambda b: self._build_attention(b, B, H, S, D),
                         {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})

    def test_multi_batch_head(self):
        """(2, 8, 32, 16) — multiple batches and heads."""
        np.random.seed(1)
        B, H, S, D = 2, 8, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check(lambda b: self._build_attention(b, B, H, S, D),
                         {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})

    def test_larger_seq(self):
        """(1, 2, 64, 32) — larger sequence triggers S-dim tiling."""
        np.random.seed(2)
        B, H, S, D = 1, 2, 64, 32
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check(lambda b: self._build_attention(b, B, H, S, D),
                         {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})

    def test_odd_heads(self):
        """(1, 3, 32, 16) — odd number of heads (not power of 2)."""
        np.random.seed(3)
        B, H, S, D = 1, 3, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check(lambda b: self._build_attention(b, B, H, S, D),
                         {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})

    @pytest.mark.hw
    def test_multi_batch_head_hw(self, compile_and_run):
        """(2, 8, 32, 16) on HW."""
        np.random.seed(1)
        B, H, S, D = 2, 8, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_attention(b, B, H, S, D),
                            {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})

    @pytest.mark.hw
    def test_larger_seq_hw(self, compile_and_run):
        """(1, 2, 64, 32) on HW — larger sequence."""
        np.random.seed(2)
        B, H, S, D = 1, 2, 64, 32
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_attention(b, B, H, S, D),
                            {"q": q, "k": k, "v": v}, {"out": (B, H, S, D)})


# ---------------------------------------------------------------------------
# Feedforward (SwiGLU) — shape coverage
# ---------------------------------------------------------------------------

class TestFeedforwardShapes:
    """SwiGLU feedforward with tiling and rank variations."""

    def _build_ffn(self, b, x_shape, D, intermediate):
        x = b.add_input("x", x_shape)
        gate_up_w = b.add_input("gate_up_w", (D, intermediate * 2))
        down_w = b.add_input("down_w", (intermediate, D))
        mm = b.matmul(x, gate_up_w)
        gate, up = b.split(mm, 2, axis=-1)
        hidden = b.mul(b.silu(gate), up)
        out = b.matmul(hidden, down_w)
        b.set_outputs({"y": out})

    def test_rank2_small(self):
        """(32, 64) — rank-2, fits in single tile."""
        np.random.seed(0)
        S, D, I = 32, 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank2_p_tiled(self):
        """(256, 64) — P-axis tiled (256/128=2)."""
        np.random.seed(1)
        S, D, I = 256, 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank2_p_imperfect(self):
        """(200, 64) — P imperfect (200/128)."""
        np.random.seed(2)
        S, D, I = 200, 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank3_bsd(self):
        """(2, 16, 128) — rank-3 (B, S, D)."""
        np.random.seed(3)
        B, S, D, I = 2, 16, 128, 256
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    def test_rank3_large_k(self):
        """(1, 32, 256) with intermediate=512 — K-tiling in matmul (256>128)."""
        np.random.seed(4)
        B, S, D, I = 1, 32, 256, 512
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    # --- Rank-2: tiling with remainder on each possible axis ---

    def test_rank2_p_remainder(self):
        """(300, 64) — P-axis remainder (300/128 = 2 full + 44 partial)."""
        np.random.seed(10)
        S, D, I = 300, 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank2_k_remainder(self):
        """(32, 200) — K-axis (D=200) remainder (200/128 = 1 full + 72 partial)."""
        np.random.seed(11)
        S, D, I = 32, 200, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank2_n_remainder(self):
        """(32, 64) with I=384 — N-axis (I*2=768) exceeds PSUM, remainder on F-tile."""
        np.random.seed(12)
        S, D, I = 32, 64, 384
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    # --- Rank-3: tiling with remainder on each possible axis ---

    def test_rank3_p_remainder(self):
        """(2, 100, 64) — flattened P=200, remainder (200/128 = 1 full + 72)."""
        np.random.seed(13)
        B, S, D, I = 2, 100, 64, 128
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    def test_rank3_k_remainder(self):
        """(1, 16, 200) — K-axis (D=200) remainder in K-tiled matmul."""
        np.random.seed(14)
        B, S, D, I = 1, 16, 200, 128
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    def test_rank3_n_remainder(self):
        """(1, 32, 64) with I=300 — N-axis (I*2=600) exceeds PSUM, F-tile remainder."""
        np.random.seed(15)
        B, S, D, I = 1, 32, 64, 300
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    # --- Rank-4: tiling with remainder on each possible axis ---

    def test_rank4_p_remainder(self):
        """(2, 3, 25, 64) — flattened P=150, remainder (150/128 = 1 full + 22)."""
        np.random.seed(16)
        B, H, S, D, I = 2, 3, 25, 64, 128
        x = np.random.randn(B, H, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, H, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, H, S, D)})

    def test_rank4_k_remainder(self):
        """(1, 1, 16, 200) — K-axis (D=200) remainder in K-tiled matmul."""
        np.random.seed(17)
        B, H, S, D, I = 1, 1, 16, 200, 128
        x = np.random.randn(B, H, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, H, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, H, S, D)})

    def test_rank4_n_remainder(self):
        """(1, 1, 16, 64) with I=300 — N-axis remainder on F-tile."""
        np.random.seed(18)
        B, H, S, D, I = 1, 1, 16, 64, 300
        x = np.random.randn(B, H, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, H, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, H, S, D)})

    # --- Large matrices ---

    def test_rank2_large_p_and_k(self):
        """(512, 200) with I=128 — P tiled (4 tiles), K remainder (200/128)."""
        np.random.seed(20)
        S, D, I = 512, 200, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank2_large_p_remainder_and_k(self):
        """(500, 200) with I=128 — P remainder (500/128), K remainder (200/128)."""
        np.random.seed(21)
        S, D, I = 500, 200, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    def test_rank3_large(self):
        """(1, 256, 200) with I=128 — large rank-3 P tiled + K remainder."""
        np.random.seed(22)
        B, S, D, I = 1, 256, 200, 128
        x = np.random.randn(B, S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check(lambda b: self._build_ffn(b, (B, S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (B, S, D)})

    # --- BF16 dtype ---

    def _build_ffn_bf16(self, b, x_shape, D, intermediate):
        x = b.add_input("x", x_shape, DType.BF16)
        gate_up_w = b.add_input("gate_up_w", (D, intermediate * 2), DType.BF16)
        down_w = b.add_input("down_w", (intermediate, D), DType.BF16)
        mm = b.matmul(x, gate_up_w)
        gate, up = b.split(mm, 2, axis=-1)
        hidden = b.mul(b.silu(gate), up)
        out = b.matmul(hidden, down_w)
        b.set_outputs({"y": out})

    def test_rank2_bf16_small(self):
        """(32, 64) BF16 — single tile, tests dtype propagation."""
        np.random.seed(30)
        S, D, I = 32, 64, 128
        x = np.random.randn(S, D).astype(np.float16).view(np.dtype('bfloat16'))
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        dw = (np.random.randn(I, D) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        _lower_and_check(lambda b: self._build_ffn_bf16(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)},
                         atol=0.05, rtol=0.05)

    def test_rank2_bf16_p_remainder(self):
        """(200, 64) BF16 — P-axis remainder with reduced precision."""
        np.random.seed(31)
        S, D, I = 200, 64, 128
        x = np.random.randn(S, D).astype(np.float16).view(np.dtype('bfloat16'))
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        dw = (np.random.randn(I, D) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        _lower_and_check(lambda b: self._build_ffn_bf16(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)},
                         atol=0.05, rtol=0.05)

    def test_rank2_bf16_k_remainder(self):
        """(32, 200) BF16 — K-axis remainder with K-tiling."""
        np.random.seed(32)
        S, D, I = 32, 200, 128
        x = np.random.randn(S, D).astype(np.float16).view(np.dtype('bfloat16'))
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        dw = (np.random.randn(I, D) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        _lower_and_check(lambda b: self._build_ffn_bf16(b, (S, D), D, I),
                         {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)},
                         atol=0.05, rtol=0.05)

    # --- HW execution tests ---

    @pytest.mark.hw
    def test_rank2_p_remainder_hw(self, compile_and_run):
        """(300, 64) on HW — P-axis remainder."""
        np.random.seed(10)
        S, D, I = 300, 64, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_ffn(b, (S, D), D, I),
                            {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    @pytest.mark.hw
    def test_rank2_k_remainder_hw(self, compile_and_run):
        """(32, 200) on HW — K-axis remainder."""
        np.random.seed(11)
        S, D, I = 32, 200, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_ffn(b, (S, D), D, I),
                            {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    @pytest.mark.hw
    def test_rank2_large_hw(self, compile_and_run):
        """(512, 200) with I=128 on HW — large matrix P+K tiled."""
        np.random.seed(20)
        S, D, I = 512, 200, 128
        x = np.random.randn(S, D).astype(np.float32)
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float32)
        dw = (np.random.randn(I, D) * 0.02).astype(np.float32)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_ffn(b, (S, D), D, I),
                            {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)})

    @pytest.mark.hw
    def test_rank2_bf16_hw(self, compile_and_run):
        """(32, 64) BF16 on HW — dtype handling on real hardware."""
        np.random.seed(30)
        S, D, I = 32, 64, 128
        x = np.random.randn(S, D).astype(np.float16).view(np.dtype('bfloat16'))
        guw = (np.random.randn(D, I * 2) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        dw = (np.random.randn(I, D) * 0.02).astype(np.float16).view(np.dtype('bfloat16'))
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_ffn_bf16(b, (S, D), D, I),
                            {"x": x, "gate_up_w": guw, "down_w": dw}, {"y": (S, D)},
                            atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# RoPE — shape coverage
# ---------------------------------------------------------------------------

class TestRopeShapes:
    """RoPE with various rank and shape configurations."""

    def _build_rope(self, b, x_shape, S, half):
        BS = x_shape[0]
        H = x_shape[2]
        D = x_shape[3]
        xq = b.add_input("xq", x_shape)
        freqs_cos = b.add_input("freqs_cos", (S, half))
        freqs_sin = b.add_input("freqs_sin", (S, half))
        fc = b.broadcast_to(b.reshape(freqs_cos, (1, S, 1, half)), (BS, S, H, half))
        fs = b.broadcast_to(b.reshape(freqs_sin, (1, S, 1, half)), (BS, S, H, half))
        x1 = b.slice(xq, starts=(0, 0, 0, 0), stops=(BS, S, H, half))
        x2 = b.slice(xq, starts=(0, 0, 0, half), stops=(BS, S, H, D))
        rot1 = b.sub(b.mul(x1, fc), b.mul(x2, fs))
        rot2 = b.add(b.mul(x1, fs), b.mul(x2, fc))
        out = b.concat([rot1, rot2], axis=3)
        b.set_outputs({"out": out})

    def _make_freqs(self, S, D):
        half = D // 2
        base = 10000
        freqs = 1.0 / (base ** (np.arange(0, D, 2)[:half] / D))
        t = np.arange(S, dtype=np.float32)
        freqs = np.outer(t, freqs)
        return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def test_small(self):
        """(1, 8, 2, 16) — minimal shape."""
        np.random.seed(0)
        BS, S, H, D = 1, 8, 2, 16
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        cos_c, sin_c = self._make_freqs(S, D)
        _lower_and_check(lambda b: self._build_rope(b, (BS, S, H, D), S, D // 2),
                         {"xq": x, "freqs_cos": cos_c, "freqs_sin": sin_c},
                         {"out": (BS, S, H, D)})

    def test_larger_batch(self):
        """(4, 16, 4, 32) — larger batch and heads."""
        np.random.seed(1)
        BS, S, H, D = 4, 16, 4, 32
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        cos_c, sin_c = self._make_freqs(S, D)
        _lower_and_check(lambda b: self._build_rope(b, (BS, S, H, D), S, D // 2),
                         {"xq": x, "freqs_cos": cos_c, "freqs_sin": sin_c},
                         {"out": (BS, S, H, D)})

    def test_odd_batch(self):
        """(3, 16, 4, 32) — odd batch size (not power of 2)."""
        np.random.seed(2)
        BS, S, H, D = 3, 16, 4, 32
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        cos_c, sin_c = self._make_freqs(S, D)
        _lower_and_check(lambda b: self._build_rope(b, (BS, S, H, D), S, D // 2),
                         {"xq": x, "freqs_cos": cos_c, "freqs_sin": sin_c},
                         {"out": (BS, S, H, D)})

    @pytest.mark.hw
    def test_larger_batch_hw(self, compile_and_run):
        """(4, 16, 4, 32) on HW."""
        np.random.seed(1)
        BS, S, H, D = 4, 16, 4, 32
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        cos_c, sin_c = self._make_freqs(S, D)
        _lower_and_check_hw(compile_and_run,
                            lambda b: self._build_rope(b, (BS, S, H, D), S, D // 2),
                            {"xq": x, "freqs_cos": cos_c, "freqs_sin": sin_c},
                            {"out": (BS, S, H, D)})
