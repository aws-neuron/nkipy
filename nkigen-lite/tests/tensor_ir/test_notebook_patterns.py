"""End-to-end lowering tests for the patterns used in nkigen_lite/notebooks/.

These intentionally use the realistic shapes that appear in real Qwen3
forward passes — rank-3/4 inputs, rank-1 weights, fused gate+up matmul
followed by split. They exercise the gaps documented in
``nkigen_lite/docs/LOWERING_ISSUES.md`` and must pass for the demo notebooks
to use realistic shapes without workarounds.

Each pattern has both an interpreter test (always run) and a HW test
(only runs if `nki` is installed and Trainium cores are available).
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


def _lower_and_check(build_fn, inputs, out_shapes, atol=1e-4, rtol=1e-4):
    b = Builder("t")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)
    nki_graph = lower_to_nki(b.graph)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    nki_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            nki_inputs[name] = arr.reshape(expected)
        else:
            nki_inputs[name] = arr
    for name, shape in out_shapes.items():
        key = f"{name}_out"
        expected = graph_input_shapes.get(key, shape)
        nki_inputs[key] = np.zeros(expected, dtype=np.float32)
    actual = nki_run(nki_graph, nki_inputs)
    for k in ref:
        actual_k = actual[k]
        ref_k = ref[k]
        if actual_k.shape != ref_k.shape:
            actual_k = actual_k.reshape(ref_k.shape)
        np.testing.assert_allclose(actual_k, ref_k, atol=atol, rtol=rtol)


def _lower_and_check_hw(
    compile_and_run, build_fn, inputs, out_shapes, atol=1e-3, rtol=1e-3,
):
    b = Builder("t")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)
    nki_graph = lower_to_nki(b.graph)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    nki_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            nki_inputs[name] = arr.reshape(expected)
        else:
            nki_inputs[name] = arr
    nki_outputs = {}
    for n, sh in out_shapes.items():
        key = f"{n}_out"
        expected = graph_input_shapes.get(key, sh)
        nki_outputs[key] = np.zeros(expected, dtype=np.float32)
    hw_result = compile_and_run(nki_graph, nki_inputs, nki_outputs)
    for k in ref:
        hw_k = hw_result[f"{k}_out"]
        ref_k = ref[k]
        if hw_k.shape != ref_k.shape:
            hw_k = hw_k.reshape(ref_k.shape)
        np.testing.assert_allclose(hw_k, ref_k, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# RMSNorm — the rmsnorm_demo notebook pattern
# ---------------------------------------------------------------------------

class TestRmsnormPatterns:
    def test_rank3_input_rank1_weight(self):
        """The shape used by Qwen3: x is (B, S, D), w is (D,)."""
        np.random.seed(0)
        B, S, D = 2, 16, 256
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)

        def build(b):
            xv = b.add_input("x", (B, S, D))
            wv = b.add_input("w", (D,), DType.F32)
            xs = b.mul(xv, xv)
            mean_sq = b.reduce(xs, axis=2, keepdims=True, kind="mean")
            eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
            rstd = b.rsqrt(b.add(mean_sq, eps))
            out = b.mul(b.mul(xv, rstd), wv)
            b.set_outputs({"y": out})

        _lower_and_check(build, {"x": x, "w": w}, {"y": (B, S, D)}, atol=1e-3)

    @pytest.mark.hw
    def test_rank3_input_rank1_weight_hw(self, compile_and_run):
        np.random.seed(0)
        B, S, D = 2, 16, 256
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D).astype(np.float32)

        def build(b):
            xv = b.add_input("x", (B, S, D))
            wv = b.add_input("w", (D,), DType.F32)
            xs = b.mul(xv, xv)
            mean_sq = b.reduce(xs, axis=2, keepdims=True, kind="mean")
            eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
            rstd = b.rsqrt(b.add(mean_sq, eps))
            out = b.mul(b.mul(xv, rstd), wv)
            b.set_outputs({"y": out})

        _lower_and_check_hw(compile_and_run, build, {"x": x, "w": w}, {"y": (B, S, D)})

    def test_rank2_with_d_above_sbuf_split(self):
        """D=768 forces F-axis tiling, exercising broadcast-load with stride 0.

        This is the case that today breaks the interpreter: when F is
        tiled, the rank-2 weight broadcast emits dma_copy strides=(0, 1).
        """
        np.random.seed(0)
        S, D = 128, 768
        x = np.random.randn(S, D).astype(np.float32)
        w = np.random.randn(1, D).astype(np.float32)

        def build(b):
            xv = b.add_input("x", (S, D))
            wv = b.add_input("w", (1, D))
            xs = b.mul(xv, xv)
            mean_sq = b.reduce(xs, axis=1, keepdims=True, kind="mean")
            eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
            rstd = b.rsqrt(b.add(mean_sq, eps))
            out = b.mul(b.mul(xv, rstd), wv)
            b.set_outputs({"y": out})

        _lower_and_check(build, {"x": x, "w": w}, {"y": (S, D)}, atol=1e-3)


# ---------------------------------------------------------------------------
# Attention — multi-head with rank-4 (B, H, S, D)
# ---------------------------------------------------------------------------

class TestAttentionPatterns:
    def test_multihead_attention_rank4(self):
        """Multi-head SDPA with rank-4 Q/K/V — Q @ K^T is batched matmul."""
        np.random.seed(42)
        B, H, S, D = 2, 4, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)

        def build(b):
            qv = b.add_input("q", (B, H, S, D))
            kv = b.add_input("k", (B, H, S, D))
            vv = b.add_input("v", (B, H, S, D))
            kt = b.transpose(kv, (0, 1, 3, 2))             # (B,H,D,S)
            scores = b.matmul(qv, kt)                       # (B,H,S,S)
            scale = b.constant(1.0 / (D ** 0.5),
                               scores.type.shape, DType.F32)
            scaled = b.mul(scores, scale)
            s_max = b.reduce(scaled, axis=-1, keepdims=True, kind="max")
            s_exp = b.exp(b.sub(scaled, s_max))
            s_sum = b.reduce(s_exp, axis=-1, keepdims=True, kind="sum")
            weights = b.mul(s_exp, b.reciprocal(s_sum))
            out = b.matmul(weights, vv)                     # (B,H,S,D)
            b.set_outputs({"output": out})

        _lower_and_check(
            build, {"q": q, "k": k, "v": v}, {"output": (B, H, S, D)},
            atol=1e-3, rtol=1e-3,
        )

    @pytest.mark.hw
    def test_multihead_attention_rank4_hw(self, compile_and_run):
        np.random.seed(42)
        B, H, S, D = 2, 4, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)

        def build(b):
            qv = b.add_input("q", (B, H, S, D))
            kv = b.add_input("k", (B, H, S, D))
            vv = b.add_input("v", (B, H, S, D))
            kt = b.transpose(kv, (0, 1, 3, 2))
            scores = b.matmul(qv, kt)
            scale = b.constant(1.0 / (D ** 0.5),
                               scores.type.shape, DType.F32)
            scaled = b.mul(scores, scale)
            s_max = b.reduce(scaled, axis=-1, keepdims=True, kind="max")
            s_exp = b.exp(b.sub(scaled, s_max))
            s_sum = b.reduce(s_exp, axis=-1, keepdims=True, kind="sum")
            weights = b.mul(s_exp, b.reciprocal(s_sum))
            out = b.matmul(weights, vv)
            b.set_outputs({"output": out})

        _lower_and_check_hw(
            compile_and_run, build, {"q": q, "k": k, "v": v},
            {"output": (B, H, S, D)},
        )


# ---------------------------------------------------------------------------
# RoPE — rank-4 with rank-2 broadcast operands
# ---------------------------------------------------------------------------

class TestRopePatterns:
    def test_rope_rank4(self):
        """RoPE applied to (BS, S, H, D) Q with (S, D/2) cos/sin caches."""
        np.random.seed(7)
        BS, S, H, D = 2, 16, 4, 32
        half = D // 2
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        fc = np.random.randn(S, half).astype(np.float32)
        fs = np.random.randn(S, half).astype(np.float32)

        def build(b):
            xv = b.add_input("x", (BS, S, H, D))
            fcv = b.add_input("freqs_cos", (S, half))
            fsv = b.add_input("freqs_sin", (S, half))
            fcb = b.broadcast_to(b.reshape(fcv, (1, S, 1, half)),
                                 (BS, S, H, half))
            fsb = b.broadcast_to(b.reshape(fsv, (1, S, 1, half)),
                                 (BS, S, H, half))
            x1 = b.slice(xv, starts=(0, 0, 0, 0), stops=(BS, S, H, half))
            x2 = b.slice(xv, starts=(0, 0, 0, half), stops=(BS, S, H, D))
            rot1 = b.sub(b.mul(x1, fcb), b.mul(x2, fsb))
            rot2 = b.add(b.mul(x1, fsb), b.mul(x2, fcb))
            out = b.concat([rot1, rot2], axis=3)
            b.set_outputs({"x_out": out})

        _lower_and_check(
            build,
            {"x": x, "freqs_cos": fc, "freqs_sin": fs},
            {"x_out": (BS, S, H, D)},
            atol=1e-3,
        )

    @pytest.mark.hw
    def test_rope_rank4_hw(self, compile_and_run):
        np.random.seed(7)
        BS, S, H, D = 2, 16, 4, 32
        half = D // 2
        x = np.random.randn(BS, S, H, D).astype(np.float32)
        fc = np.random.randn(S, half).astype(np.float32)
        fs = np.random.randn(S, half).astype(np.float32)

        def build(b):
            xv = b.add_input("x", (BS, S, H, D))
            fcv = b.add_input("freqs_cos", (S, half))
            fsv = b.add_input("freqs_sin", (S, half))
            fcb = b.broadcast_to(b.reshape(fcv, (1, S, 1, half)),
                                 (BS, S, H, half))
            fsb = b.broadcast_to(b.reshape(fsv, (1, S, 1, half)),
                                 (BS, S, H, half))
            x1 = b.slice(xv, starts=(0, 0, 0, 0), stops=(BS, S, H, half))
            x2 = b.slice(xv, starts=(0, 0, 0, half), stops=(BS, S, H, D))
            rot1 = b.sub(b.mul(x1, fcb), b.mul(x2, fsb))
            rot2 = b.add(b.mul(x1, fsb), b.mul(x2, fcb))
            out = b.concat([rot1, rot2], axis=3)
            b.set_outputs({"x_out": out})

        _lower_and_check_hw(
            compile_and_run, build,
            {"x": x, "freqs_cos": fc, "freqs_sin": fs},
            {"x_out": (BS, S, H, D)},
        )


# ---------------------------------------------------------------------------
# Feedforward (SwiGLU) — fused gate+up matmul followed by split
# ---------------------------------------------------------------------------

class TestFeedforwardPatterns:
    def test_swiglu_fused_gate_up_split(self):
        """SwiGLU as the model implements it: matmul to (S, 2*I), split, silu, mul, matmul."""
        np.random.seed(42)
        B, S, D = 1, 16, 128
        intermediate = 256

        x_np = np.random.randn(B, S, D).astype(np.float32)
        gu_w = np.random.randn(D, 2 * intermediate).astype(np.float32) * 0.02
        d_w = np.random.randn(intermediate, D).astype(np.float32) * 0.02

        def build(b):
            x = b.add_input("x", (B, S, D))
            gate_up_w = b.add_input("gate_up_w", (D, 2 * intermediate))
            down_w = b.add_input("down_w", (intermediate, D))
            mm = b.matmul(x, gate_up_w)
            gate, up = b.split(mm, 2, axis=2)
            hidden = b.mul(b.silu(gate), up)
            out = b.matmul(hidden, down_w)
            b.set_outputs({"result": out})

        _lower_and_check(
            build,
            {"x": x_np, "gate_up_w": gu_w, "down_w": d_w},
            {"result": (B, S, D)},
            atol=1e-3, rtol=1e-3,
        )

    @pytest.mark.hw
    def test_swiglu_fused_gate_up_split_hw(self, compile_and_run):
        np.random.seed(42)
        B, S, D = 1, 16, 128
        intermediate = 256

        x_np = np.random.randn(B, S, D).astype(np.float32)
        gu_w = np.random.randn(D, 2 * intermediate).astype(np.float32) * 0.02
        d_w = np.random.randn(intermediate, D).astype(np.float32) * 0.02

        def build(b):
            x = b.add_input("x", (B, S, D))
            gate_up_w = b.add_input("gate_up_w", (D, 2 * intermediate))
            down_w = b.add_input("down_w", (intermediate, D))
            mm = b.matmul(x, gate_up_w)
            gate, up = b.split(mm, 2, axis=2)
            hidden = b.mul(b.silu(gate), up)
            out = b.matmul(hidden, down_w)
            b.set_outputs({"result": out})

        _lower_and_check_hw(
            compile_and_run, build,
            {"x": x_np, "gate_up_w": gu_w, "down_w": d_w},
            {"result": (B, S, D)},
        )
