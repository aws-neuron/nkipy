"""End-to-end lowering tests for playground/tensor_layout_solver patterns.

Exercises the full tensor_ir → nki_ir pipeline using the graph-builder
patterns defined in nkigen_lite.tensor_ir.patterns.
Each pattern is tested at both interpreter and HW level.

Patterns that cannot yet be lowered (due to F-axis concat, broadcast_to
rank mismatch, or reshape-based multi-output splits) are marked xfail
with the specific limitation documented.
"""
from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.tensor_ir.ir import run as tensor_run
from nkigen_lite.tensor_ir.passes.lower_to_nki import lower_to_nki
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel
from nkigen_lite.tensor_ir.patterns import (
    build_rmsnorm,
    build_softmax,
    build_layernorm,
    build_residual_add,
    build_cross_lane_reduce,
    build_fused_scale_bias_activation,
    build_matmul_with_epilogue,
    build_ffn,
    build_swiglu_gate,
    build_attention,
    build_cross_entropy_loss,
    build_elementwise_merge_for_utilization,
    build_rope,
    build_kv_cache_update,
    build_multi_head_projection,
    build_gqa_attention,
    build_linear_attention_deltanet,
    build_elementwise_rank_change,
    build_elementwise_split_for_batched_mm,
    build_qk_norm,
    build_transformer_layer,
)

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


def _lower_and_check(graph, inputs, atol=1e-3, rtol=1e-3):
    """Lower graph to nki_ir, run both interpreters, compare."""
    ref = tensor_run(graph, inputs)
    nki_graph = lower_to_nki(graph)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    nki_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            nki_inputs[name] = arr.reshape(expected)
        else:
            nki_inputs[name] = arr
    for name in graph.outputs:
        key = f"{name}_out"
        expected = graph_input_shapes.get(key)
        if expected is not None:
            nki_inputs[key] = np.zeros(expected, dtype=np.float32)
        else:
            nki_inputs[key] = np.zeros(ref[name].shape, dtype=np.float32)
    actual = nki_run(nki_graph, nki_inputs)
    for k in ref:
        actual_k = actual[k]
        ref_k = ref[k]
        if actual_k.shape != ref_k.shape:
            actual_k = actual_k.reshape(ref_k.shape)
        np.testing.assert_allclose(actual_k, ref_k, atol=atol, rtol=rtol)
    return nki_graph


def _lower_and_check_hw(compile_and_run, graph, inputs, atol=1e-3, rtol=1e-3):
    """Lower graph, verify interpreter, then run on HW."""
    ref = tensor_run(graph, inputs)
    nki_graph = lower_to_nki(graph)
    graph_input_shapes = {v.name: v.type.shape for v in nki_graph.inputs}
    # Interpreter check first
    nki_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            nki_inputs[name] = arr.reshape(expected)
        else:
            nki_inputs[name] = arr
    for name in graph.outputs:
        key = f"{name}_out"
        expected = graph_input_shapes.get(key)
        if expected is not None:
            nki_inputs[key] = np.zeros(expected, dtype=np.float32)
        else:
            nki_inputs[key] = np.zeros(ref[name].shape, dtype=np.float32)
    interp_result = nki_run(nki_graph, nki_inputs)
    for k in ref:
        actual_k = interp_result[k]
        ref_k = ref[k]
        if actual_k.shape != ref_k.shape:
            actual_k = actual_k.reshape(ref_k.shape)
        np.testing.assert_allclose(
            actual_k, ref_k, atol=atol, rtol=rtol,
            err_msg=f"Interpreter mismatch on '{k}' (must pass before HW)",
        )
    # HW execution
    hw_inputs = {}
    for name, arr in inputs.items():
        expected = graph_input_shapes.get(name)
        if expected is not None and arr.shape != expected:
            hw_inputs[name] = arr.reshape(expected)
        else:
            hw_inputs[name] = arr
    hw_outputs = {}
    for name in graph.outputs:
        key = f"{name}_out"
        expected = graph_input_shapes.get(key)
        if expected is not None:
            hw_outputs[key] = np.zeros(expected, dtype=np.float32)
        else:
            hw_outputs[key] = np.zeros(ref[name].shape, dtype=np.float32)
    hw_result = compile_and_run(nki_graph, hw_inputs, hw_outputs)
    for k in ref:
        hw_k = hw_result[f"{k}_out"]
        ref_k = ref[k]
        if hw_k.shape != ref_k.shape:
            hw_k = hw_k.reshape(ref_k.shape)
        np.testing.assert_allclose(hw_k, ref_k, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Normalization patterns
# ---------------------------------------------------------------------------

class TestNormalizationPatterns:

    def test_rmsnorm_rank2(self):
        np.random.seed(0)
        g = build_rmsnorm((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "w": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_rmsnorm_rank3(self):
        np.random.seed(1)
        g = build_rmsnorm((2, 32, 128))
        inputs = {
            "x": np.random.randn(2, 32, 128).astype(np.float32),
            "w": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_layernorm_rank2(self):
        np.random.seed(2)
        g = build_layernorm((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "gamma": np.random.randn(128).astype(np.float32),
            "beta": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_layernorm_rank3(self):
        np.random.seed(3)
        g = build_layernorm((2, 32, 256))
        inputs = {
            "x": np.random.randn(2, 32, 256).astype(np.float32),
            "gamma": np.random.randn(256).astype(np.float32),
            "beta": np.random.randn(256).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_rmsnorm_rank3_hw(self, compile_and_run):
        np.random.seed(1)
        g = build_rmsnorm((2, 32, 128))
        inputs = {
            "x": np.random.randn(2, 32, 128).astype(np.float32),
            "w": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_layernorm_rank3_hw(self, compile_and_run):
        np.random.seed(3)
        g = build_layernorm((2, 32, 256))
        inputs = {
            "x": np.random.randn(2, 32, 256).astype(np.float32),
            "gamma": np.random.randn(256).astype(np.float32),
            "beta": np.random.randn(256).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)



# ---------------------------------------------------------------------------
# Softmax / cross-entropy
# ---------------------------------------------------------------------------

class TestSoftmaxPatterns:

    def test_softmax_rank2(self):
        np.random.seed(10)
        g = build_softmax((64, 128))
        inputs = {"x": np.random.randn(64, 128).astype(np.float32)}
        _lower_and_check(g, inputs)

    def test_softmax_rank3(self):
        np.random.seed(11)
        g = build_softmax((2, 32, 128))
        inputs = {"x": np.random.randn(2, 32, 128).astype(np.float32)}
        _lower_and_check(g, inputs)

    def test_softmax_rank4(self):
        np.random.seed(12)
        g = build_softmax((1, 4, 32, 64))
        inputs = {"x": np.random.randn(1, 4, 32, 64).astype(np.float32)}
        _lower_and_check(g, inputs)

    def test_cross_entropy_loss(self):
        np.random.seed(13)
        g = build_cross_entropy_loss(1, 16, 64)
        inputs = {"logits": np.random.randn(1, 16, 64).astype(np.float32)}
        _lower_and_check(g, inputs)

    def test_cross_entropy_loss_larger(self):
        np.random.seed(14)
        g = build_cross_entropy_loss(2, 32, 128)
        inputs = {"logits": np.random.randn(2, 32, 128).astype(np.float32)}
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_softmax_rank3_hw(self, compile_and_run):
        np.random.seed(11)
        g = build_softmax((2, 32, 128))
        inputs = {"x": np.random.randn(2, 32, 128).astype(np.float32)}
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_cross_entropy_loss_hw(self, compile_and_run):
        np.random.seed(13)
        g = build_cross_entropy_loss(1, 16, 64)
        inputs = {"logits": np.random.randn(1, 16, 64).astype(np.float32)}
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# FFN / gating patterns
# ---------------------------------------------------------------------------

class TestFFNPatterns:

    def test_ffn_rank2(self):
        np.random.seed(20)
        g = build_ffn((32, 64), intermediate=128)
        inputs = {
            "x": np.random.randn(32, 64).astype(np.float32),
            "gate_up_w": np.random.randn(64, 256).astype(np.float32) * 0.02,
            "down_w": np.random.randn(128, 64).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_ffn_rank3(self):
        np.random.seed(21)
        g = build_ffn((2, 16, 128), intermediate=256)
        inputs = {
            "x": np.random.randn(2, 16, 128).astype(np.float32),
            "gate_up_w": np.random.randn(128, 512).astype(np.float32) * 0.02,
            "down_w": np.random.randn(256, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_swiglu_gate_rank2(self):
        np.random.seed(22)
        g = build_swiglu_gate((32, 64), intermediate=128)
        inputs = {
            "x": np.random.randn(32, 64).astype(np.float32),
            "W_gate": np.random.randn(64, 128).astype(np.float32) * 0.02,
            "W_up": np.random.randn(64, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_swiglu_gate_rank3(self):
        np.random.seed(23)
        g = build_swiglu_gate((1, 32, 128), intermediate=256)
        inputs = {
            "x": np.random.randn(1, 32, 128).astype(np.float32),
            "W_gate": np.random.randn(128, 256).astype(np.float32) * 0.02,
            "W_up": np.random.randn(128, 256).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_ffn_rank2_hw(self, compile_and_run):
        np.random.seed(20)
        g = build_ffn((32, 64), intermediate=128)
        inputs = {
            "x": np.random.randn(32, 64).astype(np.float32),
            "gate_up_w": np.random.randn(64, 256).astype(np.float32) * 0.02,
            "down_w": np.random.randn(128, 64).astype(np.float32) * 0.02,
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_ffn_rank3_hw(self, compile_and_run):
        np.random.seed(21)
        g = build_ffn((2, 16, 128), intermediate=256)
        inputs = {
            "x": np.random.randn(2, 16, 128).astype(np.float32),
            "gate_up_w": np.random.randn(128, 512).astype(np.float32) * 0.02,
            "down_w": np.random.randn(256, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_swiglu_gate_hw(self, compile_and_run):
        np.random.seed(22)
        g = build_swiglu_gate((32, 64), intermediate=128)
        inputs = {
            "x": np.random.randn(32, 64).astype(np.float32),
            "W_gate": np.random.randn(64, 128).astype(np.float32) * 0.02,
            "W_up": np.random.randn(64, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttentionPatterns:

    def test_attention_rank4_small(self):
        np.random.seed(30)
        g = build_attention((1, 2, 32, 16))
        inputs = {
            "q": np.random.randn(1, 2, 32, 16).astype(np.float32),
            "k": np.random.randn(1, 2, 32, 16).astype(np.float32),
            "v": np.random.randn(1, 2, 32, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_attention_rank4_multi_batch(self):
        np.random.seed(31)
        g = build_attention((2, 4, 32, 16))
        inputs = {
            "q": np.random.randn(2, 4, 32, 16).astype(np.float32),
            "k": np.random.randn(2, 4, 32, 16).astype(np.float32),
            "v": np.random.randn(2, 4, 32, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_attention_rank3(self):
        np.random.seed(32)
        g = build_attention((4, 32, 16))
        inputs = {
            "q": np.random.randn(4, 32, 16).astype(np.float32),
            "k": np.random.randn(4, 32, 16).astype(np.float32),
            "v": np.random.randn(4, 32, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_attention_rank4_hw(self, compile_and_run):
        np.random.seed(30)
        g = build_attention((1, 2, 32, 16))
        inputs = {
            "q": np.random.randn(1, 2, 32, 16).astype(np.float32),
            "k": np.random.randn(1, 2, 32, 16).astype(np.float32),
            "v": np.random.randn(1, 2, 32, 16).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_attention_multi_batch_hw(self, compile_and_run):
        np.random.seed(31)
        g = build_attention((2, 4, 32, 16))
        inputs = {
            "q": np.random.randn(2, 4, 32, 16).astype(np.float32),
            "k": np.random.randn(2, 4, 32, 16).astype(np.float32),
            "v": np.random.randn(2, 4, 32, 16).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# Residual / projection patterns
# ---------------------------------------------------------------------------

class TestResidualProjectionPatterns:

    def test_residual_add_rank2(self):
        np.random.seed(40)
        g = build_residual_add((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "W": np.random.randn(128, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_residual_add_rank3(self):
        np.random.seed(41)
        g = build_residual_add((2, 32, 64))
        inputs = {
            "x": np.random.randn(2, 32, 64).astype(np.float32),
            "W": np.random.randn(64, 64).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_residual_add_rank2_hw(self, compile_and_run):
        np.random.seed(40)
        g = build_residual_add((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "W": np.random.randn(128, 128).astype(np.float32) * 0.02,
        }
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# Elementwise / activation patterns
# ---------------------------------------------------------------------------

class TestElementwisePatterns:

    def test_fused_scale_bias_activation_rank2(self):
        np.random.seed(50)
        g = build_fused_scale_bias_activation((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "scale": np.random.randn(128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_fused_scale_bias_activation_rank3(self):
        np.random.seed(51)
        g = build_fused_scale_bias_activation((2, 32, 128))
        inputs = {
            "x": np.random.randn(2, 32, 128).astype(np.float32),
            "scale": np.random.randn(128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_matmul_with_epilogue(self):
        np.random.seed(52)
        g = build_matmul_with_epilogue((64, 128), N=64)
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "W": np.random.randn(128, 64).astype(np.float32) * 0.02,
            "bias": np.random.randn(64).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_matmul_with_epilogue_rank3(self):
        np.random.seed(53)
        g = build_matmul_with_epilogue((2, 32, 64), N=128)
        inputs = {
            "x": np.random.randn(2, 32, 64).astype(np.float32),
            "W": np.random.randn(64, 128).astype(np.float32) * 0.02,
            "bias": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_elementwise_merge_for_utilization(self):
        np.random.seed(54)
        g = build_elementwise_merge_for_utilization()
        inputs = {
            "x": np.random.randn(4, 32, 64).astype(np.float32),
            "W": np.random.randn(64, 128).astype(np.float32) * 0.02,
            "bias": np.random.randn(128).astype(np.float32),
            "scale": np.random.randn(128).astype(np.float32),
            "W2": np.random.randn(128, 64).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_fused_scale_bias_activation_hw(self, compile_and_run):
        np.random.seed(50)
        g = build_fused_scale_bias_activation((64, 128))
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "scale": np.random.randn(128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_matmul_with_epilogue_hw(self, compile_and_run):
        np.random.seed(52)
        g = build_matmul_with_epilogue((64, 128), N=64)
        inputs = {
            "x": np.random.randn(64, 128).astype(np.float32),
            "W": np.random.randn(128, 64).astype(np.float32) * 0.02,
            "bias": np.random.randn(64).astype(np.float32),
        }
        _lower_and_check_hw(compile_and_run, g, inputs)

    @pytest.mark.hw
    def test_elementwise_merge_hw(self, compile_and_run):
        np.random.seed(54)
        g = build_elementwise_merge_for_utilization()
        inputs = {
            "x": np.random.randn(4, 32, 64).astype(np.float32),
            "W": np.random.randn(64, 128).astype(np.float32) * 0.02,
            "bias": np.random.randn(128).astype(np.float32),
            "scale": np.random.randn(128).astype(np.float32),
            "W2": np.random.randn(128, 64).astype(np.float32) * 0.02,
        }
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# Reduction patterns
# ---------------------------------------------------------------------------

class TestReductionPatterns:

    def test_cross_lane_reduce_rank2(self):
        np.random.seed(60)
        g = build_cross_lane_reduce((64, 128))
        inputs = {"x": np.random.randn(64, 128).astype(np.float32)}
        _lower_and_check(g, inputs)

    def test_cross_lane_reduce_rank3(self):
        np.random.seed(61)
        g = build_cross_lane_reduce((4, 32, 64))
        inputs = {"x": np.random.randn(4, 32, 64).astype(np.float32)}
        _lower_and_check(g, inputs)

    @pytest.mark.hw
    def test_cross_lane_reduce_hw(self, compile_and_run):
        np.random.seed(60)
        g = build_cross_lane_reduce((64, 128))
        inputs = {"x": np.random.randn(64, 128).astype(np.float32)}
        _lower_and_check_hw(compile_and_run, g, inputs)


# ---------------------------------------------------------------------------
# Patterns with known lowering limitations (xfail)
# ---------------------------------------------------------------------------

class TestKnownLimitations:

    def test_rope_rank2(self):
        np.random.seed(70)
        g = build_rope((64, 32))
        inputs = {
            "x": np.random.randn(64, 32).astype(np.float32),
            "cos": np.random.randn(64, 16).astype(np.float32),
            "sin": np.random.randn(64, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_kv_cache_update(self):
        np.random.seed(71)
        g = build_kv_cache_update(1, 2, 16, 4, 16)
        inputs = {
            "cached_k": np.random.randn(1, 2, 16, 16).astype(np.float32),
            "new_k": np.random.randn(1, 2, 4, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_gqa_attention(self):
        np.random.seed(72)
        g = build_gqa_attention(1, 4, 2, 16, 16)
        inputs = {
            "q": np.random.randn(1, 4, 16, 16).astype(np.float32),
            "k": np.random.randn(1, 2, 16, 16).astype(np.float32),
            "v": np.random.randn(1, 2, 16, 16).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_linear_attention_deltanet(self):
        np.random.seed(73)
        g = build_linear_attention_deltanet()
        inputs = {
            "q": np.random.randn(1, 4, 64, 32).astype(np.float32),
            "k": np.random.randn(1, 4, 64, 32).astype(np.float32),
            "v": np.random.randn(1, 4, 64, 32).astype(np.float32),
            "beta_logits": np.random.randn(1, 4, 64).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_elementwise_rank_change(self):
        np.random.seed(74)
        g = build_elementwise_rank_change()
        inputs = {
            "x": np.random.randn(2, 64, 128).astype(np.float32),
            "W": np.random.randn(128, 256).astype(np.float32) * 0.02,
            "V": np.random.randn(2, 256, 32).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_elementwise_split_for_batched_mm(self):
        np.random.seed(75)
        g = build_elementwise_split_for_batched_mm()
        inputs = {
            "x": np.random.randn(128, 128).astype(np.float32),
            "W": np.random.randn(128, 64).astype(np.float32) * 0.02,
            "K": np.random.randn(2, 64, 32).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_multi_head_projection(self):
        np.random.seed(76)
        g = build_multi_head_projection(1, 16, 64, 4)
        inputs = {
            "x": np.random.randn(1, 16, 64).astype(np.float32),
            "W_qkv": np.random.randn(64, 192).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)

    def test_qk_norm(self):
        np.random.seed(77)
        g = build_qk_norm(1, 32, 4, 64)
        inputs = {
            "q": np.random.randn(1, 32, 4, 64).astype(np.float32),
            "k": np.random.randn(1, 32, 4, 64).astype(np.float32),
            "q_norm_w": np.random.randn(64).astype(np.float32),
            "k_norm_w": np.random.randn(64).astype(np.float32),
        }
        _lower_and_check(g, inputs)

    def test_transformer_layer(self):
        np.random.seed(78)
        g = build_transformer_layer(1, 16, 32, 2, 64)
        inputs = {
            "x": np.random.randn(1, 16, 32).astype(np.float32) * 0.1,
            "attn_norm_w": np.random.randn(32).astype(np.float32),
            "W_qkv": np.random.randn(32, 96).astype(np.float32) * 0.02,
            "W_o": np.random.randn(32, 32).astype(np.float32) * 0.02,
            "ffn_norm_w": np.random.randn(32).astype(np.float32),
            "gate_up_w": np.random.randn(32, 128).astype(np.float32) * 0.02,
            "down_w": np.random.randn(64, 32).astype(np.float32) * 0.02,
        }
        _lower_and_check(g, inputs)
