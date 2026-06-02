"""Tests for the orchestrated direct lowering pass.

Ported from test_fusion_tile_lower.py — all patterns lowered via
direct_lower.lower_graph with HBM boundaries between op segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder as TensorBuilder, run as tensor_run
from nkigen_lite.tensor_ir.passes.canonicalize import canonicalize
from nkigen_lite.tensor_ir.passes.decompose import decompose
from nkigen_lite.tensor_ir.passes.layout_solver import solve_graph
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower import lower_graph

try:
    import nki.compiler.kernel_builder as nb_kb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

pytestmark = pytest.mark.hw

from nkigen_lite.tensor_ir.patterns import (
    build_rmsnorm,
    build_softmax,
    build_ffn,
    build_attention,
    build_full_attention,
    build_layernorm,
    build_gqa_attention,
    build_rope,
    build_residual_add,
    build_kv_cache_update,
    build_swiglu_gate,
    build_multi_head_projection,
    build_output_projection,
    build_cross_entropy_loss,
    build_linear_attention_deltanet,
    build_cross_lane_reduce,
    build_fused_scale_bias_activation,
    build_matmul_with_epilogue,
    build_elementwise_rank_change,
    build_elementwise_merge_for_utilization,
    build_elementwise_split_for_batched_mm,
    build_qk_norm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_interp_then_hw(nki_graph, graph, inputs, ref, atol):
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required")

    nki_inputs = dict(inputs)
    for out_name, out_val in graph.outputs.items():
        nki_inputs[f"{out_name}_out"] = np.zeros(out_val.type.shape, dtype=np.float32)
    interp = nki_run(nki_graph, nki_inputs)
    for k in ref:
        np.testing.assert_allclose(
            interp[k], ref[k], atol=atol, rtol=atol,
            err_msg=f"Interpreter mismatch on {k!r} (must pass before HW)",
        )

    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_inputs = dict(inputs)
    hw_outputs = {
        f"{out_name}_out": np.zeros(out_val.type.shape, dtype=np.float32)
        for out_name, out_val in graph.outputs.items()
    }
    nb_kb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    for k in ref:
        np.testing.assert_allclose(
            hw_outputs[f"{k}_out"], ref[k], atol=atol, rtol=atol,
            err_msg=f"HW mismatch on {k!r}",
        )


def _lower_and_check(build_fn, inputs, atol=1e-2):
    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph
    layouts = solve_graph(graph)
    nki_graph = lower_graph(graph, layouts)
    ref = tensor_run(graph, inputs)
    _check_interp_then_hw(nki_graph, graph, inputs, ref, atol)


def _lower_pattern_and_check(build_fn, input_gen, atol=1e-2):
    graph = build_fn()
    canonicalize(graph)
    decompose(graph)
    layouts = solve_graph(graph)
    nki_graph = lower_graph(graph, layouts)
    inputs = input_gen(graph)
    ref = tensor_run(graph, inputs)
    _check_interp_then_hw(nki_graph, graph, inputs, ref, atol)


def _random_inputs(graph, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return {v.name: rng.standard_normal(v.type.shape).astype(np.float32) for v in graph.inputs}


# ---------------------------------------------------------------------------
# Basic elementwise
# ---------------------------------------------------------------------------


class TestBasicElementwise:
    def test_add_relu(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            bias = b.add_input("bias", (128, 256), DType.F32)
            y = b.relu(b.add(x, bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
            "bias": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_gelu_bias(self):
        rng = np.random.default_rng(42)

        def build(b):
            x = b.add_input("x", (512, 1024), DType.F32)
            bias = b.add_input("bias", (512, 1024), DType.F32)
            y = b.gelu(b.add(x, bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((512, 1024)).astype(np.float32),
            "bias": rng.standard_normal((512, 1024)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Shape coverage
# ---------------------------------------------------------------------------


class TestShapeCoverage:
    @pytest.mark.parametrize("shape", [
        (1, 1), (1, 700), (128, 1), (129, 33), (300, 700),
        (7, 13, 5), (5, 200, 97), (4, 128, 256), (2, 3, 64, 50),
    ])
    def test_add_relu_shapes(self, shape):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", shape, DType.F32)
            bias = b.add_input("bias", shape, DType.F32)
            y = b.relu(b.add(x, bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal(shape).astype(np.float32),
            "bias": rng.standard_normal(shape).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Patterns (ported from test_fusion_tile_lower.py)
# ---------------------------------------------------------------------------


class TestFusedScaleBiasActivation:
    def test_rank2(self):
        _lower_pattern_and_check(
            lambda: build_fused_scale_bias_activation((128, 256)), _random_inputs)

    def test_rank3(self):
        _lower_pattern_and_check(
            lambda: build_fused_scale_bias_activation((4, 128, 256)), _random_inputs)


class TestRMSNorm:
    @pytest.mark.parametrize("shape", [
        (128, 512), (4, 128, 512), (2, 4, 128, 512),
    ])
    def test_rmsnorm(self, shape):
        _lower_pattern_and_check(lambda: build_rmsnorm(shape), _random_inputs)


class TestLayerNorm:
    @pytest.mark.parametrize("shape", [
        (128, 512), (4, 128, 512), (2, 4, 128, 512),
    ])
    def test_layernorm(self, shape):
        _lower_pattern_and_check(lambda: build_layernorm(shape), _random_inputs)


class TestSoftmax:
    @pytest.mark.parametrize("shape", [
        (128, 512), (4, 128, 512), (2, 4, 128, 512),
    ])
    def test_softmax(self, shape):
        _lower_pattern_and_check(lambda: build_softmax(shape), _random_inputs)


class TestCrossEntropyLoss:
    def test_cross_entropy(self):
        _lower_pattern_and_check(
            lambda: build_cross_entropy_loss(2, 64, 1024), _random_inputs)


class TestCrossLaneReduce:
    @pytest.mark.parametrize("shape", [(128, 512)])
    def test_cross_lane_reduce(self, shape):
        _lower_pattern_and_check(
            lambda: build_cross_lane_reduce(shape), _random_inputs)


class TestQKNorm:
    def test_qk_norm(self):
        _lower_pattern_and_check(
            lambda: build_qk_norm(1, 32, 4, 64), _random_inputs)


class TestMatmulEpilogue:
    @pytest.mark.parametrize("shape,N", [
        ((128, 256), 512),
        ((4, 128, 256), 256),
        ((128, 256), 1024),
    ])
    def test_matmul_epilogue(self, shape, N):
        _lower_pattern_and_check(
            lambda: build_matmul_with_epilogue(shape, N=N), _random_inputs)


class TestSwiGLU:
    @pytest.mark.parametrize("shape,intermediate", [
        ((64, 256), 512),
        ((2, 64, 256), 512),
    ])
    def test_swiglu(self, shape, intermediate):
        _lower_pattern_and_check(
            lambda: build_swiglu_gate(shape, intermediate=intermediate), _random_inputs)


class TestFFN:
    @pytest.mark.parametrize("shape,intermediate", [
        ((64, 256), 512),
        ((2, 64, 256), 512),
    ])
    def test_ffn(self, shape, intermediate):
        _lower_pattern_and_check(
            lambda: build_ffn(shape, intermediate=intermediate), _random_inputs,
            atol=0.1)


class TestAttention:
    @pytest.mark.parametrize("shape", [
        (4, 32, 64),
        (2, 8, 64, 64),
    ])
    def test_attention(self, shape):
        _lower_pattern_and_check(
            lambda: build_attention(shape), _random_inputs)


class TestRoPE:
    @pytest.mark.parametrize("shape", [
        (4, 64, 64),
        (2, 8, 64, 64),
    ])
    def test_rope(self, shape):
        _lower_pattern_and_check(lambda: build_rope(shape), _random_inputs)


class TestKVCacheUpdate:
    def test_kv_cache(self):
        _lower_pattern_and_check(
            lambda: build_kv_cache_update(B=1, H=8, S_cached=128, S_new=16, D=64),
            _random_inputs)


class TestOutputProjection:
    def test_output_proj(self):
        _lower_pattern_and_check(
            lambda: build_output_projection(B=2, H=8, S=64, D_h=32, D=256),
            _random_inputs)


class TestMultiHeadProjection:
    def test_multi_head_proj(self):
        _lower_pattern_and_check(
            lambda: build_multi_head_projection(B=2, S=64, D=256, H=8),
            _random_inputs)


class TestGQAAttention:
    def test_gqa(self):
        _lower_pattern_and_check(
            lambda: build_gqa_attention(B=1, H_q=8, H_kv=2, S=64, D=64),
            _random_inputs)


class TestDeltaNet:
    def test_deltanet(self):
        _lower_pattern_and_check(
            lambda: build_linear_attention_deltanet(), _random_inputs)


class TestResidualAdd:
    @pytest.mark.parametrize("shape", [(64, 256)])
    def test_residual(self, shape):
        _lower_pattern_and_check(
            lambda: build_residual_add(shape), _random_inputs)


class TestElementwiseRankChange:
    def test_rank_change(self):
        _lower_pattern_and_check(build_elementwise_rank_change, _random_inputs)


class TestElementwiseMerge:
    def test_merge(self):
        _lower_pattern_and_check(
            build_elementwise_merge_for_utilization, _random_inputs)


class TestElementwiseSplit:
    def test_split(self):
        _lower_pattern_and_check(
            build_elementwise_split_for_batched_mm, _random_inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-q"])
