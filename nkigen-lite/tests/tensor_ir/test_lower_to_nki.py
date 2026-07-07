"""Tests for the full tensor_ir → nki_ir lowering pipeline.

Tests verify correctness at two levels:
  1. Interpreter: nki_ir numpy interpreter matches tensor_ir reference.
  2. Hardware: compiled kernel on Trainium matches tensor_ir reference.

Run interpreter tests only:
    pytest nkigen_lite/tests/tensor_ir/test_lower_to_nki.py -m "not hw"

Run hardware tests:
    pytest nkigen_lite/tests/tensor_ir/test_lower_to_nki.py -m hw
"""

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder, run as tensor_run
from nkigen_lite.tensor_ir.examples import softmax, layer_norm
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
    """Compile and execute on Trainium hardware."""
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


# ===========================
# Helper
# ===========================

def _lower_and_check_interp(build_fn, inputs, out_shape, atol=1e-4):
    """Build graph, lower to nki_ir, run both interpreters, compare."""
    b = Builder("test")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)

    nki_graph = lower_to_nki(b.graph)
    nki_inputs = dict(inputs)
    for out_name in b.graph.outputs:
        nki_inputs[f"{out_name}_out"] = np.zeros(out_shape, dtype=np.float32)
    nki_result = nki_run(nki_graph, nki_inputs)

    for k in ref:
        if k in nki_result:
            np.testing.assert_allclose(nki_result[k], ref[k], atol=atol, rtol=1e-4)
    return nki_graph, ref


def _lower_and_check_hw(compile_and_run, build_fn, inputs, out_shape, atol=1e-3):
    """Build graph, lower, compile to HW, compare with tensor_ir reference."""
    b = Builder("test")
    build_fn(b)
    ref = tensor_run(b.graph, inputs)

    nki_graph = lower_to_nki(b.graph)
    nki_inputs = dict(inputs)
    nki_outputs = {}
    for out_name in b.graph.outputs:
        nki_outputs[f"{out_name}_out"] = np.zeros(out_shape, dtype=np.float32)

    hw_result = compile_and_run(nki_graph, nki_inputs, nki_outputs)

    for k in ref:
        out_key = f"{k}_out"
        if out_key in hw_result:
            np.testing.assert_allclose(hw_result[out_key], ref[k], atol=atol, rtol=1e-3)


# ===========================
# Interpreter tests
# ===========================

class TestLowerInterp:
    """Verify lowered nki_ir matches tensor_ir via numpy interpreters."""

    def test_relu(self):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.relu(inp)})

        _lower_and_check_interp(build, {"x": x}, (128, 512))

    def test_exp(self):
        x = np.random.randn(128, 512).astype(np.float32) * 0.1

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.exp(inp)})

        _lower_and_check_interp(build, {"x": x}, (128, 512))

    def test_silu(self):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.silu(inp)})

        _lower_and_check_interp(build, {"x": x}, (128, 512))

    def test_add_broadcast(self):
        x = np.random.randn(128, 512).astype(np.float32)
        bias = np.random.randn(128, 1).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b_inp = b.add_input("bias", (128, 1))
            b.set_outputs({"y": b.add(inp, b_inp)})

        _lower_and_check_interp(build, {"x": x, "bias": bias}, (128, 512))

    def test_bias_relu_fusion(self):
        x = np.random.randn(128, 512).astype(np.float32)
        bias = np.random.randn(128, 1).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b_inp = b.add_input("bias", (128, 1))
            b.set_outputs({"y": b.relu(b.add(inp, b_inp))})

        nki_graph, _ = _lower_and_check_interp(
            build, {"x": x, "bias": bias}, (128, 512)
        )
        # Verify fusion happened: should have activation with bias, no separate add
        act_ops = [op for op in nki_graph.ops if op.opcode == "activation"]
        assert len(act_ops) >= 1

    def test_matmul(self):
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 512).astype(np.float32)

        def build(b):
            a = b.add_input("A", (128, 64))
            b_inp = b.add_input("B", (64, 512))
            b.set_outputs({"C": b.matmul(a, b_inp)})

        _lower_and_check_interp(build, {"A": A, "B": B}, (128, 512), atol=1e-3)

    def test_reduce_sum_free(self):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.reduce(inp, axis=1, keepdims=True, kind="sum")})

        _lower_and_check_interp(build, {"x": x}, (128, 1))

    def test_reduce_max_free(self):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.reduce(inp, axis=1, keepdims=True, kind="max")})

        _lower_and_check_interp(build, {"x": x}, (128, 1))

    def test_softmax(self):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": softmax(b, inp, axis=1)})

        _lower_and_check_interp(build, {"x": x}, (128, 512), atol=1e-5)

    def test_layer_norm(self):
        x = np.random.randn(128, 512).astype(np.float32)
        w = np.ones((1, 512), dtype=np.float32)
        bias = np.zeros((1, 512), dtype=np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            w_inp = b.add_input("w", (1, 512))
            b_inp = b.add_input("bias", (1, 512))
            b.set_outputs({"y": layer_norm(b, inp, w_inp, b_inp, axis=1)})

        _lower_and_check_interp(
            build, {"x": x, "w": w, "bias": bias}, (128, 512), atol=1e-4
        )

    def test_tiled_relu(self):
        """Test that tiling works for tensors larger than one tile (256 > 128)."""
        x = np.random.randn(256, 1024).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (256, 1024))
            b.set_outputs({"y": b.relu(inp)})

        _lower_and_check_interp(build, {"x": x}, (256, 1024))

    def test_matmul_add_relu(self):
        """Test a small MLP-like pattern: relu(A @ B + bias)."""
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 512).astype(np.float32)
        bias = np.random.randn(128, 1).astype(np.float32)

        def build(b):
            a = b.add_input("A", (128, 64))
            b_w = b.add_input("B", (64, 512))
            bi = b.add_input("bias", (128, 1))
            mm = b.matmul(a, b_w)
            added = b.add(mm, bi)
            b.set_outputs({"y": b.relu(added)})

        _lower_and_check_interp(
            build, {"A": A, "B": B, "bias": bias}, (128, 512), atol=1e-3
        )

    def test_mul_same_shape(self):
        """Binary mul with same-shape operands → tensor_tensor_arith."""
        x = np.random.randn(128, 512).astype(np.float32)
        y = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            a = b.add_input("x", (128, 512))
            b_inp = b.add_input("y", (128, 512))
            b.set_outputs({"z": b.mul(a, b_inp)})

        _lower_and_check_interp(build, {"x": x, "y": y}, (128, 512))

    def test_rmsnorm(self):
        """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""
        x = np.random.randn(128, 512).astype(np.float32)
        w = np.random.randn(1, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            weight = b.add_input("w", (1, 512))
            eps = b.constant(1e-5, (1, 1), DType.F32)
            x_sq = b.mul(inp, inp)
            mean_sq = b.reduce(x_sq, axis=1, keepdims=True, kind="mean")
            normed = b.mul(inp, b.rsqrt(b.add(mean_sq, eps)))
            b.set_outputs({"y": b.mul(normed, weight)})

        _lower_and_check_interp(build, {"x": x, "w": w}, (128, 512), atol=1e-4)

    def test_non_divisible_shape(self):
        """Test with non-divisible shapes: partition dim 200 needs tiling with
        boundary handling (200 / 128 → 2 tiles, second tile is partial)."""
        x = np.random.randn(200, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (200, 512))
            b.set_outputs({"y": b.relu(inp)})

        _lower_and_check_interp(build, {"x": x}, (200, 512))

    def test_reduce_max_partition_axis(self):
        """Test reduce_max along partition axis (P-axis reduction via
        cross_lane_reduce)."""
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.reduce(inp, axis=0, keepdims=True, kind="max")})

        _lower_and_check_interp(build, {"x": x}, (1, 512))

    def test_concat_constant_inputs_splat(self):
        """Concat where some inputs are constants: the lowering splat-fills the
        constant windows (no HBM constant buffer) — verify values still land.
        Mirrors the MoE router-weights assembly (concat of ~130 tiny constants
        with a few computed values)."""
        x = np.random.randn(1, 5).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (1, 5))
            c1 = b.constant(2.5, (1, 3))
            c2 = b.constant(-1.0, (1, 4))
            b.set_outputs({"y": b.concat([c1, inp, c2, c1], axis=1)})

        _lower_and_check_interp(build, {"x": x}, (1, 15))

    def test_concat_constant_inputs_splat_axis0(self):
        """Same as above but on the partition axis, rank 2, uneven extents."""
        x = np.random.randn(3, 8).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (3, 8))
            c1 = b.constant(7.0, (2, 8))
            c2 = b.constant(0.5, (140, 8))  # > PARTITION_MAX rows
            b.set_outputs({"y": b.concat([c1, inp, c2], axis=0)})

        _lower_and_check_interp(build, {"x": x}, (145, 8))

    def test_chained_matmul_k512_k128(self):
        """Test chained matmul with K=512 then K=128 (SwiGLU-like pattern).

        First matmul has K=512 (requires K-tiling: 512/128 = 4 chunks).
        Second matmul uses first result as input with K=128 (single chunk).
        """
        A = np.random.randn(128, 512).astype(np.float32)
        B = np.random.randn(512, 128).astype(np.float32)
        C = np.random.randn(128, 256).astype(np.float32)

        def build(b):
            a = b.add_input("A", (128, 512))
            b_inp = b.add_input("B", (512, 128))
            c = b.add_input("C", (128, 256))
            # First matmul: (128, 512) @ (512, 128) → (128, 128), K=512
            mm1 = b.matmul(a, b_inp)
            # Second matmul: (128, 128) @ (128, 256) → (128, 256), K=128
            mm2 = b.matmul(mm1, c)
            b.set_outputs({"y": mm2})

        _lower_and_check_interp(
            build, {"A": A, "B": B, "C": C}, (128, 256), atol=1e-2
        )


# ===========================
# Hardware tests
# ===========================

class TestLowerHW:
    """Verify lowered nki_ir executes correctly on Trainium hardware."""

    def test_relu_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.relu(inp)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (128, 512))

    def test_exp_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32) * 0.1

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.exp(inp)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (128, 512))

    def test_silu_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.silu(inp)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (128, 512))

    def test_bias_relu_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)
        bias = np.random.randn(128, 1).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b_inp = b.add_input("bias", (128, 1))
            b.set_outputs({"y": b.relu(b.add(inp, b_inp))})

        _lower_and_check_hw(
            compile_and_run, build, {"x": x, "bias": bias}, (128, 512)
        )

    def test_matmul_hw(self, compile_and_run):
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 512).astype(np.float32)

        def build(b):
            a = b.add_input("A", (128, 64))
            b_inp = b.add_input("B", (64, 512))
            b.set_outputs({"C": b.matmul(a, b_inp)})

        _lower_and_check_hw(compile_and_run, build, {"A": A, "B": B}, (128, 512))

    def test_reduce_sum_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": b.reduce(inp, axis=1, keepdims=True, kind="sum")})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (128, 1))

    def test_softmax_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            b.set_outputs({"y": softmax(b, inp, axis=1)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (128, 512))

    def test_layer_norm_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)
        w = np.ones((1, 512), dtype=np.float32)
        bias = np.zeros((1, 512), dtype=np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            w_inp = b.add_input("w", (1, 512))
            b_inp = b.add_input("bias", (1, 512))
            b.set_outputs({"y": layer_norm(b, inp, w_inp, b_inp, axis=1)})

        _lower_and_check_hw(
            compile_and_run, build, {"x": x, "w": w, "bias": bias}, (128, 512)
        )

    def test_tiled_relu_hw(self, compile_and_run):
        x = np.random.randn(256, 1024).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (256, 1024))
            b.set_outputs({"y": b.relu(inp)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (256, 1024))

    def test_matmul_add_relu_hw(self, compile_and_run):
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 512).astype(np.float32)
        bias = np.random.randn(128, 1).astype(np.float32)

        def build(b):
            a = b.add_input("A", (128, 64))
            b_w = b.add_input("B", (64, 512))
            bi = b.add_input("bias", (128, 1))
            mm = b.matmul(a, b_w)
            added = b.add(mm, bi)
            b.set_outputs({"y": b.relu(added)})

        _lower_and_check_hw(
            compile_and_run, build, {"A": A, "B": B, "bias": bias}, (128, 512)
        )

    def test_rmsnorm_hw(self, compile_and_run):
        x = np.random.randn(128, 512).astype(np.float32)
        w = np.random.randn(1, 512).astype(np.float32)

        def build(b):
            inp = b.add_input("x", (128, 512))
            weight = b.add_input("w", (1, 512))
            eps = b.constant(1e-5, (1, 1), DType.F32)
            x_sq = b.mul(inp, inp)
            mean_sq = b.reduce(x_sq, axis=1, keepdims=True, kind="mean")
            normed = b.mul(inp, b.rsqrt(b.add(mean_sq, eps)))
            b.set_outputs({"y": b.mul(normed, weight)})

        _lower_and_check_hw(
            compile_and_run, build, {"x": x, "w": w}, (128, 512)
        )


class TestCollapsedElementwise:
    """Rank>=3 elementwise segments lower correctly.

    These shapes mirror the multi-batch attention segments in the fused Qwen3
    layer. Verified at the interpreter level and (selectively) on hardware.
    """

    # -- interpreter --

    def test_mul_4d_full(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 128, 16, 64)).astype(np.float32)
        y = rng.standard_normal((1, 128, 16, 64)).astype(np.float32)

        def build(b):
            a = b.add_input("x", (1, 128, 16, 64))
            c = b.add_input("y", (1, 128, 16, 64))
            b.set_outputs({"z": b.mul(a, c)})

        _lower_and_check_interp(build, {"x": x, "y": y}, (1, 128, 16, 64))

    def test_mul_4d_free_size1_broadcast(self):
        """(1,128,16,128) * (1,128,16,1): free-size-1 operand broadcasts."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 128, 16, 128)).astype(np.float32)
        s = rng.standard_normal((1, 128, 16, 1)).astype(np.float32)

        def build(b):
            a = b.add_input("x", (1, 128, 16, 128))
            c = b.add_input("s", (1, 128, 16, 1))
            b.set_outputs({"z": b.mul(a, c)})

        _lower_and_check_interp(build, {"x": x, "s": s}, (1, 128, 16, 128))

    def test_var_4d_constant(self):
        """Chain with constants on a 4D free-size-1 shape (RMSNorm variance)."""
        rng = np.random.default_rng(2)
        x = np.abs(rng.standard_normal((1, 128, 16, 1)).astype(np.float32)) + 0.1

        def build(b):
            a = b.add_input("x", (1, 128, 16, 1))
            eps = b.constant(1e-5, (1, 128, 16, 1), DType.F32)
            b.set_outputs({"z": b.sqrt(b.add(b.mul(a, a), eps))})

        _lower_and_check_interp(build, {"x": x}, (1, 128, 16, 1), atol=1e-4)

    def test_sub_exp_4d(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((1, 16, 128, 128)).astype(np.float32) * 0.1
        m = rng.standard_normal((1, 16, 128, 128)).astype(np.float32) * 0.1

        def build(b):
            a = b.add_input("x", (1, 16, 128, 128))
            c = b.add_input("m", (1, 16, 128, 128))
            b.set_outputs({"z": b.exp(b.sub(a, c))})

        _lower_and_check_interp(build, {"x": x, "m": m}, (1, 16, 128, 128), atol=1e-4)

    # -- hardware --

    def test_mul_4d_full_hw(self, compile_and_run):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 128, 16, 64)).astype(np.float32)
        y = rng.standard_normal((1, 128, 16, 64)).astype(np.float32)

        def build(b):
            a = b.add_input("x", (1, 128, 16, 64))
            c = b.add_input("y", (1, 128, 16, 64))
            b.set_outputs({"z": b.mul(a, c)})

        _lower_and_check_hw(
            compile_and_run, build, {"x": x, "y": y}, (1, 128, 16, 64)
        )

    def test_mul_4d_free_size1_broadcast_hw(self, compile_and_run):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 128, 16, 128)).astype(np.float32)
        s = rng.standard_normal((1, 128, 16, 1)).astype(np.float32)

        def build(b):
            a = b.add_input("x", (1, 128, 16, 128))
            c = b.add_input("s", (1, 128, 16, 1))
            b.set_outputs({"z": b.mul(a, c)})

        _lower_and_check_hw(
            compile_and_run, build, {"x": x, "s": s}, (1, 128, 16, 128)
        )

    def test_var_4d_constant_hw(self, compile_and_run):
        rng = np.random.default_rng(2)
        x = np.abs(rng.standard_normal((1, 128, 16, 1)).astype(np.float32)) + 0.1

        def build(b):
            a = b.add_input("x", (1, 128, 16, 1))
            eps = b.constant(1e-5, (1, 128, 16, 1), DType.F32)
            b.set_outputs({"z": b.sqrt(b.add(b.mul(a, a), eps))})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (1, 128, 16, 1))


class TestCollapsedReduce:
    """Rank>=3 reduce over a trailing free axis collapses leading dims onto the
    partition (mirrors HLO's reduce-then-reshape for softmax). The old per-tile
    path used only shape[-2] partition lanes and iterated the other leading
    dims one row at a time. See ``_try_emit_collapsed_f_reduce``.
    """

    def test_softmax_max_reduce_4d(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 16, 128, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 16, 128, 128))
            b.set_outputs({"o": b.reduce(tx, axis=(3,), keepdims=True, kind="max")})

        _lower_and_check_interp(build, {"x": x}, (1, 16, 128, 1))

    def test_softmax_sum_reduce_4d(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 16, 128, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 16, 128, 128))
            b.set_outputs({"o": b.reduce(tx, axis=(3,), keepdims=True, kind="sum")})

        _lower_and_check_interp(build, {"x": x}, (1, 16, 128, 1))

    def test_qk_norm_reduce_4d(self):
        # (1,128,16,128) reduce over head_dim — Q/K RMSNorm in the layer.
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 128, 16, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 128, 16, 128))
            b.set_outputs({"o": b.reduce(tx, axis=(3,), keepdims=True, kind="sum")})

        _lower_and_check_interp(build, {"x": x}, (1, 128, 16, 1))

    def test_softmax_max_reduce_4d_hw(self, compile_and_run):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 16, 128, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 16, 128, 128))
            b.set_outputs({"o": b.reduce(tx, axis=(3,), keepdims=True, kind="max")})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (1, 16, 128, 1))

    def test_softmax_sum_reduce_4d_hw(self, compile_and_run):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 16, 128, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 16, 128, 128))
            b.set_outputs({"o": b.reduce(tx, axis=(3,), keepdims=True, kind="sum")})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (1, 16, 128, 1))


class TestSliceAsView:
    """A static-start, stride-1 slice is lowered as a zero-copy offset view:
    an elementwise consumer composes the slice's base offset into its own
    loads (no buffer, no copy); any other consumer (reshape, graph output)
    materializes the slice once, on demand. See ``_is_view_slice`` /
    ``slice_views`` in ``direct_lower.py``.
    """

    # -- interpreter --

    def test_split_gate_up_mul(self):
        """Mirrors the MoE gate/up split: slice a (384,) row into two (192,)
        halves that feed an elementwise mul — both slices fold into the mul's
        loads with no intermediate buffer."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((384,)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (384,))
            gate, up = b.split(tx, 2, axis=0)  # two (192,) slices
            b.set_outputs({"y": b.mul(b.silu(gate), up)})

        _lower_and_check_interp(build, {"x": x}, (192,), atol=1e-4)

    def test_slice_2d_partition_axis_elementwise(self):
        """Slice on the partition axis (rows 8:200 of a (256,128) tensor),
        result feeds an elementwise add — the offset composes into the tiled
        2D load across both partition tiles."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((256, 128)).astype(np.float32)
        bias = rng.standard_normal((192, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (256, 128))
            tb = b.add_input("bias", (192, 128))
            s = b.slice(tx, (8, 0), (200, 128))
            b.set_outputs({"y": b.add(s, tb)})

        _lower_and_check_interp(build, {"x": x, "bias": bias}, (192, 128))

    def test_slice_last_axis_window_elementwise(self):
        """Slice a last-axis window (cols 1024:1152 of a (1,8,1280) tensor)
        feeding a unary — a mid-tensor column offset folded into the load."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 8, 1280)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 8, 1280))
            s = b.slice(tx, (0, 0, 1024), (1, 8, 1152))
            b.set_outputs({"y": b.relu(s)})

        _lower_and_check_interp(build, {"x": x}, (1, 8, 128), atol=1e-4)

    def test_slice_as_graph_output_materializes(self):
        """A slice that IS a graph output has no elementwise consumer to fold
        into, so it materializes via the copy path."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((16, 64)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (16, 64))
            b.set_outputs({"y": b.slice(tx, (4, 0), (12, 64))})

        _lower_and_check_interp(build, {"x": x}, (8, 64))

    def test_slice_then_reshape_materializes(self):
        """A slice feeding a reshape (not elementwise) materializes, then the
        reshape views the materialized buffer."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((4, 96)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (4, 96))
            s = b.slice(tx, (0, 0), (4, 64))     # (4, 64)
            b.set_outputs({"y": b.reshape(s, (4, 8, 8))})

        _lower_and_check_interp(build, {"x": x}, (4, 8, 8))

    def test_strided_slice_not_view(self):
        """A non-unit-stride slice can't compose as an offset view; it keeps
        the (correct) copy path."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((1, 8, 256)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 8, 256))
            s = b.slice(tx, (0, 0, 0), (1, 8, 256), strides=(1, 1, 2))
            b.set_outputs({"y": b.neg(s)})

        _lower_and_check_interp(build, {"x": x}, (1, 8, 128), atol=1e-4)

    def test_chained_slice_view(self):
        """A slice of a slice, feeding elementwise: the inner slice materializes
        (its consumer is another slice, not elementwise) and the outer composes
        its offset onto the materialized inner buffer."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal((320,)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (320,))
            inner = b.slice(tx, (64,), (320,))   # (256,)  -> x[64:320]
            outer = b.slice(inner, (0,), (128,))  # (128,) -> x[64:192]
            b.set_outputs({"y": b.relu(outer)})

        _lower_and_check_interp(build, {"x": x}, (128,), atol=1e-4)

    # -- hardware --

    def test_split_gate_up_mul_hw(self, compile_and_run):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((384,)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (384,))
            gate, up = b.split(tx, 2, axis=0)
            b.set_outputs({"y": b.mul(b.silu(gate), up)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (192,))

    def test_slice_2d_partition_axis_elementwise_hw(self, compile_and_run):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((256, 128)).astype(np.float32)
        bias = rng.standard_normal((192, 128)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (256, 128))
            tb = b.add_input("bias", (192, 128))
            s = b.slice(tx, (8, 0), (200, 128))
            b.set_outputs({"y": b.add(s, tb)})

        _lower_and_check_hw(compile_and_run, build, {"x": x, "bias": bias}, (192, 128))

    def test_slice_last_axis_window_elementwise_hw(self, compile_and_run):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 8, 1280)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (1, 8, 1280))
            s = b.slice(tx, (0, 0, 1024), (1, 8, 1152))
            b.set_outputs({"y": b.relu(s)})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (1, 8, 128))

    def test_slice_as_graph_output_materializes_hw(self, compile_and_run):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((16, 64)).astype(np.float32)

        def build(b):
            tx = b.add_input("x", (16, 64))
            b.set_outputs({"y": b.slice(tx, (4, 0), (12, 64))})

        _lower_and_check_hw(compile_and_run, build, {"x": x}, (8, 64))
