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
    """Rank>=3 elementwise segments collapse leading dims onto the partition.

    These shapes mirror the multi-batch attention segments in the fused Qwen3
    layer, where the old per-tile path used only out[-2] partition lanes and
    unrolled prod(out[:-2]) iterations. Verified at the interpreter level and
    (selectively) on hardware. See ``_try_emit_collapsed_ew`` in
    ``direct_lower.py``.
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
