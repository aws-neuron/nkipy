"""Tests for ops required by nkipy HLO parity that were previously missing.

These cover: floor, ceil, abs, sign, power, floor_divide, mod,
cast to f16/bf16, and 3D transpose — the operations that caused failures
when running nkipy's HLO test suite with the nkigen-lite backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir import Builder, TensorType, run


# ===========================
# Unary: floor, ceil, abs, sign
# ===========================


class TestFloorCeil:
    def test_floor_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        b.set_outputs({"y": b.floor(x)})

        inp = np.array([[-1.7, 2.3, 0.0, -0.5]] * 4, dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.floor(inp))

    def test_ceil_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        b.set_outputs({"y": b.ceil(x)})

        inp = np.array([[-1.7, 2.3, 0.0, -0.5]] * 4, dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.ceil(inp))

    def test_floor_negative_fractions(self):
        b = Builder("t")
        x = b.add_input("x", (8,), DType.F32)
        b.set_outputs({"y": b.floor(x)})

        inp = np.array([-2.9, -1.1, -0.1, 0.0, 0.1, 1.1, 2.9, 3.0], dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.floor(inp))

    def test_ceil_negative_fractions(self):
        b = Builder("t")
        x = b.add_input("x", (8,), DType.F32)
        b.set_outputs({"y": b.ceil(x)})

        inp = np.array([-2.9, -1.1, -0.1, 0.0, 0.1, 1.1, 2.9, 3.0], dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.ceil(inp))

    def test_floor_2d(self):
        rng = np.random.default_rng(42)
        b = Builder("t")
        x = b.add_input("x", (32, 64), DType.F32)
        b.set_outputs({"y": b.floor(x)})

        inp = rng.uniform(-10.0, 10.0, (32, 64)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.floor(inp))


class TestAbsSign:
    def test_abs_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.abs(x)})

        inp = np.array([-3.0, -1.0, 0.0, 2.5], dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.abs(inp))

    def test_sign_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.sign(x)})

        inp = np.array([-3.0, -0.0, 0.0, 2.5], dtype=np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.sign(inp))

    def test_abs_2d(self):
        rng = np.random.default_rng(42)
        b = Builder("t")
        x = b.add_input("x", (64, 128), DType.F32)
        b.set_outputs({"y": b.abs(x)})

        inp = rng.standard_normal((64, 128)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_allclose(result["y"], np.abs(inp))

    def test_sign_2d(self):
        rng = np.random.default_rng(7)
        b = Builder("t")
        x = b.add_input("x", (64, 128), DType.F32)
        b.set_outputs({"y": b.sign(x)})

        inp = rng.standard_normal((64, 128)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.sign(inp))


# ===========================
# Binary: power, floor_divide, mod
# ===========================


class TestPower:
    def test_power_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        b.set_outputs({"z": b.power(x, y)})

        x_np = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y_np = np.array([3.0, 2.0, 0.5, 1.0], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["z"], np.power(x_np, y_np), rtol=1e-5)

    def test_power_broadcast(self):
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        y = b.add_input("y", (1, 4), DType.F32)
        # Broadcast y to match x
        y_bc = b.broadcast_to(y, (4, 4))
        b.set_outputs({"z": b.power(x, y_bc)})

        rng = np.random.default_rng(42)
        x_np = rng.uniform(0.1, 3.0, (4, 4)).astype(np.float32)
        y_np = rng.uniform(0.5, 2.0, (1, 4)).astype(np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(
            result["z"], np.power(x_np, np.broadcast_to(y_np, (4, 4))), rtol=1e-5
        )

    def test_power_square(self):
        b = Builder("t")
        x = b.add_input("x", (128, 128), DType.F32)
        two = b.constant(2.0, (128, 128), DType.F32)
        b.set_outputs({"z": b.power(x, two)})

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((128, 128)).astype(np.float32)
        result = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(result["z"], x_np ** 2, rtol=1e-5)


class TestFloorDivide:
    def test_floor_divide_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        b.set_outputs({"z": b.floor_divide(x, y)})

        x_np = np.array([7.0, 10.0, -7.0, -10.0], dtype=np.float32)
        y_np = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_array_equal(result["z"], np.floor_divide(x_np, y_np))

    def test_floor_divide_2d(self):
        rng = np.random.default_rng(42)
        b = Builder("t")
        x = b.add_input("x", (32, 32), DType.F32)
        y = b.add_input("y", (32, 32), DType.F32)
        b.set_outputs({"z": b.floor_divide(x, y)})

        x_np = rng.uniform(-10.0, 10.0, (32, 32)).astype(np.float32)
        y_np = rng.uniform(0.5, 5.0, (32, 32)).astype(np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_array_equal(result["z"], np.floor_divide(x_np, y_np))

    def test_floor_divide_negative(self):
        b = Builder("t")
        x = b.add_input("x", (6,), DType.F32)
        y = b.add_input("y", (6,), DType.F32)
        b.set_outputs({"z": b.floor_divide(x, y)})

        x_np = np.array([7.0, -7.0, 7.0, -7.0, 0.0, 1.0], dtype=np.float32)
        y_np = np.array([3.0, 3.0, -3.0, -3.0, 3.0, 3.0], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_array_equal(result["z"], np.floor_divide(x_np, y_np))


class TestMod:
    def test_mod_basic(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        b.set_outputs({"z": b.mod(x, y)})

        x_np = np.array([7.0, 10.0, 5.5, 3.0], dtype=np.float32)
        y_np = np.array([3.0, 3.0, 2.0, 5.0], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["z"], np.mod(x_np, y_np), rtol=1e-5)

    def test_mod_negative(self):
        b = Builder("t")
        x = b.add_input("x", (6,), DType.F32)
        y = b.add_input("y", (6,), DType.F32)
        b.set_outputs({"z": b.mod(x, y)})

        x_np = np.array([7.0, -7.0, 7.0, -7.0, 0.0, 1.5], dtype=np.float32)
        y_np = np.array([3.0, 3.0, -3.0, -3.0, 3.0, 3.0], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["z"], np.mod(x_np, y_np), rtol=1e-5)

    def test_mod_2d(self):
        rng = np.random.default_rng(42)
        b = Builder("t")
        x = b.add_input("x", (32, 32), DType.F32)
        y = b.add_input("y", (32, 32), DType.F32)
        b.set_outputs({"z": b.mod(x, y)})

        x_np = rng.uniform(-10.0, 10.0, (32, 32)).astype(np.float32)
        y_np = rng.uniform(0.5, 5.0, (32, 32)).astype(np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["z"], np.mod(x_np, y_np), rtol=1e-5)


# ===========================
# Cast: f32 -> f16, f16 -> f32, f32 -> bf16
# ===========================


class TestCast:
    def test_cast_f32_to_f16(self):
        b = Builder("t")
        x = b.add_input("x", (32, 32), DType.F32)
        b.set_outputs({"y": b.cast(x, DType.F16)})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((32, 32)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        assert result["y"].dtype == np.float16
        np.testing.assert_allclose(result["y"], inp.astype(np.float16), rtol=0)

    def test_cast_f16_to_f32(self):
        b = Builder("t")
        x = b.add_input("x", (32, 32), DType.F16)
        b.set_outputs({"y": b.cast(x, DType.F32)})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((32, 32)).astype(np.float16)
        result = run(b.graph, {"x": inp})
        assert result["y"].dtype == np.float32
        np.testing.assert_allclose(result["y"], inp.astype(np.float32), rtol=0)

    def test_cast_f32_to_bf16(self):
        import ml_dtypes
        b = Builder("t")
        x = b.add_input("x", (16, 16), DType.F32)
        b.set_outputs({"y": b.cast(x, DType.BF16)})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((16, 16)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        assert result["y"].dtype == ml_dtypes.bfloat16

    def test_cast_chain(self):
        """f32 -> f16 -> f32 roundtrip."""
        b = Builder("t")
        x = b.add_input("x", (64, 64), DType.F32)
        y_f16 = b.cast(x, DType.F16)
        y_f32 = b.cast(y_f16, DType.F32)
        b.set_outputs({"y": y_f32})

        rng = np.random.default_rng(42)
        inp = rng.uniform(-1.0, 1.0, (64, 64)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        expected = inp.astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(result["y"], expected, rtol=0)


# ===========================
# Transpose: 3D and higher rank
# ===========================


class TestTranspose:
    def test_transpose_3d(self):
        b = Builder("t")
        x = b.add_input("x", (2, 3, 4), DType.F32)
        b.set_outputs({"y": b.transpose(x, (2, 0, 1))})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((2, 3, 4)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.transpose(inp, (2, 0, 1)))
        assert result["y"].shape == (4, 2, 3)

    def test_transpose_3d_identity(self):
        b = Builder("t")
        x = b.add_input("x", (2, 3, 4), DType.F32)
        b.set_outputs({"y": b.transpose(x, (0, 1, 2))})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((2, 3, 4)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], inp)

    def test_transpose_4d(self):
        b = Builder("t")
        x = b.add_input("x", (2, 3, 4, 5), DType.F32)
        b.set_outputs({"y": b.transpose(x, (3, 1, 2, 0))})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], np.transpose(inp, (3, 1, 2, 0)))
        assert result["y"].shape == (5, 3, 4, 2)

    def test_transpose_2d_swap(self):
        b = Builder("t")
        x = b.add_input("x", (64, 128), DType.F32)
        b.set_outputs({"y": b.transpose(x, (1, 0))})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((64, 128)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_array_equal(result["y"], inp.T)


# ===========================
# Composed ops: floor_divide via floor+div, mod via sub+mul+floor_divide
# ===========================


class TestComposedOps:
    def test_floor_divide_as_floor_of_div(self):
        """floor_divide(a, b) == floor(a / b) for positive values."""
        b = Builder("t")
        x = b.add_input("x", (8,), DType.F32)
        y = b.add_input("y", (8,), DType.F32)
        q = b.floor(b.div(x, y))
        b.set_outputs({"q": q})

        x_np = np.array([7.0, 10.0, 5.5, 3.0, 15.0, 1.0, 100.0, 0.5], dtype=np.float32)
        y_np = np.array([3.0, 3.0, 2.0, 5.0, 4.0, 1.0, 7.0, 0.3], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["q"], np.floor_divide(x_np, y_np), rtol=1e-5)

    def test_mod_as_sub_mul_floor_divide(self):
        """mod(a, b) == a - b * floor_divide(a, b)."""
        bld = Builder("t")
        x = bld.add_input("x", (6,), DType.F32)
        y = bld.add_input("y", (6,), DType.F32)
        q = bld.floor_divide(x, y)
        r = bld.sub(x, bld.mul(y, q))
        bld.set_outputs({"r": r})

        x_np = np.array([7.0, 10.0, 5.5, -7.0, -10.0, 0.0], dtype=np.float32)
        y_np = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32)
        result = run(bld.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(result["r"], np.mod(x_np, y_np), rtol=1e-5)

    def test_reshape_then_transpose_then_cast(self):
        """Combined operation: reshape -> transpose -> cast (f32 -> f16)."""
        b = Builder("t")
        x = b.add_input("x", (256, 256), DType.F32)
        reshaped = b.reshape(x, (64, 1024))
        transposed = b.transpose(reshaped, (1, 0))
        casted = b.cast(transposed, DType.F16)
        b.set_outputs({"y": casted})

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((256, 256)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        expected = inp.reshape(64, 1024).T.astype(np.float16)
        assert result["y"].shape == (1024, 64)
        assert result["y"].dtype == np.float16
        np.testing.assert_allclose(result["y"], expected, rtol=0)


# ===========================
# Broadcasting edge cases
# ===========================


class TestBroadcasting:
    def test_scalar_broadcast_binary(self):
        """Binary op with scalar constant broadcast to tensor."""
        b = Builder("t")
        x = b.add_input("x", (4, 8), DType.F32)
        two = b.constant(2.0, (1, 1), DType.F32)
        two_bc = b.broadcast_to(two, (4, 8))
        b.set_outputs({"y": b.power(x, two_bc)})

        rng = np.random.default_rng(42)
        inp = rng.uniform(0.1, 3.0, (4, 8)).astype(np.float32)
        result = run(b.graph, {"x": inp})
        np.testing.assert_allclose(result["y"], inp ** 2, rtol=1e-5)

    def test_row_broadcast_floor_divide(self):
        """floor_divide with row vector broadcast."""
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        y = b.add_input("y", (1, 4), DType.F32)
        y_bc = b.broadcast_to(y, (4, 4))
        b.set_outputs({"z": b.floor_divide(x, y_bc)})

        x_np = np.array([[7, 10, 5, 3]] * 4, dtype=np.float32)
        y_np = np.array([[3, 4, 2, 5]], dtype=np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_array_equal(
            result["z"], np.floor_divide(x_np, np.broadcast_to(y_np, (4, 4)))
        )

    def test_col_broadcast_mod(self):
        """mod with column vector broadcast."""
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        y = b.add_input("y", (4, 1), DType.F32)
        y_bc = b.broadcast_to(y, (4, 4))
        b.set_outputs({"z": b.mod(x, y_bc)})

        rng = np.random.default_rng(42)
        x_np = rng.uniform(1.0, 10.0, (4, 4)).astype(np.float32)
        y_np = rng.uniform(1.0, 4.0, (4, 1)).astype(np.float32)
        result = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(
            result["z"], np.mod(x_np, np.broadcast_to(y_np, (4, 4))), rtol=1e-5
        )


# ===========================
# Error handling
# ===========================


class TestErrors:
    def test_power_shape_mismatch(self):
        b = Builder("t")
        x = b.add_input("x", (4, 4), DType.F32)
        y = b.add_input("y", (3, 4), DType.F32)
        with pytest.raises(ValueError, match="not broadcastable"):
            b.power(x, y)

    def test_floor_divide_dtype_mismatch(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F16)
        with pytest.raises(ValueError, match="dtype mismatch"):
            b.floor_divide(x, y)

    def test_mod_dtype_mismatch(self):
        b = Builder("t")
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.I32)
        with pytest.raises(ValueError, match="dtype mismatch"):
            b.mod(x, y)
