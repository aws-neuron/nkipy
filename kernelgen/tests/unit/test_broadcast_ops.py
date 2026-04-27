"""
Tests for broadcasting operations.

These tests verify that NumPy-style broadcasting works correctly with various
shape combinations. Broadcasting follows NumPy rules:
1. Align shapes from the right (trailing dimensions)
2. Dimensions with size 1 can broadcast to any size
3. Missing dimensions are treated as size 1

All broadcast ops emit linalg.generic (not named linalg ops).
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import nkipy_kernelgen_test, run_kernel_test, Mode


# ============================================================================
# Shape expansion (one operand has size-1 dims)
# ============================================================================

@pytest.mark.parametrize("op,shape_a,shape_b", [
    (np.add, (128, 256), (128, 1)),             # column broadcast
    (np.multiply, (128, 256), (1, 1)),           # scalar-shaped broadcast
    (np.subtract, (64, 128, 256), (64, 1, 256)), # middle dim broadcast
])
def test_broadcast_expand(op, shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        return op(a, b)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Dimension addition (fewer dims in one operand)
# ============================================================================

@pytest.mark.parametrize("op,shape_a,shape_b", [
    (np.add, (128, 256), (256,)),              # 1D to 2D
    (np.multiply, (64, 128, 256), (256,)),     # 1D to 3D
])
def test_broadcast_add_dims(op, shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        return op(a, b)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Divide with broadcasting (custom inputs to avoid div-by-zero)
# ============================================================================

@pytest.mark.parametrize("shape_a,shape_b", [
    ((128, 256), (1, 256)),             # row broadcast
    ((64, 128, 256), (128, 256)),       # 2D to 3D
    ((64, 128, 256), (128, 1)),         # complex 3D
])
def test_broadcast_divide(shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        return np.divide(a, b)

    np.random.seed(42)
    A = np.random.randn(*shape_a).astype(np.float32)
    B = np.random.randn(*shape_b).astype(np.float32) + 1.0

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        inputs=[A, B],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Both operands broadcast
# ============================================================================

@pytest.mark.parametrize("op,shape_a,shape_b", [
    (np.multiply, (1, 256), (128, 1)),
    (np.add, (256,), (128, 1)),
])
def test_broadcast_both_operands(op, shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        return op(a, b)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# 1D broadcast with expansion
# ============================================================================

@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32"), ((1,), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_broadcast_1d_to_2d_with_expansion(A, B):
    """Broadcasting (1,) to (128, 256) - add dimension AND expand."""
    return np.add(A, B)


# ============================================================================
# Float16 broadcast
# ============================================================================

@pytest.mark.parametrize("op,shape_a,shape_b", [
    (np.add, (128, 256), (128, 1)),
    (np.multiply, (128, 256), (256,)),
])
def test_broadcast_f16(op, shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f16"), (shape_b, "f16")])
    def kernel(a, b):
        return op(a, b)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        rtol=0.01, atol=0.01,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Incompatible shapes (error tests)
# ============================================================================

def test_incompatible_shapes_no_size_1():
    """(128, 256) and (128,) are incompatible (no size-1 to expand)."""
    def add_func(A, B):
        return np.add(A, B)

    traced = trace(input_specs=[((128, 256), "f32"), ((128,), "f32")])(add_func)

    try:
        mlir_module = traced.to_mlir()
        A = np.random.randn(128, 256).astype(np.float32)
        B = np.random.randn(128).astype(np.float32)
        try:
            add_func(A, B)
            assert True
        except ValueError:
            assert False, "Tracer should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Incompatible" in str(e) or "broadcast" in str(e).lower()


def test_incompatible_shapes_mismatch():
    """(128, 256) and (64, 256) are incompatible."""
    def mul_func(A, B):
        return np.multiply(A, B)

    traced = trace(input_specs=[((128, 256), "f32"), ((64, 256), "f32")])(mul_func)

    try:
        mlir_module = traced.to_mlir()
        A = np.random.randn(128, 256).astype(np.float32)
        B = np.random.randn(64, 256).astype(np.float32)
        try:
            mul_func(A, B)
            assert True
        except ValueError:
            assert False, "Tracer should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Incompatible" in str(e) or "broadcast" in str(e).lower()


# ============================================================================
# Real-world patterns
# ============================================================================

def test_rmsnorm_pattern():
    """RMS normalization pattern: (M, N) / (M, 1)."""
    @trace(input_specs=[((128, 256), "f32"), ((128, 1), "f32")])
    def kernel(values, rms):
        return np.divide(values, rms)

    np.random.seed(42)
    values = np.random.randn(128, 256).astype(np.float32)
    rms = np.random.randn(128, 1).astype(np.float32) + 1.0

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        inputs=[values, rms],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


@nkipy_kernelgen_test(
    input_specs=[((2, 64, 128, 128), "f32"), ((1, 64, 1, 1), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_batch_normalization_pattern(A, bias):
    """Batch normalization pattern: (B, C, H, W) with (1, C, 1, 1)."""
    return np.add(A, bias)


@nkipy_kernelgen_test(
    input_specs=[((2, 4, 128, 128), "f32"), ((1, 1, 1, 1), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_attention_scale_pattern(qk, scale):
    """Attention scaling pattern: (B, H, S, S) * scalar."""
    return np.multiply(qk, scale)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
