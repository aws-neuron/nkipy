"""
Tests for the prepare-arithmetic pass.

This pass converts division operations into multiplication by reciprocal
because NISA's tensor_tensor_arith doesn't support DIVIDE directly.

Patterns tested:
  - linalg.div(A, B) -> linalg.mul(A, linalg.reciprocal(B))
  - linalg.generic with scalar divf -> mulf with reciprocal constant
  - linalg.generic with broadcast divf(block_arg, block_arg)
      -> linalg.reciprocal(rhs) + linalg.generic with mulf

Run with: python -m pytest tests/passes/prepare_arithmetic/test_basic.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Named linalg.div (same-shape tensor / tensor)
# ============================================================================

def test_tensor_div_tensor_same_shape():
    """
    Same-shape division: linalg.div -> linalg.mul + linalg.reciprocal.

    After prepare-arithmetic, arith.divf should be gone, replaced by
    linalg.reciprocal and linalg.mul.
    """
    shape = (128, 256)
    tile_size = [64, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def kernel(a, b):
        result = np.divide(a, b)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.random.randn(*shape).astype(np.float32)
    B = (np.abs(np.random.randn(*shape)) + 0.5).astype(np.float32)

    # After prepare-arithmetic: div replaced by reciprocal + mul
    check_patterns = """
    CHECK: func.func
    CHECK: linalg.reciprocal
    CHECK: linalg.mul
    CHECK-NOT: linalg.div
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        inputs=[A, B],
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Scalar division: tensor / scalar constant
# ============================================================================

def test_tensor_div_scalar():
    """
    Tensor / scalar: linalg.generic { divf(%arg, %cst) }
      -> linalg.generic { mulf(%arg, 1/cst) }.

    The divf in the body is replaced with mulf using the reciprocal constant.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x / 2.0
        knob.knob(result, tile_size=tile_size)
        return result

    # After prepare-arithmetic: divf replaced by mulf in body
    check_patterns = """
    CHECK: func.func
    CHECK: linalg.generic
    CHECK: arith.mulf
    CHECK-NOT: arith.divf
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Scalar / tensor (reciprocal pattern)
# ============================================================================

def test_scalar_div_tensor():
    """
    Scalar / tensor: linalg.generic { divf(%cst, %arg) }
      -> linalg.reciprocal(input).

    The entire generic is replaced with a reciprocal op.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = 1.0 / x
        knob.knob(result, tile_size=tile_size)
        return result

    # After prepare-arithmetic: replaced by linalg.reciprocal
    check_patterns = """
    CHECK: func.func
    CHECK: linalg.reciprocal
    CHECK-NOT: arith.divf
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Broadcast division: tensor<MxN> / tensor<Mx1>
# ============================================================================

def test_broadcast_div_column():
    """
    Broadcast column division: tensor<256x256> / tensor<256x1>.

    The tracer emits linalg.generic with broadcast indexing maps and
    arith.divf between block args. After prepare-arithmetic, the divf
    should be replaced by linalg.reciprocal on the rhs + arith.mulf
    in the body.
    """
    shape_a = (256, 256)
    shape_b = (256, 1)
    tile_size = [128, 128]

    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        result = np.divide(a, b)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.random.randn(*shape_a).astype(np.float32)
    B = (np.abs(np.random.randn(*shape_b)) + 0.5).astype(np.float32)

    # After prepare-arithmetic: reciprocal of rhs, mulf in generic body
    check_patterns = """
    CHECK: func.func
    CHECK: linalg.reciprocal
    CHECK: linalg.generic
    CHECK: arith.mulf
    CHECK-NOT: arith.divf
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        inputs=[A, B],
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_broadcast_div_row():
    """
    Broadcast row division: tensor<128x256> / tensor<1x256>.
    """
    shape_a = (128, 256)
    shape_b = (1, 256)
    tile_size = [64, 128]

    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(a, b):
        result = np.divide(a, b)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.random.randn(*shape_a).astype(np.float32)
    B = (np.abs(np.random.randn(*shape_b)) + 0.5).astype(np.float32)

    check_patterns = """
    CHECK: func.func
    CHECK: linalg.reciprocal
    CHECK: linalg.generic
    CHECK: arith.mulf
    CHECK-NOT: arith.divf
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        inputs=[A, B],
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_broadcast_div_rmsnorm_pattern():
    """
    RMSNorm-like pattern: tensor<128x256> / tensor<128x1>.

    This is the most common real-world use case for broadcast division,
    where each row is divided by its RMS norm value.
    """
    shape_a = (128, 256)
    shape_b = (128, 1)
    tile_size = [64, 128]

    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(values, rms):
        result = np.divide(values, rms)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.random.randn(*shape_a).astype(np.float32)
    B = (np.abs(np.random.randn(*shape_b)) + 0.5).astype(np.float32)

    check_patterns = """
    CHECK: func.func
    CHECK: linalg.reciprocal
    CHECK: linalg.generic
    CHECK: arith.mulf
    CHECK-NOT: arith.divf
    CHECK: return
    """
    run_kernel_test(
        kernel,
        stop_after='prepare-arithmetic',
        check_patterns=check_patterns,
        inputs=[A, B],
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
