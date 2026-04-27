"""
Tests for knob-driven-tiling pass with elementwise operations.

These tests verify simple N-dimensional tiling for elementwise ops like add, sub, etc.
Run with: python -m pytest tests/passes/knob_driven_tiling/test_elementwise.py -v
Or directly: python tests/passes/knob_driven_tiling/test_elementwise.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Configurations
# ============================================================================

# (shape, tile_size, test_id)
ADD_TEST_CONFIGS = [
    pytest.param((256, 256), [128, 128], id="add_256x256_tile128"),
    pytest.param((512, 512), [128, 128], id="add_512x512_tile128"),
    pytest.param((256, 256), [64, 64], id="add_256x256_tile64"),
    pytest.param((128, 256, 64), [64, 128, 32], id="add_3d_128x256x64"),
]


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.parametrize("shape,tile_size", ADD_TEST_CONFIGS)
def test_add_tiling(shape, tile_size):
    """
    Test knob-driven-tiling on linalg.add.

    Pattern: result = A + B

    Elementwise ops use simple single-level tiling (no blocking).
    For 2D: for i in [0, dim0, tile0): for j in [0, dim1, tile1): add_tile
    """
    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob.knob(result, tile_size=tile_size)
        return result

    # Build FileCheck patterns based on dimensionality
    # Elementwise ops have one scf.for loop per dimension
    # With SBUF promotion, we expect:
    # 1. scf.for loops for tiling
    # 2. tensor.extract_slice for each input/output
    # 3. bufferization.alloc_tensor for SBUF promotion
    # 4. linalg.add on SBUF tensors
    # 5. tensor.insert_slice to write back
    #
    # FileCheck regex uses {{.*}} - in f-strings, {{{{.*}}}} produces {{.*}}
    check_patterns = "CHECK: func.func\n"
    for i, (dim, tile) in enumerate(zip(shape, tile_size)):
        check_patterns += f"    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim}{{{{.*}}}} step %c{tile}\n"

    # Tiled tensor shapes - IR uses [dim1, dim2] with comma separator
    tile_shape_comma = ", ".join(str(t) for t in tile_size)
    tile_shape_x = "x".join(str(t) for t in tile_size)

    # With SBUF promotion, we expect alloc_tensor ops with memory_space
    check_patterns += f"    CHECK: tensor.extract_slice {{{{.*}}}} [{tile_shape_comma}]\n"
    check_patterns += f"    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32\n"
    check_patterns += f"    CHECK: tensor.extract_slice {{{{.*}}}} [{tile_shape_comma}]\n"
    check_patterns += f"    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32\n"
    check_patterns += f"    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32\n"
    check_patterns += f"    CHECK: linalg.add {{{{.*}}}} tensor<{tile_shape_x}xf32\n"
    check_patterns += f"    CHECK: tensor.insert_slice\n"
    check_patterns += f"    CHECK: scf.yield\n"

    run_kernel_test(
        add_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_sub_2d():
    """
    Test knob-driven-tiling on linalg.sub.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def sub_kernel(a, b):
        result = a - b
        knob.knob(result, tile_size=tile_size)
        return result

    run_kernel_test(
        sub_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


def test_mul_2d():
    """
    Test knob-driven-tiling on linalg.mul (element-wise multiplication).
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def mul_kernel(a, b):
        result = a * b
        knob.knob(result, tile_size=tile_size)
        return result

    run_kernel_test(
        mul_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


def test_add_simple():
    """
    Simple test for 256x256 add to verify basic elementwise functionality.

    For 256x256 with tile_size=[128, 128]:
      for i in [0, 256, 128): for j in [0, 256, 128): add([128, 128])
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob.knob(result, tile_size=tile_size)
        return result

    # FileCheck patterns for 2D elementwise tiling with SBUF promotion
    # After promotion, we expect:
    # 1. extract_slice for input tiles
    # 2. alloc_tensor with 3 : i32 for SBUF copies
    # 3. linalg.add on promoted tensors
    # 4. insert_slice to write back
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.add {{{{.*}}}} tensor<{tile0}x{tile1}xf32
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        add_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Scalar Constant Tests
# ============================================================================
# When one operand is a scalar constant, the tracer generates a linalg.generic
# with the arith.constant embedded in the region body. This has 1 DPS input
# (the tensor) and 1 DPS init (the output), so tiling promotes 1 input + 1
# output to SBUF (2 alloc_tensors, not 3 like binary tensor-tensor ops).


def test_tensor_add_scalar():
    """
    Test tiling of tensor + scalar constant.

    Pattern: result = x + 2.0
    Generated IR: linalg.generic with arith.addf and embedded arith.constant.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x + 2.0
        knob.knob(result, tile_size=tile_size)
        return result

    # linalg.generic with 1 input: 1 extract_slice + 2 alloc_tensors (1 input + 1 output)
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.generic
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_scalar_minus_tensor():
    """
    Test tiling of scalar - tensor (non-commutative, scalar on LHS).

    Pattern: result = 5.0 - x
    Generated IR: linalg.generic with arith.subf, scalar_is_lhs=True.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = 5.0 - x
        knob.knob(result, tile_size=tile_size)
        return result

    # 1 DPS input -> 1 extract_slice, 2 alloc_tensors (1 input + 1 output)
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.generic
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_tensor_mul_scalar():
    """
    Test tiling of tensor * scalar constant.

    Pattern: result = x * 3.0
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x * 3.0
        knob.knob(result, tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.generic
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_tensor_div_scalar():
    """
    Test tiling of tensor / scalar constant.

    Pattern: result = x / 2.0
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x / 2.0
        knob.knob(result, tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.generic
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_scalar_div_tensor():
    """
    Test tiling of scalar / tensor (non-commutative, scalar on LHS).

    Pattern: result = 1.0 / x  (reciprocal-like pattern used in sigmoid)
    prepare-arithmetic converts this to linalg.reciprocal.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = 1.0 / x
        knob.knob(result, tile_size=tile_size)
        return result

    # prepare-arithmetic converts 1.0/x into linalg.reciprocal
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: tensor.extract_slice {{{{.*}}}} [{tile0}, {tile1}]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.reciprocal
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
