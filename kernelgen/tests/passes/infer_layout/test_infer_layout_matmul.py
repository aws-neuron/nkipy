"""
Tests for the infer-layout pass: matmul seeding and propagation.

Verifies that infer-layout correctly seeds matmul operands with the
hardware-specific layout rules:
  - Result C [M,N]:  partition_dim=0 (M), tile=[min(M,128), min(N,512)]
  - Operand A [M,K]: partition_dim=1 (K), tile=[min(M,128), min(K,128)]
  - Operand B [K,N]: partition_dim=0 (K), tile=[min(K,128), min(N,512)]

Also tests forward propagation from matmul operands to downstream
elementwise consumers.

Run with: pytest tests/passes/infer_layout/test_infer_layout_matmul.py -v
"""

import numpy as np
import pytest

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: Matmul seeding produces correct operand layouts
# ============================================================================

def test_matmul_seed_result_layout():
    """
    A matmul with a user knob on its result should keep that annotation.
    Operands that are produced by linalg ops should get matmul-specific
    layouts via backward propagation.

    matmul(exp(a), b) with knob on result:
      - result C: user annotation preserved
      - operand A (exp result): partition_dim=1, tile=[128, 128]
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        a_exp = np.exp(a)
        mm = np.matmul(a_exp, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=[128, 128], reduction_tile=[128])
        return mm

    # exp(a) should get matmul operand A layout: partition_dim=1
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "partition_dim = 1",         # operand A gets partition_dim=1
            "tile_size = array<i64: 128, 128>",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_matmul_no_annotation_auto_seeds():
    """
    A matmul with NO user annotation should get auto-seeded defaults.

    matmul(a, b) -> result [M, N]:
      tile_size=[min(M,128), min(N,512)], partition_dim=0, reduction_tile=[min(K,128)]
    KnobDrivenTiling dynamically adjusts blocking based on dim/tile ratio.
    """
    M, N, K = 256, 512, 128

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        return np.matmul(a, b)

    # Result C: tile=[min(256,128), min(512,512)] = [128, 512]
    # reduction_tile=[min(128,128)] = [128]
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 512>",
            "reduction_tile = array<i64: 128>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_matmul_auto_seed_large_dims():
    """
    Verify matmul auto-seeding respects hardware limits for large dims.

    matmul [1024, 256] x [256, 2048] -> [1024, 2048]
      Result: tile=[min(1024/2,128), min(2048/2,512)] = [128, 512]
      reduction_tile=[min(256,128)] = [128]

    TODO: Once KnobDrivenTiling supports non-blocked matmul for small dims,
    the BF=2 divisor can be removed.
    """
    M, N, K = 1024, 2048, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        return np.matmul(a, b)

    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 512>",
            "reduction_tile = array<i64: 128>",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: Matmul + elementwise chain propagation
# ============================================================================

def test_matmul_forward_propagates_to_elementwise():
    """
    After matmul seeding, forward propagation should reach downstream
    elementwise ops.

    matmul(a, b) -> exp -> result
    Only matmul result is seeded. exp should get layout via forward
    propagation from the matmul result.
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        mm = np.matmul(a, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=[128, 128], reduction_tile=[128])
        return np.exp(mm)

    # exp should get an annotation via forward propagation
    check_patterns = """
    CHECK: linalg.matmul
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_matmul_elementwise_chain_executes():
    """
    Verify that matmul -> elementwise chain with auto-seeded layouts
    tiles and executes correctly.
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        mm = np.matmul(a, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=[128, 128], reduction_tile=[128])
        y = np.exp(mm)
        knob.knob(y, mem_space="SharedHbm", tile_size=[128, 128])
        return y

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )


# ============================================================================
# Test: Matmul operand A backward propagation through elementwise chain
# ============================================================================

def test_matmul_operand_backward_chain():
    """
    Matmul operand A is produced by an elementwise chain:
      x -> exp -> square -> matmul(square, b)

    The matmul seeding should set operand A (square) to partition_dim=1.
    Backward propagation from square should reach exp with partition_dim=1.
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(x, b):
        y = np.exp(x)
        z = np.square(y)
        mm = np.matmul(z, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=[128, 128], reduction_tile=[128])
        return mm

    # Both exp and square should get partition_dim=1 (matmul operand A layout)
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    CHECK: linalg.square
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    CHECK: linalg.matmul
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test: Compatible tile sizes across producer-consumer boundary
# ============================================================================

def test_compatible_tile_sizes_no_conflict():
    """
    Two ops with different but compatible tile sizes (one divides the other)
    should NOT trigger a conflict error.

    add1 tile=[128,128], add2 tile=[64,64]: 64 divides 128 -> compatible.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32"), (shape, "f32"), (shape, "f32")])
    def kernel(a, b, c):
        x = a + b
        knob.knob(x, mem_space="Sbuf", tile_size=[128, 128])
        y = x + c
        knob.knob(y, mem_space="SharedHbm", tile_size=[64, 64])
        return y

    # Should compile without conflict
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 128>",
            "tile_size = array<i64: 64, 64>",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_compatible_tile_sizes_executes():
    """
    Verify that compatible but different tile sizes tile and execute correctly.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32"), (shape, "f32"), (shape, "f32")])
    def kernel(a, b, c):
        x = a + b
        knob.knob(x, mem_space="Sbuf", tile_size=[128, 128])
        y = x + c
        knob.knob(y, mem_space="SharedHbm", tile_size=[64, 64])
        return y

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


# ============================================================================
# Test: Forward propagation through elementwise (no matmul)
# ============================================================================

def test_forward_propagation_elementwise():
    """
    When the first op in a chain has a knob, forward propagation should
    reach downstream unannotated ops.

    exp(x) [knob] -> square -> result
    square should get layout from forward propagation.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        y = np.exp(x)
        knob.knob(y, mem_space="Sbuf", tile_size=[128, 128])
        z = np.square(y)
        return z

    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.square
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test: Elementwise fallback defaults
# ============================================================================

def test_fallback_3d_defaults():
    """
    For a 3D tensor with no annotations, fallback should produce:
      partition_dim=0, tile_size=[min(dim0,128), 1, dim_last]

    Shape [128, 4, 256]:
      tile = [min(128,128), 1, 256] = [128, 1, 256]
    """
    shape = (128, 4, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        return np.exp(x)

    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 1, 256>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_fallback_small_partition_dim():
    """
    When dim 0 < 128, tile_size[0] should be the actual dim size.

    Shape [64, 512]:
      tile = [min(64,128), 512] = [64, 512]
    """
    shape = (64, 512)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        return np.exp(x)

    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 64, 512>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_fallback_chain_no_annotations():
    """
    A chain of ops with no annotations should all get fallback defaults
    and tile/execute correctly.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        y = np.exp(x)
        z = np.square(y)
        return z

    # All ops should get annotations
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.square
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_fallback_chain_executes():
    """
    Verify that a chain with only fallback defaults tiles and runs correctly.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        y = np.exp(x)
        z = np.square(y)
        return z

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


# ============================================================================
# Test: Mixed user annotation + fallback
# ============================================================================

def test_partial_annotation_fills_gaps():
    """
    When only some ops are annotated, the pass should:
    1. Propagate from user annotations
    2. Fill remaining gaps with fallback defaults

    exp(x) [knob] -> square [no knob] -> add(square, y) [no knob]
    exp has user annotation; square gets it via forward propagation;
    add gets it via forward propagation from square.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def kernel(x, y):
        a = np.exp(x)
        knob.knob(a, mem_space="Sbuf", tile_size=[128, 128])
        b = np.square(a)
        c = b + y
        return c

    # All three ops should have annotations
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.square
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.add
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test: Return value gets SharedHbm
# ============================================================================

def test_return_value_gets_shared_hbm():
    """
    Values that flow to func.return should get mem_space=SharedHbm.
    The fallback default for non-return values is SBUF (mem_space=3),
    but return values need SharedHbm (mem_space=4).
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        return np.exp(x)

    # mem_space = 4 is SharedHbm
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "mem_space = 4",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
