"""
Tests for the infer-layout pass.

The infer-layout pass infers layout information (tiling and placement) for
elementwise operations that lack explicit annotations. It propagates tile_size
and mem_space from annotated elementwise ops to adjacent unannotated ones
along the SSA use-def chain.

Run with: python -m pytest tests/passes/infer_layout/test_infer_layout.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: SiLU chain propagation (the motivating case)
# ============================================================================

def test_silu_chain_annotations():
    """
    Verify that infer-layout propagates tiling and placement to all intermediate
    elementwise ops in a compact SiLU expression.

    Input:  gated = gate / (1.0 + exp(-gate)) * up
            knob only on final 'gated' (linalg.mul)

    After prepare-arithmetic, the chain is:
      linalg.generic (negate)  -> no layout
      linalg.exp               -> no layout
      linalg.generic (add 1.0) -> no layout
      linalg.reciprocal        -> no layout
      linalg.mul (x * recip)   -> no layout
      linalg.mul (swish * up)  -> HAS layout (tile_size=[128, 128], mem_space=Sbuf)

    After infer-layout, ALL elementwise ops should have nkipy.annotate with
    tile_size = [128, 128].
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def silu_kernel(gate, up):
        gated = gate / (1.0 + np.exp(-gate)) * up
        knob.knob(gated, mem_space="Sbuf", tile_size=tile_size)
        return gated

    # After infer-layout, we expect nkipy.annotate ops with tile_size on every
    # elementwise op in the chain. Each op should be followed by one.
    check_patterns = """
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.reciprocal
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.mul
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.mul
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        silu_kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_silu_chain_tiling_executes():
    """
    Verify that after infer-layout, knob-driven-tiling can tile ALL ops in the
    SiLU chain and the result is numerically correct via LLVM JIT.

    This is the end-to-end correctness check: inferred layout -> tiling -> JIT.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def silu_kernel(gate, up):
        gated = gate / (1.0 + np.exp(-gate)) * up
        knob.knob(gated, mem_space="Sbuf", tile_size=tile_size)
        return gated

    run_kernel_test(
        silu_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: Simple chain (A -> B -> C, only C annotated)
# ============================================================================

def test_simple_chain():
    """
    Test backward propagation through a simple 2-op chain.

    Input:  y = exp(x), z = y + 1.0, knob on z
    After infer-layout: exp should also get a layout annotation.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def chain_kernel(x):
        y = np.exp(x)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=tile_size)
        return z

    # After infer-layout, both linalg.exp and linalg.generic(add) should
    # have nkipy.annotate with tile_size
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        chain_kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_simple_chain_tiling_executes():
    """
    Verify that a simple chain with inferred layout tiles and executes correctly.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def chain_kernel(x):
        y = np.exp(x)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=tile_size)
        return z

    run_kernel_test(
        chain_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


# ============================================================================
# Test: Already-annotated ops should not be overridden
# ============================================================================

def test_existing_annotations_preserved():
    """
    If a producer op already has a layout annotation, infer-layout should NOT
    override it. Each op retains its original annotation.

    Input:  y = exp(x) with knob tile=[128, 128]
            z = y + 1.0 with knob tile=[128, 128]
    Both ops already annotated -> no new annotations should be created.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        y = np.exp(x)
        knob.knob(y, mem_space="Sbuf", tile_size=tile_size)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=tile_size)
        return z

    # Both ops already have annotations. After infer-layout, the pass should
    # report 0 inferred annotations. We verify the IR still has the same
    # structure: exp then annotate, then generic then annotate.
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test: Non-elementwise boundary (matmul -> elementwise chain)
# ============================================================================

def test_stops_at_matmul_boundary():
    """
    Verify that layout propagation does NOT cross non-elementwise ops.

    Input: mm = matmul(a, b) with its own matmul knob
           y = exp(mm)       -> no layout
           z = y + 1.0       -> knob on z

    The infer-layout pass should propagate from z to y (both elementwise),
    but should NOT create a new annotation on mm (it's a matmul, not elementwise).
    """
    shape = (256, 256)
    matmul_tile = [128, 128]
    ew_tile = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def kernel(a, b):
        mm = np.matmul(a, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=[128])
        y = np.exp(mm)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=ew_tile)
        return z

    # After infer-layout:
    # - matmul should still have its own annotate with tile_size [128,128,128]
    # - exp should get an inferred annotate with tile_size [128,128]
    # - generic(add) should have its original annotate with tile_size [128,128]
    check_patterns = """
    CHECK: linalg.matmul
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_matmul_plus_elementwise_tiling_executes():
    """
    Verify that matmul followed by elementwise chain with inferred layout
    tiles and executes correctly.
    """
    shape = (256, 256)
    matmul_tile = [128, 128]
    ew_tile = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def kernel(a, b):
        mm = np.matmul(a, b)
        knob.knob(mm, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=[128])
        y = np.exp(mm)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=ew_tile)
        return z

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )


# ============================================================================
# Test: No annotations (pass should be a no-op)
# ============================================================================

def test_no_annotations_generates_defaults():
    """
    When no ops have layout annotations, infer-layout should generate
    default annotations: partition_dim=0, tile_size=[min(dim0,128), dim_last],
    mem_space=SBUF for intermediates.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        return np.exp(x)

    # Default tile: partition_dim=0, tile_size=[min(256,128), 256] = [128, 256].
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 256>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: 3D partition_dim BFS propagation
# ============================================================================

def test_3d_partition_dim_inferred_from_tile():
    """
    Verify that both ops retain partition_dim=1 from their explicit knob
    annotations.  Both exp and mul are annotated with partition_dim=1,
    so infer-layout should preserve both.
    """
    B, S, D = 4, 128, 64
    tile_size = [1, 128, 64]

    @trace(input_specs=[((B, S, D), "f32"), ((B, S, D), "f32")])
    def kernel_3d_pdim(a, b):
        intermediate = np.exp(a)
        knob.knob(intermediate, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)
        result = intermediate * b
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size,
                  partition_dim=1)
        return result

    # After infer-layout, ALL ops should have partition_dim = 1 since
    # tile dim 1 = 128 = MAX_PARTITION_DIM.
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    CHECK: linalg.mul
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    """
    run_kernel_test(
        kernel_3d_pdim,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_3d_partition_dim_propagation_unannotated():
    """
    Verify that BFS backward propagation copies partition_dim=1 from an
    annotated consumer to an unannotated producer.

    Chain: x → exp(x) → exp(x) + bias → result
    Only result has a knob with partition_dim=1.  BFS should propagate
    partition_dim=1 to the intermediate exp op.
    """
    B, S, D = 4, 128, 64
    tile_size = [1, 128, 64]

    @trace(input_specs=[((B, S, D), "f32"), ((B, S, D), "f32")])
    def kernel_3d_chain(x, bias):
        y = np.exp(x)
        # Only annotate the final result — y gets tile_size via BFS
        z = y + bias
        knob.knob(z, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)
        return z

    # Both ops should have partition_dim = 1 after infer-layout
    check_patterns = """
    CHECK: linalg.exp
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    CHECK: linalg.add
    CHECK: nkipy.annotate{{.*}}partition_dim = 1
    """
    run_kernel_test(
        kernel_3d_chain,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_3d_partition_dim_enables_canonicalize():
    """
    Verify that BFS-propagated partition_dim=1 enables
    canonicalize-partition-dim to insert transposes.

    Chain: x → exp(x) → exp(x) + bias → result
    Only result has explicit partition_dim=1.  After BFS propagation
    and canonicalize-partition-dim, transposes should be inserted to
    move partition_dim=1 to position 0.
    """
    B, S, D = 4, 128, 64
    tile_size = [1, 128, 64]

    @trace(input_specs=[((B, S, D), "f32"), ((B, S, D), "f32")])
    def kernel_3d_canon(x, bias):
        y = np.exp(x)
        z = y + bias
        knob.knob(z, mem_space="SharedHbm", tile_size=tile_size,
                  partition_dim=1)
        return z

    # After canonicalize-partition-dim, transposes should be inserted
    # to move partition_dim=1 to dim 0. This proves that infer-layout
    # correctly filled partition_dim on the unannotated exp op.
    check_patterns = """
    CHECK: func.func @kernel_3d_canon
    CHECK: linalg.transpose
    CHECK: linalg.exp
    CHECK: linalg.add
    """
    run_kernel_test(
        kernel_3d_canon,
        stop_after='canonicalize-partition-dim',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
