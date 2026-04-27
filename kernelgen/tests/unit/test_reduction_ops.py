"""
Tests for reduction operations: sum, mean, max, min.

These tests verify that MLIR/LLVM execution matches NumPy CPU execution
and that the tracer emits linalg.generic with reduction iterator types.
Tests with knobs also verify KnobDrivenTiling + apply-and-strip-transforms
produces tiled scf.for loops with SBUF promotion, and linalg-to-nisa
converts reduction generics to nisa.tensor_reduce_arith.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import nkipy_kernelgen_test, run_kernel_test, Mode


# ============================================================================
# np.sum → linalg.generic (arith.addf)
# ============================================================================

@pytest.mark.parametrize("shape,axis,keepdims,tile_size,reduction_tile", [
    ((128, 256), -1, True,  [64],     [128]),    # last axis, keepdims
    ((128, 256), 0,  False, [128],    [64]),      # first axis
    ((128, 256), 1,  False, [64],     [128]),     # last axis
    ((128, 256), 1,  True,  [64],     [128]),     # last axis, keepdims
    ((64, 128, 64), -1, True,  [32, 64], [32]),   # 3D, last axis, keepdims
    ((64, 128, 64), 0,  False, [64, 32], [32]),   # 3D, first axis
    ((64, 128, 64), 1,  False, [32, 32], [64]),   # 3D, middle axis
])
def test_sum_axis(shape, axis, keepdims, tile_size, reduction_tile):
    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        result = np.sum(a, axis=axis, keepdims=keepdims)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage verification once we fix:
    #   - 1D output shapes (keepdims=False) crash nisa.dma_copy (needs 2D tiles)
    #   - non-rightmost reductions (axis=0) not supported by LinalgGenericReductionToNisaPattern
    #   - linalg.fill on HBM not yet lowered to NISA
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=["nisa.tensor_reduce_arith"],
    #     modes=Mode.STRING_CHECK,
    # )


@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    check_ir_not_contains=["linalg.reduce"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_sum_full_reduction(a):
    """Sum all elements to a scalar (no parallel dims, not tileable)."""
    return np.sum(a)


# ============================================================================
# np.mean → linalg.generic (sum) + scalar divide
# ============================================================================

@pytest.mark.parametrize("shape,axis,keepdims,tile_size,reduction_tile", [
    ((128, 256), -1, True,  [64],     [128]),    # last axis, keepdims
    ((128, 256), 0,  False, [128],    [64]),      # first axis
    ((128, 256), 1,  False, [64],     [128]),     # last axis
    ((128, 256), 1,  True,  [64],     [128]),     # last axis, keepdims
    ((64, 128, 64), -1, True,  [32, 64], [32]),   # 3D, last axis, keepdims
    ((64, 128, 64), 0,  False, [64, 32], [32]),   # 3D, first axis
    ((64, 128, 64), 1,  False, [32, 32], [64]),   # 3D, middle axis
])
def test_mean_axis(shape, axis, keepdims, tile_size, reduction_tile):
    norm_axis = axis % len(shape)
    N = shape[norm_axis]

    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        # Decompose mean as sum * (1/N) so we can annotate the reduction
        sm = np.sum(a, axis=axis, keepdims=keepdims)
        knob.knob(sm, tile_size=tile_size, reduction_tile=reduction_tile)
        return sm * np.float32(1.0 / N)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage verification (same blockers as test_sum_axis,
    #   plus scalar multiply after reduction needs all intermediate allocs annotated)
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=["nisa.tensor_reduce_arith", "nisa.tensor_scalar_arith"],
    #     modes=Mode.STRING_CHECK,
    # )


@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    check_ir_not_contains=["linalg.reduce"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_mean_full_reduction(a):
    """Mean of all elements to a scalar (no parallel dims, not tileable)."""
    return np.mean(a)


# ============================================================================
# np.max → linalg.generic (arith.maximumf)
# ============================================================================

@pytest.mark.parametrize("shape,axis,keepdims,tile_size,reduction_tile", [
    ((128, 256), -1, True,  [64],     [128]),    # last axis, keepdims
    ((128, 256), 0,  False, [128],    [64]),      # first axis
    ((128, 256), 1,  False, [64],     [128]),     # last axis
    ((128, 256), 1,  True,  [64],     [128]),     # last axis, keepdims
    ((64, 128, 64), -1, True,  [32, 64], [32]),   # 3D, last axis, keepdims
    ((64, 128, 64), 0,  False, [64, 32], [32]),   # 3D, first axis
    ((64, 128, 64), 1,  False, [32, 32], [64]),   # 3D, middle axis
])
def test_max_axis(shape, axis, keepdims, tile_size, reduction_tile):
    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        result = np.max(a, axis=axis, keepdims=keepdims)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage verification (same blockers as test_sum_axis)
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=["nisa.tensor_reduce_arith"],
    #     modes=Mode.STRING_CHECK,
    # )


@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    check_ir_not_contains=["linalg.reduce"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_max_full_reduction(a):
    """Max of all elements to a scalar (no parallel dims, not tileable)."""
    return np.max(a)


# ============================================================================
# np.min → linalg.generic (arith.minimumf)
# ============================================================================

@pytest.mark.parametrize("shape,axis,keepdims,tile_size,reduction_tile", [
    ((128, 256), -1, True,  [64],     [128]),    # last axis, keepdims
    ((128, 256), 0,  False, [128],    [64]),      # first axis
    ((128, 256), 1,  False, [64],     [128]),     # last axis
    ((128, 256), 1,  True,  [64],     [128]),     # last axis, keepdims
    ((64, 128, 64), -1, True,  [32, 64], [32]),   # 3D, last axis, keepdims
    ((64, 128, 64), 0,  False, [64, 32], [32]),   # 3D, first axis
    ((64, 128, 64), 1,  False, [32, 32], [64]),   # 3D, middle axis
])
def test_min_axis(shape, axis, keepdims, tile_size, reduction_tile):
    @trace(input_specs=[(shape, "f32")])
    def kernel(a):
        result = np.min(a, axis=axis, keepdims=keepdims)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage verification (same blockers as test_sum_axis)
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=["nisa.tensor_reduce_arith"],
    #     modes=Mode.STRING_CHECK,
    # )


@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32")],
    stop_after="trace",
    check_ir_contains=["linalg.generic"],
    check_ir_not_contains=["linalg.reduce"],
    modes=Mode.LLVM | Mode.STRING_CHECK,
)
def test_min_full_reduction(a):
    """Min of all elements to a scalar (no parallel dims, not tileable)."""
    return np.min(a)


# ============================================================================
# Chained reductions (real-world patterns)
# ============================================================================

def test_sum_of_squares():
    """Pattern: sum(x^2) - used in RMSNorm variance computation."""
    @trace(input_specs=[((128, 256), "f32")])
    def kernel(a):
        sq = np.square(a)
        knob.knob(sq, tile_size=[64, 128])

        result = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(result, tile_size=[64], reduction_tile=[128])
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.square", "linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage — crashes because intermediate
    #   tensor.empty() (from chaining square→reduce) gets memref.alloc without
    #   NISA memory space annotation
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=["nisa.activation", "nisa.tensor_reduce_arith"],
    #     modes=Mode.STRING_CHECK,
    # )


def test_mean_of_squares():
    """Pattern: mean(x^2) = sum(x^2) * (1/N) - variance without centering."""
    N = 256

    @trace(input_specs=[((128, N), "f32")])
    def kernel(a):
        sq = np.square(a)
        knob.knob(sq, tile_size=[64, 128])

        sm = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(sm, tile_size=[64], reduction_tile=[128])
        return sm * np.float32(1.0 / N)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.square", "linalg.generic"],
        check_ir_not_contains=["linalg.reduce"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage — same intermediate alloc issue as
    #   test_sum_of_squares, plus scalar multiply chain
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     check_ir_contains=[
    #         "nisa.activation",
    #         "nisa.tensor_reduce_arith",
    #         "nisa.tensor_scalar_arith",
    #     ],
    #     modes=Mode.STRING_CHECK,
    # )


def test_softmax_reductions():
    """Pattern: exp(x - max(x)) / sum(exp(x - max(x))) - softmax."""
    @trace(input_specs=[((128, 256), "f32")])
    def kernel(a):
        a_max = np.max(a, axis=-1, keepdims=True)
        knob.knob(a_max, tile_size=[64], reduction_tile=[128])

        exp_a = np.exp(a - a_max)

        exp_sum = np.sum(exp_a, axis=-1, keepdims=True)
        knob.knob(exp_sum, tile_size=[64], reduction_tile=[128])

        return exp_a / exp_sum

    np.random.seed(42)
    A = (np.random.randn(128, 256) * 0.5).astype(np.float32)

    run_kernel_test(
        kernel, stop_after="trace",
        inputs=[A],
        modes=Mode.LLVM,
    )

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        inputs=[A],
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.generic"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    # TODO: enable linalg-to-nisa stage — multi-output chained kernel with
    #   intermediate allocs that lack NISA memory space annotation
    # run_kernel_test(
    #     kernel,
    #     stop_after='linalg-to-nisa',
    #     inputs=[A],
    #     check_ir_contains=["nisa.tensor_reduce_arith", "nisa.activation"],
    #     modes=Mode.STRING_CHECK,
    # )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
