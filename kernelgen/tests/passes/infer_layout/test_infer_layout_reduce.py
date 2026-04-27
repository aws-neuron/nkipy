"""
Tests for the infer-layout pass with reduction operations.

Verifies that infer-layout propagates layout annotations backward through
reduction generics (linalg.generic with reduction iterator types).

np.mean decomposes into sum (reduction generic) + divide (elementwise generic).
A knob on the final mean result should propagate backward to the sum.

Run with: pytest tests/passes/infer_layout/test_infer_layout_reduce.py -v
"""

import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


M, N = 256, 256
TILE_SIZE = [128, 128]


def test_mean_propagates_to_sum():
    """
    np.mean = sum (linalg.generic reduction) + divide (linalg.generic elementwise).
    A knob on the mean result should propagate to the intermediate sum.

    Chain: linalg.square -> linalg.generic(sum) -> linalg.generic(div) -> result
                                                                          ^ knob

    After infer-layout, all three ops should have nkipy.annotate with tile_size.
    The sum's inferred tile_size should be [128] (not [128, 1]) — for reduction
    ops, tile_size only covers the non-reduced dimensions. If [128, 1] is
    propagated instead, knob-driven-tiling will error out.
    """
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x.astype(np.float32))
        knob.knob(sq, mem_space="Sbuf", tile_size=TILE_SIZE)

        result = np.mean(sq, axis=-1, keepdims=True)
        knob.knob(
            result,
            mem_space="SharedHbm",
            tile_size=[128, 1],
            reduction_tile=[128],
        )
        return result

    # Verify annotation propagation ordering:
    # square (user knob) -> sum (inferred, tile=[128]) -> div (user knob, tile=[128,1])
    # The sum is a reduction generic — its tile_size must be [128] (partition dim only),
    # NOT [128, 1], otherwise knob-driven-tiling will fail.
    check_patterns = """
    CHECK: linalg.square
    CHECK: nkipy.annotate{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size = array<i64: 128>
    CHECK: linalg.generic
    CHECK: nkipy.annotate{{.*}}tile_size = array<i64: 128, 1>
    """
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )

    # Verify numerical correctness after infer-layout
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )

    # Verify knob-driven-tiling succeeds with the inferred sum tile_size.
    # If InferLayout propagates [128, 1] instead of [128], this will error out.
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )


def test_rmsnorm_reduction_knob_propagation():
    """
    RMSNorm pattern: square -> sum -> scale -> add_eps -> sqrt -> broadcast div.

    The sum reduction has an explicit knob with reduction_tile. InferLayout
    should NOT propagate a knob without reduction_tile to the sum op, which
    would cause knob-driven-tiling to fail with:
      "Invalid tile configuration: reduction op requires reduction_tile, got none"
    """
    tile_size = [128, 128]
    eps = 1e-6

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x)
        knob.knob(sq, mem_space="Sbuf", tile_size=tile_size)
        mean_sq = np.sum(sq, axis=1, keepdims=True) / 256.0
        rms = np.sqrt(mean_sq + eps)
        result = np.divide(x, rms)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    # Verify the reduction op gets tile_size=[128] with reduction_tile,
    # not an incorrectly propagated [128, 128] without reduction_tile
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128>",
            "reduction_tile = array<i64: 128>",
        ],
        modes=Mode.STRING_CHECK,
    )

    # Verify knob-driven-tiling succeeds (would fail if reduction_tile is missing)
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
