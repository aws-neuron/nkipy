"""
Tests for the infer-layout pass with reduction operations.

Verifies that infer-layout propagates layout annotations backward through
reduction generics (linalg.generic with reduction iterator types).

np.mean decomposes into sum (reduction generic) + divide (elementwise generic).
A knob on the final mean result should propagate backward to the sum.

Run with: pytest tests/passes/infer_layout/test_infer_layout_reduce.py -v
"""

import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


M, N = 256, 256
TILE_SIZE = [128, 128]


def test_mean_propagates_to_sum():
    """
    np.mean = sum (linalg.generic reduction) + divide (linalg.generic elementwise).
    A knob on the mean result should propagate to the intermediate sum.

    Chain: linalg.square -> linalg.generic(sum) -> linalg.generic(div) -> result
                                                                          ^ knob

    After infer-layout, all three ops should have nkipy.layout with tile_size.
    The sum (keepdims=True) result has shape [M, 1]; its layout tile is
    [128, 1] (value-shape) and its tile_op carries [128, 128] (iter-space:
    parallel + reduction).
    """
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x.astype(np.float32))
        knob.knob(sq).tile_op(tile_size=TILE_SIZE).layout(mem_space="Sbuf")

        result = np.mean(sq, axis=-1, keepdims=True)
        # divide(mean) is rank-2; the user supplies the elementwise output
        # tile here.
        knob.knob(result).tile_op(tile_size=[128, 1]).layout(mem_space="SharedHbm")
        return result

    # Verify annotation propagation ordering:
    # square (user knob) -> sum (inferred) -> div (user knob).
    # Sum (keepdims=True): layout tile=[128,1], tile_op=[128,128] iter-space.
    # Divide is rank-2 elementwise: layout tile=[128,1].
    check_patterns = """
    CHECK: linalg.square
    CHECK: nkipy.layout{{.*}}tile_size
    CHECK: linalg.generic
    CHECK: nkipy.layout{{.*}}tile_size = array<i64: 128, 1>
    CHECK: linalg.generic
    CHECK: nkipy.layout{{.*}}tile_size = array<i64: 128, 1>
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
        knob.knob(sq).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")
        mean_sq = np.sum(sq, axis=1, keepdims=True) / 256.0
        rms = np.sqrt(mean_sq + eps)
        result = np.divide(x, rms)
        knob.knob(result).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm")
        return result

    # Verify the sum (keepdims=True) reduction's nkipy.layout gets the
    # value-shape tile [128, 1] and the iter-space loop_tile_size [128, 128].
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=[
            "tile_size = array<i64: 128, 1>",
            "loop_tile_size = array<i64: 128, 128>",
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
