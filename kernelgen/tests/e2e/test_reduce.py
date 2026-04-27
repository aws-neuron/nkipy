"""
End-to-end tests for reduction kernels (sum, mean).

Exercises the full reduce pipeline:
1. Element-wise square
2. Reduction over last axis with keepdims=True
3. Tiled accumulation (tensor_reduce_arith + tensor_tensor_arith)

Run with: pytest tests/e2e/test_reduce.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


M, N = 256, 256
TILE_SIZE = [128, 128]


# ============================================================================
# Trace-level tests (LLVM JIT) — verify linalg lowering
# ============================================================================


@pytest.mark.parametrize("reduce_fn", ["sum", "mean"])
def test_reduce_square_trace(reduce_fn):
    """
    Test np.sum / np.mean of squared input at trace level.

    Verifies tracing produces a single linalg.generic
    instead of linalg.reduce + tensor.reshape.
    """
    reduce_op = getattr(np, reduce_fn)

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x.astype(np.float32))
        knob.knob(sq, mem_space="Sbuf", tile_size=TILE_SIZE)

        result = reduce_op(sq, axis=-1, keepdims=True)
        knob.knob(
            result,
            mem_space="SharedHbm",
            tile_size=[128, 1],
            reduction_tile=[128],
        )
        return result

    run_kernel_test(
        kernel,
        stop_after="trace",
        check_ir_contains=["linalg.generic"],
        check_ir_not_contains=["linalg.reduce", "tensor.reshape"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
        rtol=1e-3,
        atol=1e-3,
    )


# ============================================================================
# BIR simulation tests — verify full pipeline correctness
# ============================================================================


def test_reduce_sum_sim():
    """
    BIR simulation: np.sum of squared input.

    Verifies the tiled reduction accumulation pattern:
      tensor_reduce_arith(dst=temp, src=tile)     -- partial reduce
      tensor_tensor_arith(accum += temp)           -- accumulate
    """

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x.astype(np.float32))
        knob.knob(sq, mem_space="Sbuf", tile_size=TILE_SIZE)

        result = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(
            result,
            mem_space="SharedHbm",
            tile_size=[128],
            reduction_tile=[128],
        )
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.tensor_reduce_arith", "nisa.tensor_tensor_arith"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )


def test_reduce_mean_sim():
    """
    BIR simulation: np.mean of squared input.

    Mean is expressed as sum * (1/N) with separate knobs on the
    intermediate sum and the final multiply, since they use different
    memory spaces and tile configurations.
    """

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        sq = np.square(x.astype(np.float32))
        knob.knob(sq, mem_space="Sbuf", tile_size=TILE_SIZE)

        sm = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(
            sm,
            mem_space="SharedHbm",
            tile_size=[128],
            reduction_tile=[128],
        )

        result = sm * np.float32(1.0 / N)
        knob.knob(
            result,
            mem_space="SharedHbm",
            tile_size=[128, 1],
        )
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=[
            "nisa.tensor_reduce_arith",
            "nisa.tensor_tensor_arith",
            "nisa.tensor_scalar_arith",
        ],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
