"""
End-to-end tests for auto-inferred layouts (no user annotations).

These tests verify that the infer-layout pass can automatically determine
tile sizes, partition dims, and memory spaces, and that the resulting code
compiles and runs correctly through BIR simulation and hardware.

Run with: pytest tests/e2e/test_auto_layout.py -v
"""

import numpy as np
import pytest

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Elementwise: no user annotations at all
# ============================================================================

def test_exp_no_annotations():
    """
    exp(x) with no user knobs. The pass should auto-infer:
      partition_dim=0, tile_size=[128, 256], mem_space=SharedHbm (return val)
    """
    M, N = 128, 256

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.exp(x)

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )


def test_elementwise_chain_no_annotations():
    """
    exp -> square -> add_scalar chain with no user annotations.
    All ops should get auto-inferred layouts and compile/run correctly.
    """
    M, N = 256, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        y = np.exp(x)
        z = np.square(y)
        return z + 1.0

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )


def test_sigmoid_no_annotations():
    """
    sigmoid(x) = 1 / (1 + exp(-x)) with no user annotations.
    This exercises: negate, exp, add_scalar, reciprocal, mul_scalar.
    """
    M, N = 128, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return 1.0 / (1.0 + np.exp(-x))

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Matmul: no user annotations
# ============================================================================

def test_matmul_no_annotations():
    """
    matmul(a, b) with no user knobs. The pass should auto-seed:
      Result C [M,N]: partition_dim=0, tile=[128, 128], reduction_tile=[128]
      Operand A [M,K]: partition_dim=1
      Operand B [K,N]: partition_dim=0

    TODO: Once KnobDrivenTiling supports non-blocked matmul for small dims,
    the auto-seeded tile can be larger (e.g., [128, 256] for N=512).
    """
    M, N, K = 256, 256, 128

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        return np.matmul(a, b)

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )


def test_matmul_add_no_annotations():
    """
    matmul(a, b) + c with no user annotations. Tests auto-seeding of
    matmul and forward propagation to the elementwise add.

    Dims must be large enough to satisfy KnobDrivenTiling's blocking
    factor=2 constraint: tile * 2 <= dim for each tiled dimension.

    TODO: Once KnobDrivenTiling supports non-blocked matmul for small dims,
    smaller dimensions can be used here.
    """
    M, N, K = 256, 512, 128

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, c):
        return np.matmul(a, b) + c

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
