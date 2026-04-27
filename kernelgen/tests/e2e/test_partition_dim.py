"""
End-to-end tests for non-zero partition_dim.

These tests verify that the canonicalize-partition-dim pass correctly inserts
transposes so that kernels with partition_dim != 0 produce numerically correct
results through the full compilation pipeline.

Run with: pytest tests/e2e/test_partition_dim.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: 2D elementwise with partition_dim=1
# ============================================================================

def test_exp_partition_dim_1():
    """
    exp(x) with partition_dim=1 through full pipeline to BIR simulation.
    Verifies transposes are correctly inserted and the result matches NumPy.

    Shape (64, 128) with partition_dim=1: dim 1 (size 128) is the partition dim.
    After canonicalization: tensor becomes (128, 64) with partition_dim=0.
    """
    M, N = 64, 128
    tile_size = [M, N]

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        result = np.exp(x)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size,
                  partition_dim=1)
        return result

    run_kernel_test(
        kernel,

        check_ir_contains=["nisa.activation", "op=exp"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


# ============================================================================
# Test: 2D elementwise chain with partition_dim=1
# ============================================================================

def test_sigmoid_partition_dim_1():
    """
    Sigmoid with partition_dim=1: sigmoid(x) = 1 / (1 + exp(-x))

    The entire elementwise chain should be rewritten with permuted shapes,
    and the result should match NumPy through BIR simulation.

    Shape (64, 128) with partition_dim=1: dim 1 (size 128) is the partition dim.
    After canonicalization: tensors become (128, 64) with partition_dim=0.
    """
    M, N = 64, 128
    tile_size = [M, N]

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        neg_x = -x
        knob.knob(neg_x, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)

        exp_neg_x = np.exp(neg_x)
        knob.knob(exp_neg_x, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)

        one_plus_exp = 1.0 + exp_neg_x
        knob.knob(one_plus_exp, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)

        result = 1.0 / one_plus_exp
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size,
                  partition_dim=1)

        return result

    run_kernel_test(
        kernel,

        check_ir_contains=["nisa.activation", "op=exp"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
