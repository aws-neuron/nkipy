"""
End-to-end tests for sigmoid activation and tensor-scalar arithmetic.

Sigmoid: sigmoid(x) = 1.0 / (1.0 + exp(-x))

This exercises the full Path 2 implementation:
1. Division converted to multiply + reciprocal (prepare-arithmetic pass)
2. Tensor-scalar operations for the 1.0 constants (nisa.tensor_scalar_arith)
3. Negation for -x
4. Exponential via nisa.activation(op=exp)
5. Reciprocal via nisa.reciprocal

Run with: pytest tests/e2e/test_sigmoid.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test Cases
# ============================================================================

def test_exp_activation():
    """
    Test exp(x) lowering to nisa.activation.

    This verifies:
    - linalg.exp is converted to nisa.activation with op=exp
    - Scalar bias=0.0 and scale=1.0 are used
    """
    M, N = 128, 256
    tile_size = [128, 128]

    @trace(input_specs=[((M, N), "f32")])
    def exp_kernel(x):
        result = np.exp(x)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    run_kernel_test(
        exp_kernel,

        check_ir_contains=["nisa.activation", "op=exp"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_tensor_add_scalar():
    """
    Test tensor + scalar arithmetic.

    This verifies:
    - Constant 2.0 is broadcast via linalg.fill with CONSTANT memspace
    - linalg.add detects CONSTANT operand and emits nisa.tensor_scalar_arith
    """
    M, N = 128, 256
    tile_size = [128, 128]
    scalar_value = 2.0

    @trace(input_specs=[((M, N), "f32")])
    def add_scalar_kernel(x):
        result = x + scalar_value
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    run_kernel_test(
        add_scalar_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.target",
            "nisa.tensor_scalar_arith", "op0=add",
        ],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_sigmoid():
    """
    Test sigmoid activation: sigmoid(x) = 1.0 / (1.0 + exp(-x))

    This is the full test that exercises:
    1. Negation: -x (via linalg.negf or multiply by -1)
    2. Exponential: exp(-x) via nisa.activation(op=exp)
    3. Addition with scalar: 1.0 + exp(-x) via nisa.tensor_scalar_arith(op=add)
    4. Division: 1.0 / (result) converted to reciprocal by prepare-arithmetic
    5. Multiply by 1.0 (or direct reciprocal output)
    """
    M, N = 128, 256
    tile_size = [128, 128]

    @trace(input_specs=[((M, N), "f32")])
    def sigmoid_kernel(x):
        # Sigmoid: 1 / (1 + exp(-x))
        neg_x = -x
        knob.knob(neg_x, mem_space="Sbuf", tile_size=tile_size)

        exp_neg_x = np.exp(neg_x)
        knob.knob(exp_neg_x, mem_space="Sbuf", tile_size=tile_size)

        one_plus_exp = 1.0 + exp_neg_x
        knob.knob(one_plus_exp, mem_space="Sbuf", tile_size=tile_size)

        # Division gets converted to reciprocal by prepare-arithmetic
        result = 1.0 / one_plus_exp
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)

        return result

    run_kernel_test(
        sigmoid_kernel,

        check_ir_contains=["nisa.activation", "op=exp"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_scalar_minus_tensor():
    """
    Test scalar - tensor arithmetic (reverse operands).

    This verifies:
    - When lhs is CONSTANT (scalar), reverse_operands=first is set
    - nisa.tensor_scalar_arith correctly computes scalar - tensor
    """
    M, N = 128, 256
    tile_size = [128, 128]
    scalar_value = 5.0

    @trace(input_specs=[((M, N), "f32")])
    def sub_scalar_kernel(x):
        # scalar - tensor requires reverse_operands
        result = scalar_value - x
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    run_kernel_test(
        sub_scalar_kernel,
        check_ir_contains=[
            "nisa.tensor_scalar_arith", "op0=subtract", "reverse_operands=first",
        ],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_division_to_reciprocal():
    """
    Test division conversion to multiply + reciprocal.

    x / 2.0 is converted by prepare-arithmetic to:
    x * reciprocal(broadcast(2.0))
    """
    M, N = 128, 256
    tile_size = [128, 128]
    divisor = 2.0

    @trace(input_specs=[((M, N), "f32")])
    def div_kernel(x):
        result = x / divisor
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    run_kernel_test(
        div_kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
