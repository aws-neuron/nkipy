import numpy as np


def silu_kernel_(x):
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


def feedforward_kernel(
    x, gate_up_weight, down_weight, gate_up_bias=None, down_bias=None
):
    """Feed-forward network kernel with SiLU activation and gating."""
    # Gate and Up projection: x @ [gate_weight, up_weight]
    mm_gup = np.matmul(x, gate_up_weight)

    # Add bias if provided
    if gate_up_bias is not None:
        mm_gup = mm_gup + gate_up_bias

    # Split into gate and up components
    split_axis = mm_gup.ndim - 1
    xg, x_V = np.split(mm_gup, 2, axis=split_axis)

    # Apply SiLU activation to gate
    swish = silu_kernel_(xg)

    # Element-wise multiplication (gating)
    x0 = swish * x_V

    # Down projection
    output = x0 @ down_weight

    # Add down bias if provided
    if down_bias is not None:
        output = output + down_bias

    return output
