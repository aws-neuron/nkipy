"""Feed-forward network kernel with SiLU activation."""

import numpy as np


def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


def feedforward_kernel(
    x, gate_up_weight, down_weight, gate_up_bias=None, down_bias=None
):
    """Feed-forward network with SiLU-gated linear unit (SwiGLU).

    Args:
        x: Input tensor [batch_size, seq_len, hidden_size]
        gate_up_weight: Fused gate and up projection [hidden_size, 2*intermediate_size]
        down_weight: Down projection [intermediate_size, hidden_size]
        gate_up_bias: Optional bias for gate/up projection
        down_bias: Optional bias for down projection

    Returns:
        Output tensor [batch_size, seq_len, hidden_size]
    """
    # Gate and Up projection
    mm_gup = np.matmul(x, gate_up_weight)

    if gate_up_bias is not None:
        mm_gup = mm_gup + gate_up_bias

    # Split into gate and up components
    split_axis = mm_gup.ndim - 1
    xg, x_up = np.split(mm_gup, 2, axis=split_axis)

    # SwiGLU: silu(gate) * up
    swish = silu(xg)

    x0 = swish * x_up

    # Down projection
    output = x0 @ down_weight

    if down_bias is not None:
        output = output + down_bias

    return output
