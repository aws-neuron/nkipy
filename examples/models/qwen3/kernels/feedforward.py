import numpy as np


def silu_kernel_(x):
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


def feedforward_kernel(x, gate_up_weight, down_weight):
    """Feed-forward network kernel with SiLU activation and gating."""
    mm_gup = np.matmul(x, gate_up_weight)

    split_axis = mm_gup.ndim - 1
    xg, x_V = np.split(mm_gup, 2, axis=split_axis)

    swish = silu_kernel_(xg)

    x0 = swish * x_V
    return x0 @ down_weight
