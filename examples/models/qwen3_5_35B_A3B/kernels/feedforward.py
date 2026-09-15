import numpy as np


def silu_kernel_(x):
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


def feedforward_kernel(x, gate_up_weight, down_weight):
    """Feed-forward network kernel with SiLU activation and gating."""
    mm_gup = np.matmul(x, gate_up_weight)
    xg, x_V = np.split(mm_gup, 2, axis=-1)
    swish = silu_kernel_(xg)
    x0 = swish * x_V
    return x0 @ down_weight


def shared_expert_kernel(x, gate_proj_weight, up_proj_weight, down_proj_weight):
    """Shared expert FFN: SiLU(x @ gate) * (x @ up) @ down."""
    gate = silu_kernel_(np.matmul(x, gate_proj_weight))
    up = np.matmul(x, up_proj_weight)
    return np.matmul(gate * up, down_proj_weight)
