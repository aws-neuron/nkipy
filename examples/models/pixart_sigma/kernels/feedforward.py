"""PixArt DiT feed-forward: Linear -> GELU(approximate=tanh) -> Linear.

Matches diffusers ``FeedForward(activation_fn="gelu-approximate")`` used by
PixArt. Unlike the qwen3 SiLU-gated MLP, there is no gating here.
"""

import numpy as np


def _gelu_tanh(x):
    xf = x.astype(np.float32)
    inner = np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf * xf * xf)
    out = 0.5 * xf * (1.0 + np.tanh(inner))
    return out.astype(x.dtype)


def feedforward_kernel(x, up_weight, up_bias, down_weight, down_bias):
    """Args:
        x: (B, L, hidden)
        up_weight: (hidden, intermediate), up_bias: (intermediate,)
        down_weight: (intermediate, hidden), down_bias: (hidden,)
    """
    h = np.matmul(x, up_weight)
    if up_bias is not None:
        h = h + up_bias
    h = _gelu_tanh(h)
    out = np.matmul(h, down_weight)
    if down_bias is not None:
        out = out + down_bias
    return out
