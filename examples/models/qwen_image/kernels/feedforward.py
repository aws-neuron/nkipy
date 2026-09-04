"""Per-stream feed-forward for the Qwen-Image MMDiT.

Matches diffusers ``FeedForward(activation_fn="gelu-approximate")``: a GELU
(tanh approximation) MLP, no gating. Both ``img_mlp`` and ``txt_mlp`` use it with
``dim_out == dim`` and inner dim ``4 * dim``.
"""

import numpy as np


def _gelu_tanh(x):
    xf = x.astype(np.float32)
    inner = np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf * xf * xf)
    out = 0.5 * xf * (1.0 + np.tanh(inner))
    return out.astype(x.dtype)


def feedforward_kernel(x, up_weight, up_bias, down_weight, down_bias,
                       all_reduce_fn):
    """Args:
        x: (B, L, dim)
        up_weight: (dim, inner), up_bias: (inner,)     [column-parallel under TP]
        down_weight: (inner, dim), down_bias: (dim,)   [row-parallel under TP]
        all_reduce_fn: callable summing the row-parallel output across ranks;
            always applied (TP is required). ``inner`` is the local (sharded)
            intermediate size and ``down_bias`` is the full replicated (dim,),
            added once after the reduction.
    """
    h = np.matmul(x, up_weight)
    if up_bias is not None:
        h = h + up_bias
    h = _gelu_tanh(h)
    out = np.matmul(h, down_weight)
    out = all_reduce_fn(out)
    if down_bias is not None:
        out = out + down_bias
    return out
