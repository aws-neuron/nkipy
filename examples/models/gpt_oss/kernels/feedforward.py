import numpy as np


def clamped_swiglu(gate, up, alpha, limit):
    """gpt-oss clamped SwiGLU gating.

    Mirrors GptOssExperts._apply_gate:
        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, limit)
        glu  = gate * sigmoid(alpha * gate)
        out  = (up + 1) * glu
    """
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    glu = gate * (1.0 / (1.0 + np.exp(-alpha * gate)))
    return (up + 1.0) * glu


def feedforward_kernel(
    x, gate_up_weight, gate_up_bias, down_weight, down_bias, alpha, limit
):
    """Single-expert feed-forward with clamped SwiGLU and biases.

    `gate_up_weight`/`gate_up_bias` are pre-arranged at weight-prep time so the
    gate half comes first and the up half second (de-interleaved from the HF
    layout), letting us split in half here.
    """
    mm_gup = np.matmul(x, gate_up_weight) + gate_up_bias

    xg, x_up = np.split(mm_gup, 2, axis=-1)

    gated = clamped_swiglu(xg, x_up, alpha, limit)

    return np.matmul(gated, down_weight) + down_bias
