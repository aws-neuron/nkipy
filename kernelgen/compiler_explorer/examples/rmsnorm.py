"""
Example NKIPy kernel: RMSNorm

RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight

This exercises:
1. Element-wise square (multiply)
2. Mean reduction over last axis
3. Addition with scalar epsilon
4. Reciprocal square root (rsqrt)
5. Element-wise multiply with weight
"""
import numpy as np
from nkipy_kernelgen import trace, knob

# Hardcoded dimensions
M = 256
N = 256

# Tile sizes
tile_size = [128, 128]  # TILE_M, TILE_N

# RMSNorm epsilon
eps = 1e-6

@trace(input_specs=[((M, N), "f32"), ((N, 1), "f32")])
def rmsnorm_kernel(x, weight):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight"""
    x_fp32 = x.astype(np.float32)
    w_fp32 = weight.astype(np.float32)

    sq = np.square(x_fp32)
    knob.knob(sq, mem_space="Sbuf", tile_size=tile_size)

    sum_sq = np.sum(sq, axis=-1, keepdims=True)
    knob.knob(
        sum_sq,
        mem_space="Sbuf",
        tile_size=[128],
        reduction_tile=[128],
    )

    mean_sq = sum_sq * np.float32(1.0 / N)
    knob.knob(
        mean_sq,
        mem_space="Sbuf",
        tile_size=[128, 1],
    )

    normed = x_fp32 / np.sqrt(mean_sq + eps)
    knob.knob(normed, mem_space="Sbuf", tile_size=tile_size)

    result = normed * w_fp32
    knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)

    return result
