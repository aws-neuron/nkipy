import numpy as np


def rmsnorm_kernel(
    x,
    weight,
    eps: float,
    compute_dtype=np.float32,
):
    """RMSNorm for Qwen3.5: output = (1 + weight) * (x / rms(x))"""
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)
    z = np.square(x)
    z = np.mean(z, axis=-1, keepdims=True)
    z = (z + eps).astype(x.dtype)
    z = x / np.sqrt(z)
    # Qwen3.5 uses (1 + weight) scaling (weight initialized to 0)
    res = z * (1.0 + weight)
    res = res.astype(original_dtype)
    return res
