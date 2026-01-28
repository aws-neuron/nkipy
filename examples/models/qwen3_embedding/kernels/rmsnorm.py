import numpy as np


def rmsnorm(x, weight, eps: float):
    # Use float32 to reduce numerical error
    compute_dtype = np.float32
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)

    z = np.square(x)
    z = np.mean(z, axis=-1, keepdims=True)
    z = x / np.sqrt(z + eps)
    res = z * weight
    res = res.astype(original_dtype)
    return res
