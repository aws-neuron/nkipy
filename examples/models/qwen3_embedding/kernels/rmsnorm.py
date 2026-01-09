import numpy as np


def rmsnorm(x, weight, eps: float):
    # Use float32 to reduce numerical error
    compute_dtype = np.float32
    """
    z: Array["B, L or 1, 1"] = (x**2).mean(-1, keepdims=True) + self.eps
    z: Array["B, L or 1, D"] = x / np.sqrt(z)
    ret = z * self.weight
    """
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)

    z = np.square(x)
    z = np.mean(z, axis=-1, keepdims=True)
    z = x / np.sqrt(z + eps)
    res = z * weight
    res = res.astype(original_dtype)
    return res
