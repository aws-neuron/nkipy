import numpy as np


def rmsnorm_kernel(
    x,
    weight,
    eps: float,
    compute_dtype=np.float32,  # reduce numerical error
):
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)
    z = np.square(x)
    z = np.mean(z, axis=-1, keepdims=True)

    z = (z + eps).astype(x.dtype)
    z = x / np.sqrt(z)

    res = z * weight
    res = res.astype(original_dtype)
    return res
