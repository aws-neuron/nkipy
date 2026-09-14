"""Non-affine LayerNorm for the Qwen-Image MMDiT blocks.

Both streams use ``LayerNorm(elementwise_affine=False)`` before adaLN-style
modulation, so there are no learnable gain/bias. Computed in fp32 to limit
numerical error, then cast back to the input dtype.
"""

import numpy as np


def layernorm_kernel(x, eps: float = 1e-6, compute_dtype=np.float32):
    original_dtype = x.dtype
    x = x.astype(compute_dtype)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    z = (x - mean) / np.sqrt(var + eps)

    return z.astype(original_dtype)
