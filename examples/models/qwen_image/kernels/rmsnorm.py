"""RMSNorm for Qwen-Image.

Used in two places: the text-stream input norm (``txt_norm``, over
``joint_attention_dim``) and the per-head QK-RMSNorm inside joint attention
(``norm_q/norm_k`` and ``norm_added_q/norm_added_k``, over ``head_dim``).

Matches the qwen3 example's ``rmsnorm_kernel``; computed in fp32 to limit
numerical error, then cast back to the input dtype.
"""

import numpy as np


def rmsnorm_kernel(x, weight, eps: float, compute_dtype=np.float32):
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)

    z = np.mean(np.square(x), axis=-1, keepdims=True)
    z = (z + eps).astype(x.dtype)
    z = x / np.sqrt(z)

    res = z * weight
    return res.astype(original_dtype)
