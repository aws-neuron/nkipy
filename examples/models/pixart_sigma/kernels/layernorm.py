import numpy as np


def layernorm_kernel(x, weight=None, bias=None, eps: float = 1e-6,
                     compute_dtype=np.float32):
    """LayerNorm over the last axis.

    PixArt's DiT blocks use *non-affine* LayerNorm (elementwise_affine=False)
    before the adaLN modulation, so ``weight``/``bias`` are optional and
    default to the pure normalization. Computed in fp32 to limit numerical
    error, then cast back to the input dtype (matching rmsnorm_kernel in the
    qwen3 example).
    """
    original_dtype = x.dtype
    x = x.astype(compute_dtype)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    z = (x - mean) / np.sqrt(var + eps)

    if weight is not None:
        z = z * weight.astype(compute_dtype)
    if bias is not None:
        z = z + bias.astype(compute_dtype)

    return z.astype(original_dtype)
