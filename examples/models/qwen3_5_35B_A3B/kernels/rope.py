import numpy as np


def compute_cos_sin_cache(
    rotary_dim: int, max_seq_len: int, base: float = 10000000.0, dtype=np.float32
):
    """Compute cosine and sine cache for partial RoPE.

    For Qwen3.5, rotary_dim = head_dim * partial_rotary_factor (e.g. 256 * 0.25 = 64).
    The cache has rotary_dim/2 frequencies.
    """
    freqs = 1.0 / (
        base ** (np.arange(0, rotary_dim, 2)[: (rotary_dim // 2)] / rotary_dim)
    )
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return (
        np.cos(freqs, dtype=np.float32).astype(dtype),
        np.sin(freqs, dtype=np.float32).astype(dtype),
    )


def apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin):
    """Apply rotary position embedding to query and key tensors.

    Uses the rotate_half convention: (x * cos) + (rotate_half(x) * sin)
    where rotate_half(x) = [-x2, x1] for x = [x1, x2].
    """
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    half_h = xq.shape[-1] // 2
    xq0 = xq[:, :, :, :half_h]
    xq1 = xq[:, :, :, half_h:]
    xk0 = xk[:, :, :, :half_h]
    xk1 = xk[:, :, :, half_h:]

    xq_out_0 = xq0 * freqs_cos - xq1 * freqs_sin
    xq_out_1 = xq0 * freqs_sin + xq1 * freqs_cos
    xk_out_0 = xk0 * freqs_cos - xk1 * freqs_sin
    xk_out_1 = xk0 * freqs_sin + xk1 * freqs_cos

    xq_out = np.concatenate([xq_out_0, xq_out_1], axis=-1)
    xk_out = np.concatenate([xk_out_0, xk_out_1], axis=-1)
    return xq_out, xk_out
