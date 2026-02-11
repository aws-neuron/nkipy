import numpy as np


def compute_cos_sin_cache(
    head_dim: int, max_seq_len: int, base: int = 10000, dtype=np.float32
):
    """Compute cosine and sine cache for RoPE (Rotary Position Embedding).

    Comptime: This function uses only numpy operations on constant arguments (not
    runtime tensor inputs). When compiled for Trainium, these computations execute
    at compile time and the resulting arrays are baked into the HLO graph as
    constants. On CPU, they execute as regular numpy.
    """
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)

    return np.cos(freqs, dtype=np.float32).astype(dtype), np.sin(
        freqs, dtype=np.float32
    ).astype(dtype)


def apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin):
    """Apply rotary position embedding to query and key tensors."""
    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    # Split the hidden states into two halves
    half_h = xq.shape[-1] // 2
    xq0 = xq[:, :, :, :half_h]
    xq1 = xq[:, :, :, half_h:]

    xk0 = xk[:, :, :, :half_h]
    xk1 = xk[:, :, :, half_h:]

    # Apply rotary embedding between first and second halves
    xq_out_0 = xq0 * freqs_cos - xq1 * freqs_sin
    xq_out_1 = xq0 * freqs_sin + xq1 * freqs_cos

    xk_out_0 = xk0 * freqs_cos - xk1 * freqs_sin
    xk_out_1 = xk0 * freqs_sin + xk1 * freqs_cos

    # Concatenate the results back together to form the final output
    xq_out = np.concatenate([xq_out_0, xq_out_1], axis=-1)
    xk_out = np.concatenate([xk_out_0, xk_out_1], axis=-1)

    return xq_out, xk_out
