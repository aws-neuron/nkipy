import numpy as np


def compute_qwen3_cos_sin(max_model_len: int, head_dim: int, theta: float = 1000000.0):
    """
    Compute cos/sin tables for RoPE specific to Qwen3

    Args:
        max_model_len: Maximum sequence length
        head_dim: Dimension of each attention head
        theta: RoPE theta parameter

    Returns:
        cos, sin: Precomputed cos/sin tables [seq_len, head_dim]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

    # Compute position indices
    t = np.arange(max_model_len, dtype=np.float32)

    # Compute frequencies for each position
    freqs = np.outer(t, inv_freq)  # [seq_len, head_dim/2]

    # Compute cos and sin
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    return cos, sin


def rope_qwen3(xq, xk, freqs_cos, freqs_sin):
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
