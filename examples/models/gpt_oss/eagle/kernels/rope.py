"""llama3 RoPE for the P-EAGLE drafter.

Same rotate-halves formulation as the gpt-oss base model; only the inverse
frequencies differ (llama3 scaling, precomputed on the host in EagleConfig).
"""

import numpy as np


def compute_cos_sin_cache(
    inv_freq, max_seq_len, attention_scaling=1.0, dtype=np.float32
):
    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    cos = (np.cos(freqs, dtype=np.float32) * attention_scaling).astype(dtype)
    sin = (np.sin(freqs, dtype=np.float32) * attention_scaling).astype(dtype)
    return cos, sin


def apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin):
    """Rotate-halves RoPE on query/key tensors of shape (B, S, H, D)."""
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    half_h = xq.shape[-1] // 2
    xq0 = xq[:, :, :, :half_h]
    xq1 = xq[:, :, :, half_h:]
    xk0 = xk[:, :, :, :half_h]
    xk1 = xk[:, :, :, half_h:]

    xq_out_0 = xq0 * freqs_cos - xq1 * freqs_sin
    xq_out_1 = xq1 * freqs_cos + xq0 * freqs_sin
    xk_out_0 = xk0 * freqs_cos - xk1 * freqs_sin
    xk_out_1 = xk1 * freqs_cos + xk0 * freqs_sin

    xq_out = np.concatenate([xq_out_0, xq_out_1], axis=-1)
    xk_out = np.concatenate([xk_out_0, xk_out_1], axis=-1)
    return xq_out, xk_out
