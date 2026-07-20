import numpy as np


def compute_cos_sin_cache(
    inv_freq, max_seq_len, attention_scaling=1.0, dtype=np.float32
):
    """Compute cosine/sine cache for RoPE from precomputed inverse frequencies.

    gpt-oss uses YaRN scaling, so `inv_freq` (length head_dim/2) and the
    `attention_scaling` post-multiplier are computed once on the host from the HF
    config and passed in here.

    Comptime: this runs on constant arguments only, so under Trainium compilation
    the resulting arrays bake into the HLO graph as constants. On CPU it is plain
    numpy.
    """
    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)

    cos = (np.cos(freqs, dtype=np.float32) * attention_scaling).astype(dtype)
    sin = (np.sin(freqs, dtype=np.float32) * attention_scaling).astype(dtype)
    return cos, sin


def apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin):
    """Apply rotary position embedding to query and key tensors.

    Matches gpt-oss `_apply_rotary_emb`: split each vector into first/second
    halves and rotate, without interleaving.
    """
    # Reshape `freqs_cos` and `freqs_sin` for broadcasting over (B, S, H, D/2).
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
    xq_out_1 = xq1 * freqs_cos + xq0 * freqs_sin

    xk_out_0 = xk0 * freqs_cos - xk1 * freqs_sin
    xk_out_1 = xk1 * freqs_cos + xk0 * freqs_sin

    # Concatenate the results back together to form the final output
    xq_out = np.concatenate([xq_out_0, xq_out_1], axis=-1)
    xk_out = np.concatenate([xk_out_0, xk_out_1], axis=-1)

    return xq_out, xk_out
