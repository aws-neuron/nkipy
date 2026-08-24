import math
from typing import Optional

import nkipy.core.typing as nt
import numpy as np
from config import Config


def apply_rotary_emb(
    x: nt.tensor,
    cos: nt.tensor,
    sin: nt.tensor,
):
    """Apply rotary position embedding to query and key tensors.

    Args:
        x: SBD
        cos: SD
        sin: SD
    """
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    x1, x2 = np.split(x, 2, axis=-1)

    # Apply rotary embedding between first and second halves
    o0 = x1 * cos - x2 * sin
    o1 = x1 * sin + x2 * cos

    # Concatenate the results back together to form the final output
    return np.concatenate([o0, o1], axis=-1)

def compute_cos_sin(
    max_model_len,
):
    concentration, inv_freq = compute_concentration_and_inv_freq(
        head_dim=Config.head_dim,
        base=Config.rope_theta,
        initial_context_length=Config.initial_context_length,
        scaling_factor=Config.rope_scaling_factor,
        ntk_alpha=Config.rope_ntk_alpha,
        ntk_beta=Config.rope_ntk_beta,
    )
    t = np.arange(max_model_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    cos = np.cos(freqs) * concentration
    sin = np.sin(freqs) * concentration
    return cos, sin


def compute_concentration_and_inv_freq(
    head_dim,
    base,
    initial_context_length,
    scaling_factor,
    ntk_alpha,
    ntk_beta,
):
    """Compute concentration factor and inverse frequencies for RoPE.
    See YaRN paper: https://arxiv.org/abs/2309.00071"""

    freq = (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)).astype(np.float32)

    if scaling_factor > 1.0:
        concentration = (
            0.1 * math.log(scaling_factor) + 1.0
        )  # YaRN concentration

        d_half = head_dim / 2
        # NTK by parts
        low = (
                d_half
                * math.log(initial_context_length / (ntk_beta * 2 * math.pi))
                / math.log(base)
            )
        high = (
            d_half
            * math.log(initial_context_length / (ntk_alpha * 2 * math.pi))
            / math.log(base)
        )
        assert 0 < low < high < d_half - 1

        interpolation = 1.0 / (scaling_factor * freq)
        extrapolation = 1.0 / freq

        ramp = (
            np.arange(d_half, dtype=np.float32) - low
        ) / (high - low)
        mask = 1 - np.clip(ramp, 0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq
    
    return concentration, inv_freq


def rope_yarn(
    query: nt.tensor,
    key: nt.tensor,
    cos: nt.tensor,
    sin: nt.tensor,
    start_pos: Optional[nt.tensor],
):
    """Apply yarn-style RoPE to query and key tensors (numpy implementation).
    
    Args:
        query, key: SB*D
        sin, cos: SD
    """
    if start_pos is None:
        # prefill
        _, num_tokens, _, head_dim = query.shape
        query = query.transpose(1, 0, 2, 3)  # BSHD -> SBHD
        key = key.transpose(1, 0, 2, 3)  # BSHD -> SBHD

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, head_dim)
        if num_tokens > 1:
            # when seq_len == 1, below has no effect but compiler error
            cos = cos[:num_tokens]
            sin = sin[:num_tokens]
        cos = np.expand_dims(cos, axis=-2)
        sin = np.expand_dims(sin, axis=-2)
        query = apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)
        
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, head_dim)
        key = apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)

        query = query.transpose(1, 0, 2, 3)  # SBHD -> BSHD
        key = key.transpose(1, 0, 2, 3)  # SBHD -> BSHD
    else:
        # decode
        batch_size, _, _, head_dim = query.shape
        cos = np.expand_dims(cos[start_pos], axis=-2)
        sin = np.expand_dims(sin[start_pos], axis=-2)
        query_shape = query.shape
        query = query.reshape(batch_size, -1, head_dim)
        query = apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)
        key_shape = key.shape
        key = key.reshape(batch_size, -1, head_dim)
        key = apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)

    return query, key