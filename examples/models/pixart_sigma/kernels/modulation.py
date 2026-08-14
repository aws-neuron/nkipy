"""adaLN-single modulation helpers for PixArt DiT blocks.

Each block owns a learnable ``scale_shift_table`` of shape (6, hidden). Adding
the broadcast shared ``timestep`` embedding (B, 6*hidden) and splitting by 6
yields the per-block (shift/scale/gate) x (msa/mlp) modulation parameters.

This mirrors diffusers' BasicTransformerBlock in ``ada_norm_single`` mode:

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        scale_shift_table[None] + timestep.reshape(B, 6, -1)
    ).chunk(6, dim=1)
"""

import numpy as np


def block_modulation(scale_shift_table, timestep, hidden_size):
    """Produce the six modulation tensors for one DiT block.

    Args:
        scale_shift_table: (6, hidden) per-block learnable table.
        timestep: (B, 6*hidden) shared adaLN projection.
        hidden_size: int.

    Returns:
        Six tensors each (B, 1, hidden): shift_msa, scale_msa, gate_msa,
        shift_mlp, scale_mlp, gate_mlp — shaped to broadcast over the token axis.
    """
    B = timestep.shape[0]
    table = np.expand_dims(scale_shift_table, axis=0)  # (1, 6, hidden)
    mod = table + timestep.reshape(B, 6, hidden_size)  # (B, 6, hidden)

    parts = np.split(mod, 6, axis=1)  # each (B, 1, hidden)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def modulate(x, shift, scale):
    """Apply adaLN modulation: x * (1 + scale) + shift.

    ``x`` is (B, L, hidden); ``shift``/``scale`` are (B, 1, hidden) and
    broadcast over the token axis.
    """
    return x * (1 + scale) + shift


def final_modulation(scale_shift_table, embedded_timestep, hidden_size):
    """Produce (shift, scale) for the final output layer.

    Mirrors PixArtTransformer2DModel.forward:
        shift, scale = (scale_shift_table[None] + embedded_timestep[:, None]).chunk(2)

    Args:
        scale_shift_table: (2, hidden) learnable table for the final layer.
        embedded_timestep: (B, hidden).

    Returns:
        (shift, scale) each (B, 1, hidden).
    """
    B = embedded_timestep.shape[0]
    table = np.expand_dims(scale_shift_table, axis=0)  # (1, 2, hidden)
    mod = table + np.expand_dims(embedded_timestep, axis=1)  # (B, 2, hidden)
    shift, scale = np.split(mod, 2, axis=1)  # each (B, 1, hidden)
    return shift, scale
