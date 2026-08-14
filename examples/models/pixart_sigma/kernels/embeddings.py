"""Input embeddings for the PixArt-Sigma DiT.

Three pieces run before the transformer blocks:

1. ``patch_embed`` — turn the (B, C, H, W) VAE latent into a sequence of patch
   tokens via a patch_size x patch_size strided projection, then add a fixed 2D
   sin-cos positional embedding.
2. ``timestep_embedding`` + ``adaln_single`` — sinusoidal embedding of the
   diffusion timestep, an MLP to ``hidden_size`` (``embedded_timestep``), and a
   second projection to ``6 * hidden_size`` consumed by every block's adaLN.
3. ``caption_projection`` — project the (host-computed) T5 text embeddings from
   ``caption_channels`` to ``hidden_size`` for cross-attention.

The positional and sinusoidal tables are plain numpy computed from constant
arguments, so on Trainium they bake into the graph as comptime constants (the
same pattern the qwen3 attention kernel uses for its RoPE / causal-mask
tables).
"""

import numpy as np


# ── positional embedding (comptime constant) ──────────────────────────────


def _get_1d_sincos_pos_embed(embed_dim, pos):
    """1D sin-cos embedding. ``pos`` is a 1D array of positions -> (len(pos), embed_dim).

    Matches diffusers ``get_1d_sincos_pos_embed_from_grid_np``: sin then cos.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    out = np.einsum("m,d->md", pos.reshape(-1), omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size=None, interpolation_scale=1.0):
    """2D sin-cos positional embedding matching diffusers' PatchEmbed.

    Returns (grid_size*grid_size, embed_dim). ``base_size``/``interpolation_scale``
    reproduce PixArt-Sigma's resolution-dependent grid scaling (for the 1024px
    model, base_size=64 and interpolation_scale=2). Layout follows diffusers
    exactly: meshgrid(w, h) with w first, then emb = concat(emb over grid[0],
    emb over grid[1]).
    """
    if base_size is None:
        base_size = grid_size
    grid_h = np.arange(grid_size, dtype=np.float64) / (grid_size / base_size) / interpolation_scale
    grid_w = np.arange(grid_size, dtype=np.float64) / (grid_size / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # w first (diffusers convention)
    grid = np.stack(grid, axis=0)  # (2, gs, gs)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (gs*gs, D/2)
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (gs*gs, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (gs*gs, D)


# ── patch embedding ────────────────────────────────────────────────────────


def patch_embed_kernel(latent, proj_weight, proj_bias, patch_size, pos_embed):
    """Patchify a latent and project each patch to ``hidden_size``.

    Args:
        latent: (B, C, H, W) VAE latent.
        proj_weight: conv weight (hidden_size, C, patch_size, patch_size); a
            patch_size-strided conv with no overlap is exactly a linear map over
            the flattened patch, so we reshape it to (C*patch_size*patch_size,
            hidden_size) and matmul.
        proj_bias: (hidden_size,).
        patch_size: int.
        pos_embed: (num_patches, hidden_size) comptime constant from
            ``get_2d_sincos_pos_embed``.

    Returns:
        (B, num_patches, hidden_size) tokens with positional embedding added.
    """
    B, C, H, W = latent.shape
    p = patch_size
    gh, gw = H // p, W // p

    # (B, C, gh, p, gw, p) -> (B, gh, gw, C, p, p) -> (B, gh*gw, C*p*p)
    x = latent.reshape(B, C, gh, p, gw, p)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, gh * gw, C * p * p)

    # conv weight (hidden, C, p, p) -> (C*p*p, hidden)
    hidden = proj_weight.shape[0]
    w = proj_weight.reshape(hidden, C * p * p).transpose(1, 0)

    tokens = np.matmul(x, w) + proj_bias
    tokens = tokens + np.expand_dims(pos_embed.astype(tokens.dtype), axis=0)
    return tokens


# ── timestep embedding + adaLN-single ────────────────────────────────────────


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding matching diffusers' get_timestep_embedding.

    ``timesteps`` is a runtime (B,) tensor; the frequency table is a comptime
    numpy constant. diffusers uses (cos, sin) ordering (flip_sin_to_cos=True)
    with no downscale for PixArt.

    Returns (B, dim).
    """
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)  # (half,)
    args = np.expand_dims(timesteps.astype(np.float32), axis=-1) * np.expand_dims(freqs, axis=0)
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros_like(emb[:, :1])], axis=-1)
    return emb


def _silu(x):
    return x * (1 / (1 + np.exp(-x)))


def adaln_single_kernel(
    timesteps,
    time_proj_weight,
    time_proj_bias,
    time_emb_weight,
    time_emb_bias,
    linear_weight,
    linear_bias,
    hidden_size,
    dtype,
):
    """AdaLayerNormSingle: shared timestep conditioning for every block.

    Mirrors diffusers ``AdaLayerNormSingle``:
        emb = timestep_proj(timesteps)                     # sinusoidal -> hidden
        embedded_timestep = MLP(emb)                       # (B, hidden)
        timestep = linear(silu(embedded_timestep))         # (B, 6*hidden)

    The ``emb`` MLP here is the ``timestep_embedder`` (Linear-SiLU-Linear) of
    ``PixArtAlphaCombinedTimestepSizeEmbeddings``; PixArt-Sigma at fixed
    resolution has no additional (resolution / aspect-ratio) conditions, so
    those are omitted.

    Returns:
        (timestep, embedded_timestep) with shapes (B, 6*hidden) and (B, hidden).
    """
    # sinusoidal projection dimension is the input width of time_proj_weight
    proj_dim = time_proj_weight.shape[0]
    sin_emb = timestep_embedding(timesteps, proj_dim).astype(dtype)

    # timestep_embedder MLP: Linear -> SiLU -> Linear  -> (B, hidden)
    h = np.matmul(sin_emb, time_proj_weight) + time_proj_bias
    h = _silu(h)
    embedded_timestep = np.matmul(h, time_emb_weight) + time_emb_bias  # (B, hidden)

    # adaLN linear: SiLU then project to 6*hidden
    timestep = np.matmul(_silu(embedded_timestep), linear_weight) + linear_bias
    return timestep, embedded_timestep


# ── caption (T5) projection ──────────────────────────────────────────────────


def caption_projection_kernel(caption, w1, b1, w2, b2):
    """Project T5 caption embeddings to hidden_size (Linear-GELU-Linear).

    diffusers' ``PixArtAlphaTextProjection`` uses a GELU(tanh) activation.

    Args:
        caption: (B, num_tokens, caption_channels).
    Returns:
        (B, num_tokens, hidden_size).
    """
    x = np.matmul(caption, w1) + b1
    x = _gelu_tanh(x)
    x = np.matmul(x, w2) + b2
    return x


def _gelu_tanh(x):
    xf = x.astype(np.float32)
    inner = np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf * xf * xf)
    out = 0.5 * xf * (1.0 + np.tanh(inner))
    return out.astype(x.dtype)
