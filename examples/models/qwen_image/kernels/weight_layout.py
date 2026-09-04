"""Canonical flat weight-key scheme shared by weight prep and the kernel.

The compiled ``qwenimage_forward`` kernel takes all weights as flat ``**weights``
kwargs (the tracer only turns top-level ndarrays into HLO parameters — it does
not recurse into dicts). Per-block weights are prefixed ``b{layer}_``; shared
weights use bare names. ``regroup_weights`` rebuilds the nested structure.

Each MMDiT block is dual-stream: an image stream (``img_*``) and a text stream
(``txt_*``). Both feed one joint attention. Naming maps to the diffusers
``QwenImageTransformerBlock`` submodules:

    img_mod / txt_mod  -> SiLU-Linear(dim, 6*dim) modulation  (weight+bias)
    img: to_q/to_k/to_v/to_out  + norm_q/norm_k (QK-RMSNorm)
    txt: add_q_proj/add_k_proj/add_v_proj/to_add_out + norm_added_q/norm_added_k
    img_mlp / txt_mlp  -> FeedForward gelu-approximate (fc1/fc2 weight+bias)

The layout is grouped by stream so a later tensor-parallel split (shard heads +
MLP intermediate) can slice per-stream keys uniformly.
"""

# short per-block keys -> filled by tensor_prep from the diffusers state dict.
# ``*_w`` weights, ``*_b`` biases, ``*_g`` RMSNorm gains.
BLOCK_KEYS = [
    # modulation (SiLU -> Linear to 6*hidden), one per stream
    "img_mod_w", "img_mod_b", "txt_mod_w", "txt_mod_b",
    # image-stream attention projections
    "iq_w", "iq_b", "ik_w", "ik_b", "iv_w", "iv_b", "io_w", "io_b",
    "iq_g", "ik_g",  # QK-RMSNorm gains (norm_q / norm_k)
    # text-stream attention projections
    "tq_w", "tq_b", "tk_w", "tk_b", "tv_w", "tv_b", "to_w", "to_b",
    "tq_g", "tk_g",  # QK-RMSNorm gains (norm_added_q / norm_added_k)
    # image-stream MLP (gelu-approximate)
    "iff0_w", "iff0_b", "iff2_w", "iff2_b",
    # text-stream MLP
    "tff0_w", "tff0_b", "tff2_w", "tff2_b",
]

# shared (non-block) weights
SHARED_KEYS = [
    "img_in.weight", "img_in.bias",
    "txt_norm.weight",
    "txt_in.weight", "txt_in.bias",
    "time.proj.weight", "time.proj.bias",     # TimestepEmbedding fc1
    "time.emb.weight", "time.emb.bias",        # TimestepEmbedding fc2
    "norm_out.linear.weight", "norm_out.linear.bias",  # AdaLayerNormContinuous
    "proj_out.weight", "proj_out.bias",
]


def block_key(layer_id, short_name):
    return f"b{layer_id}_{short_name}"


def regroup_weights(flat, num_layers):
    """Split a flat ``{key: tensor}`` dict into (shared, blocks).

    Returns:
        shared: dict of SHARED_KEYS -> tensor
        blocks: list (len num_layers) of dict short_name -> tensor
    """
    shared = {k: flat[k] for k in SHARED_KEYS if k in flat}
    blocks = []
    for layer_id in range(num_layers):
        blk = {}
        for short in BLOCK_KEYS:
            key = block_key(layer_id, short)
            if key in flat:
                blk[short] = flat[key]
        blocks.append(blk)
    return shared, blocks
