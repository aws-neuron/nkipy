"""Canonical flat weight-key scheme shared by weight prep and the kernel.

The compiled ``dit_forward`` kernel takes all weights as flat ``**weights``
kwargs (the tracer only turns top-level ndarrays / expanded kwargs into HLO
parameters — it does not recurse into dicts or lists). Per-block weights are
prefixed ``b{layer}_``; shared weights use bare names. ``regroup_weights``
rebuilds the nested structure the block/forward code expects.
"""

# short per-block keys -> diffusers state-dict suffixes (filled by tensor_prep)
BLOCK_KEYS = [
    "sst",  # scale_shift_table (6, hidden)
    # self-attention
    "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "o_b",
    # cross-attention
    "cq_w", "cq_b", "ck_w", "ck_b", "cv_w", "cv_b", "co_w", "co_b",
    # feed-forward
    "ff0_w", "ff0_b", "ff2_w", "ff2_b",
]

# shared (non-block) weights
SHARED_KEYS = [
    "pos_embed.proj.weight", "pos_embed.proj.bias",
    "adaln.time_proj.weight", "adaln.time_proj.bias",
    "adaln.time_emb.weight", "adaln.time_emb.bias",
    "adaln.linear.weight", "adaln.linear.bias",
    "caption.w1", "caption.b1", "caption.w2", "caption.b2",
    "scale_shift_table", "proj_out.weight", "proj_out.bias",
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
