"""Flat weight-key scheme for the T5 encoder kernel (mirrors weight_layout.py).

Per-layer keys are prefixed ``t5_b{layer}_``; shared weights use bare names.
All T5 linears are stored transposed (in, out) for ``x @ W`` (done in prep).
"""

T5_BLOCK_KEYS = ["ln0", "q", "k", "v", "o", "ln1", "wi0", "wi1", "wo"]
T5_SHARED_KEYS = ["t5_rel_bias", "t5_final_ln"]


def t5_block_key(layer_id, short):
    return f"t5_b{layer_id}_{short}"


def regroup_t5_weights(flat, num_layers):
    shared = {k: flat[k] for k in T5_SHARED_KEYS if k in flat}
    blocks = []
    for i in range(num_layers):
        blk = {}
        for short in T5_BLOCK_KEYS:
            key = t5_block_key(i, short)
            if key in flat:
                blk[short] = flat[key]
        blocks.append(blk)
    return shared, blocks
