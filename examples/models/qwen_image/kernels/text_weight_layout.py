"""Flat weight-key scheme for the Qwen2.5 text encoder.

Per-layer keys are prefixed ``l{i}.``; shared (embedding-side) weights use bare
names. As elsewhere, ``nn.Linear`` weights are transposed (out,in)->(in,out) for
the ``x @ W`` convention; RMSNorm gains and biases pass through.

The token-embedding table and LM head are **not** here — the host does the
embedding lookup (huge table, data-dependent gather) and there is no LM head in
encoder mode.
"""

# per-layer short keys
LAYER_KEYS = [
    "attn_norm", "mlp_norm",
    "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w",
    "gate_w", "up_w", "down_w",
]

SHARED_KEYS = ["final_norm"]


def layer_key(i, short):
    return f"l{i}.{short}"


def present_keys(flat, num_layers):
    keys = [k for k in SHARED_KEYS if k in flat]
    for i in range(num_layers):
        keys += [layer_key(i, s) for s in LAYER_KEYS if layer_key(i, s) in flat]
    return keys
