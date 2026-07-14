"""P-EAGLE parallel drafter forward: produce K draft-token logits in one pass.

Input construction (K positions = depths 0..K-1):
  * depth 0 (NTP): real fused target hidden (fc(cat of 3 tapped layers)) and the
    embedding of the last accepted token.
  * depth d>0 (MTP): the learnable shared hidden state `mask_hidden` (fused via
    `fc`) and the embedding of the placeholder token `ptd_token_id`.

The K positions run through the fusion midlayer then the plain layers under a
cross-depth causal mask (depth d attends to depths <= d). Each position's
post-norm hidden is projected by `lm_head` to a draft logit row; the argmax (then
`d2t` remap) gives that depth's draft token.

Weights are passed as a flat dict keyed exactly as produced by
``eagle/tensor_preparation.py``.
"""

import neuronxcc.nki.language as nl
import numpy as np
from nkipy.core import tensor_apis

from .drafter_layer import drafter_layer, drafter_layer_cached
from .rmsnorm import rmsnorm_kernel
from .rope import compute_cos_sin_cache


def _cross_depth_mask(K, dtype):
    """Additive (K, K) mask: depth d attends to depths <= d (lower-triangular)."""
    NEG = -100000.0
    return np.triu(np.ones((K, K)) * NEG, k=1).astype(dtype)


def drafter_forward(
    fused_hidden,  # (B, 1, hidden): fc-fused real target hidden for NTP position
    last_emb,  # (B, 1, hidden): embedding of last accepted token
    mask_hidden_fused,  # (1, hidden): fc(mask_hidden), shared MTP hidden
    ptd_emb,  # (1, hidden): embedding of ptd_token_id
    layer_weights,  # list[dict]: per-layer weight dicts (idx 0 = fusion midlayer)
    norm_weight,
    lm_head_weight,
    cfg,
):
    """Return draft logits of shape (B, K, draft_vocab_local)."""
    B = fused_hidden.shape[0]
    K = cfg.num_draft_tokens
    H = cfg.hidden_size

    # ── Build the K-position input stream: cat(embedding, hidden) per depth ──
    # depth 0 uses real hidden + last token embedding; depths>0 use the shared
    # mask hidden + placeholder embedding.
    emb_cols = [last_emb]
    hid_cols = [fused_hidden]
    if K > 1:
        ptd = np.broadcast_to(ptd_emb.reshape(1, 1, H), (B, K - 1, H))
        msk = np.broadcast_to(mask_hidden_fused.reshape(1, 1, H), (B, K - 1, H))
        emb_cols.append(ptd.astype(last_emb.dtype))
        hid_cols.append(msk.astype(fused_hidden.dtype))
    embeds = np.concatenate(emb_cols, axis=1)  # (B, K, H)
    hiddens = np.concatenate(hid_cols, axis=1)  # (B, K, H)

    # Fusion midlayer consumes cat(embeds, hidden) of width 2H.
    x = np.concatenate([embeds, hiddens], axis=-1)  # (B, K, 2H)

    # RoPE cache + cross-depth mask (compile-time constants).
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        cfg.rope_inv_freq,
        cfg.max_seq_len,
        cfg.rope_attention_scaling,
        dtype=nl.bfloat16,
    )
    freqs_cos = freqs_cos[0:K]
    freqs_sin = freqs_sin[0:K]
    attn_mask = tensor_apis.constant(_cross_depth_mask(K, nl.bfloat16))

    # ── Run the drafter layer stack ──
    for i, w in enumerate(layer_weights):
        is_fusion = i == 0
        x = drafter_layer(
            x,
            w,
            cfg.norm_eps,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            freqs_cos,
            freqs_sin,
            attn_mask,
            is_fusion,
        )

    # Final norm + draft lm_head.
    x = rmsnorm_kernel(x, norm_weight, cfg.norm_eps)
    logits = np.matmul(x, lm_head_weight)  # (B, K, draft_vocab_local)
    return logits


def drafter_layer_kernel(
    x,  # (B, S, 2H) for fusion, (B, S, H) for plain
    start_pos,  # (1,) int32 runtime offset, or None for prefill
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    input_weight,
    hidden_norm_weight,  # only used when is_fusion; pass a dummy otherwise
    post_attention_weight,
    gate_proj,
    up_proj,
    down_proj,
    cache_k,
    cache_v,
    cfg,
    is_fusion=False,
):
    """One KV-cached drafter layer (fusion midlayer or plain), device entry point.

    Aliases ``cache_k``/``cache_v`` (in + out): the S new positions are scattered
    into the cache at absolute positions ``start_pos + [0..S-1]`` (or ``0..S-1`` in
    prefill) and attend causally to the full cache. Same host-driven per-layer loop
    as the base model (``GptOssModel._run_layer``).
    """
    weights = {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "input_layernorm": input_weight,
        "post_attention_layernorm": post_attention_weight,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
    }
    if is_fusion:
        weights["hidden_norm"] = hidden_norm_weight

    out = drafter_layer_cached(
        x,
        weights,
        cfg.norm_eps,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.rope_inv_freq,
        cfg.rope_attention_scaling,
        cache_k,
        cache_v,
        start_pos,
        is_fusion,
    )
    return out, cache_k, cache_v


def drafter_head_kernel(h, norm_weight, lm_head_weight, cfg):
    """Final norm + draft lm_head over S positions -> logits (B, S, draft_vocab)."""
    h = rmsnorm_kernel(h, norm_weight, cfg.norm_eps)
    return np.matmul(h, lm_head_weight)


# Per-layer weight key suffixes, in the order the prep step emits them.
_FUSION_W = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "input_weight",
    "hidden_norm_weight",
    "post_attention_weight",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_PLAIN_W = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "input_weight",
    "post_attention_weight",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def drafter_kernel(
    target_hidden3,  # (B, 1, 3*target_hidden): the 3 tapped target layers, concatenated
    last_emb,  # (B, 1, hidden): embedding of last accepted token
    ptd_emb,  # (1, hidden): placeholder-token embedding
    fc_weight,  # (3*target_hidden, hidden)
    mask_hidden,  # (1, 3*target_hidden) learnable shared hidden (pre-fc)
    norm_weight,
    lm_head_weight,
    # fusion midlayer weights
    m_q_proj,
    m_k_proj,
    m_v_proj,
    m_o_proj,
    m_input_weight,
    m_hidden_norm_weight,
    m_post_attention_weight,
    m_gate_proj,
    m_up_proj,
    m_down_proj,
    # plain layer weights, stacked on a leading axis of size num_plain
    p_q_proj,
    p_k_proj,
    p_v_proj,
    p_o_proj,
    p_input_weight,
    p_post_attention_weight,
    p_gate_proj,
    p_up_proj,
    p_down_proj,
    cfg,
):
    """Device-traceable drafter entry point with a flat, static signature.

    Does the ``fc`` fusion (real hidden for NTP, ``mask_hidden`` for MTP) inside
    the kernel, then delegates to :func:`drafter_forward`. Plain-layer weights are
    stacked on a leading axis (one row per plain layer) and indexed statically.
    """
    fused_hidden = np.matmul(target_hidden3, fc_weight)  # (B, 1, hidden)
    mask_hidden_fused = np.matmul(mask_hidden, fc_weight)  # (1, hidden)

    midlayer = {
        "q_proj": m_q_proj,
        "k_proj": m_k_proj,
        "v_proj": m_v_proj,
        "o_proj": m_o_proj,
        "input_layernorm": m_input_weight,
        "hidden_norm": m_hidden_norm_weight,
        "post_attention_layernorm": m_post_attention_weight,
        "gate_proj": m_gate_proj,
        "up_proj": m_up_proj,
        "down_proj": m_down_proj,
    }
    layer_weights = [midlayer]
    num_plain = p_q_proj.shape[0]
    for i in range(num_plain):
        layer_weights.append(
            {
                "q_proj": p_q_proj[i],
                "k_proj": p_k_proj[i],
                "v_proj": p_v_proj[i],
                "o_proj": p_o_proj[i],
                "input_layernorm": p_input_weight[i],
                "post_attention_layernorm": p_post_attention_weight[i],
                "gate_proj": p_gate_proj[i],
                "up_proj": p_up_proj[i],
                "down_proj": p_down_proj[i],
            }
        )

    return drafter_forward(
        fused_hidden,
        last_emb,
        mask_hidden_fused,
        ptd_emb,
        layer_weights,
        norm_weight,
        lm_head_weight,
        cfg,
    )
