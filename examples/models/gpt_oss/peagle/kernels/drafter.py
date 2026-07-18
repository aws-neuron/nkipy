"""P-EAGLE parallel drafter forward, fused into one KV-cached device kernel.

``drafter_fused_kernel`` runs the whole drafter forward for one speculation step
in a single Neuron kernel: on-device fc-fusion, ``[embed | hidden]`` assembly,
all layers (fusion midlayer + plain layers, each aliasing its own KV cache), a
final RMSNorm, and the draft ``lm_head``. It produces the per-position draft
logits; the caller argmaxes rows ``[C-1 .. W-1]`` (NTP + K-1 MTP) and applies the
``d2t`` remap.

Weights are passed as a flat, static signature (fusion-midlayer weights prefixed
``m_``, plain-layer weights stacked on a leading axis and prefixed ``p_``), keyed
exactly as produced by ``peagle/tensor_preparation.py``.
"""

import neuronxcc.nki.language as nl
import numpy as np
from nkipy.core import tensor_apis

from .drafter_layer import drafter_layer_cached
from .rmsnorm import rmsnorm_kernel
from .rope import compute_cos_sin_cache


def drafter_fused_kernel(
    embeds,  # (B, W, H): token embeddings for all W positions (committed + ptd)
    target_hidden3,  # (B, C, 3*target_H): raw 3-tap hiddens for the C committed rows
    start_pos,  # (1,) int32 absolute offset of row 0
    fc_weight,  # (3*target_H, H)
    mask_hidden,  # (1, 3*target_H): learnable shared MTP hidden (pre-fc)
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
    # per-layer KV caches (separate named tensors, aliased in+out like the base
    # model). This checkpoint has 4 layers (1 fusion + 3 plain).
    cache_k0,
    cache_v0,
    cache_k1,
    cache_v1,
    cache_k2,
    cache_v2,
    cache_k3,
    cache_v3,
    cfg,
):
    """Whole KV-cached drafter forward in ONE kernel: fc-fuse + input assembly +
    all layers + final norm + lm_head.

    Replaces the host-driven per-layer loop (4 launches + separate head + host
    fc-fuse/concat) with a single device call. The W-position input stream is
    ``[commit_0..commit_{C-1} | ptd_0..ptd_{K-2}]``:
      * committed rows: hidden = fc(target_hidden3)
      * MTP (ptd) rows: hidden = fc(mask_hidden), broadcast across depths
    ``embeds`` (host-gathered token embeddings) supplies the embedding half for all
    rows. Returns ``(logits, cache_k, cache_v)`` with the per-layer caches updated.
    """
    B, W, H = embeds.shape
    C = target_hidden3.shape[1]
    cache_k = [cache_k0, cache_k1, cache_k2, cache_k3]
    cache_v = [cache_v0, cache_v1, cache_v2, cache_v3]

    # fc-fuse on device: committed hiddens + shared MTP hidden.
    commit_hidden = np.matmul(target_hidden3, fc_weight)  # (B, C, H)
    if W > C:
        mask_fused = np.matmul(mask_hidden, fc_weight)  # (1, H)
        mtp = np.broadcast_to(mask_fused.reshape(1, 1, H), (B, W - C, H)).astype(
            commit_hidden.dtype
        )
        hiddens = np.concatenate([commit_hidden, mtp], axis=1)  # (B, W, H)
    else:
        hiddens = commit_hidden

    # Fusion midlayer consumes cat(embeds, hidden) of width 2H.
    x = np.concatenate([embeds, hiddens], axis=-1)  # (B, W, 2H)

    # RoPE tables, gathered to the W query positions ONCE and shared across all
    # layers (every layer uses identical rope params). start_pos=0 for prefill.
    max_seq_len = cache_k[0].shape[1]
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        cfg.rope_inv_freq, max_seq_len, cfg.rope_attention_scaling, dtype=nl.bfloat16
    )
    query_pos = start_pos + np.arange(W, dtype=np.int32)
    freqs_cos = tensor_apis.constant(freqs_cos)[query_pos]
    freqs_sin = tensor_apis.constant(freqs_sin)[query_pos]

    fusion_w = {
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
    x = drafter_layer_cached(
        x,
        fusion_w,
        cfg.norm_eps,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        freqs_cos,
        freqs_sin,
        cache_k[0],
        cache_v[0],
        start_pos,
        True,
    )

    num_plain = p_q_proj.shape[0]
    for i in range(num_plain):
        plain_w = {
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
        x = drafter_layer_cached(
            x,
            plain_w,
            cfg.norm_eps,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            freqs_cos,
            freqs_sin,
            cache_k[i + 1],
            cache_v[i + 1],
            start_pos,
            False,
        )

    x = rmsnorm_kernel(x, norm_weight, cfg.norm_eps)
    logits = np.matmul(x, lm_head_weight)  # (B, W, draft_vocab)
    return (
        logits,
        cache_k[0], cache_v[0],
        cache_k[1], cache_v[1],
        cache_k[2], cache_v[2],
        cache_k[3], cache_v[3],
    )
