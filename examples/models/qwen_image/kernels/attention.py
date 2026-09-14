"""Joint (double-stream) attention for the Qwen-Image MMDiT.

Mirrors diffusers ``QwenDoubleStreamAttnProcessor2_0``. Qwen-Image runs **one**
attention over the concatenation of the text and image tokens (rather than
separate self- and cross-attention):

    1. project image tokens with to_q/to_k/to_v, text tokens with
       add_q_proj/add_k_proj/add_v_proj;
    2. reshape to heads, apply per-head QK-RMSNorm (norm_q/norm_k for image,
       norm_added_q/norm_added_k for text);
    3. apply 3D RoPE to q/k of both streams;
    4. concat in order [text, image] along the sequence axis;
    5. scaled-dot-product attention (bidirectional, no causal mask); an optional
       key-padding mask masks text padding tokens;
    6. split back, project image out via to_out, text out via to_add_out.

The score matrix is O((Ltext+Limg)^2); the image tokens dominate. A tuned
device backend can be swapped in later — this hand-rolled softmax is the
portable, CPU-testable reference.
"""

import numpy as np

from .rmsnorm import rmsnorm_kernel
from .rope3d import apply_rotary_emb
from .softmax import softmax_kernel


def _to_heads(x, n_heads, head_dim):
    """(B, L, n_heads*head_dim) -> (B, n_heads, L, head_dim)."""
    B, L, _ = x.shape
    return x.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)


def _from_heads(x):
    """(B, n_heads, L, head_dim) -> (B, L, n_heads*head_dim)."""
    B, H, L, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, H * d)


def joint_attention_kernel(
    img, txt,
    # image-stream projections
    iq_w, iq_b, ik_w, ik_b, iv_w, iv_b, io_w, io_b, iq_g, ik_g,
    # text-stream projections
    tq_w, tq_b, tk_w, tk_b, tv_w, tv_b, to_w, to_b, tq_g, tk_g,
    n_heads, head_dim, eps,
    img_cos, img_sin, txt_cos, txt_sin,
    txt_mask_bias=None, local_heads=None, all_reduce_fn=None,
):
    """Joint attention over concat([text, image]).

    Args:
        img: (B, Limg, hidden) modulated image tokens.
        txt: (B, Ltext, hidden) modulated text tokens.
        *_w / *_b: projection weights / biases. Under tensor parallelism these
            are the *sharded* slices: q/k/v are (hidden, local_heads*head_dim),
            o is (local_heads*head_dim, hidden), and the o/to biases are the full
            replicated (hidden,).
        iq_g/ik_g/tq_g/tk_g: QK-RMSNorm gains (head_dim,).
        img_cos/img_sin: (Limg, head_dim) RoPE tables for image tokens.
        txt_cos/txt_sin: (Ltext, head_dim) RoPE tables for text tokens.
        txt_mask_bias: optional (B, 1, 1, Ltext) additive bias for text padding
            (applied only over the text key positions).
        local_heads: heads owned by this rank under tensor parallelism.
        all_reduce_fn: callable summing the output projections across ranks
            (row-parallel reduction); always applied (TP is required).

    Returns:
        (img_out, txt_out): (B, Limg, hidden), (B, Ltext, hidden).
    """
    Ltext = txt.shape[1]
    h = n_heads if local_heads is None else local_heads

    # projections (head axis may be sharded -> ``h`` local heads)
    iq = _to_heads(np.matmul(img, iq_w) + iq_b, h, head_dim)
    ik = _to_heads(np.matmul(img, ik_w) + ik_b, h, head_dim)
    iv = _to_heads(np.matmul(img, iv_w) + iv_b, h, head_dim)
    tq = _to_heads(np.matmul(txt, tq_w) + tq_b, h, head_dim)
    tk = _to_heads(np.matmul(txt, tk_w) + tk_b, h, head_dim)
    tv = _to_heads(np.matmul(txt, tv_w) + tv_b, h, head_dim)

    # QK-RMSNorm over head_dim (per-head)
    iq = rmsnorm_kernel(iq, iq_g, eps=eps)
    ik = rmsnorm_kernel(ik, ik_g, eps=eps)
    tq = rmsnorm_kernel(tq, tq_g, eps=eps)
    tk = rmsnorm_kernel(tk, tk_g, eps=eps)

    # 3D RoPE
    iq = apply_rotary_emb(iq, img_cos, img_sin)
    ik = apply_rotary_emb(ik, img_cos, img_sin)
    tq = apply_rotary_emb(tq, txt_cos, txt_sin)
    tk = apply_rotary_emb(tk, txt_cos, txt_sin)

    # concat [text, image] along sequence
    q = np.concatenate([tq, iq], axis=2)
    k = np.concatenate([tk, ik], axis=2)
    v = np.concatenate([tv, iv], axis=2)

    # scaled dot-product attention (bidirectional)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)).astype(np.float32)
    scores = scores / np.float32(np.sqrt(head_dim))
    if txt_mask_bias is not None:
        # bias applies over the text key positions only; image keys unmasked.
        # Build the joint-key bias by concatenating the text bias with a zero
        # image-key block (trace-friendly: no in-place slice assignment).
        B = scores.shape[0]
        Limg = scores.shape[-1] - Ltext
        txt_b = np.broadcast_to(txt_mask_bias.astype(np.float32), (B, 1, 1, Ltext))
        img_b = np.zeros((B, 1, 1, Limg), dtype=np.float32)
        bias = np.concatenate([txt_b, img_b], axis=-1)
        scores = scores + bias
    weights = softmax_kernel(scores).astype(v.dtype)
    out = np.matmul(weights, v)  # (B, H, Ljoint, head_dim)

    joint = _from_heads(out)  # (B, Ljoint, local_heads*head_dim)
    txt_out = joint[:, :Ltext, :]
    img_out = joint[:, Ltext:, :]

    # row-parallel output projection: matmul the local slice, all-reduce the
    # partial sums, then add the (replicated) bias exactly once.
    img_out = np.matmul(img_out, io_w)
    txt_out = np.matmul(txt_out, to_w)
    img_out = all_reduce_fn(img_out)
    txt_out = all_reduce_fn(txt_out)
    if io_b is not None:
        img_out = img_out + io_b
    if to_b is not None:
        txt_out = txt_out + to_b
    return img_out, txt_out
