# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def torch_ragged_paged_attention(
    query: torch.Tensor,  # (Nq, n_q*d) or (Nq, n_q, d)  -- no batch dim
    kv_cache: torch.Tensor,  # (2, n_kv, NB, BS, d)         -- [K,V]
    context_lens: torch.Tensor,  # (B,) int64
    block_tables: torch.Tensor,  # (B, T_max) int64  (-1 padded)
    query_start_loc: torch.Tensor,  # (B+1,) int64  prefix sums; last == Nq
    # num_seqs: torch.Tensor,   # kept for signature parity
):
    # ---- normalize query to (Nq, Hq, 1, d) ----
    if query.dim() == 2:
        Nq, last = query.shape
        d = kv_cache.shape[-1]
        assert last % d == 0, "query last dim must be n_q * d"
        n_q = last // d
        q4 = query.view(Nq, n_q, 1, d)
        out_flatten = True
    elif query.dim() == 3:
        Nq, n_q, d = query.shape
        q4 = query.unsqueeze(2)  # (Nq, n_q, 1, d)
        out_flatten = False
    else:
        raise ValueError("query must be 2D (Nq, n_q*d) or 3D (Nq, n_q, d)")

    # ---- unpack KV: (2, n_kv, NB, BS, d) -> (n_kv, NB*BS, d) ----
    assert kv_cache.dim() == 5 and kv_cache.size(0) == 2, (
        "kv_cache must be (2, n_kv, NB, BS, d)"
    )
    K_cache, V_cache = kv_cache[0], kv_cache[1]
    n_kv, NB, BS, d2 = K_cache.shape
    assert d2 == d

    B, T_max = block_tables.shape
    Lk_pad = T_max * BS
    device = query.device

    K_flat = K_cache.reshape(n_kv, NB * BS, d)
    V_flat = V_cache.reshape(n_kv, NB * BS, d)

    # ---- block_tables -> flat token indices ----
    within = torch.arange(BS, device=device)
    idx = block_tables[..., None] * BS + within.view(1, 1, BS)  # (B, T_max, BS)
    block_valid = block_tables >= 0
    idx = torch.where(block_valid[..., None], idx, idx.new_zeros(()))
    idx2 = idx.reshape(B, Lk_pad)
    gather_idx = idx2[:, None, :, None]  # (B,1,Lk_pad,1)

    # ---- gather K/V for all seqs once (keep Kv heads!) ----
    K_g0 = torch.take_along_dim(
        K_flat.unsqueeze(0).expand(B, -1, -1, -1),
        gather_idx.expand(B, n_kv, Lk_pad, d),
        dim=2,
    )  # (B, n_kv, Lk_pad, d)
    V_g0 = torch.take_along_dim(
        V_flat.unsqueeze(0).expand(B, -1, -1, -1),
        gather_idx.expand(B, n_kv, Lk_pad, d),
        dim=2,
    )  # (B, n_kv, Lk_pad, d)

    # ---- per-seq padding mask -> per-token ----
    valid_from_blocks = block_valid[:, :, None].expand(B, T_max, BS).reshape(B, Lk_pad)
    tok_id = torch.arange(Lk_pad, device=device, dtype=context_lens.dtype)[None, :]
    valid_from_len = tok_id < context_lens[:, None]
    pad_mask_seq = ~(valid_from_blocks & valid_from_len)  # (B, Lk_pad), True = mask

    # ---- map each query token to its sequence WITHOUT bucketize ----
    # boundaries: (B,) = query_start_loc[1:]
    # For t in [0..Nq-1], seq_id[t] = count(boundaries <= t)
    Nq = q4.size(0)
    boundaries = query_start_loc[1:]  # (B,)
    token_ids = torch.arange(Nq, device=query.device, dtype=boundaries.dtype)  # (Nq,)
    q_seq_ids = (token_ids[:, None] >= boundaries[None, :]).sum(dim=1)  # (Nq,) int64

    # ---- pick per-token K/V and mask; preserve head dims for GQA ----
    K_tok = K_g0.index_select(0, q_seq_ids)  # (Nq, n_kv, Lk_pad, d)
    V_tok = V_g0.index_select(0, q_seq_ids)  # (Nq, n_kv, Lk_pad, d)
    m_tok = (
        pad_mask_seq.index_select(0, q_seq_ids).unsqueeze(1).unsqueeze(1)
    )  # (Nq,1,1,Lk_pad)

    # ---- SDPA (Math backend + GQA) ----
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q4, K_tok, V_tok, attn_mask=m_tok, is_causal=False, enable_gqa=True
        )  # (Nq, n_q, 1, d)

    out = out.squeeze(2)  # (Nq, n_q, d)
    return out.reshape(Nq, n_q * d) if out_flatten else out
