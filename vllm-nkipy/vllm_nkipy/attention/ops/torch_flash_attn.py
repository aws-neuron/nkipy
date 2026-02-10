# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def neuron_paged_attn_torch(
    query: torch.Tensor,  # (B, n_q, Lq, d)
    key: torch.Tensor,  # (B, n_kv, Lactive, d)
    value: torch.Tensor,  # (B, n_kv, Lactive, d)
    kv_cache: torch.Tensor,  # (2, n_kv, NB, BS, d)  <-- head-major
    active_block_table: torch.Tensor,  # (Nactive,) padded
    attn_mask: torch.Tensor,  # bool mask with [prior | active] columns
) -> torch.Tensor:
    """
    Torch implementation of paged attention for accuracy testing.
    Uses SDPA (math backend) and the same 'active block table' prior-context
    layout as the NKI kernel.

    - Prior keys/values are gathered via `active_block_table` from kv_cache.
    - Active keys/values come from `key`/`value` inputs.
    - `attn_mask` is the concatenation [prior_mask | active_mask] already built
      by the model runner; we slice it to the actual (Lq x Lk_total) window.
    """
    B, n_q, Lq, d = query.shape
    _, n_kv, Lactive, _ = key.shape

    # kv_cache: (2, n_kv, NB, BS, d)
    k_cache = kv_cache[0]  # (n_kv, NB, BS, d)
    v_cache = kv_cache[1]

    # Gather prior blocks per head and flatten to sequences
    k_prior = k_cache.index_select(1, active_block_table)  # (n_kv, Nactive, BS, d)
    v_prior = v_cache.index_select(1, active_block_table)

    # Flatten blocks*slots → (B, n_kv, Lprior, d)
    _, n_act, bs, _ = k_prior.shape
    Lprior = n_act * bs
    k_prior = k_prior.reshape(1, n_kv, Lprior, d)
    v_prior = v_prior.reshape(1, n_kv, Lprior, d)

    # ---- Concatenate prior + active along sequence length (K/V length) ----
    k_combined = torch.cat([k_prior.expand(B, -1, -1, -1), key], dim=2)
    v_combined = torch.cat([v_prior.expand(B, -1, -1, -1), value], dim=2)
    Lk_total = k_combined.shape[2]

    # ---- Build SDPA bias from provided boolean mask ----
    attn_bias = None
    # Slice to actual window (Lq x Lk_total); mask is True => allowed
    combined_mask = attn_mask[:Lq, :Lk_total]
    # Convert to 0 (keep) / -inf (mask) and cast to q.dtype
    attn_bias = torch.where(combined_mask, 0.0, float("-inf")).to(query.dtype)

    # ---- SDPA (math backend), with GQA enabled ----
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = torch.nn.functional.scaled_dot_product_attention(
            query,  # (B, n_q, Lq, d)
            k_combined,  # (B, n_kv, Lk_total, d)
            v_combined,  # (B, n_kv, Lk_total, d)
            attn_mask=attn_bias,  # broadcastable to (B, n_q, Lq, Lk_total)
            dropout_p=0.0,
            is_causal=False,  # explicit mask provided
            enable_gqa=True,
        )

    return out


def reshape_and_cache_torch(
    key: torch.Tensor,  # (T, n_kv, d)
    value: torch.Tensor,  # (T, n_kv, d)
    kv_cache: torch.Tensor,  # (2, n_kv, NB, BS, d)
    slot_mapping: torch.Tensor,  # (T,)
) -> None:
    # Flatten the (NB, BS) dimension to a single slot axis per head.
    k_cache = kv_cache[0].reshape(
        kv_cache.size(1), -1, kv_cache.size(-1)
    )  # (n_kv, S, d)
    v_cache = kv_cache[1].reshape(
        kv_cache.size(1), -1, kv_cache.size(-1)
    )  # (n_kv, S, d)

    # Move tokens to per-head batch: (n_kv, T, d)
    key_t = key.transpose(0, 1).contiguous()
    value_t = value.transpose(0, 1).contiguous()

    # Same slot_mapping for every head; write along dim=1 (slots)
    k_cache.index_copy_(1, slot_mapping, key_t)
    v_cache.index_copy_(1, slot_mapping, value_t)
