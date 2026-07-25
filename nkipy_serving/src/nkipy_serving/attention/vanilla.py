"""Vanilla (numpy) attention backend: KV cache update + scaled dot-product attention."""

from __future__ import annotations

import numpy as np

from nkipy_serving.attention.base import (
    FORWARD_MODE_EXTEND,
    AttentionMetadata,
)


def vanilla_update_kv_cache(
    k: np.ndarray,
    v: np.ndarray,
    kv_cache: np.ndarray,
    slot_mapping: np.ndarray,
) -> None:
    """Write K/V into block-based cache at positions specified by slot_mapping.

    Args:
        k: [total_tokens, num_kv_heads, head_dim]
        v: [total_tokens, num_kv_heads, head_dim]
        kv_cache: [2, num_blocks, num_kv_heads, block_size, head_dim]
        slot_mapping: [total_tokens] → flat cache slot indices
    """
    block_size = kv_cache.shape[3]
    slot_mapping = np.asarray(slot_mapping, dtype=np.int64).reshape(-1)
    block_ids = slot_mapping // block_size
    offsets = slot_mapping % block_size
    num_kv_heads = k.shape[1]
    for h in range(num_kv_heads):
        kv_cache[0, block_ids, h, offsets, :] = k[:, h, :]
        kv_cache[1, block_ids, h, offsets, :] = v[:, h, :]


def vanilla_attention_core(
    q: np.ndarray,
    kv_cache: np.ndarray,
    metadata: AttentionMetadata,
) -> np.ndarray:
    """Scaled dot-product attention with block-based KV cache and page-table indirection.

    Args:
        q: [total_tokens, num_heads, head_dim]
        kv_cache: [2, num_blocks, num_kv_heads, block_size, head_dim]
        metadata: AttentionMetadata with block_tables, seq_lens, etc.

    Returns:
        output: [total_tokens, num_heads, head_dim]
    """
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = metadata.num_kv_heads
    block_size = metadata.block_size
    batch_size = metadata.batch_size
    seq_lens = np.asarray(metadata.seq_lens, dtype=np.int64).reshape(-1)
    block_tables = np.asarray(metadata.block_tables, dtype=np.int64)
    query_start_loc = np.asarray(metadata.query_start_loc, dtype=np.int64).reshape(-1)

    # GQA: how many Q heads per KV head
    heads_per_kv = num_heads // num_kv_heads

    scale = np.float32(1.0 / np.sqrt(np.float32(head_dim)))
    output = np.zeros((total_tokens, num_heads, head_dim), dtype=np.float32)

    for seq_idx in range(batch_size):
        seq_len = int(seq_lens[seq_idx])
        q_start = int(query_start_loc[seq_idx])
        q_end = int(query_start_loc[seq_idx + 1])
        q_len = q_end - q_start

        # Gather full K/V for this sequence via block_tables.
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        k_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        v_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)

        for blk_idx in range(num_blocks_needed):
            block_id = int(block_tables[seq_idx, blk_idx])
            start = blk_idx * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            k_gathered[start:end] = (
                kv_cache[0, block_id, :, :length, :]
                .transpose(1, 0, 2)
                .astype(np.float32)
            )
            v_gathered[start:end] = (
                kv_cache[1, block_id, :, :length, :]
                .transpose(1, 0, 2)
                .astype(np.float32)
            )

        # Per Q-head attention.
        q_seq = q[q_start:q_end].astype(np.float32)  # [q_len, num_heads, head_dim]

        for h in range(num_heads):
            kv_h = h // heads_per_kv
            q_h = q_seq[:, h, :]  # [q_len, head_dim]
            k_h = k_gathered[:, kv_h, :]  # [seq_len, head_dim]
            v_h = v_gathered[:, kv_h, :]  # [seq_len, head_dim]

            # Scaled dot-product: [q_len, seq_len]
            scores = (q_h @ k_h.T) * scale

            # Causal masking for EXTEND mode.
            if metadata.forward_mode == FORWARD_MODE_EXTEND:
                # In EXTEND, the query tokens are the last q_len tokens of the sequence.
                # Token at position i in the query can attend to all KV positions
                # up to (seq_len - q_len + i) inclusive.
                for qi in range(q_len):
                    max_kv_pos = seq_len - q_len + qi
                    if max_kv_pos + 1 < seq_len:
                        scores[qi, max_kv_pos + 1 :] = -np.inf
            # DECODE: single token per sequence, attends to full context — no masking needed.

            # Softmax.
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
            attn_weights = scores_exp / scores_sum  # [q_len, seq_len]

            # Weighted sum.
            output[q_start:q_end, h, :] = attn_weights @ v_h

    return output.astype(q.dtype)
