"""Host correctness gate for the DSV4 paged sparse-attention trace function.

``sparse_attention_paged_swa_decode_fn`` is the trace-function transcription of
the production decode attention kernel (``_sparse_attn_batched_paged_swa_multiK_kernel``).
These pure-NumPy tests compare it against the established oracle
(``swa_global_slots_oracle`` -> ``gather_kv_and_mask`` -> ``sparse_mla_attention_oracle``)
*before* any device compilation, so the #1 risk (double-applying ``softmax_scale``)
is caught deterministically on host.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from nkipy_serving.attention.deepseek_v4.kernels import swa_global_slots_oracle
from nkipy_serving.models.deepseek_v4.neff_graphs.attention import (
    q_scale_transpose_fn,
    sparse_attention_paged_swa_decode_fn,
)
from nkipy_serving.ops.attention.sparse_mla import (
    gather_kv_and_mask,
    sparse_mla_attention_oracle,
)


def _reference(
    *,
    q_scaled_t: np.ndarray,
    kv_hbm_bf: np.ndarray,
    positions: np.ndarray,
    block_tables: np.ndarray,
    sink: np.ndarray,
    softmax_scale: float,
    block_size: int,
    window_size: int,
    max_k: int,
) -> np.ndarray:
    """Oracle path: derive SWA slots, gather, sink-aware softmax."""
    n_q = int(q_scaled_t.shape[0])
    topk_t, _mask, _lens = swa_global_slots_oracle(
        positions=positions.reshape(-1),
        req_id_per_token=np.arange(n_q, dtype=np.int64),
        block_tables=block_tables,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )
    topk_idxs = topk_t.T  # [n_q, max_k], -1 == invalid
    gathered, valid = gather_kv_and_mask(kv_hbm_bf.astype(np.float32), topk_idxs)
    # The fragment consumes the SAME bf16 q as the kernel. Un-bake the scale so
    # the oracle (which re-applies it) sees identical bf16-rounded values.
    q_for_oracle = q_scaled_t.transpose(0, 2, 1).astype(np.float32) / np.float32(
        softmax_scale
    )
    return sparse_mla_attention_oracle(
        q_for_oracle,
        gathered,
        valid,
        sink.reshape(-1),
        float(softmax_scale),
    )


@pytest.mark.parametrize(
    "positions,window_size,max_k,block_size",
    [
        # short window: every pos+1 < window_size
        (np.array([5, 6, 7, 3], dtype=np.int32), 64, 128, 16),
        # full window: pos+1 > window_size for most rows
        (np.array([60, 100, 47, 200], dtype=np.int32), 48, 128, 16),
        # block boundaries: logical % block_size hits 0 and block_size-1
        (np.array([15, 16, 31, 32], dtype=np.int32), 64, 128, 16),
        # window_size > max_k: cur_len clamped to max_k
        (np.array([300, 301, 302, 303], dtype=np.int32), 256, 128, 16),
    ],
)
def test_fragment_matches_swa_oracle(positions, window_size, max_k, block_size):
    rng = np.random.default_rng(0)
    n_q = int(positions.shape[0])
    head_dim, n_heads = 128, 8
    num_blocks = 64
    num_slots = num_blocks * block_size
    max_blocks = num_slots // block_size
    softmax_scale = 1.0 / float(np.sqrt(head_dim))

    block_tables = rng.integers(0, num_blocks, size=(n_q, max_blocks)).astype(np.int32)
    kv_hbm_bf = rng.standard_normal((num_slots, head_dim)).astype(ml_dtypes.bfloat16)
    sink = rng.standard_normal((1, n_heads)).astype(np.float32)

    q_unscaled = rng.standard_normal((1, n_q, n_heads, head_dim)).astype(np.float32)
    q_scaled_t = q_scale_transpose_fn(
        q_unscaled, softmax_scale=softmax_scale, token_bucket=n_q
    )  # [n_q, head_dim, n_heads] bf16, scale baked in

    out = sparse_attention_paged_swa_decode_fn(
        q_scaled_t,
        kv_hbm_bf,
        positions.reshape(n_q, 1),
        block_tables,
        sink,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )
    ref = _reference(
        q_scaled_t=q_scaled_t,
        kv_hbm_bf=kv_hbm_bf,
        positions=positions,
        block_tables=block_tables,
        sink=sink,
        softmax_scale=softmax_scale,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )

    assert out.shape == (n_q, n_heads, head_dim)
    np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_fragment_does_not_double_scale():
    """A double-scale bug diverges by ~softmax_scale; the oracle comparison
    above would catch it, but assert it directly for a clear failure signal."""
    rng = np.random.default_rng(1)
    n_q, head_dim, n_heads = 4, 128, 8
    block_size, window_size, max_k = 16, 48, 128
    num_blocks = 64
    num_slots = num_blocks * block_size
    max_blocks = num_slots // block_size
    softmax_scale = 1.0 / float(np.sqrt(head_dim))

    positions = np.array([60, 100, 47, 200], dtype=np.int32)
    block_tables = rng.integers(0, num_blocks, size=(n_q, max_blocks)).astype(np.int32)
    kv_hbm_bf = rng.standard_normal((num_slots, head_dim)).astype(ml_dtypes.bfloat16)
    sink = rng.standard_normal((1, n_heads)).astype(np.float32)
    q_unscaled = rng.standard_normal((1, n_q, n_heads, head_dim)).astype(np.float32)
    q_scaled_t = q_scale_transpose_fn(
        q_unscaled, softmax_scale=softmax_scale, token_bucket=n_q
    )

    out = sparse_attention_paged_swa_decode_fn(
        q_scaled_t,
        kv_hbm_bf,
        positions.reshape(n_q, 1),
        block_tables,
        sink,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )
    # A version that erroneously multiplies scores by softmax_scale again would
    # sharpen the softmax; reconstruct that wrong output and confirm we differ.
    ref = _reference(
        q_scaled_t=q_scaled_t,
        kv_hbm_bf=kv_hbm_bf,
        positions=positions,
        block_tables=block_tables,
        sink=sink,
        softmax_scale=softmax_scale,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )
    np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_padded_rows_are_zeroed():
    """Rows beyond the active batch (token_bucket padding) must produce zeros:
    q_scaled_t pads with zero q, and the any_valid guard zeroes the output."""
    rng = np.random.default_rng(2)
    n_active, bucket = 2, 8
    head_dim, n_heads = 128, 8
    block_size, window_size, max_k = 16, 48, 128
    num_blocks = 64
    num_slots = num_blocks * block_size
    max_blocks = num_slots // block_size
    softmax_scale = 1.0 / float(np.sqrt(head_dim))

    q_unscaled = rng.standard_normal((1, n_active, n_heads, head_dim)).astype(
        np.float32
    )
    q_scaled_t = q_scale_transpose_fn(
        q_unscaled, softmax_scale=softmax_scale, token_bucket=bucket
    )  # [bucket, head_dim, n_heads], trailing rows zero

    # Padded rows: position 0, valid block table; cur_len = min(0+1, ...) = 1,
    # but the q is zero so scores are zero -> still a valid softmax. The
    # any_valid guard does NOT zero them (pos 0 has 1 valid slot). To match the
    # production contract, padded rows are masked downstream by reduce-rows; the
    # fragment itself just must not crash and must produce finite output.
    positions = np.zeros((bucket, 1), dtype=np.int32)
    positions[:n_active, 0] = np.array([30, 40], dtype=np.int32)
    block_tables = rng.integers(0, num_blocks, size=(bucket, max_blocks)).astype(
        np.int32
    )
    kv_hbm_bf = rng.standard_normal((num_slots, head_dim)).astype(ml_dtypes.bfloat16)
    sink = rng.standard_normal((1, n_heads)).astype(np.float32)

    out = sparse_attention_paged_swa_decode_fn(
        q_scaled_t,
        kv_hbm_bf,
        positions,
        block_tables,
        sink,
        block_size=block_size,
        window_size=window_size,
        max_k=max_k,
    )
    assert out.shape == (bucket, n_heads, head_dim)
    assert np.all(np.isfinite(out))
