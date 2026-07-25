"""Contract tests for the MoE eager executors.

These cover the paths that previously had correctness bugs:

  - GPT-OSS attention must thread the learned sink tensor through attn.
  - GPT-OSS YaRN RoPE arg order must match production (slow, fast).
  - Both MoE executors must reject oversized token_bucket in decode mode
    (prefill has no MOE_BLOCK_SIZE cap).
"""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from nkipy_serving.attention.base import (
    FORWARD_MODE_DECODE,
    AttentionMetadata,
)
from nkipy_serving.models.gpt_oss import (
    GptOssEagerExecutor,
    cpu_attn_with_sink_fn,
    nki_attn_with_sink_fn,
)
from nkipy_serving.models.gpt_oss import eager_executor as gpt_oss_eager
from nkipy_serving.models.qwen3_moe import Qwen3MoeEagerExecutor

# --- GPT-OSS attention wiring --------------------------------------------


def test_gpt_oss_forward_cpu_rope_uses_slow_then_fast(monkeypatch):
    """YaRN beta order must match production: slow -> alpha, fast -> beta."""

    captured: dict[str, float] = {}

    def _fake_rope_cache(positions, **kwargs):
        captured["ntk_alpha"] = float(kwargs["ntk_alpha"])
        captured["ntk_beta"] = float(kwargs["ntk_beta"])
        return np.zeros((len(positions), 1), dtype=np.float32), np.zeros(
            (len(positions), 1),
            dtype=np.float32,
        )

    monkeypatch.setattr(
        gpt_oss_eager,
        "_build_rope_cache_for_positions_yarn",
        _fake_rope_cache,
    )

    exe = GptOssEagerExecutor.__new__(GptOssEagerExecutor)
    exe._weights = SimpleNamespace(
        tp_degree=1,
        ep_degree=1,
        head_dim=1,
        rope_theta=150000.0,
        yarn_original_max_pos=4096,
        yarn_factor=1.0,
        yarn_beta_slow=1.25,
        yarn_beta_fast=32.5,
        dtype=np.float32,
        num_hidden_layers=0,
        rms_norm_eps=1e-5,
    )
    exe._vocab_shard = SimpleNamespace(vocab_start_index=0, vocab_end_index=4)
    exe._shared_np = {
        "embeddings": np.eye(4, dtype=np.float32),
        "final_norm": np.ones((4,), dtype=np.float32),
        "lm_head": np.eye(4, dtype=np.float32),
    }
    exe._layer_np = []

    out = exe.forward_cpu(
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([8, 9], dtype=np.int32),
        [],
        _MinimalMeta(FORWARD_MODE_DECODE),  # unused for a zero-layer model
    )

    assert out.shape == (2, 4)
    assert captured == {"ntk_alpha": 1.25, "ntk_beta": 32.5}


def test_gpt_oss_sink_functions_have_correct_signatures():
    nki_params = signature(nki_attn_with_sink_fn).parameters
    assert "sink" in nki_params
    # Ordering matters; sink must follow kv_cache.
    names = list(nki_params.keys())
    assert names[4] == "sink", f"sink must be 5th positional arg, got {names[:6]}"
    cpu_params = signature(cpu_attn_with_sink_fn).parameters
    assert "sink" in cpu_params


# --- Decode bucket-cap guard ----------------------------------------------


class _MinimalMeta:
    """AttentionMetadata is a dataclass — mimic just the attrs the guards read."""

    def __init__(self, forward_mode: int):
        self.forward_mode = forward_mode


def test_moe_decode_rejects_oversized_bucket():
    """Decode MoE is capped at MOE_BLOCK_SIZE (128); prefill has no such cap."""
    for executor_cls in (Qwen3MoeEagerExecutor, GptOssEagerExecutor):
        exe = executor_cls.__new__(executor_cls)
        with pytest.raises(ValueError, match="exceeds decode MoE BLOCK_SIZE"):
            executor_cls.forward(
                exe,
                input_ids=np.zeros((256,), dtype=np.int32),
                positions=np.zeros((256,), dtype=np.int32),
                kv_caches=[],
                attn_metadata=_MinimalMeta(FORWARD_MODE_DECODE),
                token_bucket=256,
            )


# --- GPT-OSS sink CPU reference sanity (numpy-only) ----------------------


def test_gpt_oss_cpu_attn_with_sink_matches_manual_softmax():
    """Single-sequence decode with a known sink value — verify the sink
    contributes to the denominator but not the numerator.
    """
    rng = np.random.default_rng(0)
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    seq_len = 4
    block_size = 4

    q = rng.standard_normal((1, num_heads, head_dim), dtype=np.float32)
    k = rng.standard_normal((1, num_kv_heads, head_dim), dtype=np.float32)
    v = rng.standard_normal((1, num_kv_heads, head_dim), dtype=np.float32)
    kv_cache = np.zeros(
        (2, 1, num_kv_heads, block_size, head_dim),
        dtype=np.float32,
    )
    # Prefill the prior seq_len-1 tokens into the cache directly.
    prior_k = rng.standard_normal(
        (seq_len - 1, num_kv_heads, head_dim),
        dtype=np.float32,
    )
    prior_v = rng.standard_normal(
        (seq_len - 1, num_kv_heads, head_dim),
        dtype=np.float32,
    )
    for t in range(seq_len - 1):
        kv_cache[0, 0, :, t, :] = prior_k[t]
        kv_cache[1, 0, :, t, :] = prior_v[t]

    sink = np.array([[0.3], [-0.2]], dtype=np.float32)

    meta = AttentionMetadata(
        forward_mode=FORWARD_MODE_DECODE,
        seq_lens=np.array([seq_len], dtype=np.int64),
        slot_mapping=np.array([seq_len - 1], dtype=np.int64),
        block_tables=np.array([[0]], dtype=np.int64),
        query_start_loc=np.array([0, 1], dtype=np.int64),
        total_tokens=1,
        batch_size=1,
        max_seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    out = cpu_attn_with_sink_fn(q, k, v, kv_cache, sink, meta)

    # Manual reference.
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    gathered_k = np.concatenate([prior_k, k], axis=0)
    gathered_v = np.concatenate([prior_v, v], axis=0)
    expected = np.zeros_like(out, dtype=np.float32)
    for h in range(num_heads):
        scores = (q[0, h, :] @ gathered_k[:, 0, :].T) * scale
        m = np.max(scores)
        exp = np.exp(scores - m)
        sink_exp = np.exp(np.float32(sink[h, 0]) - m)
        weights = exp / (exp.sum() + sink_exp)
        expected[0, h, :] = weights @ gathered_v[:, 0, :]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
