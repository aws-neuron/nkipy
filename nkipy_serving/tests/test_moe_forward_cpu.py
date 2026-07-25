"""CPU-only forward_cpu tests for Qwen3-MoE and GPT-OSS eager executors.

These bypass the HF snapshot load by constructing tiny synthetic weights
directly. The full ``__init__`` requires device runtime + HF snapshots,
so we use ``__new__`` and populate the fields that ``forward_cpu``
reads: ``_weights``, ``_shared_np``, ``_layer_np``, and (for GPT-OSS)
``_vocab_shard``.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import FORWARD_MODE_DECODE, AttentionMetadata
from nkipy_serving.models.common.moe_cpu_ops import (
    cpu_moe_dispatch_swiglu_oai,
    cpu_moe_dispatch_swish,
)
from nkipy_serving.models.gpt_oss.config import GptOssWeights
from nkipy_serving.models.gpt_oss.eager_executor import GptOssEagerExecutor
from nkipy_serving.models.qwen3_moe.config import Qwen3MoeWeights
from nkipy_serving.models.qwen3_moe.eager_executor import Qwen3MoeEagerExecutor
from nkipy_serving.ops.moe.blockwise_index import BLOCK_SIZE as MOE_BLOCK_SIZE
from nkipy_serving.ops.vocab_parallel_embedding import get_vocab_parallel_shard

# ---------------------------------------------------------------------------
# Common mini-model parameters
# ---------------------------------------------------------------------------

_HIDDEN = 32
_HEAD_DIM = 16
_N_HEADS = 4
_N_KV = 2
_LAYERS = 2
_VOCAB = 64
_INTERMEDIATE = 32
_N_EXPERTS = 4
_TOP_K = 2
_BATCH = 2
_SEQ_LEN = 3  # prior tokens in KV cache per sequence
_BLOCK_SIZE = 4  # KV-cache paging block
_DTYPE = ml_dtypes.bfloat16


# ---------------------------------------------------------------------------
# Synthetic weight helpers
# ---------------------------------------------------------------------------


def _rand(shape, dtype=_DTYPE, scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal(shape)).astype(dtype)


def _qwen3_moe_weights() -> Qwen3MoeWeights:
    return Qwen3MoeWeights(
        model_id="synthetic-qwen3-moe",
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV,
        moe_intermediate_size=_INTERMEDIATE,
        num_experts=_N_EXPERTS,
        experts_per_token=_TOP_K,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dtype=np.dtype(_DTYPE),
        tp_degree=1,
        tp_rank=0,
        num_heads=_N_HEADS,
        num_kv_heads=_N_KV,
        local_vocab_size=_VOCAB,
        lm_head_vocab_offset=0,
        local_intermediate_size=_INTERMEDIATE,
        ep_degree=1,
        ep_rank=0,
        local_num_experts=_N_EXPERTS,
    )


def _qwen3_moe_synthetic_np(seed: int = 0) -> tuple[dict, list[dict]]:
    shared = {
        "embeddings": _rand((_VOCAB, _HIDDEN), seed=seed),
        "final_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
        "lm_head": _rand((_VOCAB, _HIDDEN), seed=seed + 1),
    }
    layers = []
    for li in range(_LAYERS):
        s = seed + 100 * (li + 1)
        layers.append(
            {
                "input_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
                "post_attn_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
                "w_q": _rand((_HIDDEN, _N_HEADS * _HEAD_DIM), seed=s),
                "w_k": _rand((_HIDDEN, _N_KV * _HEAD_DIM), seed=s + 1),
                "w_v": _rand((_HIDDEN, _N_KV * _HEAD_DIM), seed=s + 2),
                "w_o": _rand((_N_HEADS * _HEAD_DIM, _HIDDEN), seed=s + 3),
                "q_norm": np.ones((_HEAD_DIM,), dtype=_DTYPE),
                "k_norm": np.ones((_HEAD_DIM,), dtype=_DTYPE),
                "router_w": _rand((_HIDDEN, _N_EXPERTS), seed=s + 4),
                "gup_w": _rand(
                    (_N_EXPERTS, _HIDDEN, 2, _INTERMEDIATE),
                    dtype=ml_dtypes.float8_e5m2,
                    seed=s + 5,
                ),
                "gup_bias": np.zeros((_N_EXPERTS, _INTERMEDIATE, 2), dtype=np.float32),
                "down_w": _rand(
                    (_N_EXPERTS, _INTERMEDIATE, _HIDDEN),
                    dtype=ml_dtypes.float8_e5m2,
                    seed=s + 6,
                ),
                "down_bias_bc": np.zeros(
                    (_N_EXPERTS, MOE_BLOCK_SIZE, _HIDDEN), dtype=_DTYPE
                ),
            }
        )
    return shared, layers


def _gpt_oss_weights() -> GptOssWeights:
    return GptOssWeights(
        model_id="synthetic-gpt-oss",
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV,
        intermediate_size=_INTERMEDIATE,
        num_experts=_N_EXPERTS,
        experts_per_token=_TOP_K,
        rms_norm_eps=1e-5,
        rope_theta=150000.0,
        yarn_factor=1.0,  # < 1 triggers no-YaRN ramp path
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_original_max_pos=4096,
        dtype=np.dtype(_DTYPE),
        tp_degree=1,
        tp_rank=0,
        num_heads=_N_HEADS,
        num_kv_heads=_N_KV,
        local_vocab_size=_VOCAB,
        lm_head_vocab_offset=0,
        local_intermediate_size=_INTERMEDIATE,
        ep_degree=1,
        ep_rank=0,
        local_num_experts=_N_EXPERTS,
    )


def _gpt_oss_synthetic_np(seed: int = 0) -> tuple[dict, list[dict]]:
    shared = {
        "embeddings": _rand((_VOCAB, _HIDDEN), seed=seed),
        "final_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
        "lm_head": _rand((_VOCAB, _HIDDEN), seed=seed + 1),
    }
    layers = []
    for li in range(_LAYERS):
        s = seed + 100 * (li + 1)
        gup_bias = np.zeros((_N_EXPERTS, _INTERMEDIATE, 2), dtype=np.float32)
        gup_bias[:, :, 1] = 1.0  # "+1" baked into up-bias (production behavior)
        layers.append(
            {
                "input_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
                "post_attn_norm": np.ones((_HIDDEN,), dtype=_DTYPE),
                "w_q": _rand((_HIDDEN, _N_HEADS * _HEAD_DIM), seed=s),
                "w_k": _rand((_HIDDEN, _N_KV * _HEAD_DIM), seed=s + 1),
                "w_v": _rand((_HIDDEN, _N_KV * _HEAD_DIM), seed=s + 2),
                "b_q": _rand((_N_HEADS * _HEAD_DIM,), seed=s + 10, scale=0.01),
                "b_k": _rand((_N_KV * _HEAD_DIM,), seed=s + 11, scale=0.01),
                "b_v": _rand((_N_KV * _HEAD_DIM,), seed=s + 12, scale=0.01),
                "w_o": _rand((_N_HEADS * _HEAD_DIM, _HIDDEN), seed=s + 3),
                "b_o": _rand((_HIDDEN,), seed=s + 13, scale=0.01),
                "sink": _rand((_N_HEADS, 1), seed=s + 14, scale=0.1),
                "router_w": _rand((_HIDDEN, _N_EXPERTS), seed=s + 4),
                "router_b": _rand((_N_EXPERTS,), seed=s + 15, scale=0.01),
                "gup_w": _rand((_N_EXPERTS, _HIDDEN, 2, _INTERMEDIATE), seed=s + 5),
                "gup_bias": gup_bias,
                "down_w": _rand((_N_EXPERTS, _INTERMEDIATE, _HIDDEN), seed=s + 6),
                "down_bias_bc": np.zeros(
                    (_N_EXPERTS, MOE_BLOCK_SIZE, _HIDDEN), dtype=_DTYPE
                ),
            }
        )
    return shared, layers


# ---------------------------------------------------------------------------
# Attention metadata + KV cache helpers (decode: 1 new tok / sequence)
# ---------------------------------------------------------------------------


def _decode_meta_and_kv():
    batch = _BATCH
    # Each sequence has _SEQ_LEN prior tokens cached; decode appends 1 new tok.
    # With _BLOCK_SIZE=4, one block per sequence is enough (_SEQ_LEN+1 <= 4).
    seq_lens = np.full(batch, _SEQ_LEN + 1, dtype=np.int64)
    slots = np.array([i * _BLOCK_SIZE + _SEQ_LEN for i in range(batch)], dtype=np.int64)
    block_tables = np.arange(batch, dtype=np.int64).reshape(batch, 1)
    qsl = np.array([0, 1, 2], dtype=np.int64)  # 1 query token per sequence

    num_blocks = batch + 1  # extra scratch block
    kv_caches = []
    rng = np.random.default_rng(7)
    for _ in range(_LAYERS):
        kv = np.zeros(
            (2, num_blocks, _N_KV, _BLOCK_SIZE, _HEAD_DIM),
            dtype=_DTYPE,
        )
        # Populate prior tokens for each sequence.
        for s in range(batch):
            kv[0, s, :, :_SEQ_LEN, :] = (
                0.02 * rng.standard_normal((_N_KV, _SEQ_LEN, _HEAD_DIM))
            ).astype(_DTYPE)
            kv[1, s, :, :_SEQ_LEN, :] = (
                0.02 * rng.standard_normal((_N_KV, _SEQ_LEN, _HEAD_DIM))
            ).astype(_DTYPE)
        kv_caches.append(kv)

    meta = AttentionMetadata(
        forward_mode=FORWARD_MODE_DECODE,
        seq_lens=seq_lens,
        slot_mapping=slots,
        block_tables=block_tables,
        query_start_loc=qsl,
        total_tokens=batch,
        batch_size=batch,
        max_seq_len=_SEQ_LEN + 1,
        num_kv_heads=_N_KV,
        head_dim=_HEAD_DIM,
        block_size=_BLOCK_SIZE,
    )
    return meta, kv_caches


# ---------------------------------------------------------------------------
# Deterministic numeric tests — each token routes to exactly one expert,
# so the weighted sum collapses and we can compare to the explicit formula.
# All weights/inputs are float32 to avoid fp8 quantization confounds.
# ---------------------------------------------------------------------------


def _one_hot_affinities(T: int, E: int, expert_ids: list[int]) -> np.ndarray:
    """Affinities that route each token to a single expert with weight 1."""
    aff = np.zeros((T, E), dtype=np.float32)
    for t, e in enumerate(expert_ids):
        aff[t, e] = 1.0
    return aff


def test_cpu_moe_dispatch_swish_matches_formula():
    """Qwen3 variant: weight 1 → expert e's SiLU(gate) * up → down."""
    T, H, E, inter = 2, 8, 3, 4
    rng = np.random.default_rng(42)
    normed = rng.standard_normal((T, H)).astype(np.float32)
    gup = rng.standard_normal((E, H, 2, inter)).astype(np.float32) * 0.05
    dn = rng.standard_normal((E, inter, H)).astype(np.float32) * 0.05

    # Token 0 → expert 1, token 1 → expert 2.
    route = [1, 2]
    aff = _one_hot_affinities(T, E, route)

    got = cpu_moe_dispatch_swish(normed, aff, gup, dn)

    expected = np.zeros((T, H), dtype=np.float32)
    for t, e in enumerate(route):
        gate_w = gup[e, :, 0, :]
        up_w = gup[e, :, 1, :]
        gate = normed[t : t + 1] @ gate_w
        up = normed[t : t + 1] @ up_w
        silu = gate / (1.0 + np.exp(-gate))
        expected[t : t + 1] = (silu * up) @ dn[e]

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_cpu_moe_dispatch_swiglu_oai_special_terms():
    """GPT-OSS SwiGLU applies the +1 up bias, gate clamp, and down bias."""
    T, H, E, inter = 1, 4, 1, 2
    rng = np.random.default_rng(7)
    normed = rng.standard_normal((T, H)).astype(np.float32)
    gup = rng.standard_normal((E, H, 2, inter)).astype(np.float32) * 0.05
    dn = rng.standard_normal((E, inter, H)).astype(np.float32) * 0.05

    # Case A: bias with +1 baked in (production behavior).
    bias_a = np.zeros((E, inter, 2), dtype=np.float32)
    bias_a[:, :, 1] = 1.0
    # Case B: no +1 shift.
    bias_b = np.zeros((E, inter, 2), dtype=np.float32)

    dn_bias = np.zeros((E, MOE_BLOCK_SIZE, H), dtype=_DTYPE)
    aff = _one_hot_affinities(T, E, [0])

    out_a = cpu_moe_dispatch_swiglu_oai(normed, aff, gup, bias_a, dn, dn_bias)
    out_b = cpu_moe_dispatch_swiglu_oai(normed, aff, gup, bias_b, dn, dn_bias)

    # Difference = ((up_pre_a - up_pre_b) * glu) @ dn = ([1] * glu) @ dn.
    gate_w = gup[0, :, 0, :]
    up_w = gup[0, :, 1, :]
    gate_pre = normed @ gate_w
    gate = np.minimum(gate_pre, 7.0)
    glu = gate / (1.0 + np.exp(-gate * 1.702))
    up_pre_b = normed @ up_w
    up_pre_a = up_pre_b + 1.0
    up_b = np.clip(up_pre_b, -7.0, 7.0)
    up_a = np.clip(up_pre_a, -6.0, 8.0)
    expected_delta = ((up_a - up_b) * glu) @ dn[0]

    np.testing.assert_allclose(
        out_a.astype(np.float32) - out_b.astype(np.float32),
        expected_delta,
        rtol=1e-5,
        atol=1e-5,
    )

    T, H, E, inter = 1, 2, 1, 1
    # Craft a gate that explodes past the limit. normed=1, gate_w=100 → gate_pre=100.
    normed = np.ones((T, H), dtype=np.float32)
    gup = np.zeros((E, H, 2, inter), dtype=np.float32)
    gup[0, :, 0, 0] = 100.0  # huge gate
    gup[0, :, 1, 0] = 0.5  # small up
    dn = np.ones((E, inter, H), dtype=np.float32)
    bias = np.zeros((E, inter, 2), dtype=np.float32)
    bias[:, :, 1] = 1.0  # standard +1 shift
    dn_bias = np.zeros((E, MOE_BLOCK_SIZE, H), dtype=_DTYPE)
    aff = _one_hot_affinities(T, E, [0])

    got = cpu_moe_dispatch_swiglu_oai(normed, aff, gup, bias, dn, dn_bias)

    # With clamping: gate = 7.0 → glu = 7.0 * sigmoid(7*1.702) ≈ 7.0.
    # up_pre = 1*0.5*H + 1 = H+1 = 3; up clamped to [-6, 8] → 3.
    # expert_out per hidden dim = up * glu = 3 * 7 = 21.
    # out = aff * expert_out = 21 on every H dim.
    gate_clamped = np.float32(7.0)
    glu_expected = gate_clamped / (1.0 + np.exp(-gate_clamped * 1.702))
    up_pre = np.float32(H) * np.float32(0.5) + np.float32(1.0)
    up_clamped = np.clip(up_pre, -6.0, 8.0)
    expected = up_clamped * glu_expected * np.ones((T, H), dtype=np.float32)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    T, H, E, inter = 1, 4, 1, 2
    rng = np.random.default_rng(11)
    normed = rng.standard_normal((T, H)).astype(np.float32)
    gup = rng.standard_normal((E, H, 2, inter)).astype(np.float32) * 0.05
    dn = rng.standard_normal((E, inter, H)).astype(np.float32) * 0.05
    bias = np.zeros((E, inter, 2), dtype=np.float32)
    bias[:, :, 1] = 1.0

    # down_bias_bc[e, 0, :] is the per-hidden-dim bias; all rows are identical
    # after the kernel's TP zero-out step.
    db = np.array([0.7, -0.3, 0.1, 0.2], dtype=_DTYPE)
    dn_bias_zero = np.zeros((E, MOE_BLOCK_SIZE, H), dtype=_DTYPE)
    dn_bias_set = np.zeros((E, MOE_BLOCK_SIZE, H), dtype=_DTYPE)
    dn_bias_set[0, :, :] = db

    aff = _one_hot_affinities(T, E, [0])
    out_zero = cpu_moe_dispatch_swiglu_oai(normed, aff, gup, bias, dn, dn_bias_zero)
    out_set = cpu_moe_dispatch_swiglu_oai(normed, aff, gup, bias, dn, dn_bias_set)

    np.testing.assert_allclose(
        out_set.astype(np.float32) - out_zero.astype(np.float32),
        db.astype(np.float32)[None, :],
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Executor forward_cpu tests
# ---------------------------------------------------------------------------


def test_forward_cpu_runs_end_to_end_for_moe_models():
    cases = [
        (
            Qwen3MoeEagerExecutor,
            _qwen3_moe_weights(),
            _qwen3_moe_synthetic_np(),
            False,
        ),
        (
            GptOssEagerExecutor,
            _gpt_oss_weights(),
            _gpt_oss_synthetic_np(),
            True,
        ),
    ]
    for executor_cls, weights, (shared, layers), needs_vocab_shard in cases:
        exe = executor_cls.__new__(executor_cls)
        exe._weights = weights
        if needs_vocab_shard:
            exe._vocab_shard = get_vocab_parallel_shard(
                vocab_size=_VOCAB,
                rank=0,
                world_size=1,
            )
        exe._shared_np, exe._layer_np = shared, layers

        meta, kv_caches = _decode_meta_and_kv()
        logits = exe.forward_cpu(
            np.array([5, 10], dtype=np.int32),
            np.array([_SEQ_LEN, _SEQ_LEN], dtype=np.int32),
            kv_caches,
            meta,
        )
        assert logits.shape == (2, _VOCAB)
        assert np.all(np.isfinite(logits))


def test_gpt_oss_forward_cpu_sink_changes_output():
    """Sanity: sink != 0 yields different logits than sink == 0."""
    meta, kv_caches = _decode_meta_and_kv()
    input_ids = np.array([5, 10], dtype=np.int32)
    positions = np.array([_SEQ_LEN, _SEQ_LEN], dtype=np.int32)

    exe_a = GptOssEagerExecutor.__new__(GptOssEagerExecutor)
    exe_a._weights = _gpt_oss_weights()
    exe_a._vocab_shard = get_vocab_parallel_shard(
        vocab_size=_VOCAB,
        rank=0,
        world_size=1,
    )
    shared_a, layers_a = _gpt_oss_synthetic_np()
    exe_a._shared_np, exe_a._layer_np = shared_a, layers_a
    logits_a = exe_a.forward_cpu(
        input_ids, positions, [k.copy() for k in kv_caches], meta
    )

    # Zero out all sinks; everything else identical.
    exe_b = GptOssEagerExecutor.__new__(GptOssEagerExecutor)
    exe_b._weights = _gpt_oss_weights()
    exe_b._vocab_shard = get_vocab_parallel_shard(
        vocab_size=_VOCAB,
        rank=0,
        world_size=1,
    )
    shared_b, layers_b = _gpt_oss_synthetic_np()
    for lt in layers_b:
        lt["sink"] = np.zeros_like(lt["sink"])
    exe_b._shared_np, exe_b._layer_np = shared_b, layers_b
    logits_b = exe_b.forward_cpu(
        input_ids, positions, [k.copy() for k in kv_caches], meta
    )

    assert not np.allclose(logits_a, logits_b, rtol=1e-3, atol=1e-3)
