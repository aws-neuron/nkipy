"""Standalone NKI attention decode accuracy test.

Compares NKI BlockSparse Flash Attention against a vanilla numpy reference
for a decode step (1 query token attending to prior KV cache context).

No nkipy_serving serving imports — only uses the vendored NKI attention code
and nkipy runtime directly.

Usage:
    NEURON_RT_VISIBLE_CORES=0 uv run python nki_attn_tests/test_decode_accuracy.py
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

# --- NKI attention imports (vendored in nkipy-serving) ---
sys.path.insert(0, "src")
# --- nkipy runtime ---
from nkipy.core import nki_op  # noqa: F401 — register NKI ops
from nkipy.runtime import DeviceKernel, DeviceTensor

from nkipy_serving.attention.blocksparse_flash_attention.flash_paged_attn_varlen import (
    flash_paged_attention_varlen,
)
from nkipy_serving.attention.blocksparse_flash_attention.scheduler import (
    FlashAttentionPlanner,
)
from nkipy_serving.attention.nki_paged_kv_cache import update_kv_cache

# ============================================================
# Config
# ============================================================
NUM_HEADS = 2  # q heads per kv head (GQA ratio)
NUM_KV_HEADS = 1  # 1 kv head (TP=8 for 8-kv-head model)
BLOCK_SIZE = 32
LARGE_Q_TILE_SIZE = 128
LARGE_KV_TILE_SIZE = 1024
DYNAMIC_LOOP_UNROLL = 8
COMPILER_ARGS = "-O1 --tensorizer-options='--skip-pass=LateLegalizePostSplit'"

_WRITE_BACK_SKIP = 100_000_000
_B_P_SIZE = 128

TILE_FIELDS = (
    "tile_q_indices",
    "tile_block_tables",
    "tile_masks",
    "num_dynamic_loop_steps",
    "q_update_pred",
    "last_tile_indices",
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ============================================================
# Vanilla reference: decode attention
# ============================================================
def vanilla_decode_attention(
    q: np.ndarray,  # (1, num_heads, head_dim) bf16
    kv_cache: np.ndarray,  # (2, num_blocks, num_kv_heads, block_size, head_dim) bf16
    seq_len: int,  # total sequence length (context + 1)
    block_table: np.ndarray,  # (max_blocks,) int64
) -> np.ndarray:
    """Single decode token attending to full KV cache context.

    Returns: (1, num_heads, head_dim) float32
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[3]
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(float(head_dim))

    # Gather full K/V sequence from paged cache.
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    k_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
    v_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
    for blk_idx in range(num_blocks_needed):
        bid = int(block_table[blk_idx])
        start = blk_idx * block_size
        end = min(start + block_size, seq_len)
        length = end - start
        # kv_cache layout: [2, num_blocks, num_kv_heads, block_size, head_dim]
        k_gathered[start:end] = (
            kv_cache[0, bid, :, :length, :].transpose(1, 0, 2).astype(np.float32)
        )
        v_gathered[start:end] = (
            kv_cache[1, bid, :, :length, :].transpose(1, 0, 2).astype(np.float32)
        )

    output = np.zeros((1, num_heads, head_dim), dtype=np.float32)
    q_f32 = q.astype(np.float32)

    for h in range(num_heads):
        kv_h = h // heads_per_kv
        q_h = q_f32[0, h, :]  # (head_dim,)
        k_h = k_gathered[:, kv_h, :]  # (seq_len, head_dim)
        v_h = v_gathered[:, kv_h, :]  # (seq_len, head_dim)

        scores = (q_h @ k_h.T) * scale  # (seq_len,)
        scores_max = np.max(scores)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp)
        output[0, h, :] = attn_weights @ v_h

    return output


def vanilla_update_kv_cache(k, v, kv_cache, slot_mapping):
    """Write K/V into block-based cache at slot positions."""
    block_size = kv_cache.shape[3]
    for i in range(len(slot_mapping)):
        bid = int(slot_mapping[i]) // block_size
        off = int(slot_mapping[i]) % block_size
        kv_cache[0, bid, :, off, :] = k[i]
        kv_cache[1, bid, :, off, :] = v[i]


# ============================================================
# NKI attention: build tile plan + call kernel (torch-free)
# ============================================================
def _build_dummy_prefill_plan(max_num_prefill_tiles: int) -> dict[str, np.ndarray]:
    kv_blocks_per_tile = LARGE_KV_TILE_SIZE // BLOCK_SIZE
    P = int(max_num_prefill_tiles)
    lti = np.zeros((_B_P_SIZE, 2), dtype=np.int32)
    lti[:, 1] = _WRITE_BACK_SKIP
    return {
        "tile_q_indices": np.zeros((P, LARGE_Q_TILE_SIZE), dtype=np.int32),
        "tile_block_tables": np.zeros((P, kv_blocks_per_tile), dtype=np.int32),
        "tile_masks": np.zeros(
            (_B_P_SIZE, P, LARGE_Q_TILE_SIZE // _B_P_SIZE, LARGE_KV_TILE_SIZE),
            dtype=np.uint8,
        ),
        "num_dynamic_loop_steps": np.zeros((1, 1), dtype=np.int32),
        "q_update_pred": np.zeros((P, 1), dtype=np.uint8),
        "last_tile_indices": lti,
    }


def _build_decode_plan(
    *,
    seq_lens: np.ndarray,
    query_start_loc: np.ndarray,
    block_tables: np.ndarray,
    max_num_decode_tiles: int,
) -> dict[str, np.ndarray]:
    query_lens = np.diff(query_start_loc).astype(np.int32)
    context_lens = (seq_lens - query_lens).astype(np.int32)
    prompt_starts = np.asarray(query_start_loc[:-1], dtype=np.int32)
    max_seq_len = int(np.max(seq_lens)) if seq_lens.size else 0
    plan = FlashAttentionPlanner.MakeTilePlan(
        prompt_lens=query_lens,
        prior_context_lens=context_lens,
        tile_size_q=1,
        tile_size_kv=LARGE_KV_TILE_SIZE,
        block_size=BLOCK_SIZE,
        prompt_starts=prompt_starts,
        include_prompt_in_context=True,
        max_seq_len=max_seq_len,
    )
    if int(plan.num_real_tiles) > int(max_num_decode_tiles):
        raise RuntimeError(
            f"Decode plan too large: {plan.num_real_tiles=} {max_num_decode_tiles=}"
        )

    plan = plan.pad_plan(int(max_num_decode_tiles), q_pad_value=0)
    tile_q_indices = plan.build_tile_q_indices(skip_value=0).astype(np.int32)
    tile_block_tables = plan.build_tile_block_tables(
        np.asarray(block_tables, dtype=np.int32), skip_value=0
    ).astype(np.int32)
    tile_masks = plan.build_tile_masks(decode_kq_layout=True).astype(np.uint8)

    num_dynamic_loop_steps = np.zeros((1, 1), dtype=np.int32)
    num_dynamic_loop_steps[0, 0] = _ceil_div(
        int(plan.num_real_tiles), int(DYNAMIC_LOOP_UNROLL)
    )

    q_update_pred, last_tile_indices = plan.build_tile_update_indices(
        max_num_q_tiles=_B_P_SIZE
    )
    q_update_pred = np.asarray(q_update_pred, dtype=np.uint8).reshape(
        (int(plan.num_tiles), 1)
    )
    last_tile_indices = np.asarray(last_tile_indices, dtype=np.int32)

    return {
        "tile_q_indices": tile_q_indices,
        "tile_block_tables": tile_block_tables,
        "tile_masks": tile_masks,
        "num_dynamic_loop_steps": num_dynamic_loop_steps,
        "q_update_pred": q_update_pred,
        "last_tile_indices": last_tile_indices,
    }


def nki_decode_attention(
    q_dev,  # DeviceTensor (token_bucket, num_heads, head_dim)
    k_dev,  # DeviceTensor (token_bucket, num_kv_heads, head_dim)
    v_dev,  # DeviceTensor (token_bucket, num_kv_heads, head_dim)
    kv_cache_dev,  # DeviceTensor (2, num_blocks, num_kv_heads, block_size, head_dim)
    seq_lens: np.ndarray,  # (batch_size,) int64
    query_start_loc: np.ndarray,  # (batch_size+1,) int64
    block_tables: np.ndarray,  # (batch_size, max_blocks) int64
    token_bucket: int,
    real_total_tokens: int,
    num_blocks: int,
    head_dim: int,
    build_dir: str = "/tmp/nki_attn_test",
) -> np.ndarray:
    """Run NKI attention for a decode step. Returns (token_bucket, num_heads, head_dim) bf16."""
    dtype = bfloat16
    softmax_scale = 1.0 / (head_dim**0.5)

    query_lens_np = np.diff(query_start_loc)
    context_lens_np = seq_lens - query_lens_np

    print(f"  query_lens={query_lens_np}, context_lens={context_lens_np}")
    print(f"  seq_lens={seq_lens}, token_bucket={token_bucket}")

    # Unified kernel expects both prefill+decode tile plans (dummy prefill here).
    max_num_prefill_tiles = _round_up(1, DYNAMIC_LOOP_UNROLL)
    full_context_lens = query_lens_np + context_lens_np
    real_decode_tiles = int(
        sum(
            _ceil_div(int(x), LARGE_KV_TILE_SIZE) for x in full_context_lens.reshape(-1)
        )
    )
    max_num_decode_tiles = _round_up(max(real_decode_tiles, 1), DYNAMIC_LOOP_UNROLL)

    prefill_nps = _build_dummy_prefill_plan(max_num_prefill_tiles)
    decode_nps = _build_decode_plan(
        seq_lens=np.asarray(seq_lens, dtype=np.int64),
        query_start_loc=np.asarray(query_start_loc, dtype=np.int64),
        block_tables=np.asarray(block_tables, dtype=np.int64),
        max_num_decode_tiles=max_num_decode_tiles,
    )

    # Build wrapper and compile.
    def unified_wrapper(
        q,
        k,
        v,
        kv_cache,
        p_tqi,
        p_tbt,
        p_tm,
        p_ndls,
        p_qup,
        p_lti,
        d_tqi,
        d_tbt,
        d_tm,
        d_ndls,
        d_qup,
        d_lti,
    ):
        q_4d = q.reshape((1,) + q.shape).transpose(0, 2, 1, 3)
        k_4d = k.reshape((1,) + k.shape).transpose(0, 2, 3, 1)
        v_4d = v.reshape((1,) + v.shape).transpose(0, 2, 1, 3)
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]
        out = flash_paged_attention_varlen[1](
            q_4d,
            k_4d,
            v_4d,
            k_cache,
            v_cache,
            None,
            None,
            p_tqi,
            p_tbt,
            p_tm,
            p_ndls,
            p_qup,
            p_lti,
            d_tqi,
            d_tbt,
            d_tm,
            d_ndls,
            d_qup,
            d_lti,
            dynamic_loop_unroll_factor=DYNAMIC_LOOP_UNROLL,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            skip_active=True,
        )
        return out[0].transpose(1, 0, 2)

    sq = np.zeros((token_bucket, NUM_HEADS, head_dim), dtype=dtype)
    sk = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    sv = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    sc = np.zeros((2, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, head_dim), dtype=dtype)

    sample_args = [sq, sk, sv, sc]
    for f in TILE_FIELDS:
        sample_args.append(np.zeros_like(prefill_nps[f]))
    for f in TILE_FIELDS:
        sample_args.append(np.zeros_like(decode_nps[f]))

    kernel = DeviceKernel.compile_and_load(
        unified_wrapper,
        *sample_args,
        name=f"test_nki_decode_attn_unified_t{token_bucket}_b{num_blocks}_d{head_dim}",
        additional_compiler_args=COMPILER_ARGS,
        use_cached_if_exists=True,
        build_dir=build_dir,
    )

    # Upload tile plans and execute.
    inputs = {
        "q": q_dev,
        "k": k_dev,
        "v": v_dev,
        "kv_cache": kv_cache_dev,
    }

    pnames = ["p_tqi", "p_tbt", "p_tm", "p_ndls", "p_qup", "p_lti"]
    for nm, f in zip(pnames, TILE_FIELDS):
        inputs[nm] = DeviceTensor.from_numpy(prefill_nps[f], name=f"pf_{f}")

    dnames = ["d_tqi", "d_tbt", "d_tm", "d_ndls", "d_qup", "d_lti"]
    for nm, f in zip(dnames, TILE_FIELDS):
        inputs[nm] = DeviceTensor.from_numpy(decode_nps[f], name=f"dc_{f}")

    out_np = np.zeros((token_bucket, NUM_HEADS, head_dim), dtype=dtype)
    out_dev = DeviceTensor.from_numpy(out_np, name="attn_out")
    kernel(inputs=inputs, outputs={"output0": out_dev})
    return out_dev.numpy()


# ============================================================
# KV cache update via NKI kernel
# ============================================================
def _kv_update_wrapper(key, value, kv_cache, slot_mapping):
    return update_kv_cache(key, value, kv_cache, slot_mapping)


def nki_kv_cache_update(
    k_dev,
    v_dev,
    kv_cache_dev,
    slot_mapping: np.ndarray,
    token_bucket: int,
    real_total_tokens: int,
    num_blocks: int,
    head_dim: int,
    build_dir: str = "/tmp/nki_attn_test",
):
    dtype = bfloat16
    scratch_slot = (num_blocks - 1) * BLOCK_SIZE
    padded_slot = np.full(token_bucket, scratch_slot, dtype=np.int32)
    padded_slot[:real_total_tokens] = slot_mapping
    slot_dev = DeviceTensor.from_numpy(padded_slot, name="slot_mapping")

    sk = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    sv = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    sc = np.zeros((2, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, head_dim), dtype=dtype)
    ss = np.zeros((token_bucket,), dtype=np.int32)
    kernel = DeviceKernel.compile_and_load(
        _kv_update_wrapper,
        sk,
        sv,
        sc,
        ss,
        name=f"test_kv_update_t{token_bucket}_d{head_dim}",
        additional_compiler_args=COMPILER_ARGS,
        use_cached_if_exists=True,
        build_dir=build_dir,
    )
    kernel(
        inputs={
            "key": k_dev,
            "value": v_dev,
            "kv_cache.must_alias_input": kv_cache_dev,
            "slot_mapping": slot_dev,
        },
        outputs={"kv_cache": kv_cache_dev},
    )


# ============================================================
# Test
# ============================================================
def test_decode_accuracy(head_dim: int = 128):
    """Test: decode with 1 request, seq_len=6 (5 context + 1 query)."""
    print(f"\n{'#' * 60}")
    print(f"# Testing head_dim={head_dim}")
    print(f"{'#' * 60}")
    rng = np.random.default_rng(42)
    dtype = bfloat16
    token_bucket = 128
    seq_len = 6
    context_len = 5
    # Need enough blocks so the compiler's static bound check passes.
    # The NKI kernel tile has LARGE_KV_TILE_SIZE/BLOCK_SIZE = 1024/32 = 32
    # block entries per tile, so the cache must have >= 32 blocks.
    num_blocks = 64  # +1 scratch block added below
    real_total_tokens = 1

    # Generate random KV for the full sequence (6 tokens).
    all_k = (
        rng.standard_normal((seq_len, NUM_KV_HEADS, head_dim))
        .astype(np.float32)
        .astype(dtype)
    )
    all_v = (
        rng.standard_normal((seq_len, NUM_KV_HEADS, head_dim))
        .astype(np.float32)
        .astype(dtype)
    )
    q_token = (
        rng.standard_normal((1, NUM_HEADS, head_dim)).astype(np.float32).astype(dtype)
    )

    # Slot mapping: tokens 0..5 go to slots 0..5 (all in block 0).
    all_slots = np.arange(seq_len, dtype=np.int64)
    block_table = np.array([0], dtype=np.int64)  # 1 block needed

    print("=" * 60)
    print(
        "Step 1: Vanilla reference — write all 6 tokens to KV cache, run decode attention"
    )
    print("=" * 60)

    # Vanilla KV cache.
    vanilla_kv = np.zeros(
        (2, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, head_dim), dtype=dtype
    )
    # Write ALL 6 tokens (context + current) to cache.
    vanilla_update_kv_cache(all_k, all_v, vanilla_kv, all_slots)

    # Vanilla decode attention: query attends to all 6 tokens.
    vanilla_out = vanilla_decode_attention(q_token, vanilla_kv, seq_len, block_table)
    print(
        f"  Vanilla output: mean={np.mean(vanilla_out):.6f}, std={np.std(vanilla_out):.6f}"
    )
    print(f"  Vanilla output[:5]={vanilla_out[0, 0, :5]}")

    print()
    print("=" * 60)
    print(
        "Step 2: NKI path — write 5 context tokens, then write 1 current token, run NKI decode attention"
    )
    print("=" * 60)

    # NKI KV cache (+1 scratch block).
    nki_num_blocks = num_blocks + 1
    nki_kv_np = np.zeros(
        (2, nki_num_blocks, NUM_KV_HEADS, BLOCK_SIZE, head_dim), dtype=dtype
    )

    # Write context tokens (0..4) using vanilla (known correct).
    vanilla_update_kv_cache(
        all_k[:context_len],
        all_v[:context_len],
        nki_kv_np,
        all_slots[:context_len],
    )
    nki_kv_dev = DeviceTensor.from_numpy(nki_kv_np, name="kv_cache")

    # Write current token (5) using NKI kernel.
    current_k = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    current_k[:1] = all_k[context_len : context_len + 1]
    current_v = np.zeros((token_bucket, NUM_KV_HEADS, head_dim), dtype=dtype)
    current_v[:1] = all_v[context_len : context_len + 1]
    k_dev = DeviceTensor.from_numpy(current_k, name="k")
    v_dev = DeviceTensor.from_numpy(current_v, name="v")

    nki_kv_cache_update(
        k_dev,
        v_dev,
        nki_kv_dev,
        slot_mapping=all_slots[context_len : context_len + 1],
        token_bucket=token_bucket,
        real_total_tokens=1,
        num_blocks=nki_num_blocks,
        head_dim=head_dim,
    )

    # Verify KV cache write.
    nki_kv_after = nki_kv_dev.numpy()
    kv_diff = np.abs(
        nki_kv_after[:, :num_blocks].astype(np.float32)
        - vanilla_kv[:num_blocks].astype(np.float32)
    )
    print(f"  KV cache max diff (real blocks): {np.max(kv_diff):.8f}")

    # Run NKI decode attention.
    q_padded = np.zeros((token_bucket, NUM_HEADS, head_dim), dtype=dtype)
    q_padded[:1] = q_token
    q_dev = DeviceTensor.from_numpy(q_padded, name="q")
    k_dev_attn = DeviceTensor.from_numpy(current_k, name="k_attn")
    v_dev_attn = DeviceTensor.from_numpy(current_v, name="v_attn")

    nki_out = nki_decode_attention(
        q_dev,
        k_dev_attn,
        v_dev_attn,
        nki_kv_dev,
        seq_lens=np.array([seq_len], dtype=np.int64),
        query_start_loc=np.array([0, 1], dtype=np.int64),
        block_tables=np.array([[0]], dtype=np.int64),  # block 0
        token_bucket=token_bucket,
        real_total_tokens=real_total_tokens,
        num_blocks=nki_num_blocks,
        head_dim=head_dim,
    )

    nki_out_real = nki_out[:1].astype(np.float32)
    print(
        f"  NKI output: mean={np.mean(nki_out_real):.6f}, std={np.std(nki_out_real):.6f}"
    )
    print(f"  NKI output[:5]={nki_out_real[0, 0, :5]}")

    print()
    print("=" * 60)
    print("Step 3: Compare")
    print("=" * 60)
    diff = np.abs(vanilla_out - nki_out_real)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    cos_sims = []
    for h in range(NUM_HEADS):
        v_h = vanilla_out[0, h, :]
        n_h = nki_out_real[0, h, :]
        cos = np.dot(v_h, n_h) / (np.linalg.norm(v_h) * np.linalg.norm(n_h) + 1e-8)
        cos_sims.append(cos)
    print(f"  Max abs diff:   {max_diff:.6f}")
    print(f"  Mean abs diff:  {mean_diff:.6f}")
    print(f"  Cosine sim per head: {[f'{c:.6f}' for c in cos_sims]}")

    # bf16 precision: expect small numerical differences.
    if all(c > 0.95 for c in cos_sims):
        print(
            "\n  PASS: NKI decode attention matches vanilla reference (within bf16 precision)"
        )
    else:
        print("\n  FAIL: NKI decode attention does NOT match vanilla reference")
        print("  Vanilla first 10:", vanilla_out[0, 0, :10])
        print("  NKI first 10:    ", nki_out_real[0, 0, :10])
        sys.exit(1)


if __name__ == "__main__":
    for hd in [64, 128]:
        test_decode_accuracy(head_dim=hd)
    print("\nAll head_dim configurations passed.")
