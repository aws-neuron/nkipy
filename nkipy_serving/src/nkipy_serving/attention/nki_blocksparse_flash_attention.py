"""NKI BlockSparse Flash Attention backend for nkipy-serving.

Both nki_update_kv_cache and nki_blocksparse_attention compile NKI kernels
via DeviceKernel.compile_and_load. Q, K, V and the KV cache stay on device
throughout. Only tile-plan metadata (built on CPU) is uploaded per step.

This integration uses vllm-nkipy style attention execution:
  - ``include_prompt_in_ctx=True`` so prompt tokens are planned as context
    tiles (including the "active" tokens for both prefill and decode).
  - ``skip_active=True`` so the kernel does not run a separate active
    self-attention path.
  - ``active_mask=None`` always (no active block-diagonal mask tensor).

With this setup, we always call the unified mixed (prefill+decode) kernel.
Runtime batches that are decode-only or prefill-only pass a dummy fixed-shape
tile plan for the missing direction (0 loop steps + out-of-bounds write-back
indices), so the unused path becomes a no-op. The "active token" contribution
comes from the KV cache (which is updated before attention).

To avoid recompiles, attention kernels are compiled per ``token_bucket`` using
the unified mixed (prefill+decode) signature, with fixed tile-plan tensor
shapes derived from ``max_num_*_tiles``.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

from .base import AttentionMetadata
from .blocksparse_flash_attention.flash_paged_attn_varlen import (
    flash_paged_attention_varlen,
)
from .blocksparse_flash_attention.scheduler import (
    FlashAttentionPlanner,
    _kv_token_reorder_for_dge,
)
from .nki_paged_kv_cache import update_kv_cache

# ---------------------------------------------------------------------------
# Lazy runtime imports
# ---------------------------------------------------------------------------
_DeviceKernel = None
_DeviceTensor = None
_nki_op_imported = False


def _ensure_runtime():
    global _DeviceKernel, _DeviceTensor, _nki_op_imported
    if _DeviceKernel is not None:
        return
    if not _nki_op_imported:
        ensure_nki_bridge()
        _nki_op_imported = True
    from nkipy.runtime import DeviceKernel

    _DeviceKernel = DeviceKernel
    _DeviceTensor = get_device_tensor_cls()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_LARGE_Q_TILE_SIZE = 128
_LARGE_KV_TILE_SIZE = 1024
_DYNAMIC_LOOP_UNROLLING_SIZE = 8
_B_P_SIZE = 128  # nl.tile_size.pmax — constant for NeuronCore v2
_NKI_COMPILER_ARGS = "--tensorizer-options='--skip-pass=LateLegalizePostSplit'"
_DECODE_MASK_LOOKUP_CACHE: dict[int, np.ndarray] = {}

_TILE_FIELDS = (
    "tile_q_indices",
    "tile_block_tables",
    "tile_masks",
    "num_dynamic_loop_steps",
    "q_update_pred",
    "last_tile_indices",
)

# Public aliases for reuse in fused graph modes.
NKI_MIN_Q_SEQLEN = _B_P_SIZE
NKI_COMPILER_ARGS = _NKI_COMPILER_ARGS
NKI_DYNAMIC_LOOP_UNROLLING_SIZE = _DYNAMIC_LOOP_UNROLLING_SIZE


# ---------------------------------------------------------------------------
# Tile count computation
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def compute_max_tile_counts(
    token_bucket: int,
    max_context_len: int,
    max_requests: int,
    block_size: int,
) -> tuple[int, int]:
    """Compute deterministic max tile counts for a given token_bucket.

    Returns (max_num_prefill_tiles, max_num_decode_tiles).  These are used
    to pad tile plans to fixed shapes so that each token_bucket attention
    kernel compiles once regardless of batch composition.
    """
    max_num_kv_tiles = _ceil_div(max_context_len, _LARGE_KV_TILE_SIZE)

    # Decode: each request has 1 query token, up to max_requests requests.
    max_num_decode_tiles = _round_up(
        max_num_kv_tiles * max_requests,
        _DYNAMIC_LOOP_UNROLLING_SIZE,
    )

    # Prefill: token_bucket query tokens, each request up to max_context_len.
    max_sum_q_tiles = _ceil_div(token_bucket, _LARGE_Q_TILE_SIZE) + max_requests
    max_num_prefill_tiles = _round_up(
        max_num_kv_tiles * max_sum_q_tiles,
        _DYNAMIC_LOOP_UNROLLING_SIZE,
    )

    return max_num_prefill_tiles, max_num_decode_tiles


# ---------------------------------------------------------------------------
# Dummy tile plan builders
# ---------------------------------------------------------------------------

# Write-back skip value: merge_decode_buffer uses oob_mode.skip, so
# stores to out-of-bounds positions are silently dropped.  Dummy tile
# plans use this as the write-back index (column 1 of last_tile_indices)
# to ensure merge is a no-op when a direction has 0 real tiles.
_WRITE_BACK_SKIP = 100_000_000


def _build_dummy_prefill_plan(
    max_num_prefill_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Zero-filled prefill tile plan with fixed shapes and 0 loop steps."""
    kv_blocks_per_tile = _LARGE_KV_TILE_SIZE // block_size
    P = max_num_prefill_tiles
    lti = np.zeros((_B_P_SIZE, 2), dtype=np.int32)
    lti[:, 1] = _WRITE_BACK_SKIP
    return {
        "tile_q_indices": np.zeros((P, _LARGE_Q_TILE_SIZE), dtype=np.int32),
        "tile_block_tables": np.zeros((P, kv_blocks_per_tile), dtype=np.int32),
        "tile_masks": np.zeros(
            (_B_P_SIZE, P, _LARGE_Q_TILE_SIZE // _B_P_SIZE, _LARGE_KV_TILE_SIZE),
            dtype=np.uint8,
        ),
        "num_dynamic_loop_steps": np.zeros((1, 1), dtype=np.int32),
        "q_update_pred": np.zeros((P, 1), dtype=np.uint8),
        "last_tile_indices": lti,
    }


def _build_dummy_decode_plan(
    max_num_decode_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Zero-filled decode tile plan with fixed shapes and 0 loop steps."""
    kv_blocks_per_tile = _LARGE_KV_TILE_SIZE // block_size
    D = max_num_decode_tiles
    lti = np.zeros((_B_P_SIZE, 2), dtype=np.int32)
    lti[:, 1] = _WRITE_BACK_SKIP
    return {
        "tile_q_indices": np.zeros((D, 1), dtype=np.int32),
        "tile_block_tables": np.zeros((D, kv_blocks_per_tile), dtype=np.int32),
        "tile_masks": np.zeros(
            (_B_P_SIZE, D, _LARGE_KV_TILE_SIZE // _B_P_SIZE),
            dtype=np.uint8,
        ),
        "num_dynamic_loop_steps": np.zeros((1, 1), dtype=np.int32),
        "q_update_pred": np.zeros((D, 1), dtype=np.uint8),
        "last_tile_indices": lti,
    }


# ===================================================================
# KV cache update
# ===================================================================


def _kv_update_wrapper(key, value, kv_cache, slot_mapping):
    return update_kv_cache(key, value, kv_cache, slot_mapping)


def ensure_nki_kv_update_kernel(
    *,
    token_bucket: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    kernel_cache,
    build_dir: str = "/tmp/build",
):
    """Compile or fetch a cached KV-update kernel for a fixed bucket."""
    _ensure_runtime()
    dtype = bfloat16
    ck = (token_bucket, num_kv_heads, head_dim, num_blocks, block_size)
    cached = kernel_cache.get_kv_update_kernel(ck)
    if cached is not None:
        return cached
    sk = np.zeros((token_bucket, num_kv_heads, head_dim), dtype=dtype)
    sv = np.zeros((token_bucket, num_kv_heads, head_dim), dtype=dtype)
    sc = np.zeros((2, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
    ss = np.zeros((token_bucket,), dtype=np.int32)
    cached = _DeviceKernel.compile_and_load(
        _kv_update_wrapper,
        sk,
        sv,
        sc,
        ss,
        name=f"nki_kv_update_t{token_bucket}",
        additional_compiler_args=_NKI_COMPILER_ARGS,
        use_cached_if_exists=True,
        build_dir=build_dir,
    )
    kernel_cache.set_kv_update_kernel(ck, cached)
    return cached


def run_prepared_nki_kv_update(
    k_dev,
    v_dev,
    kv_cache_dev,
    *,
    slot_mapping_dev,
    token_bucket: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    kernel_cache,
    build_dir: str = "/tmp/build",
    kernel=None,
):
    """Run KV update with a caller-owned slot-mapping device tensor."""
    if kernel is None:
        kernel = ensure_nki_kv_update_kernel(
            token_bucket=token_bucket,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            kernel_cache=kernel_cache,
            build_dir=build_dir,
        )
    kernel(
        inputs={
            "key": k_dev,
            "value": v_dev,
            "kv_cache.must_alias_input": kv_cache_dev,
            "slot_mapping": slot_mapping_dev,
        },
        outputs={"kv_cache": kv_cache_dev},
    )
    return kv_cache_dev


def nki_update_kv_cache(
    k_dev,
    v_dev,
    kv_cache_dev,
    slot_mapping: np.ndarray,
    token_bucket: int,
    real_total_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    kernel_cache,
    build_dir: str = "/tmp/build",
):
    _ensure_runtime()

    scratch_slot = (num_blocks - 1) * block_size
    padded_slot = np.full(token_bucket, scratch_slot, dtype=np.int32)
    padded_slot[:real_total_tokens] = slot_mapping
    slot_dev = _DeviceTensor.from_numpy(padded_slot, name="slot_mapping")
    run_prepared_nki_kv_update(
        k_dev,
        v_dev,
        kv_cache_dev,
        slot_mapping_dev=slot_dev,
        token_bucket=token_bucket,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        kernel_cache=kernel_cache,
        build_dir=build_dir,
    )


# ===================================================================
# Attention — wrapper
# ===================================================================


def _reshape_qkv(q, k, v, kv_cache):
    """Reshape (T,H,D) → (1,H,T,D) for the NKI kernel."""
    q_4d = q.reshape((1,) + q.shape).transpose(0, 2, 1, 3)
    k_4d = k.reshape((1,) + k.shape).transpose(0, 2, 3, 1)
    v_4d = v.reshape((1,) + v.shape).transpose(0, 2, 1, 3)
    return q_4d, k_4d, v_4d, kv_cache[0], kv_cache[1]


def _make_attn_wrapper_unified(softmax_scale):
    """Both prefill and decode tile plans (mixed batch)."""

    def wrapper(
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
        q_4d, k_4d, v_4d, k_cache, v_cache = _reshape_qkv(q, k, v, kv_cache)
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
            dynamic_loop_unroll_factor=_DYNAMIC_LOOP_UNROLLING_SIZE,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            skip_active=True,
        )
        return out[0].transpose(1, 0, 2)

    return wrapper


def _make_attn_wrapper_unified_with_sink(softmax_scale):
    """Unified attention wrapper that threads through a per-head sink tensor."""

    def wrapper(
        q,
        k,
        v,
        kv_cache,
        sink,
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
        q_4d, k_4d, v_4d, k_cache, v_cache = _reshape_qkv(q, k, v, kv_cache)
        out = flash_paged_attention_varlen[1](
            q_4d,
            k_4d,
            v_4d,
            k_cache,
            v_cache,
            None,
            sink,
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
            dynamic_loop_unroll_factor=_DYNAMIC_LOOP_UNROLLING_SIZE,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            skip_active=True,
        )
        return out[0].transpose(1, 0, 2)

    return wrapper


# ---------------------------------------------------------------------------
# Traceable attention + KV-update core helpers (callable inside larger graphs)
# ---------------------------------------------------------------------------


def nki_update_kv_cache_core(key, value, kv_cache, slot_mapping):
    """Traceable KV-cache update (NKI custom-call)."""
    return update_kv_cache(key, value, kv_cache, slot_mapping)


def nki_attention_unified(
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
    softmax_scale: float,
):
    """Traceable unified (prefill+decode) paged-attention (NKI custom-call)."""
    q_4d, k_4d, v_4d, k_cache, v_cache = _reshape_qkv(q, k, v, kv_cache)
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
        dynamic_loop_unroll_factor=_DYNAMIC_LOOP_UNROLLING_SIZE,
        softmax_scale=softmax_scale,
        mixed_precision=True,
        skip_active=True,
    )
    return out[0].transpose(1, 0, 2)


def nki_attention_unified_with_sink(
    q,
    k,
    v,
    kv_cache,
    sink,
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
    softmax_scale: float,
):
    """Traceable unified paged-attention with sink tensor threaded through."""
    q_4d, k_4d, v_4d, k_cache, v_cache = _reshape_qkv(q, k, v, kv_cache)
    out = flash_paged_attention_varlen[1](
        q_4d,
        k_4d,
        v_4d,
        k_cache,
        v_cache,
        None,
        sink,
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
        dynamic_loop_unroll_factor=_DYNAMIC_LOOP_UNROLLING_SIZE,
        softmax_scale=softmax_scale,
        mixed_precision=True,
        skip_active=True,
    )
    return out[0].transpose(1, 0, 2)


# ===================================================================
# Attention — tile plan helpers (numpy-only)
# ===================================================================


def _as_int32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.int32)


def _build_tile_plan(
    *,
    prompt_lens: np.ndarray,
    prior_context_lens: np.ndarray,
    prompt_starts: np.ndarray,
    block_tables: np.ndarray,
    token_bucket: int,
    block_size: int,
    max_seq_len: int,
    tile_size_q: int,
    tile_size_kv: int,
    max_num_tiles: int,
    q_pad_value: int,
    decode_kq_layout: bool,
) -> dict[str, np.ndarray]:
    plan = FlashAttentionPlanner.MakeTilePlan(
        prompt_lens=_as_int32(prompt_lens),
        prior_context_lens=_as_int32(prior_context_lens),
        tile_size_q=int(tile_size_q),
        tile_size_kv=int(tile_size_kv),
        block_size=int(block_size),
        prompt_starts=_as_int32(prompt_starts),
        include_prompt_in_context=True,
        max_seq_len=int(max_seq_len),
    )
    if int(plan.num_real_tiles) > int(max_num_tiles):
        raise RuntimeError(
            "Tile plan exceeds max tiles for this bucket. "
            f"{plan.num_real_tiles=} {max_num_tiles=}. "
            "Increase request/token buckets or max_context_len, or recompute max tiles."
        )

    plan = plan.pad_plan(int(max_num_tiles), q_pad_value=int(q_pad_value))

    # Indices/masks.
    tile_q_indices = np.asarray(
        plan.build_tile_q_indices(skip_value=int(q_pad_value)),
        dtype=np.int32,
    )
    tile_masks = np.asarray(
        plan.build_tile_masks(decode_kq_layout=bool(decode_kq_layout)),
        dtype=np.uint8,
    )
    tile_block_tables = np.asarray(
        plan.build_tile_block_tables(_as_int32(block_tables), skip_value=0),
        dtype=np.int32,
    )

    # Dynamic loop steps: tiles processed = steps * unroll_size.
    num_dynamic_loop_steps = np.zeros((1, 1), dtype=np.int32)
    num_dynamic_loop_steps[0, 0] = _ceil_div(
        int(plan.num_real_tiles), int(_DYNAMIC_LOOP_UNROLLING_SIZE)
    )

    # Per-tile update predicate + per-q-tile last-tile indices.
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


def _build_unified_tile_plans(
    metadata: AttentionMetadata,
    *,
    token_bucket: int,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    block_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    query_lens = _as_int32(np.diff(metadata.query_start_loc))
    context_lens = _as_int32(metadata.seq_lens - query_lens)
    prompt_starts = _as_int32(metadata.query_start_loc[:-1])

    prefill_batch_indices = np.nonzero(query_lens > 1)[0].astype(np.int32)
    decode_batch_indices = np.nonzero(query_lens == 1)[0].astype(np.int32)

    prefill_nps = _build_dummy_prefill_plan(max_num_prefill_tiles, block_size)
    decode_nps = _build_dummy_decode_plan(max_num_decode_tiles, block_size)

    if prefill_batch_indices.size > 0:
        prefill_nps = _build_tile_plan(
            prompt_lens=query_lens[prefill_batch_indices],
            prior_context_lens=context_lens[prefill_batch_indices],
            prompt_starts=prompt_starts[prefill_batch_indices],
            block_tables=metadata.block_tables[prefill_batch_indices],
            token_bucket=token_bucket,
            block_size=block_size,
            max_seq_len=int(metadata.max_seq_len),
            tile_size_q=_LARGE_Q_TILE_SIZE,
            tile_size_kv=_LARGE_KV_TILE_SIZE,
            max_num_tiles=max_num_prefill_tiles,
            q_pad_value=int(token_bucket) * 10,
            decode_kq_layout=False,
        )

    if decode_batch_indices.size > 0:
        decode_nps = _build_tile_plan(
            prompt_lens=query_lens[decode_batch_indices],
            prior_context_lens=context_lens[decode_batch_indices],
            prompt_starts=prompt_starts[decode_batch_indices],
            block_tables=metadata.block_tables[decode_batch_indices],
            token_bucket=token_bucket,
            block_size=block_size,
            max_seq_len=int(metadata.max_seq_len),
            tile_size_q=1,
            tile_size_kv=_LARGE_KV_TILE_SIZE,
            max_num_tiles=max_num_decode_tiles,
            q_pad_value=0,
            decode_kq_layout=True,
        )

    return prefill_nps, decode_nps


def build_unified_tile_plans(
    metadata: AttentionMetadata,
    *,
    token_bucket: int,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    block_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Public wrapper for building fixed-shape unified tile plans on CPU."""
    return _build_unified_tile_plans(
        metadata,
        token_bucket=token_bucket,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        block_size=block_size,
    )


def build_dummy_prefill_tile_plan(
    *,
    max_num_prefill_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Public wrapper for a fixed-shape no-op prefill plan."""
    return _build_dummy_prefill_plan(max_num_prefill_tiles, block_size)


def build_dummy_decode_tile_plan(
    *,
    max_num_decode_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Public wrapper for a fixed-shape no-op decode plan."""
    return _build_dummy_decode_plan(max_num_decode_tiles, block_size)


def build_prefill_tile_plan(
    metadata: AttentionMetadata,
    *,
    token_bucket: int,
    max_num_prefill_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Build only the prefill-side tile plan for a runtime batch."""
    query_lens = _as_int32(np.diff(metadata.query_start_loc))
    context_lens = _as_int32(metadata.seq_lens - query_lens)
    prompt_starts = _as_int32(metadata.query_start_loc[:-1])
    prefill_batch_indices = np.nonzero(query_lens > 1)[0].astype(np.int32)
    if prefill_batch_indices.size == 0:
        return _build_dummy_prefill_plan(max_num_prefill_tiles, block_size)
    return _build_tile_plan(
        prompt_lens=query_lens[prefill_batch_indices],
        prior_context_lens=context_lens[prefill_batch_indices],
        prompt_starts=prompt_starts[prefill_batch_indices],
        block_tables=metadata.block_tables[prefill_batch_indices],
        token_bucket=token_bucket,
        block_size=block_size,
        max_seq_len=int(metadata.max_seq_len),
        tile_size_q=_LARGE_Q_TILE_SIZE,
        tile_size_kv=_LARGE_KV_TILE_SIZE,
        max_num_tiles=max_num_prefill_tiles,
        q_pad_value=int(token_bucket) * 10,
        decode_kq_layout=False,
    )


def build_prefill_tile_plan_inplace(
    metadata: AttentionMetadata,
    *,
    token_bucket: int,
    max_num_prefill_tiles: int,
    block_size: int,
    out: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fill a reusable host-side prefill tile plan buffer in place."""
    q_pad_value = int(token_bucket) * 10
    query_lens = _as_int32(np.diff(metadata.query_start_loc))
    context_lens = _as_int32(metadata.seq_lens - query_lens)
    prompt_starts = _as_int32(metadata.query_start_loc[:-1])
    prefill_batch_indices = np.nonzero(query_lens > 1)[0].astype(np.int32)

    tile_q_indices = out["tile_q_indices"]
    tile_block_tables = out["tile_block_tables"]
    tile_masks = out["tile_masks"]
    num_dynamic_loop_steps = out["num_dynamic_loop_steps"]
    q_update_pred = out["q_update_pred"]
    last_tile_indices = out["last_tile_indices"]

    tile_q_indices.fill(q_pad_value)
    tile_block_tables.fill(0)
    tile_masks.fill(0)
    num_dynamic_loop_steps.fill(0)
    q_update_pred.fill(0)
    last_tile_indices[:, 0].fill(0)
    last_tile_indices[:, 1].fill(_WRITE_BACK_SKIP)

    if prefill_batch_indices.size == 0:
        return out

    plan = FlashAttentionPlanner.MakeTilePlan(
        prompt_lens=query_lens[prefill_batch_indices],
        prior_context_lens=context_lens[prefill_batch_indices],
        tile_size_q=_LARGE_Q_TILE_SIZE,
        tile_size_kv=_LARGE_KV_TILE_SIZE,
        block_size=int(block_size),
        prompt_starts=prompt_starts[prefill_batch_indices],
        include_prompt_in_context=True,
        max_seq_len=int(metadata.max_seq_len),
    )
    if int(plan.num_real_tiles) > int(max_num_prefill_tiles):
        raise RuntimeError(
            "Prefill tile plan exceeds max tiles for this bucket. "
            f"{plan.num_real_tiles=} {max_num_prefill_tiles=}. "
            "Increase request/token buckets or max_context_len, or recompute max tiles."
        )

    num_tiles = int(plan.num_tiles)
    tile_q_indices[:num_tiles, :] = np.asarray(
        plan.build_tile_q_indices(skip_value=q_pad_value),
        dtype=np.int32,
    )
    tile_block_tables[:num_tiles, :] = np.asarray(
        plan.build_tile_block_tables(
            _as_int32(metadata.block_tables[prefill_batch_indices]),
            skip_value=0,
        ),
        dtype=np.int32,
    )
    tile_masks[:, :num_tiles, :, :] = np.asarray(
        plan.build_tile_masks(decode_kq_layout=False),
        dtype=np.uint8,
    )
    num_dynamic_loop_steps[0, 0] = _ceil_div(
        int(plan.num_real_tiles), int(_DYNAMIC_LOOP_UNROLLING_SIZE)
    )
    q_update_pred_np, last_tile_indices_np = plan.build_tile_update_indices(
        max_num_q_tiles=_B_P_SIZE
    )
    q_update_pred[:num_tiles, 0] = np.asarray(q_update_pred_np, dtype=np.uint8)[
        :num_tiles
    ]
    last_tile_indices[:, :] = np.asarray(last_tile_indices_np, dtype=np.int32)
    return out


def build_decode_tile_plan(
    metadata: AttentionMetadata,
    *,
    token_bucket: int,
    max_num_decode_tiles: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Build only the decode-side tile plan for a runtime batch."""
    query_lens = _as_int32(np.diff(metadata.query_start_loc))
    context_lens = _as_int32(metadata.seq_lens - query_lens)
    prompt_starts = _as_int32(metadata.query_start_loc[:-1])
    decode_batch_indices = np.nonzero(query_lens == 1)[0].astype(np.int32)
    if decode_batch_indices.size == 0:
        return _build_dummy_decode_plan(max_num_decode_tiles, block_size)
    return _build_tile_plan(
        prompt_lens=query_lens[decode_batch_indices],
        prior_context_lens=context_lens[decode_batch_indices],
        prompt_starts=prompt_starts[decode_batch_indices],
        block_tables=metadata.block_tables[decode_batch_indices],
        token_bucket=token_bucket,
        block_size=block_size,
        max_seq_len=int(metadata.max_seq_len),
        tile_size_q=1,
        tile_size_kv=_LARGE_KV_TILE_SIZE,
        max_num_tiles=max_num_decode_tiles,
        q_pad_value=0,
        decode_kq_layout=True,
    )


def _get_decode_mask_lookup(block_size: int) -> np.ndarray:
    cached = _DECODE_MASK_LOOKUP_CACHE.get(int(block_size))
    if cached is not None:
        return cached

    valid_counts = np.arange(_LARGE_KV_TILE_SIZE + 1, dtype=np.int32).reshape((-1, 1))
    kv_valid = (
        np.arange(_LARGE_KV_TILE_SIZE, dtype=np.int32).reshape((1, _LARGE_KV_TILE_SIZE))
        < valid_counts
    ).astype(np.uint8)
    kv_valid = _kv_token_reorder_for_dge(
        kv_valid,
        tile_size_kv=_LARGE_KV_TILE_SIZE,
        block_size=int(block_size),
    )
    cached = kv_valid.reshape(
        (-1, _LARGE_KV_TILE_SIZE // _B_P_SIZE, _B_P_SIZE)
    ).transpose(0, 2, 1)
    _DECODE_MASK_LOOKUP_CACHE[int(block_size)] = cached
    return cached


def build_decode_tile_plan_inplace(
    metadata: AttentionMetadata,
    *,
    max_num_decode_tiles: int,
    block_size: int,
    out: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fill a reusable host-side decode tile plan buffer in place."""
    query_lens = _as_int32(np.diff(metadata.query_start_loc))
    decode_batch_indices = np.nonzero(query_lens == 1)[0].astype(np.int32)
    if decode_batch_indices.size == 0:
        return out

    prompt_starts = _as_int32(metadata.query_start_loc[:-1])[decode_batch_indices]
    full_context_lens = _as_int32(metadata.seq_lens)[decode_batch_indices]
    block_tables = _as_int32(metadata.block_tables[decode_batch_indices])
    kv_blocks_per_tile = _LARGE_KV_TILE_SIZE // int(block_size)
    num_kv_tiles = (full_context_lens + _LARGE_KV_TILE_SIZE - 1) // _LARGE_KV_TILE_SIZE
    num_context_blocks = (full_context_lens + int(block_size) - 1) // int(block_size)
    num_real_tiles = int(num_kv_tiles.sum())
    if num_real_tiles > int(max_num_decode_tiles):
        raise RuntimeError(
            "Decode tile plan exceeds max tiles for this bucket. "
            f"{num_real_tiles=} {max_num_decode_tiles=}. "
            "Increase request buckets or max_context_len, or recompute max tiles."
        )

    tile_q_indices = out["tile_q_indices"]
    tile_block_tables = out["tile_block_tables"]
    tile_masks = out["tile_masks"]
    num_dynamic_loop_steps = out["num_dynamic_loop_steps"]
    q_update_pred = out["q_update_pred"]
    last_tile_indices = out["last_tile_indices"]

    tile_q_indices.fill(0)
    tile_block_tables.fill(0)
    tile_masks.fill(0)
    num_dynamic_loop_steps.fill(0)
    q_update_pred.fill(0)
    last_tile_indices[:, 0].fill(0)
    last_tile_indices[:, 1].fill(_WRITE_BACK_SKIP)

    if num_real_tiles == 0:
        return out

    repeated_prompt_starts = np.repeat(prompt_starts, num_kv_tiles).astype(
        np.int32, copy=False
    )
    tile_q_indices[:num_real_tiles, 0] = repeated_prompt_starts

    valid_tokens = np.empty((num_real_tiles,), dtype=np.int32)
    row_start = 0
    mask_lookup = _get_decode_mask_lookup(int(block_size))
    for seq_idx, (tiles_for_seq, blocks_for_seq, full_context_len) in enumerate(
        zip(
            num_kv_tiles.tolist(),
            num_context_blocks.tolist(),
            full_context_lens.tolist(),
        )
    ):
        row_end = row_start + int(tiles_for_seq)
        seq_block_values = np.zeros(
            (int(tiles_for_seq) * kv_blocks_per_tile,), dtype=np.int32
        )
        seq_block_values[: int(blocks_for_seq)] = block_tables[
            seq_idx, : int(blocks_for_seq)
        ]
        tile_block_tables[row_start:row_end, :] = seq_block_values.reshape(
            (int(tiles_for_seq), kv_blocks_per_tile)
        )
        tile_starts = (
            np.arange(int(tiles_for_seq), dtype=np.int32) * _LARGE_KV_TILE_SIZE
        )
        valid_tokens[row_start:row_end] = np.minimum(
            np.maximum(int(full_context_len) - tile_starts, 0),
            _LARGE_KV_TILE_SIZE,
        )
        row_start = row_end

    tile_masks[:, :num_real_tiles, :] = mask_lookup[valid_tokens].transpose(1, 0, 2)
    num_dynamic_loop_steps[0, 0] = _ceil_div(
        num_real_tiles, _DYNAMIC_LOOP_UNROLLING_SIZE
    )

    last_tile_load_indices = np.cumsum(num_kv_tiles, dtype=np.int32) - 1
    q_update_pred[:num_real_tiles, 0] = 1
    q_update_pred[last_tile_load_indices, 0] = 0
    num_real_q_tiles = int(last_tile_load_indices.size)
    last_tile_indices[:num_real_q_tiles, 0] = last_tile_load_indices
    last_tile_indices[:num_real_q_tiles, 1] = prompt_starts
    if num_real_q_tiles < _B_P_SIZE:
        last_tile_indices[num_real_q_tiles:, 0] = int(last_tile_load_indices[-1])

    return out


# ---------------------------------------------------------------------------
# Compile + padding helpers
# ---------------------------------------------------------------------------


def _compile_attention_kernel(
    *,
    token_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    build_dir: str,
    has_sink: bool,
    softmax_scale: float | None = None,
):
    _ensure_runtime()
    dtype = bfloat16
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    sq = np.zeros((token_bucket, num_heads, head_dim), dtype=dtype)
    sk = np.zeros((token_bucket, num_kv_heads, head_dim), dtype=dtype)
    sv = np.zeros((token_bucket, num_kv_heads, head_dim), dtype=dtype)
    sc = np.zeros((2, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
    ssink = np.zeros((num_heads, 1), dtype=dtype) if has_sink else None

    tag = (
        "nki_attn_ctx_sa_unified"
        f"_t{token_bucket}"
        f"_h{num_heads}"
        f"_kv{num_kv_heads}"
        f"_hd{head_dim}"
        f"_nb{num_blocks}"
        f"_bs{block_size}"
        f"_p{max_num_prefill_tiles}"
        f"_dc{max_num_decode_tiles}"
    )
    prefill_nps = _build_dummy_prefill_plan(max_num_prefill_tiles, block_size)
    decode_nps = _build_dummy_decode_plan(max_num_decode_tiles, block_size)
    if has_sink:
        wrapper_fn = _make_attn_wrapper_unified_with_sink(softmax_scale)
    else:
        wrapper_fn = _make_attn_wrapper_unified(softmax_scale)

    sample_args = [sq, sk, sv, sc]
    if has_sink:
        sample_args.append(ssink)
    for f in _TILE_FIELDS:
        sample_args.append(np.zeros_like(prefill_nps[f]))
    for f in _TILE_FIELDS:
        sample_args.append(np.zeros_like(decode_nps[f]))

    return _DeviceKernel.compile_and_load(
        wrapper_fn,
        *sample_args,
        name=tag,
        additional_compiler_args=_NKI_COMPILER_ARGS,
        use_cached_if_exists=True,
        build_dir=build_dir,
    )


def precompile_nki_attention_kernels(
    *,
    token_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    kernel_cache,
    build_dir: str = "/tmp/build",
    has_sink: bool = False,
    softmax_scale: float | None = None,
) -> None:
    """Compile and cache the unified NKI attention kernel for a bucket.

    The unified kernel always uses the mixed (prefill+decode) signature with
    fixed-shape tile plans. Runtime batches that have only prefill or only
    decode pass a dummy plan for the missing direction with 0 loop steps and
    out-of-bounds write-back indices so merging is a no-op.
    """
    ck = (
        token_bucket,
        num_heads,
        num_kv_heads,
        head_dim,
        num_blocks,
        block_size,
        max_num_prefill_tiles,
        max_num_decode_tiles,
        bool(has_sink),
        float(softmax_scale) if softmax_scale is not None else None,
    )
    if kernel_cache.get_attention_kernel(ck) is not None:
        return
    kernel = _compile_attention_kernel(
        token_bucket=token_bucket,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        build_dir=build_dir,
        has_sink=has_sink,
        softmax_scale=softmax_scale,
    )
    kernel_cache.set_attention_kernel(ck, kernel)


def ensure_nki_attention_kernel(
    *,
    token_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    kernel_cache,
    build_dir: str = "/tmp/build",
    has_sink: bool = False,
    softmax_scale: float | None = None,
):
    """Compile or fetch a cached unified attention kernel for a fixed bucket."""
    precompile_nki_attention_kernels(
        token_bucket=token_bucket,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        kernel_cache=kernel_cache,
        build_dir=build_dir,
        has_sink=has_sink,
        softmax_scale=softmax_scale,
    )
    ck = (
        token_bucket,
        num_heads,
        num_kv_heads,
        head_dim,
        num_blocks,
        block_size,
        max_num_prefill_tiles,
        max_num_decode_tiles,
        bool(has_sink),
        float(softmax_scale) if softmax_scale is not None else None,
    )
    kernel = kernel_cache.get_attention_kernel(ck)
    if kernel is None:
        raise RuntimeError("Expected compiled NKI attention kernel in cache")
    return kernel


def run_prepared_nki_blocksparse_attention(
    q_dev,
    k_dev,
    v_dev,
    kv_cache_dev,
    sink_dev,
    *,
    prepared_inputs: dict[str, object],
    out_dev,
    token_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    kernel_cache,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    build_dir: str = "/tmp/build",
    softmax_scale: float | None = None,
    kernel=None,
):
    """Run unified attention with caller-owned prepared tile-plan tensors."""
    if kernel is None:
        kernel = ensure_nki_attention_kernel(
            token_bucket=token_bucket,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_prefill_tiles=max_num_prefill_tiles,
            max_num_decode_tiles=max_num_decode_tiles,
            kernel_cache=kernel_cache,
            build_dir=build_dir,
            has_sink=(sink_dev is not None),
            softmax_scale=softmax_scale,
        )

    inputs = {
        "q": q_dev,
        "k": k_dev,
        "v": v_dev,
        "kv_cache": kv_cache_dev,
        "p_tqi": prepared_inputs["p_tqi"],
        "p_tbt": prepared_inputs["p_tbt"],
        "p_tm": prepared_inputs["p_tm"],
        "p_ndls": prepared_inputs["p_ndls"],
        "p_qup": prepared_inputs["p_qup"],
        "p_lti": prepared_inputs["p_lti"],
        "d_tqi": prepared_inputs["d_tqi"],
        "d_tbt": prepared_inputs["d_tbt"],
        "d_tm": prepared_inputs["d_tm"],
        "d_ndls": prepared_inputs["d_ndls"],
        "d_qup": prepared_inputs["d_qup"],
        "d_lti": prepared_inputs["d_lti"],
    }
    if sink_dev is not None:
        inputs["sink"] = sink_dev
    kernel(inputs=inputs, outputs={"output0": out_dev})
    return out_dev


# ===================================================================
# Attention — main entry
# ===================================================================


def nki_blocksparse_attention(
    q_dev,
    k_dev,
    v_dev,
    kv_cache_dev,
    sink_dev,
    metadata: AttentionMetadata,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    token_bucket: int,
    real_total_tokens: int,
    num_blocks: int,
    block_size: int,
    kernel_cache,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    build_dir: str = "/tmp/build",
    softmax_scale: float | None = None,
):
    """Run NKI attention via DeviceKernel.  Q, K, V, KV cache stay on device.

    Compiles and caches per (token_bucket, max_num_prefill_tiles,
    max_num_decode_tiles). Tile plans are padded to deterministic max shapes so
    runtime batches reuse the same compiled kernel.
    """
    _ensure_runtime()
    dtype = bfloat16

    # --- 1. Build tile plans on CPU ---
    prefill_nps, decode_nps = _build_unified_tile_plans(
        metadata,
        token_bucket=token_bucket,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        block_size=block_size,
    )

    out_np = np.zeros((token_bucket, num_heads, head_dim), dtype=dtype)
    out_dev = _DeviceTensor.from_numpy(out_np, name="attn_out")
    prepared_inputs = {}
    pnames = ["p_tqi", "p_tbt", "p_tm", "p_ndls", "p_qup", "p_lti"]
    for nm, f in zip(pnames, _TILE_FIELDS):
        prepared_inputs[nm] = _DeviceTensor.from_numpy(prefill_nps[f], name=f"pf_{f}")
    dnames = ["d_tqi", "d_tbt", "d_tm", "d_ndls", "d_qup", "d_lti"]
    for nm, f in zip(dnames, _TILE_FIELDS):
        prepared_inputs[nm] = _DeviceTensor.from_numpy(decode_nps[f], name=f"dc_{f}")
    return run_prepared_nki_blocksparse_attention(
        q_dev,
        k_dev,
        v_dev,
        kv_cache_dev,
        sink_dev,
        prepared_inputs=prepared_inputs,
        out_dev=out_dev,
        token_bucket=token_bucket,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        kernel_cache=kernel_cache,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        build_dir=build_dir,
        softmax_scale=softmax_scale,
    )
