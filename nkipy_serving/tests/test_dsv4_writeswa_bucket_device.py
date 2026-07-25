"""Device accuracy test for the bucketed prefill fused cache-write.

Validates that ``run_write_swa_dual_kv_score_state_owner_clen_device`` compiled
at a token bucket whose ``clen`` exceeds the real prompt's compressed length
writes EXACTLY the real rows into all 5 caches and never corrupts a live slot
with padding (the bucketed-prefill correctness claim).

Run with: NEURON_RT_VISIBLE_CORES=0 pytest --run-integration --run-device-dsv4 \
  tests/test_dsv4_writeswa_bucket_device.py -v
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

pytest.importorskip("neuronxcc.nki")

from nkipy.runtime.device_tensor import DeviceTensor

from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_write_swa_dual_kv_score_state_owner_clen_device,
    run_write_swa_kv_score_state_owner_clen_device,
)

pytestmark = [pytest.mark.integration, pytest.mark.device_dsv4]


@pytest.mark.parametrize(
    "bsz,real_seqlen,bucket",
    [
        (1, 9, 16),
        (2, 9, 16),
        (1, 5, 16),
        (2, 6, 8),
        # Long prefill (real > window): only the last-window tokens may write
        # SWA (older positions collide mod window with no scatter ordering).
        (1, 200, 256),
        (1, 300, 512),
        (1, 256, 256),
    ],
)
def test_bucketed_write_matches_real_length(bsz, real_seqlen, bucket, tmp_path):
    rng = np.random.default_rng(0)
    ratio = 4
    window = 128
    head_dim = 16
    idx_head_dim = 8
    max_batch = max(2, bsz)
    guard_owner = max_batch  # caches sized for (max_batch + 1) owners
    n_owners = max_batch + 1
    max_clen = bucket // ratio + 2  # room for real + guard
    # Use overlap semantics consistent with ratio==4 served path:
    keep = min(real_seqlen, ratio + real_seqlen % ratio)
    ring = 2 * ratio  # overlap ring
    MAXKEEP = 2 * ratio - 1
    real_clen = real_seqlen // ratio
    buck_clen = bucket // ratio

    width = head_dim
    idx_width = idx_head_dim

    # ---- caches (zeroed; guard owner block included) ----
    swa_cache = np.zeros((n_owners * window, head_dim), dtype=np.float32)
    state_cache = np.zeros((n_owners * ring, 2 * width), dtype=np.float32)
    comp_cache = np.zeros((n_owners * max_clen, head_dim), dtype=np.float32)
    idx_state_cache = np.zeros((n_owners * ring, 2 * idx_width), dtype=np.float32)
    idx_comp_cache = np.zeros((n_owners * max_clen, idx_head_dim), dtype=np.float32)
    ape = rng.standard_normal((ratio, width)).astype(np.float32)
    idx_ape = rng.standard_normal((ratio, idx_width)).astype(np.float32)

    # ---- bucketed inputs: SWA at bsz*bucket, state at bsz*MAXKEEP, cache at bsz*buck_clen ----
    n_swa = bsz * bucket
    n_state = bsz * MAXKEEP
    n_cache = bsz * buck_clen

    swa_kv = rng.standard_normal((n_swa, head_dim)).astype(np.float32)
    # state value rows: per request, first `keep` are real tail, rest padding
    kv = rng.standard_normal((n_state, width)).astype(np.float32)
    score = rng.standard_normal((n_state, width)).astype(np.float32)
    idx_kv = rng.standard_normal((n_state, idx_width)).astype(np.float32)
    idx_score = rng.standard_normal((n_state, idx_width)).astype(np.float32)
    comp_rows = rng.standard_normal((n_cache, head_dim)).astype(np.float32)
    idx_comp_rows = rng.standard_normal((n_cache, idx_head_dim)).astype(np.float32)

    # ---- owner/position arrays with guard-owner padding ----
    # Use the production builder: only the last min(real, window) tokens own
    # their SWA row; older positions collide mod window and must go to guard.
    from nkipy_serving.ops.deepseek_v4.compressor_state import (
        _bucketed_prefill_swa_owner_pos,
    )

    swa_owner, swa_pos = _bucketed_prefill_swa_owner_pos(
        bsz=bsz,
        bucket_seqlen=bucket,
        real_seqlen=real_seqlen,
        window_size=window,
        guard_owner=guard_owner,
    )

    state_owner = np.empty(n_state, np.int32)
    state_pos = np.empty(n_state, np.int32)
    for b in range(bsz):
        for j in range(MAXKEEP):
            i = b * MAXKEEP + j
            if j < keep:
                state_owner[i] = b
                state_pos[i] = real_seqlen - keep + j
            else:
                state_owner[i] = guard_owner
                state_pos[i] = 0

    # cache owner ids: request-major; owner_id_stride lets req map to owner.
    # Here owner_id_stride=1 and cache_owner_ids[req] = req.
    cache_owner = np.arange(n_owners, dtype=np.int32)

    # ---- reorder state value rows so row j of request b is the j-th MAXKEEP slot ----
    # (the test arrays above are already laid out [bsz*MAXKEEP]; oracle reads src=b*keep+j
    #  but kernel reads src=b*MAXKEEP+j, so align oracle to MAXKEEP layout)
    kv_k = kv.reshape(bsz, MAXKEEP, width)
    score_k = score.reshape(bsz, MAXKEEP, width)
    idx_kv_k = idx_kv.reshape(bsz, MAXKEEP, idx_width)
    idx_score_k = idx_score.reshape(bsz, MAXKEEP, idx_width)

    # reference caches (real rows only)
    ref_swa = swa_cache.copy()
    ref_state = state_cache.copy()
    ref_comp = comp_cache.copy()
    ref_idx_state = idx_state_cache.copy()
    ref_idx_comp = idx_comp_cache.copy()
    for b in range(bsz):
        for p in range(max(0, real_seqlen - window), real_seqlen):
            ref_swa[b * window + p % window] = swa_kv[b * bucket + p]
        for j in range(keep):
            p = real_seqlen - keep + j
            r = b * ring + (p % ring)
            ref_state[r, :width] = kv_k[b, j]
            ref_state[r, width:] = score_k[b, j] + ape[p % ratio]
            ref_idx_state[r, :idx_width] = idx_kv_k[b, j]
            ref_idx_state[r, idx_width:] = idx_score_k[b, j] + idx_ape[p % ratio]
        for c in range(real_clen):
            ref_comp[b * max_clen + c] = comp_rows[b * buck_clen + c]
            ref_idx_comp[b * max_clen + c] = idx_comp_rows[b * buck_clen + c]

    def D(a):
        return DeviceTensor.from_numpy(np.ascontiguousarray(a))

    swa_dev = D(swa_cache)
    state_dev = D(state_cache)
    comp_dev = D(comp_cache)
    idx_state_dev = D(idx_state_cache)
    idx_comp_dev = D(idx_comp_cache)

    run_write_swa_dual_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_dev,
        kv_score_state=state_dev,
        compressed_kv_cache=comp_dev,
        indexer_kv_score_state=idx_state_dev,
        indexer_compressed_kv_cache=idx_comp_dev,
        swa_rows=D(swa_kv),
        kv_new=D(kv),
        score_new=D(score),
        compressed_rows=D(comp_rows),
        indexer_kv_new=D(idx_kv),
        indexer_score_new=D(idx_score),
        indexer_compressed_rows=D(idx_comp_rows),
        swa_owner_ids=D(swa_owner),
        swa_positions=D(swa_pos),
        state_owner_ids=D(state_owner),
        state_positions=D(state_pos),
        cache_owner_ids=D(cache_owner),
        ape=D(ape),
        indexer_ape=D(idx_ape),
        window_size=window,
        ring_size=ring,
        indexer_ring_size=ring,
        clen=buck_clen,
        owner_id_stride=1,
        max_clen=max_clen,
        indexer_max_clen=max_clen,
        cache_real_clen=real_clen,
        guard_owner=guard_owner,
        artifacts_dir=str(tmp_path),
    )

    got_swa = swa_dev.numpy()
    got_state = state_dev.numpy()
    got_comp = comp_dev.numpy()
    got_idx_state = idx_state_dev.numpy()
    got_idx_comp = idx_comp_dev.numpy()

    # Only the REAL-owner live region must match the reference exactly; the guard
    # owner block (rows >= guard_owner*stride) may hold padding garbage.
    real_swa_rows = max_batch * window
    real_state_rows = max_batch * ring
    real_comp_rows = max_batch * max_clen
    np.testing.assert_allclose(
        got_swa[:real_swa_rows],
        ref_swa[:real_swa_rows],
        rtol=1e-3,
        atol=1e-3,
        err_msg="SWA live region corrupted",
    )
    np.testing.assert_allclose(
        got_state[:real_state_rows],
        ref_state[:real_state_rows],
        rtol=1e-3,
        atol=1e-3,
        err_msg="state live region corrupted",
    )
    np.testing.assert_allclose(
        got_comp[:real_comp_rows],
        ref_comp[:real_comp_rows],
        rtol=1e-3,
        atol=1e-3,
        err_msg="compressed live region corrupted",
    )
    np.testing.assert_allclose(
        got_idx_state[:real_state_rows],
        ref_idx_state[:real_state_rows],
        rtol=1e-3,
        atol=1e-3,
        err_msg="indexer state live region corrupted",
    )
    np.testing.assert_allclose(
        got_idx_comp[:real_comp_rows],
        ref_idx_comp[:real_comp_rows],
        rtol=1e-3,
        atol=1e-3,
        err_msg="indexer compressed live region corrupted",
    )


@pytest.mark.parametrize("real_seqlen,bucket", [(200, 256), (256, 256)])
def test_bucketed_single_write_matches_real_length(real_seqlen, bucket, tmp_path):
    rng = np.random.default_rng(1)
    ratio = 128
    window = 128
    ring = ratio
    head_dim = 16
    guard_owner = 1
    n_owners = 2
    max_clen = bucket // ratio + 2
    max_keep = ratio - 1
    keep = real_seqlen % ratio
    real_clen = real_seqlen // ratio
    buck_clen = bucket // ratio

    swa_cache = np.zeros((n_owners * window, head_dim), dtype=ml_dtypes.bfloat16)
    state_cache = np.zeros((n_owners * ring, 2 * head_dim), dtype=np.float32)
    comp_cache = np.zeros((n_owners * max_clen, head_dim), dtype=ml_dtypes.bfloat16)
    ape = rng.standard_normal((ratio, head_dim)).astype(ml_dtypes.bfloat16)

    swa_kv = rng.standard_normal((bucket, head_dim)).astype(np.float32)
    kv = rng.standard_normal((max_keep, head_dim)).astype(ml_dtypes.bfloat16)
    score = rng.standard_normal((max_keep, head_dim)).astype(ml_dtypes.bfloat16)
    comp_rows = rng.standard_normal((buck_clen, head_dim)).astype(ml_dtypes.bfloat16)

    from nkipy_serving.ops.deepseek_v4.compressor_state import (
        _bucketed_prefill_swa_owner_pos,
    )

    swa_owner, swa_pos = _bucketed_prefill_swa_owner_pos(
        bsz=1,
        bucket_seqlen=bucket,
        real_seqlen=real_seqlen,
        window_size=window,
        guard_owner=guard_owner,
    )
    tail_start = max(0, real_seqlen - max_keep)
    tok = tail_start + np.arange(max_keep, dtype=np.int32)
    live = (tok >= real_seqlen - keep) & (tok < real_seqlen)
    state_owner = np.where(live, np.int32(0), np.int32(guard_owner)).astype(np.int32)
    state_pos = np.where(live, tok, np.int32(0)).astype(np.int32)
    cache_owner = np.asarray([0, guard_owner], dtype=np.int32)

    ref_swa = swa_cache.copy()
    ref_state = state_cache.copy()
    ref_comp = comp_cache.copy()
    for p in range(max(0, real_seqlen - window), real_seqlen):
        ref_swa[p % window] = swa_kv[p]
    for j in range(max_keep):
        if not bool(live[j]):
            continue
        p = int(tok[j])
        row = p % ring
        ref_state[row, :head_dim] = np.asarray(kv[j], dtype=np.float32)
        ref_state[row, head_dim:] = np.asarray(score[j], dtype=np.float32) + np.asarray(
            ape[p % ratio], dtype=np.float32
        )
    for c in range(real_clen):
        ref_comp[c] = comp_rows[c]

    def D(a):
        return DeviceTensor.from_numpy(np.ascontiguousarray(a))

    swa_dev = D(swa_cache)
    state_dev = D(state_cache)
    comp_dev = D(comp_cache)

    run_write_swa_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_dev,
        kv_score_state=state_dev,
        compressed_kv_cache=comp_dev,
        swa_rows=D(swa_kv),
        kv_new=D(kv),
        score_new=D(score),
        compressed_rows=D(comp_rows),
        swa_owner_ids=D(swa_owner),
        swa_positions=D(swa_pos),
        state_owner_ids=D(state_owner),
        state_positions=D(state_pos),
        cache_owner_ids=D(cache_owner),
        ape=D(ape),
        window_size=window,
        ring_size=ring,
        clen=buck_clen,
        owner_id_stride=1,
        max_clen=max_clen,
        cache_real_clen=real_clen,
        guard_owner=guard_owner,
        artifacts_dir=str(tmp_path),
    )

    np.testing.assert_allclose(
        np.asarray(swa_dev.numpy()[:window], dtype=np.float32),
        np.asarray(ref_swa[:window], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
        err_msg="single SWA live region corrupted",
    )
    np.testing.assert_allclose(
        np.asarray(state_dev.numpy()[:ring], dtype=np.float32),
        np.asarray(ref_state[:ring], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
        err_msg="single state live region corrupted",
    )
    np.testing.assert_allclose(
        np.asarray(comp_dev.numpy()[:max_clen], dtype=np.float32),
        np.asarray(ref_comp[:max_clen], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
        err_msg="single compressed live region corrupted",
    )
