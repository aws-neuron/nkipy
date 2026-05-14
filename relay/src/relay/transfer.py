# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-rank peer-to-peer weight transfer over RDMA.

Provides functions to push / receive device memory between engines using
Relay RDMA writes, plus a :class:`WeightServer` for the active engine
and a helper to collect device buffer descriptors from a model.
"""

import logging
import os
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist

from .endpoint import RankEndpoint, _VAHandle

# Maximum number of buffers to register in a single RDMA registration.
# MoE models (e.g. Qwen3-30B-A3B with 128 experts) can exceed EFA device
# memory registration limits when all per-layer weights are registered at
# once.  Chunking the registration avoids "Failed to register memory with
# RDMA" errors.  Tune via env var — higher values reduce chunking overhead
# but may hit registration limits on larger per-rank shards.
MAX_RDMA_BUFS = int(os.environ.get("NKIPY_MAX_RDMA_BUFS", "1024"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)

# ------------------------------------------------------------------
# Per-rank singleton
# ------------------------------------------------------------------

rank_endpoint = RankEndpoint()
"""Module-level singleton shared by the convenience wrappers below."""


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------


def collect_weight_buffers(model) -> List[Tuple[str, int, int]]:
    """Return ``(name, va, size_bytes)`` for every weight tensor in *model*.

    Delegates to ``model.weight_buffers()`` if available, which lets each
    model architecture decide which tensors participate in P2P transfer.
    """
    return list(model.weight_buffers())


# ------------------------------------------------------------------
# Low-level transfer primitives
# ------------------------------------------------------------------


def _chunk_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` index ranges for chunking *total* items."""
    return [
        (i, min(i + chunk_size, total))
        for i in range(0, total, chunk_size)
    ]


def receive_from_peer(
    ep: RankEndpoint,
    buffers: Sequence[Tuple[str, int, int]],
    peer_url: str,
    push_endpoint: str = "/nkipy/p2p_push_weights",
    preconnect_endpoint: str = "/nkipy/p2p_preconnect",
) -> None:
    """All ranks receive their buffer shard from a peer engine via P2P RDMA.

    When the number of buffers exceeds *MAX_RDMA_BUFS*, the transfer is
    automatically split into chunks.  The endpoint and RDMA connection are
    reused across chunks — only MRs are swapped per chunk.

    Sends endpoint metadata to the sender *before* MR registration so the
    sender can start the 2s RDMA connect in parallel with receiver
    registration.
    """
    t0 = time.time()
    total_bytes = sum(sz for _, _, sz in buffers)
    chunks = _chunk_ranges(len(buffers), MAX_RDMA_BUFS)

    for ci, (cs, ce) in enumerate(chunks):
        t_chunk = time.time()

        if ci == 0:
            ep.dereg_sync()
            ep._ensure_endpoint()
            ep.ep.start_passive_accept()
        t_accept = time.time()

        # Send endpoint metadata to sender BEFORE registration so the
        # sender can start RDMA connect (~2s) in parallel with our
        # MR registration (~2.3s).
        meta_hex = ep.ep.get_metadata().hex()
        gathered_meta = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(meta_hex, gathered_meta, dst=0)
        preconnect_future = None
        if dist.get_rank() == 0:
            import concurrent.futures
            base = peer_url.rstrip("/")
            _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            preconnect_future = _pool.submit(
                requests.post,
                f"{base}{preconnect_endpoint}",
                json={"per_rank": [{"remote_metadata": m} for m in gathered_meta]},
            )
            _pool.shutdown(wait=False)
        t_preconnect = time.time()

        # Register MRs — overlapped with sender connect
        if ci == 0:
            ep.register(buffers[cs:ce])
        else:
            ep.reregister(buffers[cs:ce])
        if preconnect_future is not None:
            preconnect_future.result()
        t_reg = time.time()

        # Send transfer descriptors so the sender can do the RDMA write
        serialized = ep.ep.get_serialized_descs(ep.xfer_descs)
        my_info = (meta_hex, serialized.hex())
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(my_info, gathered, dst=0)
        t_gather = time.time()

        if dist.get_rank() == 0:
            resp = requests.post(
                f"{base}{push_endpoint}",
                json={
                    "per_rank": [
                        {"remote_metadata": m, "remote_descs": d}
                        for m, d in gathered
                    ],
                    "chunk_start": cs,
                    "chunk_end": ce,
                    "is_last_chunk": ci == len(chunks) - 1,
                },
            )
            resp.raise_for_status()
        t_post = time.time()

        dist.barrier()
        t_done = time.time()

        chunk_bytes = sum(sz for _, _, sz in buffers[cs:ce])
        if not dist.is_initialized() or dist.get_rank() == 0: logger.info(
            "Rank %d: recv chunk %d/%d [%d:%d] %d bufs %.2f MB — "
            "accept %.3fs preconnect %.3fs reg %.3fs gather %.3fs "
            "post+xfer %.3fs barrier %.3fs total %.3fs",
            dist.get_rank(), ci + 1, len(chunks), cs, ce, ce - cs,
            chunk_bytes / 1e6,
            t_accept - t_chunk, t_preconnect - t_accept,
            t_reg - t_preconnect, t_gather - t_reg,
            t_post - t_gather, t_done - t_post, t_done - t_chunk,)

    elapsed = time.time() - t0
    throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: P2P receive complete — %d bufs, %.2f MB, %.2fs, %.2f Gbps",
    dist.get_rank(), len(buffers), total_bytes / 1e6, elapsed, throughput_gbps,)

    ep.dereg_async()
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: Started async MR deregistration (%d MRs)", dist.get_rank(), len(ep.xfer_descs))


def receive_from_peer_staged(
    ep: RankEndpoint,
    buffers_with_tensors: Sequence[Tuple[str, int, int, object]],
    peer_url: str,
    push_endpoint: str = "/nkipy/p2p_push_weights",
    preconnect_endpoint: str = "/nkipy/p2p_preconnect",
) -> None:
    """Host-staged receive: RDMA into host buffer, then DMA host->device.

    On Trn2, registering host memory for RDMA is much faster than device memory,
    and RDMA to host achieves ~100+ Gbps vs ~3.5 Gbps to device.

    Uses chunked transfer to overlap RDMA receive with DMA host->device:
    sender pushes in N chunks, receiver DMAs each chunk while next arrives.
    """
    import concurrent.futures
    from spike import get_spike_singleton
    from .staging import HostStagingBuffer, compute_offsets

    num_chunks = int(os.environ.get("NKIPY_RECV_CHUNKS", "2"))

    t0 = time.time()
    sizes = [size for _, _, size, _ in buffers_with_tensors]
    total_bytes = sum(sizes)
    offsets = compute_offsets(sizes)
    n = len(buffers_with_tensors)

    # Reuse pre-allocated staging buffer if available (saves mmap allocation)
    prereg = getattr(ep, '_receiver_staging', None)
    if prereg and prereg.total_size == total_bytes:
        staging = prereg.staging
    else:
        staging = HostStagingBuffer(total_bytes)
    t_alloc = time.time()

    # Setup RDMA endpoint
    ep.dereg_sync()
    ep._ensure_endpoint()
    ep.ep.start_passive_accept()

    # Send endpoint metadata to sender for RDMA connect
    meta_hex = ep.ep.get_metadata().hex()
    gathered_meta = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(meta_hex, gathered_meta, dst=0)
    preconnect_future = None
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        preconnect_future = _pool.submit(
            requests.post,
            f"{base}{preconnect_endpoint}",
            json={"per_rank": [{"remote_metadata": m} for m in gathered_meta]},
        )
        _pool.shutdown(wait=False)
    t_preconnect = time.time()

    # Register HOST staging buffer as single contiguous MR (overlaps with preconnect)
    ep.buf_info = [(f"staged_{i}", sizes[i]) for i in range(len(sizes))]
    ep.xfer_descs = ep.ep.register_contiguous_buffer(
        staging.ptr, total_bytes, offsets, sizes)
    if preconnect_future is not None:
        preconnect_future.result()
    t_reg = time.time()

    # Send transfer descriptors to sender
    serialized = ep.ep.get_serialized_descs(ep.xfer_descs)
    my_info = (meta_hex, serialized.hex())
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(my_info, gathered, dst=0)
    t_gather = time.time()

    spike = get_spike_singleton()
    chunk_size = (n + num_chunks - 1) // num_chunks
    dma_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    dma_future = None
    t_rdma_total = 0.0
    t_dma_total = 0.0

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, n)
        if c_start >= n:
            break

        # Request sender to push this chunk
        if dist.get_rank() == 0:
            base = peer_url.rstrip("/")
            resp = requests.post(
                f"{base}{push_endpoint}",
                json={
                    "per_rank": [
                        {"remote_metadata": m, "remote_descs": d}
                        for m, d in gathered
                    ],
                    "chunk_start": c_start,
                    "chunk_end": c_end,
                    "is_last_chunk": chunk_idx == num_chunks - 1 or c_end >= n,
                },
            )
            resp.raise_for_status()
        dist.barrier()
        t_chunk_rdma = time.time()
        t_rdma_total += (t_chunk_rdma - (t_reg if chunk_idx == 0 else t_chunk_rdma))

        # Wait for previous DMA to complete before starting new one
        if dma_future is not None:
            dma_future.result()

        # Start DMA for this chunk (overlaps with next chunk's RDMA)
        dma_ops = [(buffers_with_tensors[i][3].tensor_ref,
                    staging.slice_as_numpy(offsets[i], sizes[i]))
                   for i in range(c_start, c_end)]
        if chunk_idx < num_chunks - 1 and c_end < n:
            dma_future = dma_executor.submit(spike.batch_dma_write, dma_ops)
        else:
            # Last chunk: DMA synchronously
            spike.batch_dma_write(dma_ops)
            dma_future = None

    # Wait for final DMA
    if dma_future is not None:
        dma_future.result()
    dma_executor.shutdown(wait=False)
    t_dma = time.time()

    # Cache staging buffer for reuse on next wake cycle (saves mmap allocation).
    from .staging import PreregisteredStaging
    prereg_obj = PreregisteredStaging.__new__(PreregisteredStaging)
    prereg_obj.sizes = sizes
    prereg_obj.offsets = offsets
    prereg_obj.total_size = total_bytes
    prereg_obj.staging = staging
    prereg_obj.xfer_descs = ep.xfer_descs
    ep._receiver_staging = prereg_obj
    ep.dereg_async()

    elapsed = time.time() - t0
    throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "Rank %d: staged recv (chunked, %d chunks) %d bufs %.2f MB — "
            "alloc %.3fs preconnect %.3fs mr_reg %.3fs "
            "gather %.3fs transfer+dma %.3fs total %.3fs (%.2f Gbps)",
            dist.get_rank(), num_chunks, len(buffers_with_tensors), total_bytes / 1e6,
            t_alloc - t0, t_preconnect - t_alloc, t_reg - t_preconnect,
            t_gather - t_reg, t_dma - t_gather, elapsed, throughput_gbps)


def preconnect_to_peer(
    ep: RankEndpoint,
    per_rank_info: list | None = None,
) -> None:
    """All ranks establish RDMA connection to the peer (no data transfer).

    Called ahead of ``push_to_peer`` so the 2s QP handshake overlaps with
    receiver MR registration.  ``add_remote_endpoint`` caches the connection,
    so the subsequent ``push_to_peer`` reuses it instantly.
    """
    remote_metadata_hex = per_rank_info[dist.get_rank()]
    relay_ep = ep._ensure_endpoint()
    ok, _ = relay_ep.add_remote_endpoint(bytes.fromhex(remote_metadata_hex))
    assert ok, "Failed to pre-connect to remote endpoint"
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: RDMA pre-connect complete", dist.get_rank())


def push_to_peer(
    ep: RankEndpoint,
    buffers: Sequence[Tuple[str, int, int]],
    per_rank_info: list | None = None,
    is_last_chunk: bool = True,
) -> None:
    """All ranks push their buffer shard to the corresponding peer rank.

    Each worker receives *per_rank_info* (the full list) directly via
    collective_rpc and extracts its own entry — no Gloo broadcast needed.
    This avoids Gloo thread-safety issues when push runs in a background
    thread while the main thread serves inference.

    Parameters
    ----------
    ep : RankEndpoint
        The per-rank endpoint (will be registered if not already).
    buffers : sequence of (name, va, size_bytes)
        Device buffers to push from.
    per_rank_info : list
        ``[(remote_metadata_hex, remote_descs_hex), ...]`` per rank.
        Required on ALL ranks (distributed via collective_rpc).
    is_last_chunk : bool
        If True (default), destroy the endpoint after the transfer.
        Set to False to keep the endpoint alive for subsequent chunks.
    """
    remote_metadata_hex, remote_descs_hex = per_rank_info[dist.get_rank()]

    t0 = time.time()

    pre_registered = ep.registered and len(ep.xfer_descs) == len(buffers)

    relay_ep = ep._ensure_endpoint()
    ok, conn_id = relay_ep.add_remote_endpoint(bytes.fromhex(remote_metadata_hex))
    assert ok, "Failed to connect to remote endpoint"
    t_conn = time.time()

    if not pre_registered:
        handles = []
        ep.buf_info = []
        for name, va, size_bytes in buffers:
            ep.buf_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        ep.xfer_descs = relay_ep.register_memory(handles)
    t_reg = time.time()

    remote_descs = relay_ep.deserialize_descs(bytes.fromhex(remote_descs_hex))
    if len(remote_descs) != len(ep.xfer_descs):
        raise ValueError(
            f"Rank {dist.get_rank()}: desc count mismatch: "
            f"remote_descs={len(remote_descs)} vs xfer_descs={len(ep.xfer_descs)} "
            f"buffers={len(buffers)} pre_registered={pre_registered}"
        )

    ok, _ = relay_ep.transfer(conn_id, "write", ep.xfer_descs, remote_descs)
    assert ok, "RDMA write failed"
    t_xfer = time.time()

    chunk_bytes = sum(sz for _, _, sz in buffers)
    xfer_secs = t_xfer - t_conn
    xfer_gbps = (chunk_bytes * 8) / xfer_secs / 1e9 if xfer_secs > 0 else 0

    if is_last_chunk:
        if pre_registered:
            # Reset endpoint to free QPs from this connection while keeping
            # MRs registered for subsequent pushes.  Without this, QPs
            # accumulate across pushes and eventually exhaust RDMA resources.
            ep.reset_endpoint_async()
        else:
            ep.dereg_async()
    t_dereg = time.time()

    if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: pushed %d bufs %.2f MB — "
    "connect %.3fs reg %.3fs xfer %.3fs (%.2f Gbps) dereg %.3fs total %.3fs%s",
    dist.get_rank(), len(buffers), chunk_bytes / 1e6,
    t_conn - t0, t_reg - t_conn, xfer_secs, xfer_gbps,
    t_dereg - t_xfer, t_dereg - t0,
    " (pre-registered)" if pre_registered else "",)


# ------------------------------------------------------------------
# High-level model wrappers (use module-level rank_endpoint)
# ------------------------------------------------------------------


class WeightServer:
    """Runs on the active engine.  Tracks model buffer info for /weight_info."""

    def __init__(self, model):
        bufs = collect_weight_buffers(model)
        self._buf_info = [(name, sz) for name, _va, sz in bufs]
        # Ensure an endpoint exists (for metadata) but do NOT register
        # MRs — push_to_peer handles registration.  Pre-registering all
        # buffers would make rank 0 a straggler during push (it would
        # have to deregister everything first).
        rank_endpoint._ensure_endpoint()
        if not dist.is_initialized() or dist.get_rank() == 0: logger.info("WeightServer: %d tensors tracked", len(self._buf_info))

    def get_weight_info(self):
        return {
            "weights": list(self._buf_info),
            "metadata": rank_endpoint.ep.get_metadata().hex(),
        }

    def cleanup(self):
        rank_endpoint.wait()


def preregister_weights(model) -> None:
    """Pre-register all model weight buffers for RDMA in chunks.

    Call once after model tensors are allocated (load or wake-up).
    Subsequent ``push_to_peer`` / ``receive_from_peer`` calls will
    skip registration entirely.
    """
    bufs = collect_weight_buffers(model)
    if rank_endpoint.registered:
        logger.info("Rank %d: MRs already registered (%d descs), skipping",
                    dist.get_rank() if dist.is_initialized() else 0, len(rank_endpoint.xfer_descs))
        return
    t0 = time.time()
    rank_endpoint.register_chunked(bufs, MAX_RDMA_BUFS)
    elapsed = time.time() - t0
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: pre-registered %d weight MRs in %.3fs",
                    dist.get_rank(), len(bufs), elapsed)


def receive_weights(model, peer_url: str) -> None:
    """All ranks receive model weights from *peer_url* via P2P RDMA."""
    bufs = collect_weight_buffers(model)
    receive_from_peer(rank_endpoint, bufs, peer_url)


def push_weights_to_peer(model, per_rank_info=None, chunk_start=None, chunk_end=None,
                         is_last_chunk=True) -> None:
    """All ranks push model weights to the corresponding peer rank.

    When *chunk_start* / *chunk_end* are provided only that slice of
    buffers is registered and transferred (used by the chunked protocol).
    """
    if os.environ.get("NKIPY_HOST_STAGING", "0") == "1":
        push_weights_to_peer_staged(model, per_rank_info, chunk_start, chunk_end, is_last_chunk)
        return
    bufs = collect_weight_buffers(model)
    if chunk_start is not None:
        bufs = bufs[chunk_start:chunk_end]
    push_to_peer(rank_endpoint, bufs, per_rank_info, is_last_chunk=is_last_chunk)


def start_pre_dma_to_staging(model) -> None:
    """Start staged DMA device->host for all weights in background.

    Called during preconnect so DMA overlaps with receiver MR registration
    (~2s). Uses the same stage count as the push pipeline so partially-
    completed DMA allows the push to start RDMA on ready stages immediately.

    Only applicable when NKIPY_HOST_STAGING=1 (Trn2 path).
    No-op if staging buffer is not pre-registered.
    """
    prereg = getattr(rank_endpoint, '_sender_staging', None)
    if prereg is None:
        return

    # Don't start another pre-DMA if one is already in progress
    if getattr(rank_endpoint, '_pre_dma_thread', None) is not None:
        return

    import threading
    from spike import get_spike_singleton
    from .staging import compute_offsets

    all_bufs = model.weight_buffers_with_tensors()
    all_sizes = [size for _, _, size, _ in all_bufs]
    all_offsets = compute_offsets(all_sizes)
    staging = prereg.staging
    num_stages = int(os.environ.get("NKIPY_PIPELINE_STAGES", "4"))
    n = len(all_bufs)
    stage_size = (n + num_stages - 1) // num_stages

    # Track per-stage completion for the push function to consume
    stages_done = [False] * num_stages
    rank_endpoint._pre_dma_stages_done = stages_done
    rank_endpoint._pre_dma_num_stages = num_stages
    rank_endpoint._pre_dma_stage_size = stage_size

    def _bg_dma():
        spike = get_spike_singleton()
        for stage in range(num_stages):
            s_start = stage * stage_size
            s_end = min(s_start + stage_size, n)
            if s_start >= n:
                stages_done[stage] = True
                continue
            dma_ops = [(all_bufs[i][3].tensor_ref,
                        staging.slice_as_numpy(all_offsets[i], all_sizes[i]))
                       for i in range(s_start, s_end)]
            spike.batch_dma_read(dma_ops)
            stages_done[stage] = True

    t = threading.Thread(target=_bg_dma, daemon=True)
    t.start()
    rank_endpoint._pre_dma_thread = t
    rank_endpoint._pre_dma_time = time.time()

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: started pre-DMA to staging (%.2f MB, %d stages)",
                    dist.get_rank(), sum(all_sizes) / 1e6, num_stages)


def push_weights_to_peer_staged(model, per_rank_info=None, chunk_start=None,
                                chunk_end=None, is_last_chunk=True) -> None:
    """Host-staged push: DMA device->host, then RDMA host->remote_host.

    On Trn2, direct device RDMA is ~3.5 Gbps/rank. Host RDMA is ~100+ Gbps/NIC.
    This function copies device weights into a host staging buffer, then
    does the RDMA transfer from host memory.

    If start_pre_dma_to_staging() was called during preconnect, the DMA is
    already complete (or nearly so) and we skip straight to RDMA.

    Supports chunked transfer: when chunk_start/chunk_end are provided, only
    that slice is transferred using the pre-registered full staging buffer.
    """
    from spike import get_spike_singleton
    from .staging import HostStagingBuffer, compute_offsets

    all_bufs = model.weight_buffers_with_tensors()
    all_sizes = [size for _, _, size, _ in all_bufs]
    all_offsets = compute_offsets(all_sizes)
    full_size = sum(all_sizes)

    if chunk_start is not None:
        bufs_with_tensors = all_bufs[chunk_start:chunk_end]
    else:
        bufs_with_tensors = all_bufs

    t0 = time.time()

    # Check if pre-DMA was started during preconnect.
    # _pre_dma_done is set after the thread completes and persists across
    # chunked calls so subsequent chunks also skip the DMA pipeline.
    pre_dma_thread = getattr(rank_endpoint, '_pre_dma_thread', None)
    pre_dma_done = getattr(rank_endpoint, '_pre_dma_done', False)
    pre_dma_started = pre_dma_thread is not None or pre_dma_done

    # Use pre-registered staging for the FULL model buffer
    prereg = getattr(rank_endpoint, '_sender_staging', None)
    if prereg and prereg.total_size == full_size:
        staging = prereg.staging
        all_xfer_descs = prereg.xfer_descs
    else:
        staging = HostStagingBuffer(full_size)
        prereg = None
        all_xfer_descs = None

    # Register MRs as single contiguous MR (skip if pre-registered)
    relay_ep = rank_endpoint._ensure_endpoint()
    if all_xfer_descs is None:
        all_xfer_descs = relay_ep.register_contiguous_buffer(
            staging.ptr, full_size, all_offsets, all_sizes)

    # Slice descs/offsets for the chunk
    if chunk_start is not None:
        xfer_descs = all_xfer_descs[chunk_start:chunk_end]
        chunk_offsets = [all_offsets[i] for i in range(chunk_start, chunk_end)]
        chunk_sizes = [all_sizes[i] for i in range(chunk_start, chunk_end)]
    else:
        xfer_descs = all_xfer_descs
        chunk_offsets = all_offsets
        chunk_sizes = all_sizes

    rank_endpoint.xfer_descs = all_xfer_descs
    rank_endpoint.buf_info = [(name, size) for name, _, size, _ in all_bufs]
    total_size = sum(chunk_sizes)

    # Connect to remote (reuses cached connection on subsequent chunks)
    remote_metadata_hex, remote_descs_hex = per_rank_info[dist.get_rank()]
    ok, conn_id = relay_ep.add_remote_endpoint(bytes.fromhex(remote_metadata_hex))
    assert ok, "Failed to connect to remote endpoint"
    all_remote_descs = relay_ep.deserialize_descs(bytes.fromhex(remote_descs_hex))
    if chunk_start is not None:
        remote_descs = all_remote_descs[chunk_start:chunk_end]
    else:
        remote_descs = all_remote_descs
    assert len(remote_descs) == len(xfer_descs), (
        f"desc mismatch: remote={len(remote_descs)} local={len(xfer_descs)}")

    # If pre-DMA was started during preconnect, wait for full completion then
    # RDMA the entire chunk at once. Sending RDMA while DMA is still running
    # causes PCIe bus contention that drops throughput from ~32 Gbps to ~9 Gbps.
    if pre_dma_started:
        pre_dma_elapsed = time.time() - getattr(rank_endpoint, '_pre_dma_time', t0)

        # Wait for the background DMA thread to finish ALL stages
        if pre_dma_thread is not None:
            pre_dma_thread.join()
            rank_endpoint._pre_dma_thread = None
            rank_endpoint._pre_dma_done = True
        t_dma_ready = time.time()
        dma_wait = t_dma_ready - t0

        # Single RDMA write for the entire chunk (no contention = full speed)
        ok, _ = relay_ep.transfer(conn_id, "write", xfer_descs, remote_descs)
        assert ok, "RDMA write failed (pre-DMA path)"

        if is_last_chunk:
            rank_endpoint._pre_dma_done = False
            rank_endpoint._pre_dma_stages_done = None
        t_done = time.time()

        if prereg is None:
            staging.close()

        elapsed = t_done - t0
        rdma_time = t_done - t_dma_ready
        throughput_gbps = (total_size * 8) / rdma_time / 1e9 if rdma_time > 0 else 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Rank %d: staged push (pre-DMA) %d bufs %.2f MB — "
                        "dma_wait %.3fs rdma %.3fs total %.3fs (%.1f Gbps) "
                        "[pre-DMA started %.1fs ago]",
                        dist.get_rank(), len(bufs_with_tensors), total_size / 1e6,
                        dma_wait, rdma_time,
                        elapsed, throughput_gbps, pre_dma_elapsed)
        return

    # Pipelined DMA + RDMA: overlap DMA[N+1] with RDMA[N]
    spike = get_spike_singleton()
    num_stages = int(os.environ.get("NKIPY_PIPELINE_STAGES", "4"))
    n = len(bufs_with_tensors)
    stage_size = (n + num_stages - 1) // num_stages

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Stage 0: DMA (blocking)
    s0_end = min(stage_size, n)
    dma_ops_0 = [(bufs_with_tensors[i][3].tensor_ref,
                  staging.slice_as_numpy(chunk_offsets[i], chunk_sizes[i]))
                 for i in range(s0_end)]
    spike.batch_dma_read(dma_ops_0)

    for stage in range(1, num_stages):
        s_start = stage * stage_size
        s_end = min(s_start + stage_size, n)
        if s_start >= n:
            break

        # Start DMA for current stage in background
        dma_ops_s = [(bufs_with_tensors[i][3].tensor_ref,
                      staging.slice_as_numpy(chunk_offsets[i], chunk_sizes[i]))
                     for i in range(s_start, s_end)]
        dma_future = executor.submit(spike.batch_dma_read, dma_ops_s)

        # RDMA write previous stage (overlapped with DMA)
        prev_start = (stage - 1) * stage_size
        prev_end = min(prev_start + stage_size, n)
        ok, _ = relay_ep.transfer(
            conn_id, "write",
            xfer_descs[prev_start:prev_end],
            remote_descs[prev_start:prev_end])
        assert ok, f"RDMA write failed for stage {stage-1}"

        dma_future.result()  # Wait for DMA to complete

    # RDMA write last stage
    last_start = (min(num_stages, (n + stage_size - 1) // stage_size) - 1) * stage_size
    last_end = n
    ok, _ = relay_ep.transfer(
        conn_id, "write",
        xfer_descs[last_start:last_end],
        remote_descs[last_start:last_end])
    assert ok, "RDMA write failed for last stage"
    executor.shutdown(wait=False)

    t_done = time.time()

    if prereg is None:
        staging.close()

    elapsed = t_done - t0
    throughput_gbps = (total_size * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: staged push (pipelined, %d stages) %d bufs %.2f MB — "
                    "%.3fs (%.1f Gbps)",
                    dist.get_rank(), num_stages, len(bufs_with_tensors),
                    total_size / 1e6, elapsed, throughput_gbps)


def fetch_tok_embedding(peer_url: str):
    """Fetch tok_embedding from peer over HTTP and broadcast to all ranks."""
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = requests.get(f"{base}/nkipy/tok_embedding")
        resp.raise_for_status()
        shape = tuple(int(d) for d in resp.headers["X-Shape"].split(","))
        dtype_str = resp.headers["X-Dtype"]
        torch_dtype = getattr(torch, dtype_str.replace("torch.", ""), None)
        if torch_dtype is not None:
            tok_embedding = (
                torch.frombuffer(bytearray(resp.content), dtype=torch_dtype)
                .reshape(shape)
                .clone()
            )
        else:
            tok_embedding = torch.from_numpy(
                np.frombuffer(resp.content, dtype=np.dtype(dtype_str))
                .reshape(shape)
                .copy()
            )
    else:
        tok_embedding = None
    obj_list = [tok_embedding]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
