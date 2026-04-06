# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-rank peer-to-peer weight transfer over RDMA.

Provides functions to push / receive device memory between engines using
UCCL P2P RDMA writes, plus a :class:`WeightServer` for the active engine
and a helper to collect device buffer descriptors from a model.
"""

import logging
import os
import time
from typing import List, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist

from .endpoint import RankEndpoint

# Maximum number of buffers to register in a single RDMA registration.
# MoE models (e.g. Qwen3-30B-A3B with 128 experts) can exceed EFA device
# memory registration limits when all per-layer weights are registered at
# once.  Chunking the registration avoids "Failed to register memory with
# RDMA" errors.  Tune via env var — higher values reduce chunking overhead
# but may hit registration limits on larger per-rank shards.
MAX_RDMA_BUFS = int(os.environ.get("NKIPY_MAX_RDMA_BUFS", "1024"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
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
    push_endpoint: str = "/p2p_push_weights",
) -> None:
    """All ranks receive their buffer shard from a peer engine via P2P RDMA.

    When the number of buffers exceeds *MAX_RDMA_BUFS*, the transfer is
    automatically split into chunks.  The endpoint and RDMA connection are
    reused across chunks — only MRs are swapped per chunk.

    Parameters
    ----------
    ep : RankEndpoint
        The per-rank endpoint (will be registered if not already).
    buffers : sequence of (name, va, size_bytes)
        Device buffers to receive into.
    peer_url : str
        Base URL of the active peer engine.
    push_endpoint : str
        HTTP path on the peer to trigger the push.
    """
    t0 = time.time()
    total_bytes = sum(sz for _, _, sz in buffers)

    if ep.registered and len(ep.xfer_descs) == len(buffers):
        # Pre-registered — single-shot path, no chunking needed.
        ep.ep.start_passive_accept()

        serialized = ep.ep.get_serialized_descs(ep.xfer_descs)
        my_info = (ep.ep.get_metadata().hex(), serialized.hex())
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(my_info, gathered, dst=0)

        if dist.get_rank() == 0:
            base = peer_url.rstrip("/")
            resp = requests.post(
                f"{base}{push_endpoint}",
                json={
                    "per_rank": [
                        {"remote_metadata": m, "remote_descs": d}
                        for m, d in gathered
                    ],
                    "chunk_start": None,
                    "chunk_end": None,
                    "is_last_chunk": True,
                },
            )
            resp.raise_for_status()

        dist.barrier()
        elapsed = time.time() - t0
        throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
        if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: P2P receive (pre-registered) — %d bufs, %.2f MB, %.2fs, %.2f Gbps",
        dist.get_rank(), len(buffers), total_bytes / 1e6, elapsed, throughput_gbps,)
        return

    # Fallback: chunked registration path.
    chunks = _chunk_ranges(len(buffers), MAX_RDMA_BUFS)

    for ci, (cs, ce) in enumerate(chunks):
        t_chunk = time.time()

        if ci == 0:
            ep.dereg_sync()
            ep.register(buffers[cs:ce])
            ep.ep.start_passive_accept()
        else:
            ep.reregister(buffers[cs:ce])
        t_reg = time.time()

        serialized = ep.ep.get_serialized_descs(ep.xfer_descs)
        my_info = (ep.ep.get_metadata().hex(), serialized.hex())
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(my_info, gathered, dst=0)
        t_gather = time.time()

        if dist.get_rank() == 0:
            base = peer_url.rstrip("/")
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
        if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: recv chunk %d/%d [%d:%d] %d bufs %.2f MB — "
        "reg %.3fs gather %.3fs post+xfer %.3fs barrier %.3fs total %.3fs",
        dist.get_rank(), ci + 1, len(chunks), cs, ce, ce - cs,
        chunk_bytes / 1e6,
        t_reg - t_chunk, t_gather - t_reg, t_post - t_gather,
        t_done - t_post, t_done - t_chunk,)

    elapsed = time.time() - t0
    throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: P2P receive complete — %d bufs, %.2f MB, %.2fs, %.2f Gbps",
    dist.get_rank(), len(buffers), total_bytes / 1e6, elapsed, throughput_gbps,)


def push_to_peer(
    ep: RankEndpoint,
    buffers: Sequence[Tuple[str, int, int]],
    per_rank_info: list | None = None,
    is_last_chunk: bool = True,
) -> None:
    """All ranks push their buffer shard to the corresponding peer rank.

    Called on the active engine.  Rank 0 broadcasts *per_rank_info* to all
    workers; each rank connects to its peer and RDMA-writes its buffers.

    Parameters
    ----------
    ep : RankEndpoint
        The per-rank endpoint (will be registered if not already).
    buffers : sequence of (name, va, size_bytes)
        Device buffers to push from.
    per_rank_info : list, optional
        ``[(remote_metadata_hex, remote_descs_hex), ...]`` per rank.
        Only required on rank 0.
    is_last_chunk : bool
        If True (default), destroy the endpoint after the transfer.
        Set to False to keep the endpoint alive for subsequent chunks.
    """
    obj_list = [per_rank_info] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    remote_metadata_hex, remote_descs_hex = obj_list[0][dist.get_rank()]

    t0 = time.time()

    # Re-register only the chunk's buffers, keeping the endpoint alive
    # so add_remote_endpoint can reuse the cached connection.
    # Skip if already pre-registered (same buffer count).
    if len(ep.xfer_descs) != len(buffers):
        ep.reregister(buffers)
    t_reg = time.time()

    uccl_ep = ep.ep
    ok, conn_id = uccl_ep.add_remote_endpoint(bytes.fromhex(remote_metadata_hex))
    assert ok, "Failed to connect to remote endpoint"
    t_conn = time.time()

    remote_descs = uccl_ep.deserialize_descs(bytes.fromhex(remote_descs_hex))
    assert len(remote_descs) == len(ep.xfer_descs)

    ok, _ = uccl_ep.transfer(conn_id, "write", ep.xfer_descs, remote_descs)
    assert ok, "RDMA write failed"
    t_xfer = time.time()

    chunk_bytes = sum(sz for _, _, sz in buffers)
    xfer_secs = t_xfer - t_conn
    xfer_gbps = (chunk_bytes * 8) / xfer_secs / 1e9 if xfer_secs > 0 else 0

    if is_last_chunk:
        # Kick off MR deregistration in background — don't block the
        # HTTP response or the barrier on the slow ibv_dereg_mr calls.
        ep.dereg_async()
    t_dereg = time.time()

    if not dist.is_initialized() or dist.get_rank() == 0: logger.info("Rank %d: pushed %d bufs %.2f MB — "
    "reg %.3fs connect %.3fs xfer %.3fs (%.2f Gbps) dereg %.3fs total %.3fs",
    dist.get_rank(), len(buffers), chunk_bytes / 1e6,
    t_reg - t0, t_conn - t_reg, xfer_secs, xfer_gbps,
    t_dereg - t_xfer, t_dereg - t0,)

    dist.barrier()


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
        return
    rank_endpoint.register_chunked(bufs, MAX_RDMA_BUFS)


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
    bufs = collect_weight_buffers(model)
    if chunk_start is not None:
        bufs = bufs[chunk_start:chunk_end]
    push_to_peer(rank_endpoint, bufs, per_rank_info, is_last_chunk=is_last_chunk)


def fetch_tok_embedding(peer_url: str):
    """Fetch tok_embedding from peer over HTTP and broadcast to all ranks."""
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = requests.get(f"{base}/tok_embedding")
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
