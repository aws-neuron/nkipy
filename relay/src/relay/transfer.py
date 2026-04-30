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
    push_endpoint: str = "/p2p_push_weights",
    preconnect_endpoint: str = "/p2p_preconnect",
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

    if is_last_chunk and not pre_registered:
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
