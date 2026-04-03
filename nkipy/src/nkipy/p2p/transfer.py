# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-rank peer-to-peer weight transfer over RDMA.

Provides functions to push / receive device memory between engines using
UCCL P2P RDMA writes, plus a :class:`WeightServer` for the active engine
and a helper to collect device buffer descriptors from a model.
"""

import logging
import time
from typing import List, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist

from .endpoint import RankEndpoint

logger = logging.getLogger(__name__)

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


def receive_from_peer(
    ep: RankEndpoint,
    buffers: Sequence[Tuple[str, int, int]],
    peer_url: str,
    push_endpoint: str = "/p2p_push_weights",
) -> None:
    """All ranks receive their buffer shard from a peer engine via P2P RDMA.

    Each rank registers its device buffers as receive targets, gathers UCCL
    metadata to rank 0, and POSTs it to the peer engine which then RDMA-writes
    directly into device memory.

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

    ep.register(buffers)
    uccl_ep = ep.ep
    uccl_ep.start_passive_accept()

    serialized = uccl_ep.get_serialized_descs(ep.xfer_descs)

    my_info = (uccl_ep.get_metadata().hex(), serialized.hex())
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(my_info, gathered, dst=0)

    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = requests.post(
            f"{base}{push_endpoint}",
            json={
                "per_rank": [
                    {"remote_metadata": m, "remote_descs": d} for m, d in gathered
                ]
            },
        )
        resp.raise_for_status()

    dist.barrier()
    logger.info("P2P receive completed in %.2fs", time.time() - t0)


def push_to_peer(
    ep: RankEndpoint,
    buffers: Sequence[Tuple[str, int, int]],
    per_rank_info: list | None = None,
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
    """
    obj_list = [per_rank_info] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    remote_metadata_hex, remote_descs_hex = obj_list[0][dist.get_rank()]

    ep.register(buffers)

    uccl_ep = ep.ep
    ok, conn_id = uccl_ep.add_remote_endpoint(bytes.fromhex(remote_metadata_hex))
    assert ok, "Failed to connect to remote endpoint"

    remote_descs = uccl_ep.deserialize_descs(bytes.fromhex(remote_descs_hex))
    assert len(remote_descs) == len(ep.xfer_descs)

    t0 = time.time()
    ok, _ = uccl_ep.transfer(conn_id, "write", ep.xfer_descs, remote_descs)
    assert ok, "RDMA write failed"
    logger.info(
        "Rank %d: pushed %d buffers via P2P in %.2fs",
        dist.get_rank(),
        len(ep.xfer_descs),
        time.time() - t0,
    )

    # Reset endpoint so stale RDMA connections don't cause remote access
    # errors when the receiver sleeps and wakes with a new endpoint.
    ep.dereg_sync()

    dist.barrier()


# ------------------------------------------------------------------
# High-level model wrappers (use module-level rank_endpoint)
# ------------------------------------------------------------------


class WeightServer:
    """Runs on the active engine.  Registers model weights for RDMA push."""

    def __init__(self, model):
        bufs = collect_weight_buffers(model)
        rank_endpoint.register(bufs)
        logger.info(
            "WeightServer: registered %d tensors",
            len(rank_endpoint.xfer_descs),
        )

    def get_weight_info(self):
        return {
            "weights": list(rank_endpoint.buf_info),
            "metadata": rank_endpoint.ep.get_metadata().hex(),
        }

    def cleanup(self):
        rank_endpoint.wait()


def receive_weights(model, peer_url: str) -> None:
    """All ranks receive model weights from *peer_url* via P2P RDMA."""
    bufs = collect_weight_buffers(model)
    receive_from_peer(rank_endpoint, bufs, peer_url)


def push_weights_to_peer(model, per_rank_info=None) -> None:
    """All ranks push model weights to the corresponding peer rank."""
    bufs = collect_weight_buffers(model)
    push_to_peer(rank_endpoint, bufs, per_rank_info)


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
