# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL-based peer-to-peer weight transfer.

Provides push/receive functions for direct device-to-device RDMA transfers
via NIXL's LIBFABRIC backend. Uses sender-initiated RDMA WRITE (push model)
matching relay's existing transfer flow.

Transfer flow:
  1. Receiver: register VRAM, gather agent metadata + buffer VAs, POST to sender
  2. Sender: add receiver agent, RDMA WRITE local VRAM -> receiver VRAM
  3. Sender: returns HTTP response indicating completion
  4. Receiver: done — data is in device memory
"""

import logging
import os
import sys
import time
from typing import List, Sequence, Tuple

import torch.distributed as dist

from .endpoint import Endpoint, logger

# Module-level singleton
endpoint = Endpoint()


def collect_weight_buffers(model) -> List[Tuple[str, int, int]]:
    """Return ``(name, va, size_bytes)`` for every weight tensor in *model*."""
    return list(model.weight_buffers())


def _poll_xfer(agent, xfer, timeout: float = 120.0) -> str:
    """Poll a NIXL transfer until completion or timeout."""
    t0 = time.time()
    while True:
        status = agent.check_xfer_state(xfer)
        if status != "PROC":
            return status
        if time.time() - t0 > timeout:
            return "TIMEOUT"
        time.sleep(0.0001)


# ------------------------------------------------------------------
# Receiver side
# ------------------------------------------------------------------


def receive_from_peer(
    ep: Endpoint,
    buffers: Sequence[Tuple[str, int, int]],
    peer_url: str,
    push_endpoint: str = "/nkipy/push_weights",
) -> None:
    """All ranks receive buffers from a peer engine via NIXL RDMA WRITE (push).

    Flow:
    1. Register local VRAM buffers with NIXL
    2. Gather agent metadata + buffer VAs across ranks (rank 0 collects)
    3. POST to sender with receiver metadata — sender does RDMA WRITE
    4. When HTTP returns, transfer is complete
    """
    import requests

    t0 = time.time()
    total_bytes = sum(sz for _, _, sz in buffers)
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Register local VRAM
    ep.register(buffers)
    t_reg = time.time()

    # Gather agent metadata and buffer VAs from all ranks
    nc = ep._nc_idx if ep._nc_idx is not None else int(
        os.environ.get("NEURON_RT_VISIBLE_CORES", "0").split(",")[0]
    )
    local_meta_hex = ep.get_metadata().hex()
    local_info = {
        "agent_metadata": local_meta_hex,
        "agent_name": ep.agent_name,
        "nc_idx": nc,
        "buffer_vas": [(va, sz) for _, va, sz in buffers],
    }
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local_info, gathered, dst=0)
    t_gather = time.time()

    # Rank 0 posts to sender — sender does the RDMA WRITE and returns when done
    if rank == 0:
        base = peer_url.rstrip("/")
        resp = requests.post(
            f"{base}{push_endpoint}",
            json={"receivers": [gathered]},
            timeout=300,
        )
        resp.raise_for_status()
    t_xfer = time.time()

    dist.barrier()
    t_done = time.time()

    elapsed = t_done - t0
    throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if rank == 0:
        logger.info(
            "Rank %d: NIXL receive %d bufs %.2f MB — "
            "reg %.3fs gather %.3fs xfer %.3fs total %.3fs (%.1f Gbps)",
            rank, len(buffers), total_bytes / 1e6,
            t_reg - t0, t_gather - t_reg, t_xfer - t_gather,
            elapsed, throughput_gbps,
        )

    # Registration stays alive until sleep, when worker calls ep.destroy().
    # Each wake creates a fresh agent with incremented epoch, so the sender
    # adds it as a new remote agent (old entries are tiny metadata pointers).


# ------------------------------------------------------------------
# Sender side
# ------------------------------------------------------------------


def push_to_peer(
    ep: Endpoint,
    buffers: Sequence[Tuple[str, int, int]],
    per_rank_info: list,
) -> None:
    """All ranks push buffers to the corresponding peer rank via NIXL RDMA WRITE.

    Parameters
    ----------
    ep : Endpoint
    buffers : sequence of (name, va, size_bytes)
    per_rank_info : list of dict
        ``[{"agent_metadata": hex, "agent_name": str, "buffer_vas": [(va, sz), ...]}, ...]``
        One entry per rank. Each rank extracts its own receiver info.
    """
    t0 = time.time()
    rank = dist.get_rank()
    total_bytes = sum(sz for _, _, sz in buffers)

    # Ensure local buffers are registered
    if not ep.registered:
        ep.register(buffers)
    t_reg = time.time()

    # Add remote receiver agent
    my_receiver = per_rank_info[rank]
    receiver_meta = my_receiver["agent_metadata"]
    receiver_name = my_receiver["agent_name"]
    ep.add_remote_agent(bytes.fromhex(receiver_meta), name=receiver_name)

    # Build local (source) descriptors
    nc = ep._nc_idx if ep._nc_idx is not None else int(
        os.environ.get("NEURON_RT_VISIBLE_CORES", "0").split(",")[0]
    )
    local_desc_list = [(va, sz, nc) for _, va, sz in buffers]
    local_descs = ep.agent.get_xfer_descs(local_desc_list, "VRAM")

    # Build remote (destination) descriptors from receiver's buffer VAs
    remote_nc = my_receiver.get("nc_idx", nc)
    remote_vas = my_receiver["buffer_vas"]
    remote_desc_list = [(va, sz, remote_nc) for va, sz in remote_vas]
    remote_descs = ep.agent.get_xfer_descs(remote_desc_list, "VRAM")
    t_descs = time.time()

    # Issue RDMA WRITE
    xfer = ep.agent.initialize_xfer("WRITE", local_descs, remote_descs, receiver_name)
    status = ep.agent.transfer(xfer)
    if status == "PROC":
        status = _poll_xfer(ep.agent, xfer)
    t_xfer = time.time()

    if status != "DONE":
        raise RuntimeError(f"Rank {rank}: NIXL RDMA WRITE failed with status={status}")

    xfer.release()

    elapsed = t_xfer - t0
    throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if rank == 0:
        logger.info(
            "Rank %d: NIXL push %d bufs %.2f MB — "
            "reg %.3fs descs %.3fs xfer %.3fs total %.3fs (%.1f Gbps)",
            rank, len(buffers), total_bytes / 1e6,
            t_reg - t0, t_descs - t_reg, t_xfer - t_descs,
            elapsed, throughput_gbps,
        )


# ------------------------------------------------------------------
# High-level wrappers (parallel to transfer.py's convenience functions)
# ------------------------------------------------------------------


def preregister_weights(model, ep: Endpoint | None = None) -> None:
    """Pre-register all model weight buffers as VRAM for NIXL RDMA.

    Call once after model tensors are allocated. Subsequent push_to_peer
    calls will skip registration.
    """
    if ep is None:
        ep = endpoint
    if ep.registered:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info("Rank %d: NIXL VRAM already registered, skipping", rank)
        return
    bufs = collect_weight_buffers(model)
    t0 = time.time()
    ep.register(bufs)
    elapsed = time.time() - t0
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info("Rank %d: NIXL pre-registered %d weight regions in %.3fs",
                    rank, len(bufs), elapsed)


def transfer_weights(model, receivers: list[list]) -> None:
    """Push model weights to one or more receivers in parallel.

    Parameters
    ----------
    receivers : list of per_rank_info lists
        Each entry is a per_rank_info list (one dict per rank) for a single
        receiver engine. For broadcast, pass multiple entries.
    """
    bufs = collect_weight_buffers(model)
    if len(receivers) == 1:
        push_to_peer(endpoint, bufs, receivers[0])
    else:
        broadcast_to_peers(endpoint, bufs, receivers)


def broadcast_to_peers(
    ep: Endpoint,
    buffers: Sequence[Tuple[str, int, int]],
    receivers: list[list],
) -> None:
    """All ranks push buffers to multiple receivers via parallel RDMA WRITEs.

    Issues RDMA WRITEs to all receivers concurrently and polls until all
    complete. Maximum recommended broadcast degree is 5.
    """
    t0 = time.time()
    rank = dist.get_rank()
    total_bytes = sum(sz for _, _, sz in buffers)

    if not ep.registered:
        ep.register(buffers)
    t_reg = time.time()

    nc = ep._nc_idx if ep._nc_idx is not None else int(
        os.environ.get("NEURON_RT_VISIBLE_CORES", "0").split(",")[0]
    )
    local_desc_list = [(va, sz, nc) for _, va, sz in buffers]
    local_descs = ep.agent.get_xfer_descs(local_desc_list, "VRAM")

    # Issue all RDMA WRITEs
    xfers = []
    for recv_idx, per_rank_info in enumerate(receivers):
        my_receiver = per_rank_info[rank]
        receiver_meta = my_receiver["agent_metadata"]
        receiver_name = my_receiver["agent_name"]
        if rank == 0:
            logger.info("Rank 0: adding remote agent %d: name=%s, meta_len=%d",
                        recv_idx, receiver_name, len(receiver_meta) // 2)
        ep.add_remote_agent(bytes.fromhex(receiver_meta), name=receiver_name)

        remote_nc = my_receiver.get("nc_idx", nc)
        remote_vas = my_receiver["buffer_vas"]
        remote_desc_list = [(va, sz, remote_nc) for va, sz in remote_vas]
        remote_descs = ep.agent.get_xfer_descs(remote_desc_list, "VRAM")

        xfer = ep.agent.initialize_xfer("WRITE", local_descs, remote_descs, receiver_name)
        status = ep.agent.transfer(xfer)
        xfers.append((xfer, receiver_name, status))
    t_issue = time.time()

    # Poll all transfers to completion
    for xfer, receiver_name, status in xfers:
        if status == "PROC":
            status = _poll_xfer(ep.agent, xfer)
        if status != "DONE":
            raise RuntimeError(
                f"Rank {rank}: RDMA WRITE to {receiver_name} failed with status={status}"
            )
        xfer.release()
    t_done = time.time()

    elapsed = t_done - t0
    total_pushed = total_bytes * len(receivers)
    throughput_gbps = (total_pushed * 8) / elapsed / 1e9 if elapsed > 0 else 0
    if rank == 0:
        logger.info(
            "Rank %d: broadcast to %d receivers, %.2f MB each — "
            "reg %.3fs issue %.3fs poll %.3fs total %.3fs (%.1f Gbps aggregate)",
            rank, len(receivers), total_bytes / 1e6,
            t_reg - t0, t_issue - t_reg, t_done - t_issue,
            elapsed, throughput_gbps,
        )


def receive_weights(model, peer_url: str) -> None:
    """All ranks receive model weights from *peer_url* via NIXL P2P."""
    bufs = collect_weight_buffers(model)
    receive_from_peer(endpoint, bufs, peer_url)
