# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL-backed endpoint for peer-to-peer device memory transfers.

Provides :class:`NixlEndpoint`, which implements the same lifecycle as
:class:`RankEndpoint` but uses NIXL's LIBFABRIC backend for direct
device-to-device RDMA (no host staging required on Trn2).
"""

import logging
import os
import sys
import threading
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)


def _get_nc_idx() -> int:
    vis = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    return int(vis.split(",")[0])


class NixlEndpoint:
    """Per-rank NIXL agent for VRAM RDMA transfers.

    Manages NIXL agent lifecycle, VRAM memory registration, and remote agent
    connection state. Provides an interface parallel to :class:`RankEndpoint`.

    The agent is destroyed and re-created on each register() call with new VAs
    to ensure the metadata blob reflects the current memory registrations. This
    is necessary because NIXL doesn't support updating remote agent metadata —
    the sender must load a fresh metadata blob after each receiver re-register.
    """

    def __init__(self, nc_idx: int | None = None):
        self._nc_idx = nc_idx
        self.agent = None
        self.agent_name: str | None = None
        self.reg_handle = None
        self.mem_descs: list = []
        self.buf_info: List[Tuple[str, int]] = []
        self._dereg_thread: threading.Thread | None = None
        self._remote_agents: set = set()
        self._listen_port: int = int(os.environ.get("NKIPY_NIXL_PORT", "21000"))
        self._reg_span: tuple | None = None
        self._epoch: int = 0

    def _wait_dereg(self) -> None:
        if self._dereg_thread is not None:
            self._dereg_thread.join()
            self._dereg_thread = None

    def _ensure_agent(self):
        """Create NIXL agent if not already present. Applies dmabuf fix first."""
        self._wait_dereg()
        if self.agent is not None:
            return self.agent

        from .nrt_dmabuf_fix import patch_nrt_dmabuf
        patch_nrt_dmabuf()

        from nixl._api import nixl_agent, nixl_agent_config
        import torch.distributed as dist

        import socket

        nc = self._nc_idx if self._nc_idx is not None else _get_nc_idx()
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = int(os.environ.get("RANK", "0"))
        short_host = socket.gethostname().split(".")[0][-8:]
        agent_name = f"nkipy_{short_host}_r{rank}_e{self._epoch}"

        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=self._listen_port + rank,
            backends=["LIBFABRIC"],
        )
        self.agent = nixl_agent(agent_name, config)
        self.agent_name = agent_name
        logger.info("Rank %d: NIXL agent '%s' created (port %d, nc %d)",
                    rank, agent_name, self._listen_port + rank, nc)
        return self.agent

    def register(self, buffers: Sequence[Tuple[str, int, int]]) -> None:
        """Register device buffers as VRAM for RDMA transfer.

        Registers a single contiguous region spanning all buffers to minimize
        the number of dmabuf FD + ibv_reg_mr calls (one instead of N).

        If already registered and the existing span covers all new buffers,
        this is a no-op (fast path for repeated wakes with same memory layout).

        Parameters
        ----------
        buffers : sequence of (name, va, size_bytes)
        """
        self._wait_dereg()

        min_va = min(va for _, va, _ in buffers)
        max_end = max(va + sz for _, va, sz in buffers)
        total_span = max_end - min_va

        if self.reg_handle is not None and self._reg_span is not None:
            old_start, old_size = self._reg_span
            if min_va >= old_start and max_end <= old_start + old_size:
                self.buf_info = [(name, sz) for name, _, sz in buffers]
                return
            self.agent.deregister_memory(self.reg_handle, backends=["LIBFABRIC"])
            self.reg_handle = None

        agent = self._ensure_agent()
        nc = self._nc_idx if self._nc_idx is not None else _get_nc_idx()

        self.buf_info = [(name, sz) for name, _, sz in buffers]
        mem_list = [(min_va, total_span, nc, "")]

        self.reg_handle = agent.register_memory(
            mem_list, mem_type="VRAM", backends=["LIBFABRIC"]
        )
        self.mem_descs = mem_list
        self._reg_span = (min_va, total_span)
        logger.info("Rank %s: registered %d bufs as 1 region (%.2f MB, span %.2f MB)",
                    os.environ.get("RANK", "0"), len(buffers),
                    sum(sz for _, sz in self.buf_info) / 1e6,
                    total_span / 1e6)

    def get_xfer_descs(self, buffers: Sequence[Tuple[str, int, int]]):
        """Get NIXL transfer descriptors for the given buffers."""
        agent = self._ensure_agent()
        nc = self._nc_idx if self._nc_idx is not None else _get_nc_idx()
        desc_list = [(va, size_bytes, nc) for _, va, size_bytes in buffers]
        return agent.get_xfer_descs(desc_list, "VRAM")

    def add_remote_agent(self, metadata: bytes, name: str | None = None) -> None:
        """Add a remote NIXL agent from serialized metadata.

        Skips if agent with the given name is already known (same persistent
        receiver agent across sleep/wake cycles).
        """
        agent = self._ensure_agent()
        if name and name in self._remote_agents:
            return
        added_name = agent.add_remote_agent(metadata)
        self._remote_agents.add(added_name)

    def get_metadata(self) -> bytes:
        """Return this agent's serialized metadata for peer exchange."""
        agent = self._ensure_agent()
        return agent.get_agent_metadata()

    def deregister_sync(self) -> None:
        """Deregister all memory (blocking)."""
        self._wait_dereg()
        if self.agent is not None and self.reg_handle is not None:
            self.agent.deregister_memory(self.reg_handle, backends=["LIBFABRIC"])
        self.reg_handle = None
        self.mem_descs = []
        self.buf_info = []
        self._reg_span = None

    def destroy(self) -> None:
        """Tear down agent completely, freeing all VRAM registrations.

        Increments the epoch so the next _ensure_agent() creates a fresh agent
        with a new name. This allows the sender to distinguish between the old
        and new receiver without needing remove_remote_agent.
        """
        self._wait_dereg()
        if self.agent is not None:
            if self.reg_handle is not None:
                self.agent.deregister_memory(self.reg_handle, backends=["LIBFABRIC"])
                self.reg_handle = None
            del self.agent
        self.agent = None
        self.agent_name = None
        self.reg_handle = None
        self.mem_descs = []
        self.buf_info = []
        self._reg_span = None
        self._remote_agents.clear()
        self._epoch += 1

    def deregister_async(self) -> None:
        """Deregister memory in a background thread."""
        agent = self.agent
        reg = self.reg_handle
        self.reg_handle = None
        self.mem_descs = []
        self.buf_info = []
        self._reg_span = None
        if agent is not None and reg is not None:
            def _bg():
                agent.deregister_memory(reg, backends=["LIBFABRIC"])
            self._dereg_thread = threading.Thread(target=_bg, daemon=True)
            self._dereg_thread.start()

    def wait(self) -> None:
        """Wait for pending async deregistration."""
        self._wait_dereg()

    @property
    def registered(self) -> bool:
        return self.reg_handle is not None
