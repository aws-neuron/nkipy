# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Low-level RDMA endpoint management for peer-to-peer device memory transfers.

Provides :class:`RankEndpoint`, a per-rank singleton that wraps a UCCL P2P
Endpoint and manages RDMA memory registration / deregistration lifecycle.
The class is transport-generic — callers supply ``(va, size_bytes)`` pairs
and receive opaque transfer descriptors back.
"""

import logging
import os
import threading
from typing import List, Sequence, Tuple

os.environ.setdefault("UCCL_RCMODE", "1")

from uccl import p2p  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_nc_idx() -> int:
    """Return the neuron core index for the current rank."""
    vis = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    return int(vis.split(",")[0])


class _VAHandle:
    """Adapter so ``register_memory()`` can extract ptr / size."""

    __slots__ = ("_va", "_size")

    def __init__(self, va: int, size_bytes: int):
        self._va = va
        self._size = size_bytes

    def data_ptr(self) -> int:
        return self._va

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


# ------------------------------------------------------------------
# RankEndpoint
# ------------------------------------------------------------------


class RankEndpoint:
    """Persistent UCCL endpoint for a single rank.

    Manages the RDMA Endpoint lifecycle and memory registration.  Callers
    register device buffers via :meth:`register`, transfer via the raw
    endpoint, and tear down via :meth:`dereg_sync` / :meth:`dereg_async`.

    Parameters
    ----------
    nc_idx : int, optional
        Neuron core index.  Defaults to ``NEURON_RT_VISIBLE_CORES``.
    """

    def __init__(self, nc_idx: int | None = None):
        self._nc_idx = nc_idx  # None → resolve lazily from env at first use
        self.ep = None
        self.xfer_descs: list = []
        self.buf_info: List[Tuple[str, int]] = []  # [(name, size_bytes)]
        self._dereg_thread: threading.Thread | None = None

    # -- internal helpers ------------------------------------------

    def _wait_dereg(self) -> None:
        """Block until any pending async deregistration completes."""
        if self._dereg_thread is not None:
            self._dereg_thread.join()
            self._dereg_thread = None

    def _ensure_endpoint(self):
        if self.ep is None:
            nc = self._nc_idx if self._nc_idx is not None else _get_nc_idx()
            self.ep = p2p.Endpoint(nc)
        return self.ep

    # -- public API ------------------------------------------------

    def register(self, buffers: Sequence[Tuple[str, int, int]]) -> None:
        """Register device buffers for RDMA transfer.

        Parameters
        ----------
        buffers : sequence of (name, va, size_bytes)
            Each entry describes one contiguous device buffer.

        If buffers are already registered this is a no-op.
        """
        self._wait_dereg()
        if self.xfer_descs:
            return
        ep = self._ensure_endpoint()
        handles = []
        self.buf_info = []
        for name, va, size_bytes in buffers:
            self.buf_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        self.xfer_descs = ep.register_memory(handles)

    def register_chunked(self, buffers: Sequence[Tuple[str, int, int]],
                         chunk_size: int) -> None:
        """Register buffers in chunks to stay within EFA MR limits.

        Calls ``register_memory`` repeatedly with *chunk_size* buffers at a
        time, accumulating all descriptors.  The endpoint is created once
        and reused across chunks.
        """
        self._wait_dereg()
        if self.xfer_descs:
            return
        ep = self._ensure_endpoint()
        self.buf_info = []
        self.xfer_descs = []
        for i in range(0, len(buffers), chunk_size):
            chunk = buffers[i : i + chunk_size]
            handles = []
            for name, va, size_bytes in chunk:
                self.buf_info.append((name, size_bytes))
                handles.append(_VAHandle(va, size_bytes))
            self.xfer_descs.extend(ep.register_memory(handles))

    def reregister(self, buffers: Sequence[Tuple[str, int, int]]) -> None:
        """Deregister current MRs and register *buffers*, keeping the endpoint.

        Unlike :meth:`dereg_sync` followed by :meth:`register`, this does
        **not** destroy the underlying UCCL endpoint, so existing RDMA
        connections remain valid.
        """
        self._wait_dereg()
        ep = self._ensure_endpoint()
        # Deregister old MRs (if any) without destroying the endpoint.
        for desc in self.xfer_descs:
            ep.dereg(desc.mr_id)
        self.xfer_descs = []
        self.buf_info = []
        # Register new buffers on the same endpoint.
        handles = []
        for name, va, size_bytes in buffers:
            self.buf_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        self.xfer_descs = ep.register_memory(handles)

    def dereg_sync(self) -> None:
        """Deregister memory and destroy the endpoint (blocking)."""
        self._wait_dereg()
        if self.ep is not None and self.xfer_descs:
            for desc in self.xfer_descs:
                self.ep.dereg(desc.mr_id)
        self.xfer_descs = []
        self.buf_info = []
        # Destroy the endpoint so stale RDMA connections don't persist
        # across sleep/wake cycles (causes remote access errors on re-push).
        self.ep = None

    def dereg_async(self) -> None:
        """Kick off RDMA deregistration and endpoint teardown in a background thread."""
        self._wait_dereg()
        ep, descs = self.ep, self.xfer_descs
        self.ep = None
        self.xfer_descs = []
        self.buf_info = []
        if ep is not None:

            def _bg(ep, descs):
                for desc in descs:
                    ep.dereg(desc.mr_id)

            self._dereg_thread = threading.Thread(
                target=_bg, args=(ep, descs), daemon=True
            )
            self._dereg_thread.start()

    def wait(self) -> None:
        """Wait for any pending async deregistration to finish."""
        self._wait_dereg()

    @property
    def registered(self) -> bool:
        return bool(self.xfer_descs)
