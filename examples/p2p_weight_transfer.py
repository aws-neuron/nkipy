"""
Peer-to-peer weight transfer for LLM serving wake-up.

When an engine wakes from sleep, it retrieves model weights from a peer engine's
HBM via UCCL P2P RDMA write — directly from device to device, without copying
weights through CPU memory.

Protocol:
  1. Waking engine GETs /weight_info from active engine to learn weight sizes
  2. Waking engine registers its device tensor VAs with UCCL, and POSTs
     its UCCL metadata + serialized receive descriptors to active engine's
     /p2p_push_weights endpoint
  3. Active engine registers its device tensor VAs, connects, and RDMA-writes
     weights directly from its device HBM into waking engine's device HBM
  4. No CPU copies needed — transfer is device-to-device
"""

import os
import sys
import time

import numpy as np
import requests

_uccl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uccl-trn")
if _uccl_path not in sys.path:
    sys.path.insert(0, _uccl_path)

os.environ.setdefault("UCCL_RCMODE", "1")

from uccl import p2p

_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from common.utils import print_log


def _get_nc_idx():
    """Return the neuron core index for the current rank."""
    vis = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    return int(vis.split(",")[0])


class _VAHandle:
    """Minimal wrapper so register_memory() can extract ptr/size from a device VA."""

    def __init__(self, va, size_bytes):
        self._va = va
        self._size = size_bytes

    def data_ptr(self):
        return self._va

    def numel(self):
        return self._size

    def element_size(self):
        return 1


def _collect_weight_tensors(model):
    """Yield (name, DeviceTensor) for all model weights (excluding caches)."""
    for layer_idx, layer in enumerate(model.layer_tensors):
        for key, dt in layer.items():
            if key in ("cache_k", "cache_v"):
                continue
            yield f"layers.{layer_idx}.{key}", dt
    if model.norm_weight is not None:
        yield "norm_weight", model.norm_weight
    if model.lm_head_weight is not None:
        yield "lm_head_weight", model.lm_head_weight


class WeightServer:
    """Runs on the active engine. Registers device tensor VAs and pushes them on request."""

    def __init__(self, model, nc_idx=None):
        self.nc_idx = nc_idx if nc_idx is not None else _get_nc_idx()
        self.ep = p2p.Endpoint(self.nc_idx)
        self.metadata = self.ep.get_metadata()
        self.weight_info = []  # [(name, size_bytes)]
        self._xfer_descs = []

        # Register device tensor VAs directly with UCCL — no CPU copy
        t0 = time.time()
        handles = []
        for name, dt in _collect_weight_tensors(model):
            va = dt.tensor_ref.va
            size_bytes = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
            self.weight_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        self._xfer_descs = self.ep.register_memory(handles)
        print_log(f"WeightServer: registered {len(self._xfer_descs)} device tensors "
                  f"in {time.time() - t0:.2f}s")

    def get_weight_info(self):
        """Return weight metadata and UCCL endpoint metadata."""
        return {
            "weights": [(n, s) for n, s in self.weight_info],
            "metadata": self.metadata.hex(),
        }

    def push_weights(self, remote_metadata_hex, remote_descs_bytes):
        """Connect to remote endpoint and RDMA-write all weights device-to-device."""
        remote_metadata = bytes.fromhex(remote_metadata_hex)
        ok, conn_id = self.ep.add_remote_endpoint(remote_metadata)
        assert ok, "Failed to connect to remote endpoint"

        remote_descs = self.ep.deserialize_descs(remote_descs_bytes)
        assert len(remote_descs) == len(self._xfer_descs), \
            f"Descriptor count mismatch: {len(remote_descs)} vs {len(self._xfer_descs)}"

        t0 = time.time()
        ok, _ = self.ep.transfer(conn_id, "write", self._xfer_descs, remote_descs)
        assert ok, "RDMA write failed"
        print_log(f"WeightServer: pushed {len(self._xfer_descs)} tensors "
                  f"via device-to-device P2P in {time.time() - t0:.2f}s")
        return True

    def cleanup(self):
        """Deregister memory regions and destroy the UCCL endpoint."""
        if self.ep is not None:
            for desc in self._xfer_descs:
                self.ep.dereg(desc.mr_id)
            self._xfer_descs = []
            self.weight_info.clear()
            self.ep = None
            self.metadata = None


class WeightClient:
    """Runs on the waking engine. Receives weights from peer via device-to-device P2P."""

    def __init__(self, peer_url, nc_idx=None):
        self.peer_url = peer_url.rstrip("/")
        self.nc_idx = nc_idx if nc_idx is not None else _get_nc_idx()
        self.ep = p2p.Endpoint(self.nc_idx)
        self.ep.start_passive_accept()
        self.metadata = self.ep.get_metadata()
        self._recv_descs = []

    def cleanup(self):
        """Deregister memory regions and destroy the UCCL endpoint."""
        if self.ep is not None:
            for desc in self._recv_descs:
                self.ep.dereg(desc.mr_id)
            self._recv_descs = []
            self.ep = None
            self.metadata = None

    def fetch_weights(self, model):
        """Fetch weights from peer directly into model's device tensors.

        Args:
            model: BaseModel instance (with layer_tensors, norm_weight, lm_head_weight
                   already allocated on device via _prepare_tensors)
        """
        # Step 1: Get weight info from peer
        resp = requests.get(f"{self.peer_url}/weight_info")
        resp.raise_for_status()
        info = resp.json()
        weight_list = info["weights"]  # [(name, size_bytes), ...]

        # Step 2: Register device tensor VAs as receive buffers
        handles = []
        for name, dt in _collect_weight_tensors(model):
            va = dt.tensor_ref.va
            size_bytes = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
            handles.append(_VAHandle(va, size_bytes))

        assert len(handles) == len(weight_list), \
            f"Weight count mismatch: local {len(handles)} vs remote {len(weight_list)}"

        recv_descs = self.ep.register_memory(handles)
        self._recv_descs = recv_descs
        serialized = self.ep.get_serialized_descs(recv_descs)

        # Step 3: Ask peer to push weights directly into our device buffers
        t0 = time.time()
        resp = requests.post(
            f"{self.peer_url}/p2p_push_weights",
            json={
                "remote_metadata": self.metadata.hex(),
                "remote_descs": serialized.hex(),
            },
        )
        resp.raise_for_status()
        print_log(f"WeightClient: received {len(recv_descs)} tensors "
                  f"via device-to-device P2P in {time.time() - t0:.2f}s")

        # No Step 4 needed — weights are already in device HBM!

        self.cleanup()
