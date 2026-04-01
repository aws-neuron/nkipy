"""
Peer-to-peer weight transfer for LLM serving wake-up.

When an engine wakes from sleep, it retrieves model weights from a peer engine's
HBM via UCCL P2P RDMA write — directly from device to device, without copying
weights through CPU memory.

Protocol (multi-rank):
  1. All waking ranks register their device tensor VAs with UCCL
  2. Per-rank UCCL metadata is gathered to rank 0 and POSTed to the active engine
  3. Active engine broadcasts per-rank info to its workers; each rank RDMA-writes
     its weight shard directly into the corresponding waking rank's device HBM
  4. tok_embedding (CPU tensor) is fetched separately over HTTP
"""

import os
import sys
import time

import numpy as np
import requests
import torch
import torch.distributed as dist

_uccl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uccl-trn")
if _uccl_path not in sys.path:
    sys.path.insert(0, _uccl_path)

os.environ.setdefault("UCCL_RCMODE", "1")

from uccl import p2p

_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from common.utils import print_log


# --- Helpers ---

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


def _register_weight_handles(model):
    """Build _VAHandle list for all weight tensors in the model."""
    handles = []
    for _, dt in _collect_weight_tensors(model):
        va = dt.tensor_ref.va
        size_bytes = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
        handles.append(_VAHandle(va, size_bytes))
    return handles


# --- Single-rank classes ---

class WeightServer:
    """Runs on the active engine. Registers device tensor VAs and pushes them on request."""

    def __init__(self, model, nc_idx=None):
        self.nc_idx = nc_idx if nc_idx is not None else _get_nc_idx()
        self.ep = p2p.Endpoint(self.nc_idx)
        self.metadata = self.ep.get_metadata()
        self.weight_info = []  # [(name, size_bytes)]
        self._xfer_descs = []

        t0 = time.time()
        handles = []
        for name, dt in _collect_weight_tensors(model):
            va = dt.tensor_ref.va
            size_bytes = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
            self.weight_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        self._xfer_descs = self.ep.register_memory(handles)
        print_log(f"WeightServer: registered {len(self._xfer_descs)} tensors "
                  f"in {time.time() - t0:.2f}s")

    def get_weight_info(self):
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

    def cleanup(self):
        if self.ep is not None:
            for desc in self._xfer_descs:
                self.ep.dereg(desc.mr_id)
            self._xfer_descs = []
            self.weight_info.clear()
            self.ep = None
            self.metadata = None


# --- Multi-rank transfer (all ranks participate) ---

def receive_weights(model, peer_url):
    """All ranks receive their weight shard from the peer engine via P2P RDMA.

    Each rank registers its device tensor VAs as receive buffers. Per-rank UCCL
    metadata is gathered to rank 0, which POSTs it to the peer engine. The peer
    pushes weights to all ranks simultaneously.
    """
    t0 = time.time()

    # Each rank sets up a UCCL endpoint and registers receive buffers
    nc_idx = _get_nc_idx()
    ep = p2p.Endpoint(nc_idx)
    ep.start_passive_accept()

    handles = _register_weight_handles(model)
    recv_descs = ep.register_memory(handles)
    serialized = ep.get_serialized_descs(recv_descs)

    # Gather per-rank UCCL metadata to rank 0
    my_info = (ep.get_metadata().hex(), serialized.hex())
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(my_info, gathered, dst=0)

    # Rank 0 sends all per-rank info to peer
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = requests.post(
            f"{base}/p2p_push_weights",
            json={"per_rank": [{"remote_metadata": m, "remote_descs": d} for m, d in gathered]},
        )
        resp.raise_for_status()

    dist.barrier()

    # Cleanup UCCL resources
    for desc in recv_descs:
        ep.dereg(desc.mr_id)

    print_log(f"--> P2P weight transfer completed in {time.time() - t0:.2f}s")


def push_weights_to_peer(model, per_rank_info=None):
    """All ranks push their weight shard to the corresponding peer rank via P2P RDMA.

    Called on the active engine. Rank 0 receives per_rank_info from the HTTP
    endpoint and broadcasts to all ranks.
    """
    if dist.get_rank() == 0:
        obj_list = [per_rank_info]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    remote_metadata_hex, remote_descs_hex = obj_list[0][dist.get_rank()]

    server = WeightServer(model)
    server.push_weights(remote_metadata_hex, bytes.fromhex(remote_descs_hex))
    server.cleanup()
    dist.barrier()


def fetch_tok_embedding(peer_url):
    """Fetch tok_embedding from peer (rank 0 over HTTP) and broadcast to all ranks."""
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = requests.get(f"{base}/tok_embedding")
        resp.raise_for_status()
        shape = tuple(int(d) for d in resp.headers["X-Shape"].split(","))
        dtype_str = resp.headers["X-Dtype"]
        torch_dtype = getattr(torch, dtype_str.replace("torch.", ""), None)
        if torch_dtype is not None:
            tok_embedding = torch.frombuffer(
                bytearray(resp.content), dtype=torch_dtype
            ).reshape(shape).clone()
        else:
            tok_embedding = torch.from_numpy(
                np.frombuffer(resp.content, dtype=np.dtype(dtype_str)).reshape(shape).copy()
            )
    else:
        tok_embedding = None
    obj_list = [tok_embedding]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
