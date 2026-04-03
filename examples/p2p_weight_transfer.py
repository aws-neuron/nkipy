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

Optimization: Each rank keeps a persistent UCCL Endpoint and pre-registered
memory regions across sleep/wake cycles, avoiding repeated RDMA init/teardown.
"""

import os
import sys
import threading
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


# --- Persistent per-rank endpoint ---

class _RankEndpoint:
    """Persistent UCCL endpoint for a single rank. Reused across transfers."""

    def __init__(self):
        self.ep = None
        self.xfer_descs = []
        self.weight_info = []  # [(name, size_bytes)]
        self._dereg_thread = None

    def _wait_dereg(self):
        """Block until any pending async deregistration completes."""
        if self._dereg_thread is not None:
            self._dereg_thread.join()
            self._dereg_thread = None

    def ensure_endpoint(self):
        if self.ep is None:
            self.ep = p2p.Endpoint(_get_nc_idx())
        return self.ep

    def register_model(self, model):
        """Register model weight tensors if not already registered."""
        self._wait_dereg()
        if self.xfer_descs:
            return
        ep = self.ensure_endpoint()

        handles = []
        self.weight_info = []
        for name, dt in _collect_weight_tensors(model):
            va = dt.tensor_ref.va
            size_bytes = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
            self.weight_info.append((name, size_bytes))
            handles.append(_VAHandle(va, size_bytes))
        self.xfer_descs = ep.register_memory(handles)

    def _dereg_descs(self):
        if self.ep is not None and self.xfer_descs:
            for desc in self.xfer_descs:
                self.ep.dereg(desc.mr_id)
            self.xfer_descs = []
            self.weight_info = []
        # Destroy the endpoint so stale RDMA connections don't persist
        # across sleep/wake cycles (causes remote access errors on re-push).
        self.ep = None

    def dereg_descs_async(self):
        """Kick off RDMA deregistration and endpoint teardown in a background thread."""
        self._wait_dereg()
        ep, descs = self.ep, self.xfer_descs
        self.ep = None
        self.xfer_descs = []
        self.weight_info = []
        if ep is not None:
            def _bg(ep, descs):
                for desc in descs:
                    ep.dereg(desc.mr_id)
            self._dereg_thread = threading.Thread(
                target=_bg, args=(ep, descs), daemon=True
            )
            self._dereg_thread.start()

    def cleanup(self):
        self._wait_dereg()
        self._dereg_descs()


# Per-rank singleton
_rank_ep = _RankEndpoint()


# --- Single-rank classes ---

class WeightServer:
    """Runs on the active engine (rank 0). Exposes weight info and pushes on request.

    Uses the persistent _rank_ep so Endpoint and memory registration survive
    across multiple push requests.
    """

    def __init__(self, model):
        _rank_ep.register_model(model)
        print_log(f"WeightServer: registered {len(_rank_ep.xfer_descs)} tensors")

    def get_weight_info(self):
        return {
            "weights": [(n, s) for n, s in _rank_ep.weight_info],
            "metadata": _rank_ep.ep.get_metadata().hex(),
        }

    def push_weights(self, remote_metadata_hex, remote_descs_bytes):
        """Connect to remote endpoint and RDMA-write all weights device-to-device."""
        ep = _rank_ep.ep
        remote_metadata = bytes.fromhex(remote_metadata_hex)
        ok, conn_id = ep.add_remote_endpoint(remote_metadata)
        assert ok, "Failed to connect to remote endpoint"

        remote_descs = ep.deserialize_descs(remote_descs_bytes)
        assert len(remote_descs) == len(_rank_ep.xfer_descs), \
            f"Descriptor count mismatch: {len(remote_descs)} vs {len(_rank_ep.xfer_descs)}"

        t0 = time.time()
        ok, _ = ep.transfer(conn_id, "write", _rank_ep.xfer_descs, remote_descs)
        assert ok, "RDMA write failed"
        print_log(f"WeightServer: pushed {len(_rank_ep.xfer_descs)} tensors "
                  f"via device-to-device P2P in {time.time() - t0:.2f}s")

    def cleanup(self):
        _rank_ep._wait_dereg()


# --- Multi-rank transfer (all ranks participate) ---

def receive_weights(model, peer_url):
    """All ranks receive their weight shard from the peer engine via P2P RDMA.

    Each rank registers its device tensor VAs as receive buffers using the
    persistent endpoint. Per-rank UCCL metadata is gathered to rank 0, which
    POSTs it to the peer engine. The peer pushes weights to all ranks
    simultaneously.
    """
    t0 = time.time()

    _rank_ep.register_model(model)
    ep = _rank_ep.ep
    ep.start_passive_accept()

    serialized = ep.get_serialized_descs(_rank_ep.xfer_descs)

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
    print_log(f"--> P2P weight transfer completed in {time.time() - t0:.2f}s")


def push_weights_to_peer(model, per_rank_info=None):
    """All ranks push their weight shard to the corresponding peer rank via P2P RDMA.

    Called on the active engine. Uses the persistent endpoint — only connects
    and transfers, no Endpoint/memory re-creation.
    """
    if dist.get_rank() == 0:
        obj_list = [per_rank_info]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    remote_metadata_hex, remote_descs_hex = obj_list[0][dist.get_rank()]

    # Ensure this rank has registered its model weights
    _rank_ep.register_model(model)

    ep = _rank_ep.ep
    remote_metadata = bytes.fromhex(remote_metadata_hex)
    ok, conn_id = ep.add_remote_endpoint(remote_metadata)
    assert ok, "Failed to connect to remote endpoint"

    remote_descs = ep.deserialize_descs(bytes.fromhex(remote_descs_hex))
    assert len(remote_descs) == len(_rank_ep.xfer_descs)

    t0 = time.time()
    ok, _ = ep.transfer(conn_id, "write", _rank_ep.xfer_descs, remote_descs)
    assert ok, "RDMA write failed"
    print_log(f"Rank {dist.get_rank()}: pushed weights via P2P in {time.time() - t0:.2f}s")

    # Reset endpoint so stale RDMA connections don't cause remote access
    # errors when the receiver sleeps and wakes with a new endpoint.
    _rank_ep._dereg_descs()

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
