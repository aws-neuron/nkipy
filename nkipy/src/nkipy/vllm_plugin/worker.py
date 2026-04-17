# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Worker for vLLM integration."""

import gc
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)

_KERNEL_NAMES = (
    "kernel_cte", "kernel_tkg",
    "kernel_cte_greedy_sampling", "kernel_tkg_greedy_sampling",
)


class NKIPyWorker(WorkerBase):
    """Worker that bridges vLLM with NKIPy backend."""

    def __init__(
        self,
        vllm_config: Any | None = None,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = True,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.model_runner = None
        self.device = None

        # P2P state
        self._sleeping = False
        self._kernel_cache = None  # {name: (neff_path, cache_key)}
        self._model_class = None
        self._config = None

    def init_device(self) -> None:
        import atexit
        import signal

        import torch_neuronx  # noqa: F401 — registers neuron Runtime class

        core_offset = int(os.environ.get("NKIPY_CORE_OFFSET", "0"))
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(self.local_rank + core_offset)

        # Only claim neuron cores if we have weights to load.
        # Sleep-mode engines defer core init to wake_up via get_spike_singleton().
        if os.environ.get("NKIPY_CHECKPOINT"):
            from spike import get_spike_singleton
            get_spike_singleton()

        self.device = torch.device(f"neuron:{self.local_rank}")

        # Ensure Neuron cores are released on exit / signal in worker processes.
        atexit.register(self._release_neuron_cores)
        for sig in (signal.SIGINT, signal.SIGTERM):
            prev = signal.getsignal(sig)
            def _handler(signum, frame, _prev=prev):
                self._release_neuron_cores()
                if callable(_prev) and _prev not in (signal.SIG_IGN, signal.SIG_DFL):
                    _prev(signum, frame)
                else:
                    raise SystemExit(128 + signum)
            signal.signal(sig, _handler)

        self._init_distributed()
        self._patch_get_accelerator()

        from .model_runner import NKIPyModelRunner

        self.model_runner = NKIPyModelRunner(
            vllm_config=self.vllm_config, device=self.device
        )
        set_random_seed(self.model_config.seed)
        logger.info("NKIPyWorker initialized: rank=%s device=%s", self.rank, self.device)

    @staticmethod
    def _release_neuron_cores():
        """Best-effort release of Neuron cores."""
        try:
            from spike import reset as spike_reset
            spike_reset()
        except Exception:
            pass

    @staticmethod
    def _patch_get_accelerator():
        """Patch _get_accelerator to fall back to CPU.

        PyTorch's dist.barrier() unconditionally calls
        torch._C._get_accelerator() which fails when PrivateUse1HooksInterface
        is not registered. We wrap it to return CPU on failure, matching how
        vllm-neuron avoids this via torch_neuronx's full registration.
        """
        _original = torch._C._get_accelerator

        def _safe_get_accelerator():
            try:
                return _original()
            except RuntimeError:
                return torch.device("cpu")

        torch._C._get_accelerator = _safe_get_accelerator

    def _init_distributed(self) -> None:
        import vllm.distributed.parallel_state as ps
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            ensure_model_parallel_initialized,
        )

        parallel_config = self.vllm_config.parallel_config

        # Patch in_the_same_node_as to avoid PrivateUse1 barrier issue
        # (same approach as vllm-neuron)
        def _patched_in_same_node(pg, source_rank=0):
            ws = dist.get_world_size(group=pg)
            nnodes = parallel_config.nnodes
            rpn = ws // max(nnodes, 1)
            src_node = source_rank // rpn
            return [r // rpn == src_node for r in range(ws)]

        ps.in_the_same_node_as = _patched_in_same_node

        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            backend="gloo",
        )

    def load_model(self) -> None:
        from vllm.config import set_current_vllm_config

        logger.info("Loading model on NKIPyWorker")
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

        # Capture P2P state from model runner after load
        mr = self.model_runner
        self._model_class = getattr(mr, '_model_class', None)
        self._config = getattr(mr, '_config', None)
        self._kernel_cache = getattr(mr, '_kernel_cache', None)
        self._sleeping = mr._nkipy_model is None

        # Pre-register weight MRs for RDMA so push_to_peer skips registration.
        if mr._nkipy_model is not None:
            from nkipy.p2p import preregister_weights
            preregister_weights(mr._nkipy_model)

    def determine_available_memory(self) -> int:
        return 1 * (1024 ** 3)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        pass

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        self.model_runner.warmup_model()

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def get_model(self):
        return self.model_runner.model

    def get_supported_tasks(self):
        return ("generate",)

    # ------------------------------------------------------------------
    # P2P operations (called via collective_rpc on ALL ranks)
    # ------------------------------------------------------------------

    def nkipy_sleep(self) -> dict:
        """Release device resources and enter sleep mode."""
        import time as _time

        if self._sleeping:
            return {"status": "already_sleeping"}

        t_start = _time.time()

        from spike import reset as spike_reset
        from nkipy.runtime.device_kernel import _LOADED_KERNELS
        from nkipy.p2p import rank_endpoint

        # Just clear the descriptors — spike_reset() will tear down NRT runtime.
        # Avoid touching rank_endpoint.ep to prevent triggering destructor cleanup.
        rank_endpoint.xfer_descs = []
        rank_endpoint.buf_info = []
        t_endpoint = _time.time()

        # Kernel cache is already populated during load_model() or wake_up().
        # No need to access model attrs here — just verify cache exists.
        if self._kernel_cache is None:
            logger.warning("Rank %d: _kernel_cache is None during sleep, skipping cache check",
                          self.rank)
        t_cache = _time.time()

        # CRITICAL: Clear model references BEFORE spike_reset
        # This allows spike to see that Python no longer holds references,
        # enabling faster device memory cleanup
        self.model_runner._nkipy_model = None
        self.model_runner.model = None
        _LOADED_KERNELS.clear()
        t_clear_refs = _time.time()

        # Force GC to actually release the objects
        gc.collect()
        t_gc = _time.time()

        # Now spike_reset should be fast - Python refs are gone
        spike_reset()
        t_spike = _time.time()

        # Clear endpoint after spike_reset to avoid blocking cleanup.
        rank_endpoint.ep = None
        rank_endpoint._dereg_threads = []
        rank_endpoint._dereg_thread = None

        self._sleeping = True

        latency = {
            "rank": self.rank,
            "endpoint_clear_s": round(t_endpoint - t_start, 4),
            "cache_check_s": round(t_cache - t_endpoint, 4),
            "clear_refs_s": round(t_clear_refs - t_cache, 4),
            "gc_collect_s": round(t_gc - t_clear_refs, 4),
            "spike_reset_s": round(t_spike - t_gc, 4),
            "total_s": round(t_spike - t_start, 4),
        }
        logger.info("sleep latency breakdown (rank %d): %s", self.rank, latency)
        print(f"sleep latency breakdown (rank {self.rank}): {latency}", flush=True)
        return {"status": "sleeping", "latency": latency}

    @staticmethod
    def _acknowledge_rdma_writes(model):
        """Force NRT to acknowledge RDMA-written memory.

        RDMA transfers bypass NRT's normal write tracking. Reading back a sample
        of tensors forces NRT to check device memory and update its internal state,
        potentially speeding up spike_reset() cleanup.

        This is a hypothesis test - if successful, all ranks should achieve
        ~2-5s spike_reset instead of 18s.
        """
        import os
        sample_count = int(os.environ.get("NKIPY_RDMA_ACK_SAMPLES", "10"))

        if sample_count <= 0:
            return  # Disabled via env var

        checked = 0
        for layer in model.layer_tensors:
            if checked >= sample_count:
                break
            for key, tensor in layer.items():
                if key not in ("cache_k", "cache_v") and hasattr(tensor, "numpy"):
                    # Read one element to force NRT memory check
                    try:
                        _ = tensor.numpy().flat[0]
                        checked += 1
                        break
                    except Exception:
                        pass  # Skip on error

        # Force GC to release temporary numpy arrays
        gc.collect()

    def nkipy_wake_up(self, peer_url: str | None = None) -> dict:
        """Allocate tensors, receive weights from peer, reload kernels."""
        import time as _time

        if not self._sleeping:
            return {"status": "already_awake"}

        t_start = _time.time()

        from spike import get_spike_singleton
        from nkipy.runtime import DeviceKernel

        # Broadcast peer_url from rank 0 to all ranks
        obj_list = [peer_url]
        dist.broadcast_object_list(obj_list, src=0)
        actual_peer = obj_list[0]
        t_broadcast = _time.time()

        dist.barrier()
        t_pre_nrt = _time.time()
        get_spike_singleton()
        t_nrt = _time.time()
        dist.barrier()
        t_nrt_barrier = _time.time()

        # Rebuild model with empty tensors
        model = self._model_class(config=self._config, skip_kernels=True)
        model._allocate_empty_tensors()
        t_alloc = _time.time()

        if actual_peer:
            from nkipy.p2p import (
                receive_from_peer, rank_endpoint, collect_weight_buffers,
            )
            bufs = collect_weight_buffers(model)
            t_collect = _time.time()
            receive_from_peer(
                rank_endpoint, bufs, actual_peer,
                push_endpoint="/nkipy/p2p_push_weights",
            )
            t_p2p = _time.time()

            # HYPOTHESIS TEST: Force NRT to acknowledge RDMA-written memory.
            # RDMA writes bypass NRT tracking, potentially causing slow spike_reset.
            # Reading back tensors forces NRT to discover modifications and update
            # its memory state, which should speed up cleanup.
            self._acknowledge_rdma_writes(model)
            t_ack = _time.time()
        else:
            t_collect = t_alloc
            t_p2p = t_alloc
            t_ack = t_alloc

        # Reload kernels from cached NEFFs
        for name, (neff_path, cache_key) in self._kernel_cache.items():
            setattr(model, name, DeviceKernel.load_with_cache_key(
                neff_path, cache_key, name=name,
            ))
        t_kernels = _time.time()
        dist.barrier()
        t_kernel_barrier = _time.time()

        # Convert tok_embedding_device (DeviceTensor) back to CPU tensor for inference
        if model.tok_embedding is None and model.tok_embedding_device is not None:
            model.tok_embedding = model.tok_embedding_device.torch()
        t_tok = _time.time()

        self.model_runner._nkipy_model = model
        self.model_runner.model = model
        self._sleeping = False

        t_end = _time.time()

        latency = {
            "broadcast_s": round(t_broadcast - t_start, 4),
            "nrt_init_s": round(t_nrt - t_pre_nrt, 4),
            "nrt_barrier_s": round(t_nrt_barrier - t_nrt, 4),
            "alloc_tensors_s": round(t_alloc - t_nrt_barrier, 4),
            "collect_bufs_s": round(t_collect - t_alloc, 4),
            "p2p_transfer_s": round(t_p2p - t_collect, 4),
            "rdma_ack_s": round(t_ack - t_p2p, 4),
            "kernel_load_s": round(t_kernels - t_ack, 4),
            "kernel_barrier_s": round(t_kernel_barrier - t_kernels, 4),
            "tok_embedding_s": round(t_tok - t_kernel_barrier, 4),
            "total_s": round(t_end - t_start, 4),
        }
        if self.rank == 0:
            logger.info("wake_up latency breakdown (rank %d): %s", self.rank, latency)
            print(f"wake_up latency breakdown (rank {self.rank}): {latency}", flush=True)
        return {"status": "awake", "latency": latency}

    def nkipy_push_weights(
        self,
        per_rank_info: list[dict],
        chunk_start: int | None = None,
        chunk_end: int | None = None,
        is_last_chunk: bool = True,
    ) -> dict:
        """Push weight buffers to a peer engine via RDMA."""
        from nkipy.p2p import push_weights_to_peer

        model = self.model_runner._nkipy_model
        if model is None:
            return {"status": "error", "message": "model not loaded"}

        info = [
            (r["remote_metadata"], r["remote_descs"])
            for r in per_rank_info
        ]
        push_weights_to_peer(
            model, info,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_last_chunk=is_last_chunk,
        )
        return {"status": "done"}

    def nkipy_get_tok_embedding(self) -> bytes | None:
        """Return serialized tok_embedding (rank 0 only)."""
        model = self.model_runner._nkipy_model
        if model is None or model.tok_embedding is None:
            return None
        emb = model.tok_embedding
        if isinstance(emb, torch.Tensor):
            t = emb.cpu().contiguous()
            raw = t.view(torch.uint8).numpy().tobytes()
            shape = tuple(t.shape)
            dtype = str(t.dtype)
        else:
            data = np.ascontiguousarray(emb)
            raw = data.tobytes()
            shape = tuple(data.shape)
            dtype = str(data.dtype)
        return {"raw": raw, "shape": shape, "dtype": dtype}

    def nkipy_health(self) -> dict:
        """Return P2P health status."""
        return {"status": "ok", "backend": "nkipy", "sleeping": self._sleeping}

    @staticmethod
    def _fetch_tok_embedding(peer_url: str):
        """Fetch tok_embedding from peer over HTTP and broadcast to all ranks."""
        import time as _time
        import requests as _req
        rank = dist.get_rank()

        t0 = _time.time()
        if rank == 0:
            base = peer_url.rstrip("/")
            resp = _req.get(f"{base}/nkipy/tok_embedding")
            resp.raise_for_status()
            t_http = _time.time()
            shape = tuple(int(d) for d in resp.headers["X-Shape"].split(","))
            dtype_str = resp.headers["X-Dtype"]
            torch_dtype = getattr(torch, dtype_str.replace("torch.", ""), None)
            if torch_dtype is not None:
                tok_embedding = (
                    torch.frombuffer(bytearray(resp.content), dtype=torch_dtype)
                    .reshape(shape).clone()
                )
            else:
                tok_embedding = torch.from_numpy(
                    np.frombuffer(resp.content, dtype=np.dtype(dtype_str))
                    .reshape(shape).copy()
                )
            t_deser = _time.time()
            meta = [shape, str(tok_embedding.dtype)]
            size_mb = tok_embedding.numel() * tok_embedding.element_size() / 1e6
            print(f"tok_embedding: http {t_http-t0:.3f}s, deser {t_deser-t_http:.3f}s, {size_mb:.1f} MB", flush=True)
        else:
            meta = [None, None]

        dist.broadcast_object_list(meta, src=0)
        shape, dtype_str = meta

        if rank != 0:
            torch_dtype = getattr(torch, dtype_str.replace("torch.", ""), torch.float32)
            tok_embedding = torch.empty(shape, dtype=torch_dtype)

        t_pre_bcast = _time.time()
        dist.broadcast(tok_embedding, src=0)
        t_bcast = _time.time()
        if rank == 0:
            print(f"tok_embedding: broadcast {t_bcast-t_pre_bcast:.3f}s, total {t_bcast-t0:.3f}s", flush=True)
        return tok_embedding
