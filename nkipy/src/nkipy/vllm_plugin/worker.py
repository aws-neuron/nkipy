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
        import torch_neuronx  # noqa: F401 — registers neuron Runtime class

        core_offset = int(os.environ.get("NKIPY_CORE_OFFSET", "0"))
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(self.local_rank + core_offset)

        # Only claim neuron cores if we have weights to load.
        # Sleep-mode engines defer core init to wake_up via get_spike_singleton().
        if os.environ.get("NKIPY_CHECKPOINT"):
            from spike import get_spike_singleton
            get_spike_singleton()

        self.device = torch.device(f"neuron:{self.local_rank}")

        self._init_distributed()
        self._patch_get_accelerator()

        from .model_runner import NKIPyModelRunner

        self.model_runner = NKIPyModelRunner(
            vllm_config=self.vllm_config, device=self.device
        )
        set_random_seed(self.model_config.seed)
        logger.info("NKIPyWorker initialized: rank=%s device=%s", self.rank, self.device)

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
        if self._sleeping:
            return {"status": "already_sleeping"}

        from spike import get_spike_singleton, reset as spike_reset
        from nkipy.runtime.device_kernel import _LOADED_KERNELS
        from nkipy.p2p import rank_endpoint

        model = self.model_runner._nkipy_model

        # Cache kernel NEFFs
        rank_endpoint.dereg_async()
        self._kernel_cache = {}
        for name in _KERNEL_NAMES:
            kernel = getattr(model, name, None)
            if kernel is not None:
                self._kernel_cache[name] = (kernel.neff_path, kernel.cache_key)

        # Free device tensors and unload kernels
        spike = get_spike_singleton()
        for layer in model.layer_tensors:
            for t in layer.values():
                spike.free_tensor(t.tensor_ref)
        for t in (model.norm_weight, model.lm_head_weight):
            if t is not None:
                spike.free_tensor(t.tensor_ref)
        for name in _KERNEL_NAMES:
            kernel = getattr(model, name, None)
            if kernel is not None:
                spike.unload_model(kernel.model_ref)

        _LOADED_KERNELS.clear()
        gc.collect()
        spike_reset()

        self.model_runner._nkipy_model = None
        self.model_runner.model = None
        self._sleeping = True
        return {"status": "sleeping"}

    def nkipy_wake_up(self, peer_url: str | None = None) -> dict:
        """Allocate tensors, receive weights from peer, reload kernels."""
        if not self._sleeping:
            return {"status": "already_awake"}

        from spike import get_spike_singleton
        from nkipy.runtime import DeviceKernel

        # Broadcast peer_url from rank 0 to all ranks
        obj_list = [peer_url]
        dist.broadcast_object_list(obj_list, src=0)
        actual_peer = obj_list[0]

        dist.barrier()
        get_spike_singleton()
        dist.barrier()

        # Rebuild model with empty tensors
        model = self._model_class(config=self._config, skip_kernels=True)
        model._allocate_empty_tensors()

        if actual_peer:
            from nkipy.p2p import (
                receive_from_peer, rank_endpoint, collect_weight_buffers,
            )
            bufs = collect_weight_buffers(model)
            receive_from_peer(
                rank_endpoint, bufs, actual_peer,
                push_endpoint="/nkipy/p2p_push_weights",
            )

        # Reload kernels from cached NEFFs
        for name, (neff_path, cache_key) in self._kernel_cache.items():
            setattr(model, name, DeviceKernel.load_with_cache_key(
                neff_path, cache_key, name=name,
            ))
        dist.barrier()

        if model.tok_embedding is None and actual_peer:
            model.tok_embedding = self._fetch_tok_embedding(actual_peer)

        self.model_runner._nkipy_model = model
        self.model_runner.model = model
        self._sleeping = False
        return {"status": "awake"}

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
        import requests as _req
        if dist.get_rank() == 0:
            base = peer_url.rstrip("/")
            resp = _req.get(f"{base}/nkipy/tok_embedding")
            resp.raise_for_status()
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
        else:
            tok_embedding = None
        obj_list = [tok_embedding]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]
