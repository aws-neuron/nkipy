# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Worker for vLLM integration."""

import logging
import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)


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

    def init_device(self) -> None:
        import torch_neuronx  # noqa: F401 — registers neuron Runtime class

        core_offset = int(os.environ.get("NKIPY_CORE_OFFSET", "0"))
        os.environ.setdefault(
            "NEURON_RT_VISIBLE_CORES", str(self.local_rank + core_offset)
        )
        runtime = torch.classes.neuron.Runtime()
        runtime.initialize()
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
            import torch.distributed as dist
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
