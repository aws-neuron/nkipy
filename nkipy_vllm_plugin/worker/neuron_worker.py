# SPDX-License-Identifier: Apache-2.0
"""A Neuron worker class."""
import os
import subprocess
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from nkipy.core.compile import _set_build_dir
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from config import get_config, set_env
from kernels.blockwise_index import (
    add_blockwise_index_to_path,
    build_blockwise_index_cpp,
)
from nkipy_vllm_plugin.worker.nkipy_model_runner import NKIPyModelRunner
from parallel_state import get_ep_size, initialize_model_parallel

logger = init_logger(__name__)


class NeuronWorker(WorkerBase):
    """A worker class that executes the model on a neuron core."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

    def init_device(self) -> None:
        """Initialize the NKIPy device and distributed environment."""
        # Set up NKIPy device (no CUDA setup needed)
        # TODO: use nkipy as device
        self.device = torch.device("cpu")  # NKIPy uses CPU as base device

        os.environ["NEURON_RT_VISIBLE_CORES"] = str(self.rank)

        self._init_nkipy_distributed_environment()

        # TODO: currently only support single core test
        debug_rank = os.environ.get("NEURONPY_DEBUG_RANK")
        if debug_rank and int(debug_rank) == dist.get_rank():
            import signal

            import debugpy

            def signal_handler(sig, frame):
                print(
                    f"Rank {dist.get_rank()}: Caught SIGTERM, ignoring to continue debugging"
                )

            signal.signal(signal.SIGTERM, signal_handler)
            debugpy.listen(5678)
            print(f"[Rank {dist.get_rank()}] Listening for debugpy")
            # debugpy.wait_for_client()

        # Set random seed.
        set_random_seed(self.model_config.seed)
        self.model_runner: NKIPyModelRunner = NKIPyModelRunner(
            vllm_config=self.vllm_config, device=self.device
        )
        logger.debug(
            "NKIPy rank=%d finished init device",
            self.rank,
        )

    def _init_nkipy_distributed_environment(self) -> None:
        """Initialize the distributed environment for NKIPy."""
        parallel_config = self.vllm_config.parallel_config

        # Initialize distributed environment with gloo backend for NKIPy
        init_distributed_environment(
            parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "gloo",
        )

        # Initialize model parallel groups
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
        )

        logger.debug(
            "NKIPy distributed environment initialized: "
            "world_size=%d, rank=%d, tp_size=%d, pp_size=%d",
            parallel_config.world_size,
            self.rank,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )

    def load_model(self):
        model_name = os.environ.get("MODEL_NAME", None)
        checkpoint = os.environ.get("MODEL_CHECKPOINT", None)
        assert model_name
        assert checkpoint
        tp_str = checkpoint.split("-")[-1]
        assert tp_str[:2] == "TP"
        tp_size = int(tp_str[2:])
        initialize_model_parallel(tp_size=tp_size)

        # XXX: set LOCAL_RANK for nkipy model logger
        if os.environ.get("LOCAL_RANK", None) is None:
            os.environ["LOCAL_RANK"] = str(self.rank)

        ep_size = get_ep_size()
        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        max_model_len = self.vllm_config.model_config.max_model_len
        build_dir = f"/tmp/build/gpt-oss-120b-EP{ep_size}-TP{tp_size}-BS{max_batch_size}-SEQ{max_model_len}"
        _set_build_dir(build_dir)
        set_env()
        model_config = get_config(
            model_name,
            max_batch_size=max_batch_size,
            max_model_len=max_model_len,
        )
        if dist.get_rank() == 0:
            if os.environ.get("NEURONPY_NOT_CLEAR_BUILD_CACHE") != "1":
                subprocess.run(f"rm -rf {build_dir}", shell=True)
            build_blockwise_index_cpp(
                n_experts=model_config.num_experts,
                top_k=model_config.num_experts_per_tok,
                n_blocks=model_config.num_blocks,
                n_static_blocks=model_config.num_static_blocks,
            )
        dist.barrier()  # wait for build
        add_blockwise_index_to_path()
        self.model_runner.load_model(model_config, checkpoint)

    def determine_available_memory(self):
        # TODO: implement this
        return 1024 * 1024 * 1024  # 1GB

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        return None

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def check_health(self) -> None:
        """Check the health of the NKIPy worker."""
        # Basic health check - ensure model is loaded
        if self.model_runner.model is None:
            raise RuntimeError("NKIPy model is not loaded")
        # worker will always be healthy as long as it's running.

    # vLLM uses add_adapter() to add LoRA, rather than add_lora()
    def add_adapter(self, lora_request: LoRARequest) -> bool:
        # Todo: support dynamic add/remove lora
        # return
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # return self.model_runner.add_lora(lora_request)
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        # return self.model_runner.remove_lora(lora_id)
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        # return self.model_runner.pin_lora(lora_id)
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        # return self.model_runner.list_loras()
        raise NotImplementedError
