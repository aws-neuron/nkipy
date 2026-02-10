# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project

from typing import Any, Callable, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from vllm_nkipy.worker.nkipy_model_runner import NKIpyModelRunner

logger = init_logger(__name__)

_R = TypeVar("_R")


class NKIpyWorker(WorkerBase):
    """V1 Worker for NKIPy backend following vLLM v1 architecture."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        from vllm_nkipy import ops

        ops.register_dummy_fusion_op()

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.model_config.trust_remote_code:
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize KV cache with the given block counts.

        The num_gpu_blocks value comes from vLLM's KV cache configuration,
        which respects --num-gpu-blocks-override if set.
        """
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        logger.info(
            "Initializing cache: num_gpu_blocks=%d, num_cpu_blocks=%d",
            num_gpu_blocks,
            num_cpu_blocks,
        )

    def init_device(self):
        """Initialize the NKIPy device and distributed environment."""
        import os

        # Configure visible Neuron core for this worker BEFORE any NRT
        # initialization. spike_torch auto-initializes NRT on first import,
        # so NEURON_RT_VISIBLE_CORES must be set before that happens.
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(self.local_rank)
        logger.info(
            "Set NEURON_RT_VISIBLE_CORES=%s for rank=%d",
            self.local_rank,
            self.rank,
        )

        # Set up NKIPy device (no CUDA setup needed)
        # TODO: use nkipy as device
        self.device = torch.device("cpu")  # NKIPy uses CPU as base device

        logger.info("Initializing NKIPy device and distributed environment")

        # Initialize distributed environment for NKIPy
        self._init_nkipy_distributed_environment()

        # Set random seed
        set_random_seed(self.model_config.seed)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        torch._dynamo.config.cache_size_limit = 128

        # Construct the model runner
        self.model_runner: NKIpyModelRunner = NKIpyModelRunner(
            self.vllm_config, self.device
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        For NKIPy, the actual block count should be controlled via
        --num-gpu-blocks-override. This method returns a placeholder value
        that will be overridden by vLLM's KV cache configuration.

        Swapping is not supported, so num_cpu_blocks=0.
        """
        # This value will be overridden by --num-gpu-blocks-override
        # Return 0 as placeholder since vLLM v1 uses determine_available_memory()
        return 0, 0

    def determine_available_memory(self):
        """Return available memory for KV cache.

        For NKIPy, we return a large value to pass vLLM's memory checks.
        The actual num_gpu_blocks should be controlled via --num-gpu-blocks-override.
        """
        # Return a large value (1TB) so that:
        # 1. check_enough_kv_cache_memory() passes
        # 2. The calculated num_blocks will be large
        # 3. --num-gpu-blocks-override will cap it to the desired value
        return 1024 * 1024 * 1024 * 1024  # 1TB

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        with set_current_vllm_config(self.vllm_config):
            output = self.model_runner.execute_model(scheduler_output)
            # every worker's output is needed when kv_transfer_group is set up
            return output if self.is_driver_worker or has_kv_transfer_group() else None

    def execute_dummy_batch(self) -> None:
        """Execute a dummy batch for testing."""
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.execute_dummy_batch()

    def profile(self, is_start: bool = True):
        """Profile method for NKIPy worker."""
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")

        if is_start:
            logger.info("Starting NKIPy profiling")
            self.profiler.start()
        else:
            logger.info("Stopping NKIPy profiling")
            self.profiler.stop()
            # Display profiling results (adapted for CPU-only profiling)
            print(self.profiler.key_averages().table(sort_by="self_cpu_time_total"))

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add LoRA adapter (not supported in NKIPy)."""
        logger.warning("LoRA is not supported with NKIPy backend")
        return False

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model for NKIPy execution."""
        logger.info("Warming up NKIPy model...")

        with set_current_vllm_config(self.vllm_config):
            # Run profiling to warm up the model
            # self.model_runner.profile_run(self.model_runner.max_num_tokens)
            self.model_runner.capture_model()

            # Reset the seed to ensure that the random state is not affected by
            # the model initialization and profiling.
            set_random_seed(self.model_config.seed)

            logger.info("NKIPy model warm-up completed")

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache from the given configuration."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def shutdown(self) -> None:
        self.model_runner.ensure_kv_transfer_shutdown()

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

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

        ensure_kv_transfer_initialized(self.vllm_config)

        logger.info(
            "NKIPy distributed environment initialized: "
            "world_size=%d, rank=%d, tp_size=%d, pp_size=%d",
            parallel_config.world_size,
            self.rank,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )
