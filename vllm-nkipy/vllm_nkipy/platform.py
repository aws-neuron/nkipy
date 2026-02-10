# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project

from typing import TYPE_CHECKING, Optional

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum, _Backend

from vllm_nkipy.config import get_nkipy_config

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)

# Import torch_to_nkipy.backend.nkipy_backend to register the "nkipy"
# torch.compile backend. This is needed before vLLM's module-level
# @torch.compile(backend="nkipy") decorators are evaluated at import time.
# Safe to import here because device/__init__.py no longer eagerly
# initializes NRT (the register_torch_device() call was removed).
try:
    import torch_to_nkipy.backend.nkipy_backend  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import torch_to_nkipy with %r", e)


class NKIpyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"  # TODO: device
    device_type: str = "cpu"
    dist_backend: str = "gloo"
    simple_compile_backend: str = "nkipy"
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    supported_quantization: list[str] = []

    # Automatically sync with environment variables defined in envs.py
    @property
    def additional_env_vars(self) -> list[str]:
        import vllm_nkipy.envs as envs

        return list(envs.environment_variables.keys())

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
    ) -> str:
        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported.")
        backend_cls = "vllm_nkipy.attention.backends.neuron_attn.NeuronAttentionBackend"
        return backend_cls

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "nkipy"

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on NKIPy.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_nkipy.distributed.nkipy_communicator.NKIpyCommunicator"  # noqa

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel  # noqa: E402

        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        # Parse NKIPy-specific config from additional_config
        nkipy_config = get_nkipy_config(vllm_config)

        # cache config - only set NKIPy defaults if not specified by user
        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 32  # NKIPy default
                logger.info(f"Using NKIPy default block_size={cache_config.block_size}")
            if cache_config.num_gpu_blocks_override is None:
                cache_config.num_gpu_blocks_override = 512  # NKIPy default
                logger.info(
                    "Using NKIPy default "
                    f"num_gpu_blocks_override={cache_config.num_gpu_blocks_override}"
                )

        # compilation config
        compilation_config.level = CompilationLevel.NO_COMPILATION
        logger.info("Enable all custom ops.")
        compilation_config.custom_ops = ["all"]

        # model config
        # Check and update model config
        model_config = vllm_config.model_config
        if model_config is not None and not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on NKIPy backend, "
                "fallback to the eager mode."
            )
            model_config.enforce_eager = True

        # parallel config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_nkipy.worker.nkipy_worker.NKIpyWorker"
        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "mp"

        # scheduler config - derive max_num_batched_tokens from num_tokens_paddings
        max_batched_tokens = nkipy_config.get_max_num_batched_tokens()
        if max_batched_tokens is not None:
            logger.info(
                f"Setting max_num_batched_tokens={max_batched_tokens} "
                f"(max(num_tokens_paddings)={max(nkipy_config.num_tokens_paddings)})"
            )
            scheduler_config.max_num_batched_tokens = max_batched_tokens
        elif scheduler_config.max_num_batched_tokens is None:
            # Fallback default for NKIPy
            scheduler_config.max_num_batched_tokens = 2048
            logger.info(
                "Using NKIPy default "
                f"max_num_batched_tokens={scheduler_config.max_num_batched_tokens}"
            )

    @classmethod
    def is_kv_cache_dtype_supported(
        cls, kv_cache_dtype: str, model_config: "ModelConfig"
    ) -> bool:
        return True

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        return True

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        return {"cpu": ("cpu",)}

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        assert src_cache.device.type == "cpu"
        dst_cache[:, dst_block_indices, ...] = src_cache[:, src_block_indices, ...].to(
            dst_cache.device
        )

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        assert dst_cache.device.type == "cpu"
        dst_cache[:, dst_block_indices, ...] = src_cache[:, src_block_indices, ...].to(
            "cpu"
        )
