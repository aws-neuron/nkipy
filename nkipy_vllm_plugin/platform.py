# SPDX-License-Identifier: Apache-2.0
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    VllmConfig = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    supported_quantization: list[str] = []

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "nkipy"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        disable_scheduler_override = bool(
            int(os.getenv("DISABLE_NEURON_CUSTOM_SCHEDULER", "0"))
        )

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "nkipy_vllm_plugin.worker.neuron_worker.NeuronWorker"
            )

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "mp"

        if disable_scheduler_override:
            logger.warning(
                "The vLLM V1 native scheduler will be used with chunked prefill enabled. "
                "This may lead to suboptimal performance on Neuron devices."
            )
            assert (
                vllm_config.cache_config.block_size is not None
            ), "When vLLM V1 native scheduler is enabled, block_size must be set."
        else:
            logger.info(
                "The custom Neuron scheduler will disable chunked prefill and schedule requests using "
                "the continuous batching mechanism, prioritizing prefill over decode."
            )
            vllm_config.scheduler_config.scheduler_cls = (
                "nkipy_vllm_plugin.core.scheduler.ContinuousBatchingNeuronScheduler"
            )
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_seqs = 1

            if not vllm_config.cache_config.enable_prefix_caching:
                # Neuron requires block_size = max_model_len when blockwise KV cache is disabled
                vllm_config.cache_config.block_size = (
                    vllm_config.model_config.max_model_len  # type: ignore
                )
            else:
                assert (
                    vllm_config.cache_config.block_size is not None
                ), "When prefix caching is enabled, block_size must be set."

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "nkipy_vllm_plugin.distributed.nkipy_communicator.NKIPyCommunicator"  # noqa
        )

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()
