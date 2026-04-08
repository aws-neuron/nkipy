# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Platform implementation for vLLM."""

import logging
import os
from typing import TYPE_CHECKING

import torch
from vllm.platforms import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Register "neuron" as a valid torch device name via PrivateUse1.
# Importing torch_neuronx registers the PrivateUse1HooksInterface which is
# required by torch.distributed.barrier() and other PyTorch internals.
# This is the same approach used by vllm-neuron.
try:
    import torch_neuronx  # noqa: F401 — registers neuron backend
except ImportError:
    pass
torch.utils.rename_privateuse1_backend("neuron")


class NKIPyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = []
    device_control_env_var: str = "NEURON_VISIBLE_DEVICES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"neuron:{device_id}"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "nkipy.vllm_plugin.worker.NKIPyWorker"
            )
        cache_config = vllm_config.cache_config
        if cache_config.block_size is None:
            cache_config.block_size = 16

        if parallel_config.distributed_executor_backend not in ("mp", "uni"):
            parallel_config.distributed_executor_backend = "mp"

        # Disable async scheduling — our model runner returns output synchronously
        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.async_scheduling:
            scheduler_config.async_scheduling = False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        return "nkipy.vllm_plugin.attention.NKIPyAttentionBackend"
