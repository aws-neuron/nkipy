# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Model Runner for vLLM integration.

Runs models on Neuron cores via NKIPy DeviceKernel/DeviceTensor.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    """Tracks per-request state across steps."""
    req_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


class NKIPyModelRunner:
    """Model runner bridging vLLM scheduler with NKIPy on Neuron."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device | None = None):
        self.vllm_config = vllm_config
        self.device = device or torch.device("neuron:0")
        self.model: Any = None
        self.hf_config = None
        self._nkipy_model = None

        # Exposed to worker for P2P state
        self._model_class = None
        self._config = None
        self._kernel_cache = None

        model_config = vllm_config.model_config
        self.vocab_size = model_config.hf_config.vocab_size
        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        self.requests: dict[str, _RequestState] = {}

    def load_model(self) -> None:
        model_name = self.vllm_config.model_config.model
        self.hf_config = AutoConfig.from_pretrained(model_name)
        self._load_model_neuron(model_name)

    def _load_model_neuron(self, model_name: str) -> None:
        """Load model using NKIPy on Neuron cores."""
        import torch.distributed as dist
        from safetensors.torch import load_file

        from .models.config import get_config

        logger.info("Loading model %s (Neuron mode)", model_name)

        config = get_config(model_name, self.max_model_len, self.max_model_len)

        # Select model class based on architecture (MoE vs dense)
        if config.num_experts is not None:
            from .models.qwen3 import Qwen3Model as ModelClass
            logger.info("Using Qwen3Model (MoE)")
        else:
            from .models.llama import Llama3Model as ModelClass
            logger.info("Using Llama3Model (dense)")

        self._model_class = ModelClass
        self._config = config

        checkpoint = os.environ.get("NKIPY_CHECKPOINT")
        if not checkpoint:
            # No checkpoint → start in sleep mode (P2P receiver).
            # Pre-compile kernels to NEFF so wake_up can reload them fast.
            logger.info("No NKIPY_CHECKPOINT set — starting in sleep mode")
            model = ModelClass(config=config, skip_kernels=True)
            self._kernel_cache = model._compile_kernels()
            self._nkipy_model = None
            self.model = None
            return

        shard_path = os.path.join(
            checkpoint, f"shard_{dist.get_rank()}.safetensors"
        )
        logger.info("Loading shard %s", shard_path)
        weights = load_file(shard_path, device="cpu")

        self._nkipy_model = ModelClass(weights, config, skip_kernels=True)
        self.model = self._nkipy_model
        logger.info("Model weights loaded, kernels will compile during warmup")

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        block_size = self.vllm_config.cache_config.block_size
        cfg = self.hf_config
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_size = cfg.hidden_size // cfg.num_attention_heads
        specs = {}
        for i in range(cfg.num_hidden_layers):
            specs[f"layer.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
            )
        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        logger.debug("KV cache initialized (managed internally)")

    def warmup_model(self) -> None:
        """Called after all workers are synchronized. Compile kernels eagerly."""
        if self._nkipy_model is not None and self._nkipy_model.kernel_cte is None:
            ctx_len = self._nkipy_model.config.context_len or self.max_model_len
            self._nkipy_model.config.context_len = ctx_len
            self._nkipy_model.config.max_new_tokens = self.max_model_len
            self._nkipy_model._prepare_kernels()

            try:
                from nkipy.p2p import preregister_weights
                preregister_weights(self._nkipy_model)
            except Exception:
                pass

            logger.info("NKIPy warmup complete (kernels compiled)")
        else:
            logger.info("NKIPy warmup complete (no compilation needed)")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        # Sleep mode — model not loaded yet
        if self._nkipy_model is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            self.requests[new_req.req_id] = _RequestState(
                req_id=new_req.req_id,
                prompt_token_ids=list(new_req.prompt_token_ids),
                num_computed_tokens=new_req.num_computed_tokens,
            )

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in self.requests:
                self.requests[req_id].num_computed_tokens = cached.num_computed_tokens[i]

        if scheduler_output.total_num_scheduled_tokens == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT

        scheduled_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_ids: list[str] = []
        sampled_token_ids: list[list[int]] = []

        for req_id in scheduled_ids:
            state = self.requests.get(req_id)
            if state is None:
                continue

            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            end = state.num_computed_tokens + num_scheduled
            is_prefill = state.num_computed_tokens == 0

            if is_prefill:
                prompt = state.prompt_token_ids[:end]
                ctx_len = len(prompt)
                input_ids = np.array([prompt], dtype=np.int64)

                # Compile kernels for this context length if needed
                if self._nkipy_model.kernel_cte is None or self._nkipy_model.config.context_len != ctx_len:
                    self._nkipy_model.config.context_len = ctx_len
                    self._nkipy_model.config.max_new_tokens = self.max_model_len
                    self._nkipy_model._prepare_kernels()

                    try:
                        from nkipy.p2p import preregister_weights
                        preregister_weights(self._nkipy_model)
                    except Exception:
                        pass

                gen = self._nkipy_model.generate(input_ids)
                next_id_tensor = next(gen)
                next_token = next_id_tensor[0, 0].item()
                state._generator = gen
            else:
                gen = getattr(state, "_generator", None)
                if gen is None:
                    continue
                try:
                    next_id_tensor = next(gen)
                    next_token = next_id_tensor[0, 0].item()
                except StopIteration:
                    continue

            state.output_token_ids.append(next_token)
            state.num_computed_tokens = end
            req_ids.append(req_id)
            sampled_token_ids.append([next_token])

        if not req_ids:
            return EMPTY_MODEL_RUNNER_OUTPUT
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
        )
