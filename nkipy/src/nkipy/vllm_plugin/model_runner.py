# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Model Runner for vLLM integration with continuous batching.

Processes multiple concurrent requests by interleaving prefill and decode
steps. Each execute_model() call advances one request by one step.
Prefill takes priority over decode to minimize time-to-first-token.
"""

import logging
import os
from collections import OrderedDict
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
    is_prefill_done: bool = False


class NKIPyModelRunner:
    """Model runner bridging vLLM scheduler with NKIPy on Neuron."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device | None = None):
        self.vllm_config = vllm_config
        self.device = device or torch.device("neuron:0")
        self.model: Any = None
        self.hf_config = None
        self._nkipy_model = None

        self._model_class = None
        self._config = None
        self._kernel_cache = None

        model_config = vllm_config.model_config
        self.vocab_size = model_config.hf_config.vocab_size
        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Active requests, insertion-ordered for round-robin decode
        self.requests: OrderedDict[str, _RequestState] = OrderedDict()
        # Track which request currently owns the KV cache
        self._active_req_id: str | None = None

    def load_model(self) -> None:
        model_name = self.vllm_config.model_config.model
        self.hf_config = AutoConfig.from_pretrained(model_name)
        self._load_model_neuron(model_name)

    def _load_model_neuron(self, model_name: str) -> None:
        import torch.distributed as dist
        from safetensors.torch import load_file
        from .models.config import get_config

        logger.info("Loading model %s (Neuron mode)", model_name)
        config = get_config(model_name, self.max_model_len, self.max_model_len)

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
            logger.info("No NKIPY_CHECKPOINT set — starting in sleep mode")
            model = ModelClass(config=config, skip_kernels=True)
            self._kernel_cache = model._compile_kernels()
            self._nkipy_model = None
            self.model = None
            return

        shard_path = os.path.join(checkpoint, f"shard_{dist.get_rank()}.safetensors")
        logger.info("Loading shard %s", shard_path)
        weights = load_file(shard_path, device="cpu")

        self._nkipy_model = ModelClass(weights, config, skip_kernels=True)
        self.model = self._nkipy_model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        block_size = self.vllm_config.cache_config.block_size
        cfg = self.hf_config
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_size = cfg.hidden_size // cfg.num_attention_heads
        return {
            f"layer.{i}": FullAttentionSpec(
                block_size=block_size, num_kv_heads=num_kv_heads,
                head_size=head_size, dtype=torch.bfloat16,
            )
            for i in range(cfg.num_hidden_layers)
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def warmup_model(self) -> None:
        if self._nkipy_model is not None and self._nkipy_model.kernel_cte is None:
            ctx_len = self._nkipy_model.config.context_len or self.max_model_len
            self._nkipy_model.config.context_len = ctx_len
            self._nkipy_model.config.max_new_tokens = self.max_model_len
            self._nkipy_model._prepare_kernels()
            logger.info("NKIPy warmup complete (kernels compiled)")
        else:
            logger.info("NKIPy warmup complete (no compilation needed)")

    def _ensure_kernels(self, ctx_len: int) -> None:
        model = self._nkipy_model
        if model.kernel_cte is None or model.config.context_len != ctx_len:
            model.config.context_len = ctx_len
            model.config.max_new_tokens = self.max_model_len
            model._prepare_kernels()

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        if self._nkipy_model is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # --- Update request tracking ---
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            if self._active_req_id == req_id:
                self._active_req_id = None

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

        # --- Pick which request to advance ---
        # Priority: prefill new requests first, then continue active decode
        scheduled_ids = set(scheduler_output.num_scheduled_tokens.keys())
        target_state = None

        # 1) Prefill: find a scheduled request that hasn't been prefilled
        for req_id in self.requests:
            if req_id in scheduled_ids and not self.requests[req_id].is_prefill_done:
                target_state = self.requests[req_id]
                break

        # 2) Decode: continue the active request, or pick next scheduled one
        if target_state is None:
            if self._active_req_id and self._active_req_id in scheduled_ids:
                target_state = self.requests.get(self._active_req_id)
            if target_state is None:
                for req_id in self.requests:
                    if req_id in scheduled_ids and self.requests[req_id].is_prefill_done:
                        target_state = self.requests[req_id]
                        break

        if target_state is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # --- Execute one step ---
        if not target_state.is_prefill_done:
            token = self._run_prefill(target_state)
            target_state.is_prefill_done = True
            self._active_req_id = target_state.req_id
        else:
            token = self._run_decode(target_state)
            self._active_req_id = target_state.req_id

        if token is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        return ModelRunnerOutput(
            req_ids=[target_state.req_id],
            req_id_to_index={target_state.req_id: 0},
            sampled_token_ids=[[token]],
        )

    def _run_prefill(self, state: _RequestState) -> int | None:
        from nkipy.runtime import DeviceTensor

        model = self._nkipy_model
        cfg = model.config
        ctx_len = len(state.prompt_token_ids)
        self._ensure_kernels(ctx_len)

        input_ids = np.array([state.prompt_token_ids], dtype=np.int64)
        hidden = DeviceTensor.from_torch(model.embed_tokens(input_ids), "hidden_states")
        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        for i in range(cfg.num_layers):
            model.kernel_cte(
                inputs=model._build_inputs(i, hidden),
                outputs=model._build_outputs(i, hidden),
            )
        model.kernel_cte_greedy_sampling(
            inputs={"h": hidden, "norm_weight": model.norm_weight,
                    "lm_head_weight": model.lm_head_weight},
            outputs={"output0": next_id},
        )

        token = int(next_id.torch().reshape(-1)[0].item())
        state.output_token_ids.append(token)
        state.num_computed_tokens = ctx_len
        return token

    def _run_decode(self, state: _RequestState) -> int | None:
        from nkipy.runtime import DeviceTensor

        model = self._nkipy_model
        cfg = model.config
        pos = state.num_computed_tokens + len(state.output_token_ids) - 1
        if pos >= cfg.max_seq_len:
            return None

        last_token = state.output_token_ids[-1]
        input_ids = np.array([[last_token]], dtype=np.int64)
        t_sp = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
        hidden = DeviceTensor.from_torch(model.embed_tokens(input_ids), "h0/res1")
        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        for i in range(cfg.num_layers):
            model.kernel_tkg(
                inputs=model._build_inputs(i, hidden, t_sp),
                outputs=model._build_outputs(i, hidden),
            )
        model.kernel_tkg_greedy_sampling(
            inputs={"h": hidden, "norm_weight": model.norm_weight,
                    "lm_head_weight": model.lm_head_weight},
            outputs={"output0": next_id},
        )

        token = int(next_id.torch().reshape(-1)[0].item())
        state.output_token_ids.append(token)
        return token
