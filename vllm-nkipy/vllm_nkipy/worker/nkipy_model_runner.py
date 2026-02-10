# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/tpu_model_runner.py
import bisect
import gc
import os
import time
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.config import (
    ParallelConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargs,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    LayerBlockType,
    cdiv,
    is_pin_memory_available,
)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.tpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import bind_kv_cache

import vllm_nkipy  # noqa: E402
import vllm_nkipy.envs as nkipy_envs  # noqa: E402
import vllm_nkipy.ops.moe.simple_moe  # noqa: F401, E402
from vllm_nkipy.attention.backends.neuron_attn import (  # noqa: E402
    NeuronAttentionBackend,
    NeuronAttentionMetadata,
)
from vllm_nkipy.attention.ops.nki_flash_attn_legacy import (  # noqa: E402
    reorder_context_mask,
)
from vllm_nkipy.compile import (  # noqa: E402
    NKIPY_BACKEND,
    local_compile,
    set_default_split_graph,
)
from vllm_nkipy.config import (  # noqa: E402
    CompileStrategy,
    PagedAttnImpl,
    _get_paged_attn_impl,
    get_nkipy_config,
    set_global_nkipy_config,
)
from vllm_nkipy.sample.metadata import NKIPySamplingMetadata  # noqa: E402
from vllm_nkipy.sample.ops import custom_argmax  # noqa: E402
from vllm_nkipy.sample.sampler import Sampler  # noqa: E402
from vllm_nkipy.utils import install_layer_barriers  # noqa: E402

# from .utils import (
#     MultiModalBudget,
#     add_kv_sharing_layers_to_kv_cache_groups,
#     sanity_check_mm_encoder_outputs
# )


def round_up(x, y):
    """Round x up to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def is_master():
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    return rank == 0


def get_attention_backend_class():
    return NeuronAttentionBackend, NeuronAttentionMetadata


logger = init_logger(__name__)

torch._dynamo.config.ignore_logger_methods.add(
    vllm_nkipy.ops.moe.simple_moe.logger.info
)
torch._dynamo.config.ignore_logger_methods.add(print)

_PAD_SLOT_ID = 0
INVALID_TOKEN_ID = -1


#########################################################
# model hook at torch.compile
@local_compile(
    backend=NKIPY_BACKEND,
    fullgraph=True,
    dynamic=False,
    options={
        "save_ntff": bool(nkipy_envs.VLLM_NKIPY_COMPILE_SAVE_NTFF),
        "save_ntff_dir": nkipy_envs.VLLM_NKIPY_COMPILE_SAVE_NTFF_DIR,
        "save_ntff_exe_idx": nkipy_envs.VLLM_NKIPY_COMPILE_SAVE_NTFF_EXE_IDX,
    },
)
def forward_tkg_fused(
    self, x: torch.Tensor, postitions: torch.Tensor, layers
) -> torch.Tensor:
    for layer in layers:
        x = layer(
            x,
            postitions,
        )
    return x


def make_forward_dispatch(merge_step: int):
    """Factory function to create forward_dispatch with captured merge_step."""

    def forward_dispatch(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        i_step = merge_step
        (S,) = input_ids.shape

        x = self.embedding(input_ids)
        if S < 128:
            for i in range(0, len(self.layers), i_step):
                i_end = min(i + i_step, len(self.layers))
                if is_master():
                    logger.info("[DEBUG] forward_tkg_fused")
                x = forward_tkg_fused(
                    self,
                    x,
                    positions,
                    self.layers[i:i_end],
                )
        else:
            if is_master():
                logger.info("[DEBUG] S >= 128")
            for layer in self.layers:
                x = layer(x, positions)

        x = self.norm(x)
        return x

    return forward_dispatch


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selecting hidden states for each request
# 3. sampler
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class NKIpyModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: Optional[ParallelConfig] = None,
    ):
        logger.info(f"{vllm_config=}")
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.original_parallel_config = original_parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = "nkipy"

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        self.sliding_window = model_config.get_sliding_window()
        # Per-layer sliding window (may be None for full-attn layers)
        self.layer_sliding_windows: dict[str, Optional[int]] = {}

        self.is_multimodal_model = model_config.is_multimodal_model
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)

        nkipy_config = get_nkipy_config(vllm_config)
        self.min_num_seqs = nkipy_config.min_num_seqs
        # NKI blocksparse attention tile sizes (from additional_config)
        self.large_kv_tile_size = nkipy_config.large_kv_tile_size
        self.large_q_tile_size = nkipy_config.large_q_tile_size
        self.dynamic_loop_unroll = nkipy_config.dynamic_loop_unroll

        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, self.min_num_seqs)
        if nkipy_config.num_tokens_paddings is not None:
            # Use config value (from --additional-config)
            self.num_tokens_paddings = nkipy_config.num_tokens_paddings
            logger.info(
                f"Using num_tokens_paddings from config: {self.num_tokens_paddings}"
            )
        else:
            # Generate default paddings based on scheduler config
            self.num_tokens_paddings = _get_token_paddings(
                min_token_size=128,
                max_token_size=scheduler_config.max_num_batched_tokens,
                padding_gap=0,
            )
            logger.info(
                f"Using generated num_tokens_paddings: {self.num_tokens_paddings}"
            )

        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention
        )
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()

        if self.lora_config is not None:
            self.vocab_size += self.lora_config.lora_extra_vocab_size

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Initialize input batch early to avoid AttributeError in _update_states
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device="cpu",
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
        )

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        # TODO: use CpuGpuBuffer
        self.input_ids_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )

        self.positions_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )
        self.positions_np = self.positions_cpu.numpy()

        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
        )

        self.query_start_loc_cpu = torch.zeros(
            self.max_num_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=self.min_num_seqs, max_req_size=self.max_num_reqs
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory
        )

        # Get maximum number of mm items per modality (batch size).
        self.max_num_mm_items_by_modality = dict()
        if (
            self.is_multimodal_model
            and self.max_num_encoder_input_tokens > 0
            and self.encoder_cache_size > 0
        ):
            max_tokens_by_modality_dict = (
                MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(
                    self.model_config
                )
            )
            for modality, max_tokens in max_tokens_by_modality_dict.items():
                # Check how many items of this modality can be supported by
                # the encoder budget.
                encoder_budget = min(
                    self.max_num_encoder_input_tokens, self.encoder_cache_size
                )

                max_num_mm_items_encoder_budget = cdiv(encoder_budget, max_tokens)

                # Check how many items of this modality can be supported by
                # the decoder budget.
                max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
                    self.model_config
                )[modality]

                # NOTE: We do not consider max_num_batched_tokens on purpose
                # because the multimodal embeddings can be generated in advance
                # and chunked prefilled.
                max_num_mm_items_decoder_budget = (
                    self.max_num_reqs * max_mm_items_per_req
                )

                max_num_mm_items = min(
                    max_num_mm_items_encoder_budget, max_num_mm_items_decoder_budget
                )
                self.max_num_mm_items_by_modality[modality] = max_num_mm_items

        self._nkipy_backend_initialized = False

    def _init_nkipy_backend(self) -> None:
        """Initialize NKIPy backend if not already initialized."""
        if self._nkipy_backend_initialized:
            return

        try:
            from torch_to_nkipy.backend import init_nkipy_backend
            from vllm_nkipy.distributed.parallel_state import init_world_group

            world_size = 1
            rank = 0
            local_rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                # Determine local rank (rank within the node)
                # For simplicity, use rank as local_rank if not set
                local_rank = int(os.environ.get("LOCAL_RANK", rank))

            # Initialize world group with device communicator
            if torch.distributed.is_initialized():
                backend = torch.distributed.get_backend()
                ranks = list(range(world_size))
                init_world_group(
                    ranks=ranks,
                    local_rank=local_rank,
                    backend=backend,
                    use_device_communicator=True,
                )
                logger.info(f"Initialized world group with {world_size} ranks")

            # init debugpy
            if nkipy_envs.VLLM_NKIPY_DEBUGPY_RANK and int(
                nkipy_envs.VLLM_NKIPY_DEBUGPY_RANK
            ) == int(rank):
                import signal

                import debugpy

                def signal_handler(_, __):
                    print(
                        f"Rank {rank}: Caught SIGTERM, ignoring to continue debugging"
                    )

                signal.signal(signal.SIGTERM, signal_handler)
                debugpy.listen(5678)
                print(f"[Rank {rank}] Listening for debugpy")
                # debugpy.wait_for_client()

            # TODO: use "rank_x"
            cache_prefix = nkipy_envs.VLLM_NKIPY_CACHE_DIR
            if cache_prefix is None:
                cache_dir = f"./nkipy_cache_{rank}"
            else:
                cache_dir = f"{cache_prefix}_{rank}"

            init_nkipy_backend(
                nkipy_cache=cache_dir,
                # log_level=logging.DEBUG,
                rank=rank,
                world_size=world_size,
                additional_compiler_args=(
                "-O3 --model-type=transformer --lnc=1"
            ),
            )

            # Set default NKI experimental flags and opt_level for all custom NKI ops
            from torch_to_nkipy.utils.nki import NKIOpRegistry

            NKIOpRegistry.set_default_compiler_args(
                "experimental-native-scalar-support, "
                "experimental-local-tensor-parent"
            )
            NKIOpRegistry.set_default_opt_level(
                "OptLevel.skip_middle_end"
            )

            self._nkipy_backend_initialized = True
            logger.info("NKIPy backend initialized successfully")
        except ImportError as e:
            logger.error("Failed to import torch_to_nkipy: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to initialize NKIPy backend: %s", e)
            raise

    def _update_num_xla_graphs(self, case_str):
        # check_comp = self.check_recompilation and not self.enforce_eager
        # if not check_comp:
        #     return

        # total_cached_graphs = xr.get_num_cached_compilation_graph()
        # new_compiled_graphs = total_cached_graphs - self.num_xla_graphs
        # if new_compiled_graphs == 0:
        #     return

        # logger.info("Add new %d compiled XLA graphs due to %s",
        #             new_compiled_graphs, case_str)
        self.num_xla_graphs += 1

    def _verify_num_xla_graphs(self, case_str):
        """Verify no new XLA graphs are compiled during execution."""
        # For NKIPy, we don't need strict XLA graph checking
        # This is a placeholder to maintain compatibility
        pass

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None, (
                "Pooling is not supported yet."
            )
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        model = getattr(model, "_orig_mod", model)
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        model = getattr(model, "_orig_mod", model)
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            if attn_module.attn_type == AttentionType.DECODER:
                if isinstance(attn_module, ChunkedLocalAttention):
                    logger.warning_once(
                        "Using irope in is not supported yet, it will "
                        "fall back to global attention for long context."
                    )
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                    )
            elif attn_module.attn_type in (
                AttentionType.ENCODER,
                AttentionType.ENCODER_ONLY,
            ):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        dp_max_tokens: int,
        dp_max_reqs: int,
        enable_prefill_global: bool,
    ):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(
                max_num_scheduled_tokens_all_reqs, num_tokens
            )
        num_scheduled_tokens_per_req = np.array(
            num_scheduled_tokens_per_req, dtype=np.int32
        )
        assert max_num_scheduled_tokens_all_reqs > 0

        enable_prefill = enable_prefill_global

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req]
        )

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.input_ids_cpu[:total_num_scheduled_tokens],
        )

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        # req_indices: # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        block_table_indices = (
            req_indices * self.max_num_blocks_per_req + positions_np // self.block_size
        )
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(
            block_numbers * self.block_size,
            block_offsets,
            out=self.input_batch.block_table[0].slot_mapping.np[
                :total_num_scheduled_tokens
            ],
        )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(
            num_scheduled_tokens_per_req, out=self.query_start_loc_np[1 : num_reqs + 1]
        )
        self.query_start_loc_np[num_reqs + 1 :] = self.query_start_loc_np[num_reqs]

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens_per_req
        )

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, dp_max_tokens
        )

        if enable_prefill:
            padded_total_num_scheduled_tokens = self.num_tokens_paddings[-1]

        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens
        ] = 0
        # Also zero out positions
        self.positions_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens
        ] = 0
        self.input_ids = self.input_ids_cpu[:padded_total_num_scheduled_tokens].to(
            self.device
        )
        self.position_ids = self.positions_cpu[:padded_total_num_scheduled_tokens].to(
            self.device
        )
        self.input_batch.block_table[0].slot_mapping.cpu[
            total_num_scheduled_tokens:
        ] = _PAD_SLOT_ID
        slot_mapping = self.input_batch.block_table[0].slot_mapping.cpu[
            :padded_total_num_scheduled_tokens
        ]

        block_tables = self.block_table_cpu[: self.max_num_reqs]
        block_tables[:num_reqs, : self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs]
        )

        query_start_loc = self.query_start_loc_cpu[: self.max_num_reqs + 1]
        seq_lens = self.seq_lens_cpu[: self.max_num_reqs]

        attn_mask, active_block_table = None, None
        paged_attn_impl = _get_paged_attn_impl()
        if paged_attn_impl.value.endswith("masked"):
            query_lens = torch.diff(query_start_loc)
            context_lens = seq_lens - query_lens
            num_active_blocks_shifted = shift_bit_length(
                ((context_lens + self.block_size - 1) // self.block_size).sum().item()
            )
            assert num_active_blocks_shifted <= 512, num_active_blocks_shifted
            context_cap = self.sliding_window or self.max_model_len
            num_active_blocks = get_num_active_blocks(
                self.block_size, self.num_blocks, self.max_num_reqs, context_cap
            )

            context_kv_len = num_active_blocks * self.block_size

            active_block_table = get_active_block_tables(
                block_tables,
                query_lens,
                seq_lens,
                self.block_size,
                num_active_blocks,
            )

            prior_mask, active_mask = (
                BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                    query_lens=query_lens.tolist(),
                    seq_lens=seq_lens.tolist(),
                    block_size=self.block_size,
                )
            )
            attn_mask = torch.concat(
                [
                    nn.functional.pad(
                        prior_mask,
                        (
                            0,
                            context_kv_len - prior_mask.shape[1],
                            0,
                            padded_total_num_scheduled_tokens - prior_mask.shape[0],
                        ),
                        "constant",
                        0,
                    ).bool(),
                    nn.functional.pad(
                        active_mask,
                        (
                            0,
                            padded_total_num_scheduled_tokens - active_mask.shape[1],
                            0,
                            padded_total_num_scheduled_tokens - active_mask.shape[0],
                        ),
                        "constant",
                        0,
                    ).bool(),
                ],
                dim=1,
            )
            if paged_attn_impl == PagedAttnImpl.NKI_MASKED:
                attn_mask = reorder_context_mask(attn_mask, 2048, self.block_size)
        elif paged_attn_impl in (
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION,
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE,
        ):
            # Use tile sizes from config (set via --additional-config)
            large_q_tile_size = self.large_q_tile_size
            large_kv_tile_size = self.large_kv_tile_size

            # Validate that large_kv_tile_size <= max_model_len
            if large_kv_tile_size > self.max_model_len:
                raise ValueError(
                    f"large_kv_tile_size ({large_kv_tile_size}) must be <= "
                    f"max_model_len ({self.max_model_len})"
                )

            if self.max_model_len * self.max_num_reqs % large_kv_tile_size > 0:
                raise ValueError(
                    f"large_kv_tile_size ({large_kv_tile_size})"
                    f" must be a factor of "
                    f"max_model_len ({self.max_model_len})"
                    f" * max_num_reqs ({self.max_num_reqs})"
                )

            # Calculate max_num_decode_tiles based on the formula:
            # max_model_len * max_num_reqs / large_kv_tile_size

            # scheduled 1k, large_q_tile_size=128
            # => 1 request q_tiles = 8,
            #    2 request q_tiles = ...
            # sum(q) = scheduled
            # q_tiles = (q + large_q_tile_size - 1)
            #           // large_q_tile_size
            # sum(q_tile) <= sum(ceil(q / large_q_tile_size))
            #   <= sum(q) / large_q_tile_size + max_num_reqs
            #   = scheduled / large_q_tile_size + max_num_reqs
            # kv_tiles <= (self.max_model_len
            #   + large_kv_tile_size - 1) // large_kv_tile_size
            # sum(kv_tiles * q_tiles)
            #   <= max_kv_tiles * sum(q_tiles)
            max_num_kv_tiles = int(
                (self.max_model_len + large_kv_tile_size - 1) // large_kv_tile_size
            )
            max_sum_q_tiles = int(
                (padded_total_num_scheduled_tokens + large_q_tile_size - 1)
                / large_q_tile_size
                + self.max_num_reqs
            )
            max_num_decode_tiles = round_up(
                max_num_kv_tiles * self.max_num_reqs, self.dynamic_loop_unroll
            )
            max_num_prefill_tiles = round_up(
                max_num_kv_tiles * max_sum_q_tiles, self.dynamic_loop_unroll
            )
            logger.info(
                "Using large_kv_tile_size=%s, "
                "calculated max_num_decode_tiles=%s, "
                "max_num_prefill_tiles=%s",
                large_kv_tile_size,
                max_num_decode_tiles,
                max_num_prefill_tiles,
            )

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copy to avoid state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens
            )

            self.set_active_loras(self.input_batch, padded_num_scheduled_tokens_per_req)

        # seq_lens converted to list for compatibility if needed

        # Create attention metadata using selected backend
        backend_cls, metadata_cls = get_attention_backend_class()

        if paged_attn_impl.value.endswith("masked"):
            active_block_table = active_block_table.to(self.device)
            attn_mask = attn_mask.to(self.device)
        else:
            active_block_table = None
            attn_mask = None

        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            dp_max_reqs, self.max_num_reqs
        )
        # Indices at which we sample (positions of last token in the sequence).
        # Padded to avoid recompiling when `num_reqs` varies.
        logits_indices = self.query_start_loc_cpu[1 : padded_num_reqs + 1] - 1
        logits_indices = logits_indices.to(self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens
            )

            self.set_active_loras(self.input_batch, padded_num_scheduled_tokens_per_req)

        from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.prepare_nki_attention_runner import (  # noqa: E501
            prepare_nki_attention_runner,
        )

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        per_layer_attn_metadata: dict[str, NeuronAttentionMetadata] = {}

        # Cache for sharing identical attn_metadata across layers
        # Key: sliding_window value (the only per-layer parameter that could vary)
        # Value: attn_metadata
        attn_metadata_cache: dict[Optional[int], NeuronAttentionMetadata] = {}

        for layer_name, attn_module in layers.items():
            # Get layer-specific sliding window
            # (could be None for full attention layers)
            layer_sliding_window = self.layer_sliding_windows[layer_name]

            # Check if we already have attn_metadata for this configuration
            if layer_sliding_window in attn_metadata_cache:
                per_layer_attn_metadata[layer_name] = attn_metadata_cache[
                    layer_sliding_window
                ]
                continue

            nki_kernel_runner = None
            if paged_attn_impl in (
                PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION,
                PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE,
            ):
                nki_kernel_runner = prepare_nki_attention_runner(
                    scheduler_output=scheduler_output,
                    input_batch=self.input_batch,
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    large_q_tile_size=large_q_tile_size if enable_prefill else 1,
                    large_kv_tile_size=large_kv_tile_size,
                    max_model_len=self.max_model_len,
                    dynamic_loop_unrolling_size=self.dynamic_loop_unroll,
                    padded_query_length=padded_total_num_scheduled_tokens,
                    # max_num_prefill_tiles=32,
                    max_num_prefill_tiles=max_num_prefill_tiles,
                    max_num_decode_tiles=max_num_decode_tiles,
                    include_prompt_in_ctx=True,
                    skip_active=True,
                    max_num_reqs=None,
                    sliding_window=layer_sliding_window,
                )

            attn_metadata = NeuronAttentionMetadata(
                seq_lens=seq_lens.to(dtype=torch.int32).to(self.device),
                query_start_loc=query_start_loc.to(dtype=torch.int32).to(self.device),
                block_tables=block_tables.to(dtype=torch.int32).to(self.device),
                slot_mapping=slot_mapping.to(dtype=torch.int64).to(self.device),
                num_seqs=torch.tensor([num_reqs], dtype=torch.int32).to(self.device),
                enable_prefill=True,  # TODO: enable_prefill,
                active_block_table=active_block_table.to(self.device)
                if active_block_table is not None
                else active_block_table,
                attn_mask=attn_mask.to(self.device)
                if attn_mask is not None
                else attn_mask,
                nki_kernel_runner=nki_kernel_runner.to(self.device)
                if nki_kernel_runner is not None
                else nki_kernel_runner,
            )

            # Cache this attn_metadata for reuse by other layers with same config
            attn_metadata_cache[layer_sliding_window] = attn_metadata
            per_layer_attn_metadata[layer_name] = attn_metadata

        return per_layer_attn_metadata, logits_indices, padded_num_reqs

    def _scatter_placeholders(
        self,
        embeds: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return embeds

        placeholders = embeds.new_full(
            (is_embed.shape[0], embeds.shape[-1]),
            fill_value=torch.nan,
        )
        placeholders[is_embed] = embeds
        return placeholders

    def _gather_placeholders(
        self,
        placeholders: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return placeholders

        return placeholders[is_embed]

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id])
                )

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_mm_inputs,
                device=self.device,
            )

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs
            )

            if isinstance(curr_group_outputs, torch.Tensor):
                encoder_outputs.append(curr_group_outputs)
            else:
                assert isinstance(curr_group_outputs, (list, tuple))
                for output in curr_group_outputs:
                    encoder_outputs.append(output)

        # Cache the encoder outputs.
        # NOTE (NickLucche) here we diverge from logic in other runners, as we
        # assume to only have whole mm items to process. Hence we avoid the
        # intrinsic dynamism that `scatter_mm_placeholders` introduces.
        for (req_id, input_id, pos_info), output in zip(
            req_ids_pos,
            encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            assert pos_info.is_embed is None, (
                "Expected all positions to be contiguous and embeddings."
            )
            self.encoder_cache[req_id][input_id] = output

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            # TODO unroll loop and assume/enforce --disable_chunked_mm_input
            # NOTE (NickLucche) here we diverge from logic in other runners, as
            # we assume to only have whole mm items to process. Hence we avoid
            # the intrinsic dynamism that `gather_mm_placeholders` introduces.
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                assert pos_info.is_embed is None, (
                    "Expected all positions to be contiguous and embeddings."
                )
                encoder_output = self.encoder_cache[req_id][i]
                mm_embeds.append(encoder_output)
        return mm_embeds

    def _get_model_inputs(self, input_ids: torch.Tensor, mm_embeds: list[torch.Tensor]):
        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # NOTE: setup current batch's metadata for kv connector.
        # Currently, only verified with NixlConnector
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)

        local_tokens = int(scheduler_output.total_num_scheduled_tokens or 0)
        local_reqs = int(self.input_batch.num_reqs or 0)
        enable_prefill_local = False
        if local_tokens > 0:
            try:
                # same rule as before: if any req scheduled > 1 => prefill
                enable_prefill_local = (
                    max(scheduler_output.num_scheduled_tokens.values()) > 1
                )
            except Exception:
                enable_prefill_local = False

        dp_max_tokens, dp_max_reqs, enable_prefill_global = self._dp_sync_dummy_shapes(
            num_tokens_local=(local_tokens if local_tokens > 0 else 1),
            num_reqs_local=(local_reqs if local_reqs > 0 else self.min_num_seqs),
            enable_prefill_local=enable_prefill_local,
        )

        # Prepare inputs
        prepare_inputs_result = self._prepare_inputs(
            scheduler_output,
            dp_max_tokens=dp_max_tokens,
            dp_max_reqs=dp_max_reqs,
            enable_prefill_global=enable_prefill_global,
        )
        attn_metadata, logits_indices, padded_num_reqs = prepare_inputs_result
        input_ids, inputs_embeds = self._get_model_inputs(self.input_ids, mm_embeds)

        num_reqs = self.input_batch.num_reqs

        # Run the decoder
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=self.position_ids,
                inputs_embeds=inputs_embeds,
            )
        logits = self.get_logits(hidden_states, logits_indices)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            require_struct_decoding, grammar_bitmask_padded, arange = (
                self.prepare_structured_decoding_input(logits, scheduler_output)
            )
            logits = self.structured_decode(
                require_struct_decoding, grammar_bitmask_padded, logits, arange
            )

        # Sample the next token and get logprobs if needed.
        # Use NKIPy-specific padded sampling metadata.
        sampling_metadata = NKIPySamplingMetadata.from_input_batch(
            self.input_batch,
            padded_num_reqs,
            device=self.device,
            vocab_size=self.vocab_size,
        )
        selected_token_ids = self.sample_from_logits(logits, sampling_metadata)
        # logits = logits.cpu().to(dtype=torch.float32)
        # selected_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        # NOTE (NickLucche) Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs. We can't enforce it
        # due to recompilations outside torch.compiled code, so just make
        # sure `sample_from_logits` does not modify the logits in-place.
        logprobs = (
            self.gather_logprobs(logits, selected_token_ids)
            if sampling_metadata.logprobs
            else None
        )
        logprobs_lists = logprobs.tolists() if logprobs is not None else None

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfers(
            scheduler_output
        )

        # Remove padding on cpu and keep dynamic op outside of xla graph.
        selected_token_ids = selected_token_ids.cpu()[:num_reqs]

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in self.input_batch.req_ids[:num_reqs]
        ), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()

            # Mask out the sampled tokens that should not be sampled.
            # TODO: Keep in sync with gpu_model_runner.py, in particular
            #       the "else" case here
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Append sampled tokens
            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
                self.input_batch.num_tokens[i] += 1

        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist() for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            self.input_batch.num_tokens[:num_reqs] += gen_lens
            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.input_batch.token_ids_cpu[i, target_slice] = (
                    valid_sampled_token_ids[i]
                )
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        kv_connector_output = (
            None
            if (finished_sending is None and finished_recving is None)
            else KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            )
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        # Check there are no new graphs compiled - all the graphs should be
        # captured and compiled during warm up.
        self._verify_num_xla_graphs("execute_model")

        return model_runner_output

    @torch.no_grad()
    def execute_dummy_batch(self) -> None:
        # Participate in the same DP sync as execute_model ranks.
        # Use num_tokens_local=1 so dummy ranks always contribute a valid value.
        num_tokens_local = 1
        num_reqs_local = self.min_num_seqs
        enable_prefill_local = False

        dp_max_tokens, dp_max_reqs, enable_prefill_global = self._dp_sync_dummy_shapes(
            num_tokens_local=num_tokens_local,
            num_reqs_local=num_reqs_local,
            enable_prefill_local=enable_prefill_local,
        )

        # Match execute_model padding policy (DP-max driven).
        padded_tokens_final = _get_padded_token_len(
            self.num_tokens_paddings, dp_max_tokens
        )
        if enable_prefill_global:
            padded_tokens_final = self.num_tokens_paddings[-1]

        self._dummy_run(padded_tokens_final, enable_prefill=enable_prefill_global)

        # padded_num_reqs_final = _get_padded_num_reqs_with_upper_limit(
        #     dp_max_reqs, self.max_num_reqs
        # )

    def update_config(self, overrides: dict[str, Any]) -> None:
        # TODO: TPU config may need extra validation
        # https://github.com/vllm-project/vllm/pull/20095#discussion_r2201497754
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self) -> None:
        """Load and compile the model for NKIPy execution."""
        logger.info("Starting to load model %s...", self.model_config.model)

        # Get compilation settings from config early - before any compilation happens
        nkipy_config = get_nkipy_config(self.vllm_config)
        split_graph = nkipy_config.split_graph

        # Set global config for module-level access (e.g., attention backend)
        set_global_nkipy_config(nkipy_config)
        logger.info(
            "Set global NKIPy config: "
            "paged_attn_impl=%s, paged_kv_impl=%s",
            nkipy_config.paged_attn_impl.value,
            nkipy_config.paged_kv_impl.value,
        )

        # Set module-level default for split_graph BEFORE any torch.compile
        # This ensures @local_compile decorators in gpt_oss.py etc use correct backend
        set_default_split_graph(split_graph)

        # Configure Neuron runtime inspection if enabled via environment variable
        if is_master() and nkipy_envs.VLLM_NKIPY_RT_INSPECT_ENABLE:
            os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
            os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = (
                nkipy_envs.VLLM_NKIPY_RT_INSPECT_OUTPUT_DIR
            )
            logger.info(
                "Neuron runtime inspection enabled: "
                "ENABLE=%s, OUTPUT_DIR=%s",
                os.environ["NEURON_RT_INSPECT_ENABLE"],
                os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"],
            )

        # Initialize NKIPy backend first
        self._init_nkipy_backend()

        # Override vllm's model registry to use our modified gpt_oss
        from vllm.model_executor.models.registry import ModelRegistry

        # Register the vllm_nkipy version of GptOssForCausalLM
        # This will override the default vllm version
        ModelRegistry.register_model(
            "GptOssForCausalLM",
            "vllm_nkipy.model_executor.models.gpt_oss:GptOssForCausalLM",
        )
        logger.info(
            "Registered vllm_nkipy.model_executor.models"
            ".gpt_oss:GptOssForCausalLM in ModelRegistry"
        )

        # Monkey-patch vllm's fused_moe config to use our modified version
        import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer_module

        import vllm_nkipy.model_executor.layers.fused_moe.config as nkipy_moe_config

        # Replace the config module references in layer module
        fused_moe_layer_module.FusedMoEConfig = nkipy_moe_config.FusedMoEConfig
        fused_moe_layer_module.FusedMoEParallelConfig = (
            nkipy_moe_config.FusedMoEParallelConfig
        )
        logger.info("Monkey-patched vllm's fused_moe layer to use vllm_nkipy config")

        try:
            model_loader = get_model_loader(self.load_config)
            logger.info("Loading model from scratch...")
            model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Unable to load model, a likely reason is the model is "
                "too large for the current device's HBM memory. "
                "Consider switching to a smaller model "
                "or sharding the weights on more chips. "
                f"See the detailed error: {e}"
            ) from e

        if self.lora_config is not None:
            model = self.load_lora_model(model, self.vllm_config, self.device)
            replace_set_lora(model)

        # insert barrier ops (config value read from NKIPyConfig)
        # Note: nkipy_config was already fetched at the start of load_model
        # Layer barriers only apply when using split backend
        layer_barriers_group_size = nkipy_config.layer_barriers_group_size
        if split_graph and layer_barriers_group_size > 0:
            logger.info(
                f"Installing layer barriers with group_size={layer_barriers_group_size}"
            )
            install_layer_barriers(
                model.model.layers, group_size=layer_barriers_group_size
            )
        elif layer_barriers_group_size > 0:
            logger.info("Skipping layer barriers (split_graph=False)")

        # TODO: directly load to device
        model = model.to(self.device)
        # for gpt-oss
        for layer in model.model.layers:
            layer.attn.attn.impl.sinks = layer.attn.sinks

        # Apply NKIPy compilation
        logger.info("Compiling model with NKIPy backend...")

        # Get compilation settings from config (already fetched at start of load_model)
        compile_strategy = nkipy_config.compile_strategy
        merge_step = nkipy_config.merge_step
        # split_graph already set via set_default_split_graph() at start
        logger.info(
            "Using compile_strategy=%s, "
            "merge_step=%s, split_graph=%s",
            compile_strategy.value,
            merge_step,
            split_graph,
        )

        if compile_strategy == CompileStrategy.MERGE_LAYERS:
            import types

            compiled_model = model
            forward_dispatch = make_forward_dispatch(merge_step)
            compiled_model.model.forward = types.MethodType(
                forward_dispatch, compiled_model.model
            )
        else:  # FULL_FX_GRAPH
            compiled_model = torch.compile(
                model,
                backend=NKIPY_BACKEND,
                dynamic=False,
                options={
                    "split_graph": split_graph,
                    "guard_filter_fn": (
                        torch.compiler
                        .skip_guard_on_all_nn_modules_unsafe
                    ),
                },
            )

        # # Create wrapper to handle device management
        self.model = compiled_model
        logger.info("Model compilation complete")
        self.sampler = Sampler()

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."
        )
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.model, model_config=self.model_config)

    def _dp_sync_dummy_shapes(
        self,
        num_tokens_local: int,
        num_reqs_local: int,
        enable_prefill_local: bool,
    ) -> tuple[int, int, bool]:
        """DP-sync dummy shape drivers (MAX over DP group)."""
        from vllm.config import get_current_vllm_config
        from vllm.distributed import parallel_state

        vllm_config = get_current_vllm_config()
        dp_size = vllm_config.parallel_config.data_parallel_size
        if dp_size <= 1:
            return num_tokens_local, num_reqs_local, enable_prefill_local

        dp_group = parallel_state.get_dp_group()

        # [tokens, reqs, enable_prefill_int]
        local = torch.tensor(
            [num_tokens_local, num_reqs_local, int(enable_prefill_local)],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            local,
            op=torch.distributed.ReduceOp.MAX,
            group=dp_group.cpu_group,
        )

        return int(local[0]), int(local[1]), bool(int(local[2]))

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int, enable_prefill: bool) -> None:
        logger.info("Dummy run with num_tokens=%s start.", num_tokens)
        actual_num_reqs = min(num_tokens, self.max_num_reqs)
        query_lens = [1] * self.max_num_reqs
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
        query_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int32), dim=0, dtype=torch.int32
        )
        context_lens = torch.ones((self.max_num_reqs,), dtype=torch.int32)
        block_tables = torch.zeros(
            (self.max_num_reqs, self.block_table_cpu.shape[1]), dtype=torch.int32
        )
        num_seqs = torch.tensor([actual_num_reqs], dtype=torch.int32)

        active_block_table, attn_mask = None, None
        paged_attn_impl = _get_paged_attn_impl()
        if paged_attn_impl.value.endswith("masked"):
            context_cap = self.sliding_window or self.max_model_len
            num_active_blocks = get_num_active_blocks(
                self.block_size, self.num_blocks, self.max_num_reqs, context_cap
            )
            block_tables = torch.arange((num_tokens // self.block_size) + 1).unsqueeze(
                0
            )
            active_block_table = get_active_block_tables(
                block_tables,
                torch.tensor([num_tokens]),
                torch.tensor([num_tokens]),
                self.block_size,
                num_active_blocks,
            )

            context_kv_len = num_active_blocks * self.block_size

            attn_mask, _ = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                query_lens=[num_tokens], seq_lens=[num_tokens]
            )
            attn_mask = nn.functional.pad(
                attn_mask,
                (
                    0,
                    context_kv_len + num_tokens - attn_mask.shape[1],
                    0,
                    num_tokens - attn_mask.shape[0],
                ),
                "constant",
                0,
            ).bool()
        elif paged_attn_impl in (
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION,
            PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE,
        ):
            # Use tile sizes from config (set via --additional-config)
            large_q_tile_size = self.large_q_tile_size
            large_kv_tile_size = self.large_kv_tile_size

            # Validate that large_kv_tile_size <= max_model_len
            if large_kv_tile_size > self.max_model_len:
                raise ValueError(
                    f"large_kv_tile_size ({large_kv_tile_size}) must be <= "
                    f"max_model_len ({self.max_model_len})"
                )

            if self.max_model_len * self.max_num_reqs % large_kv_tile_size > 0:
                raise ValueError(
                    f"large_kv_tile_size "
                    f"({large_kv_tile_size})"
                    f" must be a factor of "
                    f"max_model_len ({self.max_model_len})"
                    f" * max_num_reqs "
                    f"({self.max_num_reqs})"
                )

            max_num_kv_tiles = int(
                (self.max_model_len + large_kv_tile_size - 1)
                // large_kv_tile_size
            )
            max_sum_q_tiles = int(
                (num_tokens + large_q_tile_size - 1)
                / large_q_tile_size
                + self.max_num_reqs
            )
            max_num_decode_tiles = round_up(
                max_num_kv_tiles * self.max_num_reqs,
                self.dynamic_loop_unroll,
            )
            max_num_prefill_tiles = round_up(
                max_num_kv_tiles * max_sum_q_tiles,
                self.dynamic_loop_unroll,
            )
            logger.info(
                "Using large_kv_tile_size=%s, "
                "calculated max_num_decode_tiles=%s, "
                "max_num_prefill_tiles=%s",
                large_kv_tile_size,
                max_num_decode_tiles,
                max_num_prefill_tiles,
            )

            from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.prepare_nki_attention_runner import (  # noqa: E501
                _prepare_nki_attention_runner,
            )

            assert len(self.num_reqs_paddings) == 1
            num_reqs = self.num_reqs_paddings[0]
            req_ids = np.array([str(i) for i in range(num_reqs)])
            num_scheduled_tokens = [
                {str(i): 1 for i in range(num_reqs)},  # decode
                {str(i): 2 for i in range(num_reqs)},  # prefill
                {str(i): i for i in range(num_reqs)},  # mixed
            ]
            num_computed_tokens_cpu = [
                np.array([128 for i in range(num_reqs)]),  # decode
                np.array([1 for i in range(num_reqs)]),  # prefill
                np.array([1 for i in range(num_reqs)]),  # mixed
            ]
            if enable_prefill:
                num_scheduled_tokens = num_scheduled_tokens[1]
                num_computed_tokens_cpu = num_computed_tokens_cpu[1]
            else:
                num_scheduled_tokens = num_scheduled_tokens[0]
                num_computed_tokens_cpu = num_computed_tokens_cpu[0]
            block_tables = torch.arange(
                self.max_num_blocks_per_req * num_reqs, dtype=torch.int32
            ).reshape(num_reqs, self.max_num_blocks_per_req)

        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = torch.zeros(
                (num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
            )
        else:
            input_ids = torch.zeros((num_tokens), dtype=torch.int32).to(self.device)
            inputs_embeds = None
        position_ids = torch.zeros(num_tokens, dtype=torch.int32).to(self.device)

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        per_layer_attn_metadata: dict[str, NeuronAttentionMetadata] = {}

        # Cache for sharing identical attn_metadata across layers
        attn_metadata_cache: dict[Optional[int], NeuronAttentionMetadata] = {}

        for layer_name, attn_module in layers.items():
            # Get layer-specific sliding window
            layer_sliding_window = self.layer_sliding_windows[layer_name]

            # Check cache first
            if layer_sliding_window in attn_metadata_cache:
                per_layer_attn_metadata[layer_name] = attn_metadata_cache[
                    layer_sliding_window
                ]
                continue

            nki_kernel_runner = None
            if paged_attn_impl in (
                PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION,
                PagedAttnImpl.NKI_BLOCKSPARSE_FLASH_ATTENTION_KV_CACHE,
            ):
                nki_kernel_runner = _prepare_nki_attention_runner(
                    req_ids=req_ids,
                    num_scheduled_tokens=num_scheduled_tokens,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    block_table=block_tables,
                    num_reqs=num_reqs,
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    large_q_tile_size=large_q_tile_size if enable_prefill else 1,
                    large_kv_tile_size=large_kv_tile_size,
                    max_model_len=self.max_model_len,
                    padded_query_length=num_tokens,
                    dynamic_loop_unrolling_size=self.dynamic_loop_unroll,
                    max_num_prefill_tiles=max_num_prefill_tiles,
                    max_num_decode_tiles=max_num_decode_tiles,
                    include_prompt_in_ctx=True,
                    skip_active=True,
                    max_num_reqs=None,
                    sliding_window=layer_sliding_window,
                )

            attn_metadata = NeuronAttentionMetadata(
                seq_lens=context_lens.to(self.device),
                query_start_loc=query_start_loc.to(self.device),
                block_tables=block_tables.to(self.device),
                slot_mapping=slot_mapping.to(self.device),
                num_seqs=num_seqs.to(self.device),
                enable_prefill=enable_prefill,
                active_block_table=active_block_table.to(torch.int32).to(self.device)
                if active_block_table is not None
                else active_block_table,
                attn_mask=attn_mask.to(self.device)
                if attn_mask is not None
                else attn_mask,
                nki_kernel_runner=nki_kernel_runner.to(self.device)
                if nki_kernel_runner is not None
                else nki_kernel_runner,
            )

            attn_metadata_cache[layer_sliding_window] = attn_metadata
            per_layer_attn_metadata[layer_name] = attn_metadata

        with (
            self.maybe_select_dummy_loras(
                self.lora_config, np.array([num_tokens], dtype=np.int32)
            ),
            set_forward_context(per_layer_attn_metadata, self.vllm_config, 0),
        ):
            out = self.model(
                input_ids=input_ids, positions=position_ids, inputs_embeds=inputs_embeds
            )
        self._hidden_states_dtype = out.dtype

        logger.info("Dummy run with num_tokens=%s end.", num_tokens)

    def _set_active_loras(
        self, prompt_lora_mapping, token_lora_mapping, lora_requests
    ) -> None:
        super()._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests
        )

    def _precompile_mm_encoder(self) -> None:
        # Pre-compile MM encoder for all supported data modalities.
        hf_config = self.vllm_config.model_config.hf_config
        for mode, max_items_by_mode in self.max_num_mm_items_by_modality.items():
            logger.info(
                "Compiling Multimodal %s Encoder with different input shapes.", mode
            )
            start = time.perf_counter()
            # No padding for MM encoder just yet.
            for num_items in range(1, max_items_by_mode + 1):
                logger.info("  -- mode: %s items: %d", mode, num_items)
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(mode, num_items)
                # Run multimodal encoder.
                mm_embeds = self.model.get_multimodal_embeddings(
                    **batched_dummy_mm_inputs
                )
                num_patches = mm_embeds[0].shape[0]
                items_size = num_patches * num_items

                # NOTE (NickLucche) pre-compile `get_input_embeddings` when mm
                # embeddings are present. We assume `--disable-mm-chunked`,
                # hence only whole items can be scheduled. This implies we just
                # need to compile when `num_items` fit the (padded) `input_ids`
                for num_tokens in self.num_tokens_paddings:
                    if num_tokens >= items_size:
                        # XLA Workaround: if torch.zeros(..device) is used, XLA
                        # compiles a scalar+expansion op, which won't match
                        # the graph generated at runtime. CPU->TPU must be used
                        placeholders_ids = torch.zeros(
                            num_tokens, dtype=torch.int32, device="cpu"
                        )
                        # Align placeholders and actual num mm_embeddings.
                        placeholders_ids[:items_size] = hf_config.image_token_index

                        placeholders_ids = placeholders_ids.to(self.device)
                        # Assign outputs or the graph will be cut short.
                        a, b = self._get_model_inputs(placeholders_ids, [mm_embeds])
                        assert a is None

            # Pre-compile `get_input_embeddings` when mm_embeddings are not
            # present. Chunk is only made of text, no mm_placeholders.
            for num_tokens in self.num_tokens_paddings:
                placeholders_ids = torch.zeros(
                    num_tokens, dtype=torch.int32, device="cpu"
                )
                placeholders_ids = placeholders_ids.to(self.device)
                a, b = self._get_model_inputs(placeholders_ids, [])
                assert a is None

            end = time.perf_counter()
            logger.info(
                "Multimodal %s Encoder compilation finished in in %.2f [secs].",
                mode,
                end - start,
            )

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings[:-1]:
            logger.info("  -- num_tokens: %d", num_tokens)
            # self._dummy_run(num_tokens, enable_prefill=True)
            self._dummy_run(num_tokens, enable_prefill=False)
        self._dummy_run(self.num_tokens_paddings[-1], enable_prefill=True)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("model backbone")

    def _precompile_get_logits(self) -> None:
        # Compile the get_logits function for bucketed
        # n_tokens x max_num_reqs.
        logger.info("Compiling get_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            dummy_hidden = torch.zeros(
                (num_tokens, hsize), dtype=self._hidden_states_dtype
            )
            dummy_hidden = dummy_hidden.to(self.device)
            # torch._dynamo.mark_dynamic(dummy_hidden, 0)
            for num_reqs in self.num_reqs_paddings:
                indices = torch.zeros(num_reqs, dtype=torch.int32)
                indices = indices.to(self.device)
                # torch._dynamo.mark_dynamic(indices, 0)
                self.get_logits(dummy_hidden, indices)
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens, num_reqs)
                # Requests can't be more than tokens. But do compile for the
                # next bigger value in case num_tokens uses bucketed padding.
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("get_logits")

    def _precompile_structured_decoding(self) -> None:
        logger.info("Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_require_struct_decoding = self.require_structured_out_cpu[
                :num_reqs
            ].to(self.device)
            dummy_grammar_bitmask = self.grammar_bitmask_cpu[:num_reqs].to(self.device)
            # The first dimension of the above 3 dummy tensors cannot be
            # mark_dynamic because some operations in structured_decode require
            # them to be static.
            arange = self.structured_decode_arange.to(self.device)
            self.structured_decode(
                dummy_require_struct_decoding,
                dummy_grammar_bitmask,
                dummy_logits,
                arange,
            )
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("structured_decoding")

    def _precompile_sample_from_logits(self) -> None:
        logger.info("Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            # The first dimension of dummy_logits cannot be mark_dynamic
            # because some operations in the sampler require it to be static.
            for all_greedy in [True]:
                generate_params_if_all_greedy = not all_greedy
                sampling_metadata = NKIPySamplingMetadata.from_input_batch(
                    self.input_batch,
                    num_reqs,
                    self.device,
                    self.vocab_size,
                    generate_params_if_all_greedy,
                )
                sampling_metadata.all_greedy = all_greedy
                with self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_reqs], dtype=np.int32)
                ):
                    self.sample_from_logits(dummy_logits, sampling_metadata)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("sample_from_logits")

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_tokens = torch.zeros((num_reqs, 1), dtype=torch.int32).to(self.device)
            with self.maybe_select_dummy_loras(
                self.lora_config, np.array([num_reqs], dtype=np.int32)
            ):
                self.gather_logprobs(dummy_logits, dummy_tokens)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("gather_logprobs")

    @torch.no_grad()
    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        with self.maybe_setup_dummy_loras(self.lora_config):
            # self._precompile_mm_encoder()
            self._precompile_backbone()
            self._precompile_get_logits()
            # self._precompile_structured_decoding()
            self._precompile_sample_from_logits()
            # self._precompile_gather_logprobs()
        gc.collect()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (
            self.is_multimodal_model
            and self.max_num_encoder_input_tokens > 0
            and self.encoder_cache_size > 0
        ):
            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            dummy_data_modality, max_num_mm_items = max(
                self.max_num_mm_items_by_modality.items(), key=lambda t: t[1]
            )

            encoder_budget = min(
                self.max_num_encoder_input_tokens, self.encoder_cache_size
            )

            logger.info(
                "Encoder cache will be initialized with a budget of %d tokens,"
                " and profiled with %s %s items of the maximum feature size.",
                encoder_budget,
                max_num_mm_items,
                dummy_data_modality,
            )

            # Create dummy batch of multimodal inputs.
            batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                dummy_data_modality, max_num_mm_items
            )

            # Run multimodal encoder.
            # Isolate encoder graph from post-processing to minimize
            # impact of recompilation until it's fixed.
            start = time.perf_counter()
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs
            )
            end = time.perf_counter()
            logger.info(
                "Multimodal Encoder profiling finished in in %.2f [secs].", end - start
            )

            assert len(dummy_encoder_outputs) == max_num_mm_items, (
                "Expected dimension 0 of encoder outputs to match the number "
                f"of multimodal data items: {max_num_mm_items}, got "
                f"{len(dummy_encoder_outputs)=} instead. This is most likely "
                "due to the 'get_multimodal_embeddings' method of the model "
                "not implemented correctly."
            )

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Trigger compilation for general shape.
        self._dummy_run(num_tokens)

        self.encoder_cache.clear()
        gc.collect()

    def maybe_setup_cross_layer_kv_sharing(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """
        Add layers that reuse KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        raise NotImplementedError
        # add_kv_sharing_layers_to_kv_cache_groups(
        #     self.shared_kv_cache_layers,
        #     kv_cache_config.kv_cache_groups,
        # )

        # for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        # ):
        #     logger.debug("%s reuses KV cache of %s", layer_name,
        #                  target_layer_name)
        #     kv_caches[layer_name] = kv_caches[target_layer_name]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.num_blocks = kv_cache_config.num_blocks

        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )

        if (
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            != self.block_size
        ):
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device="cpu",
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
            )
        # Verify dtype compatibility between block_table_cpu and input_batch
        assert (
            self.block_table_cpu.dtype
            == self.input_batch.block_table[0].get_cpu_tensor().dtype
        )

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in TPU."
            )
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                assert num_blocks == self.num_blocks
                if isinstance(kv_cache_spec, AttentionSpec):
                    # Use selected backend for KV cache shape
                    backend_cls, _ = get_attention_backend_class()
                    kv_cache_shape = backend_cls.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape, dtype=dtype)

                    # kv_caches[layer_name] = tpu_kv_cache
                    kv_caches[layer_name] = tpu_kv_cache.to(self.device)
                else:
                    raise NotImplementedError

        # Set up cross-layer KV cache sharing if needed
        self.maybe_setup_cross_layer_kv_sharing(kv_caches, kv_cache_config)

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)

        for layer_name, attn_module in get_layers_from_vllm_config(
            self.vllm_config, Attention
        ).items():
            # attn_module.sliding_window is set by vLLM when using SlidingWindowSpec
            self.layer_sliding_windows[layer_name] = attn_module.sliding_window

    @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
    @torch.no_grad()
    def get_logits(
        self, hidden_states: torch.Tensor, indices_do_sample: torch.Tensor
    ) -> torch.Tensor:
        """Select hidden states and compute logits in a single operation."""
        selected_hidden_states = hidden_states[indices_do_sample].contiguous()
        return self.model.compute_logits(selected_hidden_states)

    @torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
    @torch.no_grad()
    def sample_from_logits(
        self, logits: torch.Tensor, sampling_metadata: NKIPySamplingMetadata
    ) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if sampling_metadata.all_greedy:
            # out_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            out_tokens = custom_argmax(logits)
        else:
            # out_tokens = self.sampler(logits,
            #                           sampling_metadata).sampled_token_ids
            raise NotImplementedError
        return out_tokens

    # @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def gather_logprobs(
        self, logits: torch.Tensor, sampled_tokens: torch.Tensor
    ) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        logprobs = self.sampler.compute_logprobs(logits)
        return self.sampler.gather_logprobs(
            logprobs,
            self.model_config.max_logprobs,
            token_ids=sampled_tokens.squeeze(-1),
        )

    # @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits,
        )

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor, arange: torch.Tensor
    ):
        assert logits.shape[0] == grammar_bitmask.shape[0]
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (
                torch.bitwise_right_shift(grammar_bitmask[i][:, None], arange[None, :])
                & 1
            ) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[: self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf")
            )
        return logits_cloned

    def get_multimodal_embeddings(self, *args, **kwargs):
        return self.model.get_multimodal_embeddings(*args, **kwargs)

    def get_input_embeddings(self, *args, **kwargs):
        return self.model.get_input_embeddings(*args, **kwargs)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(req_id)
            if mask_index is None:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = torch.from_numpy(
            grammar_bitmask[mask_indices]
        )
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        struct_out_indices = torch.tensor(struct_out_indices, dtype=torch.long)
        self.require_structured_out_cpu[struct_out_indices] = True
        return (
            self.require_structured_out_cpu[:num_reqs].to(logits.device),
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
        )

    def _get_mm_dummy_batch(
        self, modality: str, batch_size: int
    ) -> BatchedTensorInputs:
        # Dummy data for pre-compiling multimodal models.
        dummy_request_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
        )
        dummy_mm_data = dummy_request_data.multi_modal_data

        # Dummy data definition in V0 may contain multiple multimodal items
        # (e.g, multiple images) for a single request, therefore here we
        # always replicate first item by max_num_mm_items times since in V1
        # they are scheduled to be processed separately.
        assert isinstance(dummy_mm_data, MultiModalKwargs), (
            "Expected dummy multimodal data to be of type "
            f"MultiModalKwargs, got {type(dummy_mm_data)=} instead. "
            "This is most likely due to the model not having a merged "
            "processor."
        )

        # When models have a merged processor, their dummy data is
        # already batched `MultiModalKwargs`, therefore we take the first
        # `MultiModalKwargsItem` from the desired modality to profile on.
        dummy_mm_item = dummy_mm_data.get_item(modality=modality, item_index=0)
        dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

        batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] * batch_size)
        return MultiModalKwargs.as_kwargs(
            batched_dummy_mm_inputs,
            device=self.device,
        )


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    """Generate request padding sizes.

    Args:
        min_req_size: Minimum request size (should be power of 2).
        max_req_size: Maximum request size.

    Returns:
        List of padding sizes.
    """
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = min_req_size
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size, min_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(
    x: int, upper_limit: int, min_num_seqs: int = 4
) -> int:
    """Get padded number of requests with upper limit.

    Args:
        x: Input value.
        upper_limit: Upper limit for the result.
        min_num_seqs: Minimum number of sequences (default: 4).

    Returns:
        Padded value clamped to upper_limit.
    """
    res = min_num_seqs if x <= min_num_seqs else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_token_paddings(
    min_token_size: int, max_token_size: int, padding_gap: int
) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def replace_set_lora(model):
    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        # TODO: The integer index leads to a recompilation, but converting it
        # to a tensor doesn't seem to work anymore. This might be fixed with a
        # later release of torch_xla.
        self._original_set_lora(index, lora_a, lora_b, embeddings_tensor, bias)

    def _tpu_reset_lora(self, index: int):
        self._original_reset_lora(index)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(module, module.__class__)


def get_num_active_blocks(block_size, num_blocks, max_num_seqs, context_cap):
    # context_cap * max_num_seqs // block_size // magic_number
    return 512


def get_active_block_tables(block_tables, query_lens, seq_lens, block_size, num_blocks):
    context_lens = seq_lens - query_lens
    blocks_per_seq = (context_lens + block_size - 1) // block_size
    num_seqs = len(seq_lens)
    active_blocks: list[int] = []
    for seq_id in range(num_seqs):
        active_blocks = (
            active_blocks + block_tables[seq_id, : blocks_per_seq[seq_id]].tolist()
        )
    return nn.functional.pad(
        torch.tensor(active_blocks, dtype=torch.int32),
        (0, num_blocks - len(active_blocks)),
        "constant",
        0,
    )


class BlockDiagonalCausalFromBottomRightMask:
    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = torch.arange(n_queries).reshape(n_queries, 1).expand(n_queries, n_keys)
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens
            )
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size
            )
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens
            )
        return prior_mask, active_mask


def shift_bit_length(x):
    return 1 << (x - 1).bit_length()
