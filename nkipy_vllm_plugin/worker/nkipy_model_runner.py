# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.utils import make_ndarray_with_pad
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput, SamplerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from nkipy_vllm_plugin.worker.nkipy_model_loader import (
    get_neuron_model,
    NeuronCausalLM,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ModelInputForNeuron:
    request_ids: Optional[list[str]]
    input_tokens: Optional[npt.NDArray[np.uint32]]
    start_position_ids: Optional[npt.NDArray[np.int32]]
    input_block_ids: Optional[npt.NDArray[np.int32]]
    full_context_lens: Optional[npt.NDArray[np.int32]]
    is_prefill: bool


# This class is used for constructing ModelInputForNeuron and
# is not frozen.
@dataclass
class IntermediateInputData:
    request_ids: list[str] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    start_position_ids: list[int] = field(default_factory=list)
    input_block_ids: list[int] = field(default_factory=list)
    full_context_lens: list[int] = field(default_factory=list)
    computed_context_lens: list[int] = field(default_factory=list)


class NKIPyModelRunner(LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device

        self.pin_memory = False
        self.block_size = cache_config.block_size
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
        )

        self.requests: dict[str, CachedRequestState] = {}
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, npt.NDArray]] = {}

        # vLLM uses lora_manager to manage LoRA modules
        self.lora_manager = None
        self.model: NeuronCausalLM = None
        self.lora_serving_config = None

        self.is_block_kv_layout = False
        self.is_prefix_caching = False
        self.is_chunked_prefill = False

        # The following fields are used to support custom sequence id mapping.
        # The goal is to retain the batch line information for contiguous kv cache.
        # A mapping of vLLM request Id to neuron sequence id.
        self.use_custom_seq_id_mapping = not self.is_chunked_prefill
        self.vllm_req_to_neuron_seq_id_mapping: Dict[str, int] = {}
        # Set of neuron sequence id that are free for use.
        self.free_seq_ids = set(range(self.max_num_reqs))

    def get_model(self):
        return self.model

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig):
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # Not required for Neuron. To satisfy the interface.
        return

    def load_model(self, model_config, checkpoint) -> None:
        assert not self.is_chunked_prefill
        assert not self.is_prefix_caching
        assert not self.is_block_kv_layout
        is_reorder_needed = not (
            self.max_num_reqs == 1 or self.is_prefix_caching or self.is_chunked_prefill
        )
        assert not is_reorder_needed
        self.model = get_neuron_model(model_config, checkpoint, is_reorder_needed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        logger.debug(f"scheduler_output: {scheduler_output}")

        # Free slots of finished requests
        # We intentionally do this before updating the cached states as
        # the _update_states method is common across all hardware platforms.
        if self.use_custom_seq_id_mapping:
            for req_id in scheduler_output.finished_req_ids:
                if req_id in self.vllm_req_to_neuron_seq_id_mapping:
                    freed_slot = self.vllm_req_to_neuron_seq_id_mapping.pop(req_id)
                    self.free_seq_ids.add(freed_slot)

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # _prepare_model_input converts the scheduler output to ModelInputForNeuron
        model_input = self._prepare_model_input(scheduler_output)
        logger.debug(f"model_input: {model_input}")

        sampler_outputs = self._execute_model_for_text(
            model_input,
            intermediate_tensors,
        )

        return self._generate_model_runner_output(sampler_outputs)

    def _generate_model_runner_output(
        self, sampler_outputs: Optional[list[SamplerOutput]]
    ) -> ModelRunnerOutput:
        if sampler_outputs is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = sampler_outputs[0].sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]

        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            raise NotImplementedError("spec decode is not supported yet")

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            # TODO: support the following fields. currently they are hardcoded to None
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            finished_sending=None,
            finished_recving=None,
            pooler_output=[],
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        return {
            "layer": FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.model.num_key_value_heads,
                head_size=self.model.head_dim,
                # TODO: take the following from the model config
                dtype=torch.bfloat16,
                use_mla=False,
                sliding_window=None,
            )
        }

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
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

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
            assert (
                new_req_data.sampling_params is not None
            ), "Pooling is not supported in TPU yet"
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
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
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
            else:
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
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        self.input_batch.condense()
        self.input_batch.refresh_metadata()

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def _execute_model_for_text(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[list[SamplerOutput]]:
        hidden_states = self.model(**asdict(model_input))

        sampled_output = self._sample(hidden_states, model_input)
        return [sampled_output]

    def _prepare_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForNeuron:
        assert not self.is_chunked_prefill
        continuous_batching_model_input, is_prefill = (
            self._prepare_continuous_batching_inputs(scheduler_output)
        )
        return self._finalize_continuous_batching_inputs(
            continuous_batching_model_input,
            is_prefill,
        )

    def _prepare_continuous_batching_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Tuple[IntermediateInputData, bool]:
        """
        This function is used to prepare the inputs for continuous batching.
          *  For NewRequestData, it is guaranteed to be a prefill request.
          *  For CachedRequestData, it is guaranteed to be a decoding request.
        """
        data = IntermediateInputData()
        is_prefill = False
        for request_data in scheduler_output.scheduled_new_reqs:
            self._process_new_request_for_continuous_batching(request_data, data)
            is_prefill = True

        cached_request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_request_data.req_ids):
            self._process_cached_request_for_continuous_batching(
                cached_request_data, i, data
            )

        return data, is_prefill

    def _process_new_request_for_continuous_batching(
        self, request_data: NewRequestData, data: IntermediateInputData
    ) -> None:
        # Assign a free sequence id to the new request.
        assert request_data.req_id not in self.vllm_req_to_neuron_seq_id_mapping, (
            "Encountered an existing request ID " "while prefilling a new request"
        )
        assert self.free_seq_ids, "No free sequence ID available!"
        assigned_slot = self.free_seq_ids.pop()
        self.vllm_req_to_neuron_seq_id_mapping[request_data.req_id] = assigned_slot

        data.request_ids.append(request_data.req_id)
        data.input_tokens.append(request_data.prompt_token_ids)
        data.start_position_ids.append(0)
        data.full_context_lens.append(len(request_data.prompt_token_ids))
        data.input_block_ids.append(assigned_slot)

    def _process_cached_request_for_continuous_batching(
        self, request_data: CachedRequestData, index: int, data: IntermediateInputData
    ) -> None:

        req_id = request_data.req_ids[index]
        assert req_id in self.vllm_req_to_neuron_seq_id_mapping, (
            "The request ID for the current decode request "
            " is not found in request to sequence ID "
            "mapping"
        )
        state = self.requests[req_id]
        data.request_ids.append(req_id)
        data.input_tokens.append([state.output_token_ids[-1]])
        data.start_position_ids.append(request_data.num_computed_tokens[index])
        data.full_context_lens.append(request_data.num_computed_tokens[index] + 1)
        data.input_block_ids.append(self.vllm_req_to_neuron_seq_id_mapping[req_id])

    def _finalize_continuous_batching_inputs(
        self,
        data: IntermediateInputData,
        is_prefill: bool,
    ) -> ModelInputForNeuron:
        if is_prefill:
            input_tokens = make_ndarray_with_pad(
                data.input_tokens,
                pad=0,
                dtype=np.uint32,
                max_len=self.model.max_model_len,
            )
        else:
            input_tokens = make_ndarray_with_pad(
                data.input_tokens,
                pad=0,
                dtype=np.uint32,
                max_len=1,
            )
        input_block_ids = np.array(data.input_block_ids, dtype=np.int32)
        start_position_ids = np.array(data.start_position_ids, dtype=np.int32)
        full_context_lens = np.array(data.full_context_lens, dtype=np.int32)
        return ModelInputForNeuron(
            request_ids=data.request_ids,
            input_tokens=input_tokens,
            start_position_ids=start_position_ids,
            input_block_ids=input_block_ids,
            full_context_lens=full_context_lens,
            is_prefill=is_prefill,
        )

    def _sample(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelInputForNeuron,
    ):
        # The following logic reorders the model output to match the incoming request order
        # First obtain the order of requests processed by Neuron hardware
        request_id_order = {
            request_id: idx for idx, request_id in enumerate(model_input.request_ids)
        }

        # Identify the correct indices for each request in the original input batch based on request ids
        reorder_indices = np.array(
            [request_id_order[request_id] for request_id in self.input_batch.req_ids]
        )

        # Reorder along the batch dimension to restore outputs into the original request order
        hidden_states = hidden_states[reorder_indices]

        # Sample the next token.
        output = self.model.sample(logits=hidden_states)
        return output
