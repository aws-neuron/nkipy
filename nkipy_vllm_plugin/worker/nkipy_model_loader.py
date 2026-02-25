# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch.nn as nn
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)


@dataclass(frozen=True)
class NeuronConfig:
    is_chunked_prefill: bool = False
    on_device_sampling_config: bool = True


class NeuronCausalLM(nn.Module):

    def __init__(self, model_config, checkpoint, is_reorder_needed) -> None:
        super().__init__()
        from model import load_gpt_oss_weights, GPTOSSModel
        weights = load_gpt_oss_weights(checkpoint, model_config)
        self.model = GPTOSSModel(weights, model_config)
        self.is_reorder_needed: bool = is_reorder_needed
        self.neuron_config = NeuronConfig()
        self.num_key_value_heads = model_config.n_kv_heads
        self.head_dim = model_config.head_dim
        self.max_model_len = model_config.max_model_len

    @contextmanager
    def _reordered(self, input_block_ids: npt.NDArray[np.int32], **tensor_inputs):
        """
        Context manager that yields reordered input_block_ids, inputs, and a restore function.
        Automatically restores output to original order if needed.

        [NOTE] This is MANADATORY for contiguous kv cache as it will impact the output accuracy.
        """
        logger.debug(f"is_reorder_needed: {self.is_reorder_needed}")
        if self.is_reorder_needed:
            sorted_indices = np.argsort(input_block_ids)
            sorted_ids = input_block_ids[sorted_indices]
            reordered_inputs = {
                k: (
                    np.take(v, sorted_indices, axis=0)
                    # having v.shape[0] > 0 to avoid reorder empty tensors
                    if isinstance(v, np.ndarray) and v.shape[0] > 0
                    else v
                )
                for k, v in tensor_inputs.items()
            }

            def restore(output: npt.NDArray) -> npt.NDArray:
                if sorted_ids.shape[0] != 1:
                    return np.take(output, np.argsort(sorted_indices), axis=0)
                return output

            yield sorted_ids, reordered_inputs, restore
        else:
            yield input_block_ids, tensor_inputs, lambda x: x

    def forward(self, input_block_ids, **kwargs):
        with self._reordered(input_block_ids, **kwargs) as (
            sorted_ids,
            inputs,
            restore,
        ):
            output = self.model.forward(
                input_ids_np=inputs["input_tokens"],
                start_pos_np=inputs["start_position_ids"],
                context_len_np=inputs["full_context_lens"],
                is_prefill=inputs["is_prefill"],
                warmup=False, # not warmup in vllm
            )

            if not self.neuron_config.on_device_sampling_config:
                if self.neuron_config.is_chunked_prefill:
                    assert kwargs.get("prefill_completion_state") is not None
                    idx_for_sampling = (
                        kwargs["prefill_completion_state"].nonzero().flatten()
                    )
                    output = output[0, idx_for_sampling, :]
                else:
                    output = output[:, -1, :]

            return restore(output)

    def sample(self, logits: npt.NDArray[np.int32]) -> Optional[SamplerOutput]:
        if self.neuron_config.on_device_sampling_config:
            return SamplerOutput(
                # The sampled tokens are expanded to 2D tensor with shape
                # [num_requests, 1], where each row represents one generated
                # token per request.
                sampled_token_ids=logits,
                logprobs_tensors=None,
            )
        else:
            raise NotImplementedError("CPU sampler not implemented")


def get_neuron_model(model_config, checkpoint, is_reorder_needed=False) -> nn.Module:
    model = NeuronCausalLM(model_config, checkpoint, is_reorder_needed)
    return model.eval()
