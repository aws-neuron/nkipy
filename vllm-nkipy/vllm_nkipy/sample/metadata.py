# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# Adapted from vLLM (https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/tpu/metadata.py)
from dataclasses import dataclass, field
from typing import Optional

import torch
from vllm.logger import init_logger
from vllm.v1.worker.tpu_input_batch import InputBatch

logger = init_logger(__name__)

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    top_k=0,
    top_p=1.0,
    # Any other knobs are NOT SUPPORTED (ignore)
)


@dataclass
class NKIPySamplingMetadata:
    temperature: torch.Tensor = None
    min_p: torch.Tensor = None  # not supported (ignore)
    top_k: torch.Tensor = None
    top_p: torch.Tensor = None

    all_greedy: bool = True
    logprobs: bool = False  # not supported (warn + ignore)

    # The rest are carried for compatibility, will be ignored
    no_penalties: bool = True
    prompt_token_ids = None
    frequency_penalties = None
    presence_penalties = None
    repetition_penalties = None
    # should use tensor
    output_token_ids: list[list[int]] = field(default_factory=lambda: list())
    min_tokens = None  # impl is not vectorized
    logit_bias: list[Optional[dict[int, float]]] = field(default_factory=lambda: list())
    allowed_token_ids_mask = None
    bad_words_token_ids = None

    # Generator not supported
    _generators: dict[int, torch.Generator] = field(default_factory=lambda: dict())

    @property
    def generators(self) -> dict[int, torch.Generator]:
        # Generator not supported
        return self._generators

    @classmethod
    def from_input_batch(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        device: torch.device,
        vocab_size: int,
        generate_params_if_all_greedy: bool = False,
    ) -> "NKIPySamplingMetadata":
        """
        TPU-style construction:
        - Pad CPU tensors to fixed size (padded_num_reqs).
        - Sanitize supported knobs:
            * temperature < 0  -> clamp to 0 (greedy) for ACTIVE rows
            * top_k < 0        -> 0 (disabled)
            * top_k > 256      -> 256 (cap)
            * top_p <=0 or >1  -> 1.0 (disabled)
        - Move to device once in fixed shape to avoid recompiles.

        If input_batch.all_greedy and not generate_params_if_all_greedy:
            return early with tensors=None to use the “fast greedy” graph.
        """
        needs_logprobs = bool(getattr(input_batch, "max_num_logprobs", 0))
        if needs_logprobs:
            logger.warning(
                "NKIPy: logprobs requested (max_num_logprobs=%r) "
                "but not supported; ignoring.",
                getattr(input_batch, "max_num_logprobs", None),
            )

        # Early return
        if input_batch.all_greedy is True and generate_params_if_all_greedy is False:
            return cls(all_greedy=True, logprobs=needs_logprobs)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_tensor: torch.Tensor, fill_val) -> torch.Tensor:
            # Pad value is the default one.
            cpu_tensor[num_reqs:padded_num_reqs] = fill_val

        fill_slice(
            input_batch.temperature_cpu_tensor, DEFAULT_SAMPLING_PARAMS["temperature"]
        )
        fill_slice(input_batch.top_k_cpu_tensor, vocab_size)
        fill_slice(input_batch.top_p_cpu_tensor, DEFAULT_SAMPLING_PARAMS["top_p"])

        t_cpu = input_batch.temperature_cpu_tensor[:padded_num_reqs]
        k_cpu = input_batch.top_k_cpu_tensor[:padded_num_reqs]
        p_cpu = input_batch.top_p_cpu_tensor[:padded_num_reqs]

        # # Unsupported: min_p — warn if any active row uses non-default.
        # if hasattr(input_batch, "min_p_cpu_tensor"):
        #     active_min_p = input_batch.min_p_cpu_tensor[:num_reqs]
        #     if bool((active_min_p != 0.0).any()):
        #         logger.warning("NKIPy: 'min_p' is not supported and will be ignored.")  # noqa

        # ---- Move to device (fixed padded shape) ----
        return cls(
            temperature=t_cpu.to(device),
            top_p=p_cpu.to(device),
            top_k=k_cpu.to(device),
            all_greedy=input_batch.all_greedy,
            logprobs=needs_logprobs,
        )


# def _warn_unsupported_fields(input_batch: InputBatch) -> None:
#     """Best-effort warnings for knobs we do NOT support."""
#     unsupported_truthy_attrs = (
#         "allowed_token_ids_mask",
#         "bad_words_token_ids",
#         "logit_bias",
#         "frequency_penalties",
#         "presence_penalties",
#         "repetition_penalties",
#         "min_tokens",
#         "output_token_ids",
#     )
#     for name in unsupported_truthy_attrs:
#         val = getattr(input_batch, name, None)
#         if val is not None:
#             logger.warning("NKIPy: '%s' is not supported and will be ignored.", name)  # noqa
