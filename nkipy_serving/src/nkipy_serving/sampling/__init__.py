"""Sampling pipeline: params, device kernels, and logits processing.

Submodules:
  params               — SamplingParams validation and normalization
  batch_info           — SamplingBatchInfo (numpy backend)
  device_batch         — DeviceSamplingBatch (device backend)
  random_state         — stateless per-request RNG
  nki_kernels          — raw NKI sampling kernels (filtered/unfiltered)
  lm_head_sampling     — device entry points (greedy top-k, NKI CDF sampler)
  logits_processor     — LogitsProcessor: unified LM-head → sampling → logprobs
  logits_processor_np  — NumpyLogitsProcessor: CPU reference (accuracy baseline)
"""

from nkipy_serving.sampling.batch_info import SamplingBatchInfo
from nkipy_serving.sampling.constants import LOGPROBS_K_MAX
from nkipy_serving.sampling.params import (
    DEFAULT_SAMPLING_SEED,
    TOP_K_ALL,
    SamplingParams,
)

__all__ = [
    "DEFAULT_SAMPLING_SEED",
    "LOGPROBS_K_MAX",
    "SamplingBatchInfo",
    "SamplingParams",
    "TOP_K_ALL",
]
