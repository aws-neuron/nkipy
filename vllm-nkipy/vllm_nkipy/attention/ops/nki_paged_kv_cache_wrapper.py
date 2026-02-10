# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch

from vllm_nkipy.attention.ops.nki_paged_kv_cache import update_kv_cache


@torch.library.custom_op("mylib::update_kv_cache_custom_op", mutates_args=())
def update_kv_cache_custom_op(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    return update_kv_cache(key, value, kv_cache, slot_mapping)


@update_kv_cache_custom_op.register_fake
def _(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(kv_cache)


# def update_kv_cache_custom_op_nkifunc(
#     key: torch.Tensor,
#     value: torch.Tensor,
#     kv_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
# ) -> torch.Tensor:
#     _, NB, n_kv_head, BS, d_head = kv_cache.shape
#     num_tokens, n_kv_head, d_head = key.shape
#     assert value.shape == (num_tokens, n_kv_head, d_head)
#     assert n_kv_head == 1
#     kv_cache = kv_cache.reshape(2, NB, BS, d_head)
#     key = key.reshape(num_tokens, d_head)
#     value = value.reshape(num_tokens, d_head)
#     kv_cache = update_kv_cache_custom_op(key, value, kv_cache, slot_mapping)
#     kv_cache = kv_cache.reshape(2, NB, 1, BS, d_head)
#     return kv_cache
