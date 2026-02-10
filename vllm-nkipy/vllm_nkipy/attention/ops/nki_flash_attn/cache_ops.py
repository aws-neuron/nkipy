# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# ruff: noqa

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import torch
from neuronxcc import nki
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (torch.Tensor): Key tensor with shape
            (num_tokens, n_kv_head, d_head)
        value (torch.Tensor): Value tensor with shape
            (num_tokens, n_kv_head, d_head)
        kv_cache (torch.Tensor): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (torch.Tensor): Mapping tensor indicating cache positions
            with shape (num_tokens)

    Returns:
        None: Updates the kv_cache tensor in-place
    """
    (num_tokens,) = slot_mapping.shape
    block_size = kv_cache.size(3)
    n_kv_head = key.size(1)

    # Calculate indices with explicit floor division
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor").to(
        dtype=torch.int32
    )
    block_offsets = (slot_mapping % block_size).to(dtype=torch.int32)

    # Create the head indices tensor
    head_indices = torch.arange(n_kv_head, device=key.device).to(dtype=torch.int32)

    # Update caches using index_put_
    kv_cache.index_put_(
        (
            torch.tensor([0], device=key.device, dtype=torch.int32),
            block_indices[:, None],
            head_indices[None, :],
            block_offsets[:, None],
        ),
        key,
    )

    kv_cache.index_put_(
        (
            torch.tensor([1], device=key.device, dtype=torch.int32),
            block_indices[:, None],
            head_indices[None, :],
            block_offsets[:, None],
        ),
        value,
    )
