# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
from typing import Optional

import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def _apply_rotary_emb_neuron(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # x1 = x[..., ::2]

        # x2 = x[..., 1::2]
        d = x.shape[-1] // 2
        x_reshaped = x.view(-1, x.shape[-1])
        x1 = x_reshaped[:, ::2].view(*x.shape[:-1], d)
        x2 = x_reshaped[:, 1::2].view(*x.shape[:-1], d)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def custom_rotary_embedding(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if offsets is not None:
        positions = positions + offsets

    # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
    # is expensive, so avoid calling it if possible
    # also see error "has no store def" in compiler
    if (
        self.cos_sin_cache.device != query.device
        or self.cos_sin_cache.dtype != query.dtype
    ):
        self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

    query = query.contiguous()
    if key is not None:
        key = key.contiguous()

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)

    if self.rotary_dim == self.head_size:
        query = _apply_rotary_emb_neuron(query, cos, sin, self.is_neox_style)
        query = query.reshape(query_shape)
        if key is not None:
            key = _apply_rotary_emb_neuron(key, cos, sin, self.is_neox_style)
            key = key.reshape(key_shape)
    else:
        head_size = query.shape[-1]
        query_reshaped = query.view(-1, head_size)
        query_pass = query_reshaped[:, self.rotary_dim :].view(
            *query.shape[:-1], head_size - self.rotary_dim
        )
        query_rot = query_reshaped[:, : self.rotary_dim].view(
            *query.shape[:-1], self.rotary_dim
        )
        query_rot = _apply_rotary_emb_neuron(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        if key is not None:
            key_reshaped = key.view(-1, head_size)
            key_pass = key_reshaped[:, self.rotary_dim :].view(
                *key.shape[:-1], head_size - self.rotary_dim
            )
            key_rot = key_reshaped[:, : self.rotary_dim].view(
                *key.shape[:-1], self.rotary_dim
            )
            key_rot = _apply_rotary_emb_neuron(key_rot, cos, sin, self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


RotaryEmbedding.forward_oot = custom_rotary_embedding
