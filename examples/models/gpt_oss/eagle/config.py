"""Configuration for the P-EAGLE parallel-drafting drafter.

The drafter is a small Llama-style model trained for a specific gpt-oss target.
It generates K draft tokens in a single forward pass (see the project memory and
arXiv 2602.01469). Its structure, from the checkpoint:

  * ``midlayer`` - the EAGLE-3 fusion decoder layer (layer 0). Its attention
    projections take 2*hidden (embedding concat hidden), and it owns the extra
    ``hidden_norm``.
  * ``layers.1 .. layers.{N-1}`` - plain Llama decoder layers.
  * ``fc`` - fuses the 3 concatenated target hidden states (3*hidden) -> hidden.
  * ``mask_hidden`` - learnable shared hidden state for MTP (depth>0) positions.
  * ``ptd_token_id`` - placeholder token id whose embedding substitutes the
    unknown previous token at MTP positions.
  * ``d2t`` / ``t2d`` - draft<->target vocab maps (identity for this checkpoint).
"""

from dataclasses import dataclass

import numpy as np
import torch.distributed as dist
from neuronxcc.nki.language import bfloat16
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

DTYPE = bfloat16


@dataclass
class EagleConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    num_layers: int  # total drafter decoder layers (incl. the fusion midlayer)
    intermediate_size: int
    # llama3 RoPE.
    rope_inv_freq: np.ndarray
    rope_attention_scaling: float
    # Target-model hidden size whose 3 tapped layers feed `fc` (3*target_hidden).
    target_hidden_size: int
    # Draft / target vocab sizes (equal for this checkpoint).
    draft_vocab_size: int
    target_vocab_size: int
    # Placeholder token id used at MTP (depth>0) positions.
    ptd_token_id: int
    # Number of draft tokens produced per parallel forward pass.
    num_draft_tokens: int = 7
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    max_batch_size: int = 1
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"


def get_eagle_config(
    draft_model_name,
    target_hidden_size,
    num_draft_tokens=7,
    max_seq_len=4096,
):
    hf = AutoConfig.from_pretrained(draft_model_name)

    # llama3 RoPE: precompute inverse frequencies + attention scaling once.
    rope_init_fn = ROPE_INIT_FUNCTIONS[hf.rope_scaling["rope_type"]]
    inv_freq, attention_scaling = rope_init_fn(hf, device=None)

    return EagleConfig(
        hidden_size=hf.hidden_size,
        num_heads=hf.num_attention_heads,
        head_dim=hf.head_dim,
        num_kv_heads=hf.num_key_value_heads,
        num_layers=hf.num_hidden_layers,
        intermediate_size=hf.intermediate_size // dist.get_world_size(),
        rope_inv_freq=np.asarray(inv_freq, dtype=np.float32),
        rope_attention_scaling=float(attention_scaling),
        target_hidden_size=target_hidden_size,
        draft_vocab_size=hf.draft_vocab_size,
        target_vocab_size=hf.vocab_size,
        ptd_token_id=getattr(hf, "ptd_token_id"),
        num_draft_tokens=num_draft_tokens,
        norm_eps=hf.rms_norm_eps,
        max_seq_len=max_seq_len,
    )
