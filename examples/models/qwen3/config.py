from dataclasses import dataclass

import numpy as np
import torch.distributed as dist
from neuronxcc.nki.language import bfloat16
from transformers import AutoConfig

# to control compiler_args
DTYPE = bfloat16


@dataclass
class Config:
    hidden_size: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    num_layers: int
    num_experts_per_tok: int
    num_experts: int
    context_len: int = None
    max_new_tokens: int = None
    max_batch_size: int = 1
    norm_eps: float = 1e-6
    intermediate_size: int = 1536
    max_seq_len: int = 4096
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"


def get_config(model_name, context_len, max_new_tokens):
    hf_config = AutoConfig.from_pretrained(model_name)
    config = Config(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.moe_intermediate_size // dist.get_world_size(),
        num_heads=hf_config.num_attention_heads,
        head_dim=hf_config.head_dim,
        num_kv_heads=hf_config.num_key_value_heads,
        norm_eps=hf_config.rms_norm_eps,
        num_layers=hf_config.num_hidden_layers,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        num_experts=hf_config.num_experts,
        context_len=context_len,
        max_new_tokens=max_new_tokens,
    )
    return config
