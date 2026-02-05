import math
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
    n_heads: int
    head_dim: int
    n_kv_heads: int
    n_rep: int
    n_layers: int
    num_experts_per_tok: int
    num_experts: int
    num_static_blocks: int = None
    context_len: int = None
    num_blocks: int = None
    max_new_tokens: int = None
    max_batch_size: int = 1
    norm_eps: float = 1e-6
    intermediate_size: int = 1536
    num_blocks_per_launch: int = 1
    max_seq_len: int = 4096
    batch_size: int = 1
    block_size: int = 128
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"


def get_n_blocks(T, TOPK, E, B, n_block_per_iter=1):
    N = math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    N = n_block_per_iter * math.ceil(N / n_block_per_iter)
    return N


def get_config(model_name, context_len, max_new_tokens):
    hf_config = AutoConfig.from_pretrained(model_name)
    config = Config(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.moe_intermediate_size // dist.get_world_size(),
        n_heads=hf_config.num_attention_heads,
        head_dim=hf_config.head_dim,
        n_kv_heads=hf_config.num_key_value_heads,
        n_rep=hf_config.num_attention_heads // hf_config.num_key_value_heads,
        norm_eps=hf_config.rms_norm_eps,
        n_layers=hf_config.num_hidden_layers,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        num_experts=hf_config.num_experts,
        context_len=context_len,
    )
    config.max_new_tokens = max_new_tokens
    config.num_blocks = get_n_blocks(
        context_len, config.num_experts_per_tok, config.num_experts, Config.block_size
    )
    config.num_static_blocks = config.num_blocks - (config.num_experts - 1)
    # at least 1 static block to prevent not modify output
    config.num_static_blocks = max(config.num_static_blocks, 1)
    return config
