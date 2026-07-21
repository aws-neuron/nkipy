from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch.distributed as dist
from neuronxcc.nki.language import bfloat16
from transformers import AutoConfig

DTYPE = bfloat16

# Layer types for Qwen3.5 hybrid architecture
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


@dataclass
class Config:
    hidden_size: int
    num_heads: int  # full attention Q heads
    head_dim: int  # full attention head dim
    num_kv_heads: int  # full attention KV heads
    num_layers: int
    num_experts_per_tok: int
    num_experts: int
    intermediate_size: int  # moe expert intermediate (per device)
    shared_expert_intermediate_size: int  # shared expert intermediate (per device)
    vocab_size: int
    # Linear attention params
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # Layer type info
    layer_types: List[str] = field(default_factory=list)
    # RoPE
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    # Sequence
    context_len: int = None
    max_new_tokens: int = None
    max_batch_size: int = 1
    max_seq_len: int = 4096
    # Norm
    norm_eps: float = 1e-6
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"


def get_config(model_name, context_len, max_new_tokens):
    hf_config = AutoConfig.from_pretrained(model_name)
    # Qwen3.5 is multimodal; text config is nested
    text_cfg = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

    ws = dist.get_world_size()
    config = Config(
        hidden_size=text_cfg.hidden_size,
        num_heads=text_cfg.num_attention_heads,
        head_dim=text_cfg.head_dim,
        num_kv_heads=text_cfg.num_key_value_heads,
        num_layers=text_cfg.num_hidden_layers,
        num_experts_per_tok=text_cfg.num_experts_per_tok,
        num_experts=text_cfg.num_experts,
        intermediate_size=text_cfg.moe_intermediate_size // ws,
        shared_expert_intermediate_size=text_cfg.shared_expert_intermediate_size // ws,
        vocab_size=text_cfg.vocab_size,
        linear_num_key_heads=text_cfg.linear_num_key_heads,
        linear_num_value_heads=text_cfg.linear_num_value_heads,
        linear_key_head_dim=text_cfg.linear_key_head_dim,
        linear_value_head_dim=text_cfg.linear_value_head_dim,
        linear_conv_kernel_dim=text_cfg.linear_conv_kernel_dim,
        layer_types=list(text_cfg.layer_types),
        partial_rotary_factor=text_cfg.partial_rotary_factor,
        rope_theta=text_cfg.rope_parameters.get("rope_theta", 10000000.0),
        norm_eps=text_cfg.rms_norm_eps,
        context_len=context_len,
        max_new_tokens=max_new_tokens,
    )
    return config
