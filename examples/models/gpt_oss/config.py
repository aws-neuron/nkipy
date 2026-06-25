from dataclasses import dataclass

import numpy as np
import torch.distributed as dist
from neuronxcc.nki.language import bfloat16
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

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
    # RoPE (YaRN) inverse frequencies and post-scaling, precomputed from HF.
    rope_inv_freq: np.ndarray
    rope_attention_scaling: float
    # Per-layer attention type: "sliding_attention" or "full_attention".
    layer_types: list
    sliding_window: int
    # Clamped-SwiGLU parameters (gpt-oss specific).
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    context_len: int = None
    max_new_tokens: int = None
    max_batch_size: int = 1
    norm_eps: float = 1e-5
    intermediate_size: int = 2880
    max_seq_len: int = 4096
    dtype: np.dtype = DTYPE
    additional_compiler_args_nkipy: str = "--lnc 1"

    def is_sliding(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == "sliding_attention"


def get_config(model_name, context_len, max_new_tokens):
    hf_config = AutoConfig.from_pretrained(model_name)

    # YaRN RoPE: precompute inverse frequencies + attention scaling factor once.
    # These are constants (independent of runtime tensors), so we bake them into
    # the kernel's cos/sin cache at compile time.
    rope_init_fn = ROPE_INIT_FUNCTIONS[hf_config.rope_parameters["rope_type"]]
    inv_freq, attention_scaling = rope_init_fn(hf_config, device=None)

    config = Config(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size // dist.get_world_size(),
        num_heads=hf_config.num_attention_heads,
        head_dim=hf_config.head_dim,
        num_kv_heads=hf_config.num_key_value_heads,
        norm_eps=hf_config.rms_norm_eps,
        num_layers=hf_config.num_hidden_layers,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        num_experts=hf_config.num_local_experts,
        rope_inv_freq=np.asarray(inv_freq, dtype=np.float32),
        rope_attention_scaling=float(attention_scaling),
        layer_types=list(hf_config.layer_types),
        sliding_window=hf_config.sliding_window,
        swiglu_alpha=getattr(hf_config, "swiglu_alpha", 1.702),
        swiglu_limit=hf_config.swiglu_limit,
        context_len=context_len,
        max_new_tokens=max_new_tokens,
    )
    return config
