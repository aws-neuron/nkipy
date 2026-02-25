import os
from dataclasses import dataclass

import numpy as np
import parallel_state
from kernels.blockwise_index import MAX_N_BLOCKS, MAX_N_EXPERTS
from neuronxcc.nki.language import bfloat16
from parallel_state import get_tp_size

@dataclass
class Config:
    hidden_size: int = 2880
    n_heads: int = 64
    head_dim: int = 64
    n_kv_heads: int = 8
    n_layers: int = 36 # 1 24
    num_experts_per_tok: int = 4
    num_experts: int = 128  # 32
    num_blocks: int = None
    num_static_blocks: int = None
    norm_eps: float = 1e-5
    intermediate_size: int = 2880

    # one bucket
    max_batch_size_per_dp: int = None
    max_model_len: int = None
    # moe
    num_blocks_per_launch: int = 20
    dtype: np.dtype = bfloat16 # we mainly live in numpy space
    
    # Additional fields for model configuration
    vocab_size: int = 201088
    swiglu_limit: float = 7.0
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: int = 150000
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


MAX_NUM_TOKENS_ALLOWED_IN_BLOCKWISE_INDEX = 32768  # int16

def set_env():
    # disable warning
    os.environ["NCCL_DEBUG"] = "ERROR" # warning: this may prevent necessary warning messages

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # disable warning
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "16"

    # cache dma ring for static dma, need to be n_layers because unique number of io tensor sets
    # FIXME: nkipy uses dge by default, but some neff still uses static dma
    # FIXME: still see nrt_dma_mem_dealloc in runtime profile
    os.environ["NEURON_RT_IO_RING_CACHE_SIZE"] = "36"

    # enable profile contains dge var. May slow down.
    # os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"

    # increase scatch pad to fit all gather/reduce scatter tensor.
    # os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] = "1024"

def get_config(model_name, **kwargs) -> Config:
    from kernels.blockwise_index import get_n_blocks  # fix circular import

    # hf_config = AutoConfig.from_pretrained(model_name)
    # config = Config(
    #     hidden_size=hf_config.hidden_size,
    #     intermediate_size=hf_config.moe_intermediate_size // get_tp_size(),
    #     n_heads=hf_config.num_attention_heads,
    #     head_dim=hf_config.head_dim,
    #     n_kv_heads=hf_config.num_key_value_heads,
    #     norm_eps=hf_config.rms_norm_eps,
    #     n_layers=hf_config.num_hidden_layers,
    #     num_experts_per_tok=hf_config.num_experts_per_tok,
    #     num_experts=hf_config.num_experts,
    #     num_blocks=Config.num_static_blocks,
    #     batch_size=1,
    # )
    config = Config(**kwargs)
    assert config.max_model_len <= MAX_NUM_TOKENS_ALLOWED_IN_BLOCKWISE_INDEX
    assert config.num_experts % parallel_state.get_prefill_ep_size() == 0
    assert config.num_experts % parallel_state.get_decode_ep_size() == 0
    assert config.max_model_len % get_tp_size() == 0
    assert (config.max_model_len // get_tp_size()) % 32 == 0, (
        "To reduce compiler error, make max_model_len_sharded divisible by 32"
    ) 
    config.num_blocks, config.num_static_blocks = get_n_blocks(
        config.max_batch_size_per_dp * config.max_model_len,
        config.num_experts_per_tok,
        config.num_experts,
    )
    assert config.num_experts <= MAX_N_EXPERTS, f"num_experts: {config.num_experts} > MAX_N_EXPERTS: {MAX_N_EXPERTS}"
    assert config.num_blocks <= MAX_N_BLOCKS, f"num_blocks: {config.num_blocks} > MAX_N_BLOCKS: {MAX_N_BLOCKS}"
    # at least 1 static block to prevent not modify output
    config.num_static_blocks = max(config.num_static_blocks, 1)
    return config
