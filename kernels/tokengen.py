import collective as cc
import nkipy.core.typing as nt
import numpy as np

# Import config from parent directory
from config import Config
from kernels.blockwise_index import BLOCK_SIZE, ControlType
from nkipy.core import tensor_apis
from nkipy.core.nki_op import wrap_nki_kernel
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
from neuronxcc.starfish.penguin.native_maths import gelu_apprx_sigmoid
from parallel_state import get_world_group

# Import kernels from the kernels directory
from .attention import attention_module
from .blockwise_nki import blockwise_nki_tokengen_one_tile_replicated_hidden_state
from .blockwise_np import blockwise_np
from .rmsnorm import rmsnorm
from .rmsnorm_nki import rmsnorm as rmsnorm_nki
from .router import router_tokengen, expert_affinities_slice, router
from .fused_rank_slice_nki import fused_rank_slice_add


def swiglu(
    x: nt.tensor,
    is_neuronpy: bool,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    x_glu, x_linear = np.split(x, 2, axis=-1)
    if is_neuronpy:
        x_glu = np.where(np.greater(x_glu, limit), limit, x_glu)
        x_linear = np.where(np.greater(x_linear, limit), limit, x_linear)
        x_linear = np.where(np.less(x_linear, -limit), -limit, x_linear)
    else:
        x_glu = np.clip(x_glu, a_min=None, a_max=limit)
        x_linear = np.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = gelu_apprx_sigmoid(x_glu).astype(x.dtype)
    # Note we add an extra bias of 1 to the linear layer
    # TODO: move add out of linear layer
    return out_glu * (x_linear + 1)


def tokengen_moe(
    hidden_states,  # [batch_size, hidden_size]
    post_attention_weight,
    router_weight,
    router_bias,
    gate_up_weight,
    gate_up_bias_plus1_T,
    down_weight,
    down_bias_broadcasted,
    ep_rank: nt.tensor,  # [1]
    ep_size: int,
    config,
    is_neuronpy,
):
    assert ep_rank.dtype == np.int32
    batch_size = hidden_states.shape[0]
    # FIXME: add blockwise index to support larger batch size
    assert batch_size <= 128
    residual = hidden_states

    # # RMSNORM NKI Version: worse performance due to IO tensor on HBM
    # rmsnorm_nki_op = wrap_nki_kernel(
    #     rmsnorm_nki,
    #     [
    #         np.empty(hidden_states.shape, dtype=hidden_states.dtype),
    #         np.empty(post_attention_weight.shape, dtype=post_attention_weight.dtype),
    #         config.norm_eps,
    #     ],
    # )
    # hidden_states = rmsnorm_nki_op(np.copy(hidden_states), post_attention_weight)
    hidden_states = rmsnorm(
        hidden_states,
        post_attention_weight,
        config.norm_eps,
        is_neuronpy=is_neuronpy,
    )
    expert_affinities_masked = router_tokengen(
        hidden_states_sharded=hidden_states,
        router_weight=router_weight,
        router_bias=router_bias,
        top_k=config.num_experts_per_tok,
        is_neuronpy=is_neuronpy,
    )

    expert_affinities_masked = expert_affinities_slice(
        expert_affinities_masked_all_experts=expert_affinities_masked,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    output = np.zeros_like(hidden_states)
    n_experts_per_ep = config.num_experts//ep_size

    if is_neuronpy:
        token_position_to_id = tensor_apis.full(
            (1, BLOCK_SIZE), ControlType.SKIP_DMA.value, dtype=np.int32
        )
        token_position_to_id[0, :batch_size] = np.arange(batch_size).astype(np.int32) + tensor_apis.zeros((batch_size,), dtype=np.int32)
        block_to_expert = np.arange(n_experts_per_ep).astype(
            np.int8
        ) + tensor_apis.zeros((n_experts_per_ep,), dtype=np.int8)
    else:
        token_position_to_id = np.full(
            (1, BLOCK_SIZE), ControlType.SKIP_DMA.value, dtype=np.int32
        )
        token_position_to_id[0, :batch_size] = np.arange(batch_size)
        block_to_expert = np.arange(n_experts_per_ep).astype(np.int8)
    token_position_to_id = np.broadcast_to(token_position_to_id, (n_experts_per_ep, BLOCK_SIZE))

    if is_neuronpy:
        expert_affinities_masked_T = expert_affinities_masked.transpose()
        nki_op = wrap_nki_kernel(
            blockwise_nki_tokengen_one_tile_replicated_hidden_state,
            [
                hidden_states,
                expert_affinities_masked_T,
                gate_up_weight,
                gate_up_bias_plus1_T,
                down_weight,
                down_bias_broadcasted,
                token_position_to_id,
                block_to_expert,
            ],
        )
        output = nki_op(
            hidden_states,
            expert_affinities_masked_T,
            gate_up_weight,
            gate_up_bias_plus1_T,
            down_weight,
            down_bias_broadcasted,
            token_position_to_id,
            block_to_expert,
        )
    else:
        output = blockwise_np(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            down_proj_weight=down_weight,
            gate_up_proj_weight=gate_up_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            dtype=config.dtype,
            gate_up_bias_plus1_T=gate_up_bias_plus1_T,
            down_bias_broadcasted=down_bias_broadcasted,
            activation_function=ActFnType.Swish,
        )

    output = cc.all_reduce(
        output,
        replica_groups=get_world_group(),
        is_neuronpy=is_neuronpy,
    )

    assert batch_size == ep_size * config.max_batch_size_per_dp, (
        f"{batch_size=} {ep_size=} {config.max_batch_size_per_dp}"
    )

    output = fused_rank_slice_add(output, residual, ep_rank, config.max_batch_size_per_dp)
    return output

def tokengen(
    hidden_states,
    start_pos,
    # weights
    qkv_weight,
    o_weight,
    input_weight,
    # kv cache
    cache_k: nt.mutable_tensor,
    cache_v: nt.mutable_tensor,
    cos,
    sin,
    post_attention_weight: nt.tensor,  # RMSNorm weights
    router_weight: nt.tensor,  # Router weights [hidden_dim, n_experts]
    router_bias: nt.tensor,
    gate_up_weight: nt.tensor,  # Expert gate_up weights [n_experts, hidden_dim, 2*ffn_dim]
    gate_up_bias_plus1_T: nt.tensor,
    down_weight: nt.tensor,  # Expert down weights [n_experts, ffn_dim, hidden_dim]
    down_bias_broadcasted: nt.tensor,
    config: Config,
    sliding_window: int,
    sink: nt.tensor,
    qkv_bias: nt.tensor,
    o_bias: nt.tensor,
    ep_rank: nt.tensor,  # [1]
    ep_size: int,
    is_neuronpy: bool,
):
    """Single layer token generation kernel."""
    attn_hidden_states, cache_k, cache_v = attention_module(
        hidden_states=hidden_states,
        input_weight=input_weight,
        qkv_weight=qkv_weight,
        cache_k=cache_k,
        cache_v=cache_v,
        o_weight=o_weight,
        sliding_window=sliding_window,
        sink=sink,
        qkv_bias=qkv_bias,
        o_bias=o_bias,
        start_pos=start_pos,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=hidden_states.dtype,
        is_neuronpy=is_neuronpy,
    )
    batch_size, seq_len, hidden_size = attn_hidden_states.shape
    assert seq_len == 1, f"{attn_hidden_states.shape=}"

    attn_hidden_states = attn_hidden_states.reshape(batch_size, hidden_size)

    final_output = tokengen_moe(
        hidden_states=attn_hidden_states,
        post_attention_weight=post_attention_weight,
        router_weight=router_weight,
        router_bias=router_bias,
        gate_up_weight=gate_up_weight,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T,
        down_weight=down_weight,
        down_bias_broadcasted=down_bias_broadcasted,
        ep_rank=ep_rank,
        ep_size=ep_size,
        config=config,
        is_neuronpy=is_neuronpy,
    )
    assert batch_size % ep_size == 0
    batch_size_per_ep = batch_size // ep_size
    final_output = final_output.reshape(batch_size_per_ep, 1, hidden_size)
    return final_output, cache_k, cache_v


def tokengen_fused_4layers(
    hidden_states,
    start_pos,
    # Layer 0 weights (even layer - sliding_window=128)
    qkv_weight_0,
    o_weight_0,
    input_weight_0,
    cache_k_0: nt.mutable_tensor,
    cache_v_0: nt.mutable_tensor,
    post_attention_weight_0,
    router_weight_0,
    router_bias_0,
    gate_up_weight_0,
    gate_up_bias_plus1_T_0,
    down_weight_0,
    down_bias_broadcasted_0,
    sink_0,
    qkv_bias_0,
    o_bias_0,
    # Layer 1 weights (odd layer - sliding_window=0)
    qkv_weight_1,
    o_weight_1,
    input_weight_1,
    cache_k_1: nt.mutable_tensor,
    cache_v_1: nt.mutable_tensor,
    post_attention_weight_1,
    router_weight_1,
    router_bias_1,
    gate_up_weight_1,
    gate_up_bias_plus1_T_1,
    down_weight_1,
    down_bias_broadcasted_1,
    sink_1,
    qkv_bias_1,
    o_bias_1,
    # Layer 2 weights (even layer - sliding_window=128)
    qkv_weight_2,
    o_weight_2,
    input_weight_2,
    cache_k_2: nt.mutable_tensor,
    cache_v_2: nt.mutable_tensor,
    post_attention_weight_2,
    router_weight_2,
    router_bias_2,
    gate_up_weight_2,
    gate_up_bias_plus1_T_2,
    down_weight_2,
    down_bias_broadcasted_2,
    sink_2,
    qkv_bias_2,
    o_bias_2,
    # Layer 3 weights (odd layer - sliding_window=0)
    qkv_weight_3,
    o_weight_3,
    input_weight_3,
    cache_k_3: nt.mutable_tensor,
    cache_v_3: nt.mutable_tensor,
    post_attention_weight_3,
    router_weight_3,
    router_bias_3,
    gate_up_weight_3,
    gate_up_bias_plus1_T_3,
    down_weight_3,
    down_bias_broadcasted_3,
    sink_3,
    qkv_bias_3,
    o_bias_3,
    # Shared tensors (not duplicated)
    cos,
    sin,
    ep_rank: nt.tensor,  # [1]
    ep_size: int,
    configs,  # Config instance
    is_neuronpy=True,
):
    """Fused 4-layer token generation kernel.
    Layer 0: even (sliding_window=128)
    Layer 1: odd (sliding_window=0)
    Layer 2: even (sliding_window=128)
    Layer 3: odd (sliding_window=0)
    """
    # Process through layer 0 (even - sliding_window=128)
    output_0, cache_k_0, cache_v_0 = tokengen(
        hidden_states=hidden_states,
        start_pos=start_pos,
        qkv_weight=qkv_weight_0,
        o_weight=o_weight_0,
        input_weight=input_weight_0,
        cache_k=cache_k_0,
        cache_v=cache_v_0,
        cos=cos,
        sin=sin,
        post_attention_weight=post_attention_weight_0,
        router_weight=router_weight_0,
        router_bias=router_bias_0,
        gate_up_weight=gate_up_weight_0,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T_0,
        down_weight=down_weight_0,
        down_bias_broadcasted=down_bias_broadcasted_0,
        config=configs,
        sliding_window=configs.sliding_window,  # 128 for even layer
        sink=sink_0,
        qkv_bias=qkv_bias_0,
        o_bias=o_bias_0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        is_neuronpy=is_neuronpy,
    )

    # Process through layer 1 (odd - sliding_window=0)
    output_1, cache_k_1, cache_v_1 = tokengen(
        hidden_states=output_0,
        start_pos=start_pos,
        qkv_weight=qkv_weight_1,
        o_weight=o_weight_1,
        input_weight=input_weight_1,
        cache_k=cache_k_1,
        cache_v=cache_v_1,
        cos=cos,
        sin=sin,
        post_attention_weight=post_attention_weight_1,
        router_weight=router_weight_1,
        router_bias=router_bias_1,
        gate_up_weight=gate_up_weight_1,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T_1,
        down_weight=down_weight_1,
        down_bias_broadcasted=down_bias_broadcasted_1,
        config=configs,
        sliding_window=0,  # 0 for odd layer
        sink=sink_1,
        qkv_bias=qkv_bias_1,
        o_bias=o_bias_1,
        ep_rank=ep_rank,
        ep_size=ep_size,
        is_neuronpy=is_neuronpy,
    )

    # Process through layer 2 (even - sliding_window=128)
    output_2, cache_k_2, cache_v_2 = tokengen(
        hidden_states=output_1,
        start_pos=start_pos,
        qkv_weight=qkv_weight_2,
        o_weight=o_weight_2,
        input_weight=input_weight_2,
        cache_k=cache_k_2,
        cache_v=cache_v_2,
        cos=cos,
        sin=sin,
        post_attention_weight=post_attention_weight_2,
        router_weight=router_weight_2,
        router_bias=router_bias_2,
        gate_up_weight=gate_up_weight_2,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T_2,
        down_weight=down_weight_2,
        down_bias_broadcasted=down_bias_broadcasted_2,
        config=configs,
        sliding_window=configs.sliding_window,  # 128 for even layer
        sink=sink_2,
        qkv_bias=qkv_bias_2,
        o_bias=o_bias_2,
        ep_rank=ep_rank,
        ep_size=ep_size,
        is_neuronpy=is_neuronpy,
    )

    # Process through layer 3 (odd - sliding_window=0)
    output_3, cache_k_3, cache_v_3 = tokengen(
        hidden_states=output_2,
        start_pos=start_pos,
        qkv_weight=qkv_weight_3,
        o_weight=o_weight_3,
        input_weight=input_weight_3,
        cache_k=cache_k_3,
        cache_v=cache_v_3,
        cos=cos,
        sin=sin,
        post_attention_weight=post_attention_weight_3,
        router_weight=router_weight_3,
        router_bias=router_bias_3,
        gate_up_weight=gate_up_weight_3,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T_3,
        down_weight=down_weight_3,
        down_bias_broadcasted=down_bias_broadcasted_3,
        config=configs,
        sliding_window=0,  # 0 for odd layer
        sink=sink_3,
        qkv_bias=qkv_bias_3,
        o_bias=o_bias_3,
        ep_rank=ep_rank,
        ep_size=ep_size,
        is_neuronpy=is_neuronpy,
    )

    return (
        output_3,
        cache_k_0,
        cache_v_0,
        cache_k_1,
        cache_v_1,
        cache_k_2,
        cache_v_2,
        cache_k_3,
        cache_v_3,
    )
