"""
Test module for tokengen kernel implementation.
"""

import ml_dtypes
import numpy as np
import pytest
import torch
from config import Config
from convert_torch_numpy_dtype import numpy_to_torch_type
from nkipy.runtime import baremetal_jit
from kernels.tokengen import attention_module, tokengen, tokengen_moe
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import TILE_SIZE
from reference import MLPBlock
from test_attention_prefill import setup_attention_test
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def setup_tokengen_test(sliding_window, dtype=bfloat16):
    (
        config,
        x_np,
        _,
        _,
        _,
        qkv_weight,
        o_weight,
        qkv_bias,
        o_bias,
        sink,
        ref_norm_scale,
        cache_k,
        cache_v,
        cos,
        sin,
        seq_len,
        ref_transformer_block,
    ) = setup_attention_test(sliding_window, for_tokengen=True, dtype=dtype)

    with torch.no_grad():
        # random init
        x_np = np.concatenate(
            [x_np, np.random.randn(1, 1, x_np.shape[2]).astype(config.dtype)], axis=1
        )
        torch_dtype = numpy_to_torch_type[dtype]
        ref_output, ref_k, ref_v, ref_attn_hidden_states = ref_transformer_block(
            torch.from_numpy(x_np[0].astype(np.float32)).to(dtype=torch_dtype)
        )
        ref_output = ref_output.float().numpy().astype(config.dtype)
        ref_k = ref_k.float().numpy().astype(config.dtype)
        ref_v = ref_v.float().numpy().astype(config.dtype)
        ref_attn_hidden_states = ref_attn_hidden_states.float().numpy().astype(config.dtype)
    # use same kvcache
    cache_k[:, :seq_len] = ref_k[:seq_len]
    cache_v[:, :seq_len] = ref_v[:seq_len]
    start_pos = np.array([seq_len], dtype=np.int32)
    post_attention_weight = ref_transformer_block.mlp.norm.scale.data.float().numpy().astype(config.dtype)
    router_weight = (
        ref_transformer_block.mlp.gate.weight.data.T.float()
        .numpy()
        .astype(config.dtype)
    )  # Transpose to match our format
    router_bias = ref_transformer_block.mlp.gate.bias.data.float().numpy().astype(config.dtype)
    mlp1_weight = ref_transformer_block.mlp.mlp1_weight.data.float().numpy().astype(config.dtype)  # [num_experts, intermediate_size*2, hidden_size]
    gate_up_bias = ref_transformer_block.mlp.mlp1_bias.data.float().numpy().astype(config.dtype).reshape(config.num_experts, 2, config.intermediate_size)
    mlp2_weight = ref_transformer_block.mlp.mlp2_weight.data.float().numpy().astype(config.dtype)  # [num_experts, hidden_size, intermediate_size]
    down_bias = ref_transformer_block.mlp.mlp2_bias.data.float().numpy().astype(config.dtype)
    
    gate_up_weight = mlp1_weight.transpose(0, 2, 1).reshape(
            config.num_experts,
            config.hidden_size,
            2, 
            config.intermediate_size,
        )
    down_weight = mlp2_weight.transpose(0, 2, 1)
    
    return (
        config,
        x_np[:, seq_len : seq_len + 1, :],
        ref_output[seq_len],
        ref_norm_scale,
        qkv_weight,
        o_weight,
        qkv_bias,
        o_bias,
        sink,
        post_attention_weight,
        router_weight,
        router_bias,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        start_pos,
        cache_k,
        cache_v,
        cos,
        sin,
        ref_k,
        ref_v,
        ref_attn_hidden_states[seq_len],
        seq_len,
    )


@pytest.mark.skip
@pytest.mark.parametrize(
    "sliding_window",
    [
        (0),
        (128),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "device",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        bfloat16,
    ],
)
def test_tokengen_layer(sliding_window, device, dtype):    
    (
        config,
        x_np,
        ref_output,
        ref_norm_scale,
        qkv_weight,
        o_weight,
        qkv_bias,
        o_bias,
        sink,
        post_attention_weight,
        router_weight,
        router_bias,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        start_pos,
        cache_k,
        cache_v,
        cos,
        sin,
        ref_k,
        ref_v,
        ref_attn_hidden_states,
        seq_len,
    ) = setup_tokengen_test(sliding_window, dtype=dtype)

    if device == "cpu":
        output, cache_k_out, cache_v_out = tokengen(
            hidden_states=x_np,
            start_pos=start_pos,
            qkv_weight=qkv_weight,
            o_weight=o_weight,
            input_weight=ref_norm_scale,
            cache_k=cache_k,
            cache_v=cache_v,
            cos=cos,
            sin=sin,
            post_attention_weight=post_attention_weight,
            router_weight=router_weight,
            router_bias=router_bias,
            gate_up_weight=gate_up_weight,
            gate_up_bias_plus1_T=gate_up_bias,
            down_weight=down_weight,
            down_bias_broadcasted=down_bias,
            config=config,
            sliding_window=sliding_window,
            sink=sink,
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            is_neuronpy=False,
            ep_rank=np.array([0], dtype=np.int32),
            ep_size=16,
        )
    elif device == "device":
        jit_tokengen = baremetal_jit(tokengen)
        output, cache_k_out, cache_v_out = jit_tokengen(
            hidden_states=x_np,
            start_pos=start_pos,
            qkv_weight=qkv_weight,
            o_weight=o_weight,
            input_weight=ref_norm_scale,
            cache_k=cache_k,
            cache_v=cache_v,
            cos=cos,
            sin=sin,
            post_attention_weight=post_attention_weight,
            router_weight=router_weight,
            router_bias=router_bias,
            gate_up_weight=gate_up_weight,
            gate_up_bias_plus1_T=gate_up_bias,
            down_weight=down_weight,
            down_bias_broadcasted=down_bias,
            config=config,
            sliding_window=sliding_window,
            sink=sink,
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            is_neuronpy=True,
            ep_rank=np.array([0], dtype=np.int32),
            ep_size=16,
        )

    # FIXME: reduce tolerance
    atol = 2e-2 if dtype == bfloat16 else None
    assert_allclose(ref_output, output[0, 0], atol=atol)
    assert_allclose(ref_k, cache_k_out[0, : seq_len + 1], atol=atol)
    assert_allclose(ref_v, cache_v_out[0, : seq_len + 1], atol=atol)

@pytest.mark.skip
@pytest.mark.parametrize(
    "batch_size, ep_size, tp_size",
    [
        (32, 16, 8),
        (1, 1, 32),
        (2, 2, 32),
        (16, 16, 8),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "device",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        # FIXME: fp32 has a few compiler errors
        # np.float32,
        bfloat16,
    ],
)
def test_tokengen_moe(batch_size, ep_size, tp_size, device, dtype):
    assert batch_size % ep_size == 0
    config = Config(max_batch_size_per_dp=batch_size // ep_size)
    config.intermediate_size = config.intermediate_size // tp_size
    config.dtype = dtype
    ref_moe = MLPBlock(config)
    # set all weight/bias except ep0 to be 0 to for EP
    n_experts_per_ep = config.num_experts // ep_size
    ref_moe.mlp2_weight.data[n_experts_per_ep:] = 0
    ref_moe.mlp2_bias.data[n_experts_per_ep:] = 0
    ep_rank = 0

    torch_dtype = numpy_to_torch_type[dtype]
    hidden_states = torch.randn(
        batch_size, config.hidden_size, dtype=torch_dtype
    )
    # only need ep rank 0
    assert batch_size % ep_size == 0
    batch_size_per_ep = batch_size // ep_size
    ref_output = ref_moe(hidden_states[ep_rank * batch_size_per_ep : (ep_rank + 1) * batch_size_per_ep])
    ref_output = ref_output.detach().float().numpy().astype(config.dtype)
    hidden_states = hidden_states.float().numpy().astype(config.dtype)
    post_attention_weight = ref_moe.norm.scale.data.float().numpy().astype(config.dtype)
    router_weight = ref_moe.gate.weight.data.T.float().numpy().astype(config.dtype)
    router_bias = ref_moe.gate.bias.data.float().numpy().astype(config.dtype)
    mlp1_weight = (
        ref_moe.mlp1_weight.data.float().numpy().astype(config.dtype)
    )  # [num_experts, intermediate_size*2, hidden_size]
    gate_up_bias = (
        ref_moe.mlp1_bias.data.float()
        .numpy()
        .astype(config.dtype)
        .reshape(config.num_experts, 2, config.intermediate_size)
    )
    gate_up_bias_plus1_T = gate_up_bias.transpose(0, 2, 1)
    gate_up_bias_plus1_T[..., 1] += 1
    mlp2_weight = (
        ref_moe.mlp2_weight.data.float().numpy().astype(config.dtype)
    )  # [num_experts, hidden_size, intermediate_size]
    down_bias = ref_moe.mlp2_bias.data.float().numpy().astype(config.dtype)
    down_bias_boardcasted = np.expand_dims(down_bias, axis=[1])
    down_bias_boardcasted = np.broadcast_to(
        down_bias_boardcasted, (config.num_experts, TILE_SIZE, config.hidden_size)
    )  # boardcast to reduce the cost of nc_stream_shuffle

    gate_up_weight = mlp1_weight.transpose(0, 2, 1).reshape(
        config.num_experts,
        config.hidden_size,
        2,
        config.intermediate_size,
    )
    down_weight = mlp2_weight.transpose(0, 2, 1)

    if device == "cpu":
        output = tokengen_moe(
            hidden_states=hidden_states,
            post_attention_weight=post_attention_weight,
            router_weight=router_weight,
            router_bias=router_bias,
            gate_up_weight=gate_up_weight,
            gate_up_bias_plus1_T=gate_up_bias_plus1_T,
            down_weight=down_weight,
            down_bias_broadcasted=down_bias_boardcasted,
            ep_rank=np.array([ep_rank], dtype=np.int32),
            ep_size=ep_size,
            config=config,
            is_neuronpy=False,
        )
    elif device == "device":
        jit_tokengen_moe = baremetal_jit(tokengen_moe)
        output = jit_tokengen_moe(
            hidden_states=hidden_states,
            post_attention_weight=post_attention_weight,
            router_weight=router_weight,
            router_bias=router_bias,
            gate_up_weight=gate_up_weight,
            gate_up_bias_plus1_T=gate_up_bias_plus1_T,
            down_weight=down_weight,
            down_bias_broadcasted=down_bias_boardcasted,
            ep_rank=np.array([ep_rank], dtype=np.int32),
            ep_size=ep_size,
            config=config,
            is_neuronpy=True,
        )

    atol = 1e-4
    assert_allclose(ref_output, output, atol=atol)

@pytest.mark.skip
@pytest.mark.parametrize(
    "sliding_window",
    [
        (0),
        (128),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "device",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        bfloat16,
    ],
)
def test_tokengen_attention(sliding_window, device, dtype):    
    (
        config,
        x_np,
        _,
        ref_norm_scale,
        qkv_weight,
        o_weight,
        qkv_bias, 
        o_bias,
        sink,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        start_pos,
        cache_k,
        cache_v,
        cos,
        sin,
        ref_k,
        ref_v,
        ref_attn_hidden_states,
        seq_len,
    ) = setup_tokengen_test(sliding_window, dtype=dtype)

    if device == "cpu":
        attn_output, cache_k_out, cache_v_out = attention_module(
            hidden_states=x_np,
            input_weight=ref_norm_scale,
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
            compute_dtype=x_np.dtype,
            is_neuronpy=False,
        )
    elif device == "device":
        jit_attention_module = baremetal_jit(attention_module)
        attn_output, cache_k_out, cache_v_out = jit_attention_module(
            hidden_states=x_np,
            input_weight=ref_norm_scale,
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
            compute_dtype=x_np.dtype,
            is_neuronpy=True,
        )

    atol = 1e-4
    assert_allclose(ref_attn_hidden_states, attn_output, atol=atol)
    assert_allclose(ref_k, cache_k_out[0, : seq_len + 1], atol=atol)
    assert_allclose(ref_v, cache_v_out[0, : seq_len + 1], atol=atol)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
