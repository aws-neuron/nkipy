"""
Test module for attention_prefill kernel implementation.
"""
import ml_dtypes
import numpy as np
import pytest
import torch
from config import Config
from conftest import TEST_TP_FOR_SHAPE
from convert_torch_numpy_dtype import numpy_to_torch_type
from nkipy.runtime import baremetal_jit
from kernels.attention import attention_module
from kernels.rope import compute_cos_sin
from reference import TransformerBlock
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def setup_attention_test(sliding_window, for_tokengen, dtype):
    # TODO: test batch_size > 1
    seq_len = 1024
    config = Config(max_model_len=seq_len+1024)
    config.n_heads = max(config.n_heads // TEST_TP_FOR_SHAPE, 1)
    config.n_kv_heads = max(config.n_kv_heads // TEST_TP_FOR_SHAPE, 1)
    config.intermediate_size = config.intermediate_size // TEST_TP_FOR_SHAPE
    config.dtype = dtype
    ref_transformer_block = TransformerBlock(config, sliding_window)
    ref_attention = ref_transformer_block.attn

    # Create test input - use dtype from config
    torch_dtype = numpy_to_torch_type[dtype]
    x_torch = torch.randn(seq_len, config.hidden_size, dtype=torch_dtype)

    # Run reference
    with torch.no_grad():
        ref_output, ref_k, ref_v = ref_attention(x_torch)
        ref_output = ref_output.float().numpy().astype(config.dtype)
        ref_k = ref_k.float().numpy().astype(config.dtype)
        ref_v = ref_v.float().numpy().astype(config.dtype)

    # Convert to numpy for our implementation - keep in bfloat16
    x_np = x_torch.float().numpy().astype(config.dtype)
    x_np = x_np[None, :, :]  # Add batch dimension: [1, seq_len, hidden_size]

    # Extract weights and biases from reference
    qkv_weight_torch = ref_attention.qkv.weight.data.float()
    o_weight_torch = ref_attention.out.weight.data.float()

    qkv_weight = qkv_weight_torch.T.numpy().astype(config.dtype)
    o_weight = o_weight_torch.T.numpy().astype(config.dtype)
    qkv_bias = ref_attention.qkv.bias.data.float().numpy().astype(config.dtype)
    o_bias = ref_attention.out.bias.data.float().numpy().astype(config.dtype)

    # Extract attention sink
    sink = ref_attention.sinks.data.float().numpy().astype(config.dtype)

    # Get norm weights
    ref_norm_scale = ref_attention.norm.scale.data.float().numpy().astype(config.dtype)

    # Create KV cache
    cache_k = np.zeros(
        (
            config.max_batch_size_per_dp,
            config.max_model_len,
            config.n_kv_heads,
            config.head_dim,
        )
    ).astype(config.dtype)
    cache_v = np.zeros(
        (
            config.max_batch_size_per_dp,
            config.max_model_len,
            config.n_kv_heads,
            config.head_dim,
        )
    ).astype(config.dtype)
    cos, sin = compute_cos_sin(config.max_model_len)

    res = (
        config,
        x_np,
        ref_output,
        ref_k,
        ref_v,
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
    )
    if for_tokengen:
        res = res + (
            seq_len,
            ref_transformer_block,
        )
    return res

@pytest.mark.skip
@pytest.mark.parametrize(
    "sliding_window",
    [
        (0),
        (128),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        bfloat16,
    ],
)
def test_cpu(sliding_window, dtype):
    (
        config,
        x_np,
        ref_output,
        ref_k,
        ref_v,
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
    ) = setup_attention_test(sliding_window, for_tokengen=False, dtype=dtype)
    seq_len = x_np.shape[1]
    # Test our implementation with sliding window
    output, cache_k, cache_v = attention_module(
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
        start_pos=None,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=config.dtype,
        is_neuronpy=False,
    )

    # Set ATOL based on dtype - only override for bf16
    atol = 0.1 if dtype == bfloat16 else None
    assert_allclose(ref_k, cache_k.squeeze(0)[:seq_len], atol=atol)
    assert_allclose(ref_v, cache_v.squeeze(0)[:seq_len], atol=atol)
    assert_allclose(ref_output, output.squeeze(0), atol=atol)

@pytest.mark.skip
@pytest.mark.parametrize(
    "sliding_window",
    [
        (0),
        (128),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        bfloat16,
    ],
)
def test_device(sliding_window, dtype):
    (
        config,
        x_np,
        ref_output,
        ref_k,
        ref_v,
        qkv_weight,
        o_weight,
        qkv_bias,
        o_bias,
        sink_orig,
        ref_norm_scale,
        cache_k,
        cache_v,
        cos,
        sin,
    ) = setup_attention_test(sliding_window, for_tokengen=False, dtype=dtype)
    seq_len = x_np.shape[1]
    sink = sink_orig.reshape(config.n_heads, 1)

    output_device, cache_k_device, cache_v_device = baremetal_jit(attention_module)(
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
        start_pos=None,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=config.dtype,
        is_neuronpy=True,
    )

    atol = 0.1 if dtype == bfloat16 else None
    assert_allclose(ref_k, cache_k_device.squeeze(0)[:seq_len], atol=atol)
    assert_allclose(ref_v, cache_v_device.squeeze(0)[:seq_len], atol=atol)
    assert_allclose(ref_output, output_device.squeeze(0), atol=atol)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
