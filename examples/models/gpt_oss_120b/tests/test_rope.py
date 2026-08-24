import numpy as np
import pytest
import torch
from config import Config
from nkipy.runtime import baremetal_jit
from kernels.rope import compute_cos_sin, rope_yarn
from reference import RotaryEmbedding
from utils import assert_allclose


def setup():
    config = Config(max_batch_size_per_dp=2, max_model_len=256)
    batch_size = config.max_batch_size_per_dp
    seq_len = config.max_model_len
    n_heads = config.n_heads
    head_dim = config.head_dim

    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    cos, sin = compute_cos_sin(seq_len)

    rope_torch = RotaryEmbedding(
        head_dim=head_dim,
        base=config.rope_theta,
    )

    q_ref = q.permute(1, 0, 2, 3).reshape(seq_len, batch_size * n_heads, head_dim)
    k_ref = k.permute(1, 0, 2, 3).reshape(seq_len, batch_size * n_heads, head_dim)
    ref_q_out, ref_k_out = rope_torch(q_ref, k_ref)

    ref_q_prefill = ref_q_out.reshape(seq_len, batch_size, n_heads, head_dim).permute(1, 0, 2, 3).contiguous()
    ref_k_prefill = ref_k_out.reshape(seq_len, batch_size, n_heads, head_dim).permute(1, 0, 2, 3).contiguous()

    return (
        q.numpy(),
        k.numpy(),
        cos,
        sin,
        ref_q_prefill.numpy(),
        ref_k_prefill.numpy(),
    )

def test_cpu():
    q, k, cos, sin, ref_q_out, ref_k_out = setup()

    query_out, key_out = rope_yarn(
        query=q,
        key=k,
        cos=cos,
        sin=sin,
        start_pos=None,
    )

    assert_allclose(ref_q_out, query_out)
    assert_allclose(ref_k_out, key_out)


def test_device():
    q, k, cos, sin, ref_q_out, ref_k_out = setup()

    query_device, key_device = baremetal_jit(rope_yarn)(
        query=q,
        key=k,
        cos=cos,
        sin=sin,
        start_pos=None,
    )

    assert_allclose(ref_q_out, query_device)
    assert_allclose(ref_k_out, key_device)

if __name__ == "__main__":
    pytest.main(["-s", __file__])
