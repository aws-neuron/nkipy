import os
import numpy as np
import torch
import pytest
from ml_dtypes import bfloat16

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa

from nkipy.runtime.device_kernel import DeviceKernel
from nkipy.runtime.device_tensor import DeviceTensor
from kernels.attention_nki.prefill_kernel import (
    flash_attn_prefill,
    attention_prefill_sw128,
)
from kernels.attention_nki.decode_kernel import (
    flash_attn_decode,
    load_qkv_and_apply_rope,
    update_kv_cache,
)


def ref_sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    mask = torch.triu(
        Q.new_full((n_tokens, n_tokens), -float("inf")),
        diagonal=1,
    )
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-sliding_window,
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    if S is not None:
        S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
    else:
        W = torch.softmax(QK, dim=-1)
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def ref_apply_rope(x, sin, cos):
    """Reference implementation of RoPE (Rotary Position Embedding)"""
    half_d = sin.shape[-1]
    assert x.shape[-1] == half_d * 2
    x0 = x[..., :half_d]
    x1 = x[..., half_d:]
    rotated_x0 = x0 * cos - x1 * sin
    rotated_x1 = x0 * sin + x1 * cos
    return torch.cat([rotated_x0, rotated_x1], dim=-1)


def ref_rope(qkv, sin, cos, position_ids, h, k_h):
    """Reference implementation of rope function"""
    _, total_heads, _ = qkv.shape
    assert h + k_h * 2 == total_heads

    # Extract query, key, value from concatenated qkv
    q = qkv[:, :h, :]  # query heads
    k = qkv[:, h : h + k_h, :]  # key heads
    v = qkv[:, h + k_h : h + k_h * 2, :]  # value heads

    # Get sin/cos values for each batch position
    batch_sin = sin[position_ids]  # [batch_size, half_d]
    batch_cos = cos[position_ids]  # [batch_size, half_d]

    # Expand to match head dimensions
    batch_sin = batch_sin.unsqueeze(1)  # [batch_size, 1, half_d]
    batch_cos = batch_cos.unsqueeze(1)  # [batch_size, 1, half_d]

    # Apply RoPE to query and key
    q_rope = ref_apply_rope(q, batch_sin, batch_cos)
    k_rope = ref_apply_rope(k, batch_sin, batch_cos)

    return q_rope, k_rope, v


def ref_sdpa_decode(Q, K, V, position_ids, sm_scale, S=None, sliding_window=0):
    # sliding_window == 0 means no sliding window
    batch_size, n_heads, q_mult, d_head = Q.shape
    max_model_len = K.shape[1]
    assert K.shape == (batch_size, max_model_len, n_heads, d_head)
    assert V.shape == (batch_size, max_model_len, n_heads, d_head)
    Q = Q[:, None, :, :, :]
    K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    Q = Q.reshape(batch_size, 1, n_heads * q_mult, d_head)
    K = K.reshape(batch_size, max_model_len, n_heads * q_mult, d_head)
    V = V.reshape(batch_size, max_model_len, n_heads * q_mult, d_head)
    mask = torch.triu(
        Q.new_full((max_model_len, max_model_len), -float("inf")),
        diagonal=1,
    )
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((max_model_len, max_model_len), -float("inf")),
            diagonal=-sliding_window,
        )
    mask = mask[position_ids]  # batch, n_tokens
    QK = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    QK *= sm_scale
    QK += mask[:, None, None, :]
    if S is not None:
        S = (
            S.reshape(1, n_heads, q_mult, 1, 1)
            .expand(batch_size, -1, -1, -1, -1)
            .reshape(batch_size, n_heads * q_mult, 1, 1)
        )
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
    else:
        W = torch.softmax(QK, dim=-1)
    attn = torch.einsum("bhqk,bkhd->bqhd", W, V)
    return attn.reshape(batch_size, n_heads * q_mult, d_head)


def _sample_tensor(shape, dtype):
    t = torch.empty(*shape, dtype=dtype)
    t.uniform_(-1, 1)
    return t


def torch_to_numpy(x):
    if x.dtype == torch.bfloat16:
        x = x.float().numpy().astype(bfloat16)
    else:
        x = x.numpy()
    return x


def convert_torch_tensors_to_numpy(input_kwargs):
    new_input_kwargs = {}
    for arg_name in input_kwargs:
        arg = input_kwargs[arg_name]
        if isinstance(arg, torch.Tensor):
            arg = torch_to_numpy(arg)
        new_input_kwargs[arg_name] = arg
    return new_input_kwargs


def _get_default_compiler_flags():
    compiler_flags = [
        "-O1",
        "--lnc=1",
        "--tensorizer-options='--skip-pass=LateLegalizePostSplit'",
        # "--enable-internal-data-race-checker",
        # "--internal-compiler-debug-mode=all",
        # "--tensorizer-options='--print-stats --dump-after=All'",
    ]
    return compiler_flags


def prepare_nki_kernel_args_prefill(
    query,
    key,
    value,
    sink,
    sliding_window,
    softmax_scale,
    disable_attention_sw=False,
):
    seqlen, kv_head, q_h_per_k_h, head_size = query.shape
    q_head = q_h_per_k_h * kv_head
    # BSHD -> BDHS
    query = query.view(1, seqlen, q_head, head_size).permute(0, 3, 2, 1)
    # BSHD -> BHDS
    key = key.view(1, seqlen, kv_head, head_size).permute(0, 2, 3, 1)
    value = value.view(1, seqlen, kv_head, head_size).permute(0, 2, 3, 1)
    if sliding_window == 0 or disable_attention_sw:
        KV_TILE_SIZE = 2048
        assert seqlen % KV_TILE_SIZE == 0
        mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.uint8))
        if sliding_window > 0:
            mask = np.triu(mask, k=-(sliding_window - 1)).astype(np.uint8)

        kernel_args = dict(
            q=query,
            k=key,
            v=value,
            sink=sink,
            mask=mask,
            softmax_scale=softmax_scale,
            LARGE_KV_TILE_SIZE=KV_TILE_SIZE,
        )
        func = flash_attn_prefill
    else:
        assert seqlen % 128 == 0
        kernel_args = dict(
            q=query,
            k=key,
            v=value,
            sink=sink,
            softmax_scale=softmax_scale,
        )
        func = attention_prefill_sw128
    kernel_args = convert_torch_tensors_to_numpy(kernel_args)
    return func, kernel_args


def prepare_nki_kernel_args_decode(
    qkv,
    cos,
    sin,
    key_cache,
    value_cache,
    sink,
    position_ids,
    softmax_scale,
    tile_masks,
):
    # BSHD -> BHDS
    key_cache = key_cache.permute(0, 2, 3, 1)
    kernel_args = dict(
        qkv=qkv,
        cos=cos,
        sin=sin,
        cache_k=key_cache,
        cache_v=value_cache,
        sink=sink,
        position_ids=position_ids,
        softmax_scale=softmax_scale,
        tile_masks=tile_masks,
    )
    kernel_args = convert_torch_tensors_to_numpy(kernel_args)
    return kernel_args

@pytest.mark.skip
@pytest.mark.parametrize("sliding_window", [0, 128])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_attention_nki_prefill(sliding_window, dtype):
    benchmark = os.environ.get("BENCHMARK", "0") != "0"
    torch.manual_seed(12345)
    torch.set_printoptions(sci_mode=False)
    q_head = 2
    kv_head = 1
    seqlen = 10240
    head_size = 64
    assert q_head % kv_head == 0
    q_h_per_k_h = q_head // kv_head
    dtype = getattr(torch, dtype)
    query = _sample_tensor(
        shape=(seqlen, kv_head, q_h_per_k_h, head_size),
        dtype=dtype,
    )
    key = _sample_tensor(
        shape=(seqlen, kv_head, head_size),
        dtype=dtype,
    )
    value = _sample_tensor(
        shape=(seqlen, kv_head, head_size),
        dtype=dtype,
    )
    sink = _sample_tensor(
        shape=(q_head, 1),
        dtype=dtype,
    )

    # run ref version on CPU
    softmax_scale = 1.0 / (head_size**0.5)
    if not benchmark:
        out_ref = ref_sdpa(
            Q=query,
            K=key,
            V=value,
            S=sink,
            sm_scale=softmax_scale,
            sliding_window=sliding_window,
        )

    # run nki
    func, kernel_args = prepare_nki_kernel_args_prefill(
        query,
        key,
        value,
        sink,
        sliding_window,
        softmax_scale,
    )

    compiler_flags_str = " ".join(_get_default_compiler_flags())
    os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
    if benchmark:
        bench_func_ = nki.benchmark(
            warmup=5,
            iters=10,
        )(
            func
        )[1, kv_head]
        bench_func_(**kernel_args)
        latency_res = bench_func_.benchmark_result.nc_latency
        p90 = latency_res.get_latency_percentile(90)
        print(f"p90: {p90}")
    else:
        out_nki = nki.baremetal(
            func,
            # artifacts_dir="./_artifacts",
            # save_neff_name="file.neff",
            # save_trace_name="profile.ntff",
            # debug_kernel=True,
        )[1, kv_head](**kernel_args)
        out_nki = torch.tensor(out_nki.astype(np.float32)).to(query.dtype)
        # BHSD -> BSHD
        out_nki = out_nki.permute(0, 2, 1, 3).reshape(seqlen, q_head * head_size)

        torch.testing.assert_close(out_nki, out_ref, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size,max_model_len", [(1, 10240), (8, 1024)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("feed_mask", [False, True])
@pytest.mark.parametrize("sliding_window", [128, 0])
def test_attention_nki_decode(
    request,
    batch_size,
    max_model_len,
    sliding_window,
    dtype,
    feed_mask,
):
    torch.manual_seed(12345)
    torch.set_printoptions(sci_mode=False)
    q_head = 8
    kv_head = 1
    q_h_per_k_h = q_head // kv_head
    head_size = 64
    assert q_head % kv_head == 0
    dtype = getattr(torch, dtype)
    qkv = _sample_tensor(
        shape=(batch_size, q_head + 2 * kv_head, head_size),
        dtype=dtype,
    )
    half_d = head_size // 2
    cos = _sample_tensor((max_model_len, half_d), dtype=dtype)
    sin = _sample_tensor((max_model_len, half_d), dtype=dtype)
    position_ids = torch.randint(1, max_model_len, (batch_size,), dtype=torch.int)
    query, key, value = ref_rope(qkv, sin, cos, position_ids, q_head, kv_head)

    key_cache = _sample_tensor(
        shape=(batch_size, max_model_len, kv_head, head_size),
        dtype=dtype,
    )
    value_cache = _sample_tensor(
        shape=(batch_size, max_model_len, kv_head, head_size),
        dtype=dtype,
    )
    sink = _sample_tensor(
        shape=(q_head, 1),
        dtype=dtype,
    )
    if feed_mask:
        tile_masks = torch.zeros(batch_size, max_model_len, dtype=torch.uint8)
        for b_i in range(batch_size):
            lower_bound = 0
            if sliding_window > 0:
                # XXX: +1 due to kernel adopts decomposed attention
                lower_bound = max(lower_bound, position_ids[b_i] - sliding_window + 1)
            tile_masks[b_i, lower_bound : position_ids[b_i]] = 1
        LARGE_KV_TILE_SIZE = min(2048, max_model_len)
        B_P_SIZE = 128
        tile_masks = (
            tile_masks.reshape(
                batch_size,
                max_model_len // LARGE_KV_TILE_SIZE,
                B_P_SIZE,
                LARGE_KV_TILE_SIZE // B_P_SIZE,
            )
            .permute(2, 1, 0, 3)
            .contiguous()
        )
    else:
        tile_masks = None

    # run ref version on CPU
    softmax_scale = 1.0 / (head_size**0.5)
    key_cache_copy = key_cache.clone()
    value_cache_copy = value_cache.clone()
    # update key/value cache with new key value token using position_ids
    for batch_idx in range(batch_size):
        pos = position_ids[batch_idx]
        key_cache_copy[batch_idx, pos] = key[batch_idx]
        value_cache_copy[batch_idx, pos] = value[batch_idx]

    out_ref = (
        ref_sdpa_decode(
            Q=query.view(batch_size, kv_head, q_h_per_k_h, head_size),
            K=key_cache_copy,
            V=value_cache_copy,
            position_ids=position_ids,
            S=sink,
            sm_scale=softmax_scale,
            sliding_window=sliding_window,
        )
        .float()
        .numpy()
        .astype(bfloat16)
    )

    kernel_args = prepare_nki_kernel_args_decode(
        qkv,
        cos,
        sin,
        key_cache,
        value_cache,
        sink,
        position_ids,
        softmax_scale,
        tile_masks,
    )
    qkv = kernel_args["qkv"]
    cos = kernel_args["cos"]
    sin = kernel_args["sin"]
    key_cache = kernel_args["cache_k"]
    value_cache = kernel_args["cache_v"]
    sink = kernel_args["sink"]
    position_ids = kernel_args["position_ids"]
    if tile_masks is not None:
        tile_masks = kernel_args["tile_masks"]

    expected_cache_k = key_cache.copy()
    key = torch_to_numpy(key)
    value = torch_to_numpy(value)
    np.put_along_axis(
        expected_cache_k,
        position_ids.reshape(batch_size, 1, 1, 1),
        key.astype(key_cache.dtype).reshape(batch_size, kv_head, head_size, 1),
        axis=3,
    )
    expected_cache_v = value_cache.copy()
    np.put_along_axis(
        expected_cache_v,
        position_ids.reshape(batch_size, 1, 1, 1),
        value.astype(value_cache.dtype).reshape(batch_size, 1, kv_head, head_size),
        axis=1,
    )

    compiler_flags_str = " ".join(_get_default_compiler_flags())
    os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
    test_name = request.node.name
    decode_attn_kv_cache_update_kernel = DeviceKernel.compile_and_load(
        flash_attn_decode[kv_head],
        name=f"flash_attn_decode_{test_name}",
        qkv=qkv,
        cos=cos,
        sin=sin,
        cache_k=key_cache,
        cache_v=value_cache,
        sink=sink,
        position_ids=position_ids,
        sliding_window=sliding_window,
        tile_masks=tile_masks,
    )
    qkv_nkipy = DeviceTensor.from_numpy(qkv)
    cos_nkipy = DeviceTensor.from_numpy(cos)
    sin_nkipy = DeviceTensor.from_numpy(sin)
    cache_k_nkipy = DeviceTensor.from_numpy(key_cache)
    cache_v_nkipy = DeviceTensor.from_numpy(value_cache)
    sink_nkipy = DeviceTensor.from_numpy(sink)
    pos_nkipy = DeviceTensor.from_numpy(position_ids)
    o_nkipy = DeviceTensor.from_numpy(
        np.empty((batch_size, q_head, head_size), dtype=qkv.dtype)
    )
    inputs = {
        "qkv": qkv_nkipy,
        "cos": cos_nkipy,
        "sin": sin_nkipy,
        "cache_k.must_alias_input": cache_k_nkipy,
        "cache_v.must_alias_input": cache_v_nkipy,
        "sink": sink_nkipy,
        "position_ids": pos_nkipy,
    }
    if tile_masks is not None:
        tile_masks_nkipy = DeviceTensor.from_numpy(tile_masks)
        inputs["tile_masks"] = tile_masks_nkipy
    decode_attn_kv_cache_update_kernel(
        inputs=inputs,
        outputs={
            "o": o_nkipy,
            "cache_k": cache_k_nkipy,
            "cache_v": cache_v_nkipy,
        },
    )
    out_nki = o_nkipy.numpy()
    updated_cache_k = cache_k_nkipy.numpy()
    updated_cache_v = cache_v_nkipy.numpy()

    o_match = np.allclose(
        out_nki.astype(np.float32), out_ref.astype(np.float32), rtol=5e-3, atol=1e-3
    )
    k_match = np.allclose(
        updated_cache_k.astype(np.float32),
        expected_cache_k.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    v_match = np.allclose(
        updated_cache_v.astype(np.float32),
        expected_cache_v.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    print(f"Output correct: {o_match}")
    print(f"K cache update correct: {k_match}")
    print(f"V cache update correct: {v_match}")
    assert o_match and k_match and v_match


@nki.compiler.skip_middle_end_transformations
@nki.jit
def rope_nki(qkv, sin, cos, position_ids, h, k_h):
    batch_size, total_heads, d = qkv.shape
    assert batch_size <= 128 and h + k_h * 2 == total_heads
    assert sin.shape[1] == cos.shape[1] == d // 2
    q_h_per_k_h = h // k_h

    spmd_ndim = nl.program_ndim()
    assert spmd_ndim <= 1, f"{nl.program_ndim()=} > 1"
    if spmd_ndim == 0:
        assert k_h == 1
        kv_head_id = 0
    else:
        assert nl.num_programs(axes=0) == k_h
        kv_head_id = nl.program_id(axis=0)

    q_rope, k_rope, v_sbuf = load_qkv_and_apply_rope(
        qkv, sin, cos, position_ids, h, k_h, kv_head_id
    )
    q_out = nl.ndarray((batch_size, h, d), dtype=q_rope.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((batch_size, k_h, d), dtype=k_rope.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((batch_size, k_h, d), dtype=v_sbuf.dtype, buffer=nl.shared_hbm)
    nl.store(q_out[:, nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h), :], q_rope)
    nl.store(k_out[:, nl.ds(kv_head_id, 1), :], k_rope)
    nl.store(v_out[:, nl.ds(kv_head_id, 1), :], v_sbuf)
    return q_out, k_out, v_out


@pytest.mark.skip
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("q_head,kv_head", [(8, 1), (8, 2)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_decode_rope(batch_size, q_head, kv_head, dtype):
    torch.manual_seed(12345)
    torch.set_printoptions(sci_mode=False)
    max_model_len = 1024
    head_size = 64
    assert q_head % kv_head == 0
    dtype = getattr(torch, dtype)
    half_d = head_size // 2
    cos = _sample_tensor((max_model_len, half_d), dtype=dtype)
    sin = _sample_tensor((max_model_len, half_d), dtype=dtype)
    qkv = _sample_tensor((batch_size, q_head + kv_head * 2, head_size), dtype=dtype)
    position_ids = torch.randint(1, max_model_len, (batch_size,), dtype=torch.int)

    kernel_args = dict(
        qkv=qkv,
        sin=sin,
        cos=cos,
        position_ids=position_ids,
        h=q_head,
        k_h=kv_head,
    )
    kernel_args = convert_torch_tensors_to_numpy(kernel_args)
    q_nki, k_nki, v_nki = nki.baremetal(
        rope_nki,
        # artifacts_dir="./_artifacts",
        # debug_kernel=True,
    )[kv_head](**kernel_args)

    # reference version

    q_ref, k_ref, v_ref = ref_rope(
        torch.from_numpy(kernel_args["qkv"].astype(np.float32)).to(dtype),
        torch.from_numpy(kernel_args["sin"].astype(np.float32)).to(dtype),
        torch.from_numpy(kernel_args["cos"].astype(np.float32)).to(dtype),
        torch.from_numpy(kernel_args["position_ids"]),
        kernel_args["h"],
        kernel_args["k_h"],
    )

    # Convert NKI outputs to torch tensors for comparison
    q_nki_torch = torch.from_numpy(q_nki.astype(np.float32)).to(dtype)
    k_nki_torch = torch.from_numpy(k_nki.astype(np.float32)).to(dtype)
    v_nki_torch = torch.from_numpy(v_nki.astype(np.float32)).to(dtype)

    # Compare results
    torch.testing.assert_close(q_nki_torch, q_ref, atol=5e-3, rtol=1e-3)
    torch.testing.assert_close(k_nki_torch, k_ref, atol=5e-3, rtol=1e-3)
    torch.testing.assert_close(v_nki_torch, v_ref, atol=5e-3, rtol=1e-3)


@nki.compiler.skip_middle_end_transformations
@nki.jit(experimental_flags="enable-mutable-parameter")
def decode_update_kv_cache(
    position_ids,  # [batch_size]
    k,  # [batch_size, num_heads, head_size]
    v,  # [batch_size, num_heads, head_size]
    cache_k: nt.tensor[nt.mutable],  # [batch_size, num_heads, head_size, max_model_len]
    cache_v: nt.tensor[nt.mutable],  # [batch_size, max_model_len, num_heads, head_size]
):
    batch_size, num_heads, head_size, _ = cache_k.shape
    spmd_ndim = nl.program_ndim()
    assert spmd_ndim <= 1, f"{nl.program_ndim()=} > 1"
    if spmd_ndim == 0:
        assert num_heads == 1
        kv_head_id = 0
    else:
        assert nl.num_programs(axes=0) == num_heads
        kv_head_id = nl.program_id(axis=0)
    k_sbuf = nl.ndarray((head_size, batch_size), dtype=k.dtype)
    k_sbuf[...] = nisa.dma_transpose(src=k[:, kv_head_id, :], axes=(1, 0))
    position_ids_sbuf = nl.load(
        position_ids.reshape((1, batch_size)),
        dtype=nl.int32,
    )
    v_sbuf = nl.load(v[:, kv_head_id, :])
    update_kv_cache(
        kv_head_id=kv_head_id,
        k=k_sbuf,
        v=v_sbuf,
        cache_k=cache_k,
        cache_v=cache_v,
        position_ids=position_ids_sbuf,
    )
    return cache_k, cache_v


@pytest.mark.parametrize(
    "batch_size,num_heads",
    [
        (8, 1),
        (8, 8),
        (32, 2),
        (128, 1),
    ],
)
def test_decode_update_kv_cache(batch_size, num_heads):
    """Unit test for decode_update_k_cache kernel"""
    # Test parameters
    max_model_len = 2048
    num_heads = 1
    head_size = 64

    # Generate test data
    np.random.seed(42)  # For reproducible results

    # Position IDs for each request in the batch (where to insert the new token)
    position_ids = np.random.randint(
        0, max_model_len, size=(batch_size,), dtype=np.int32
    )

    # New K and V tensors to be inserted (1 token per request)
    k = np.random.randn(batch_size, num_heads, head_size).astype(np.float32)
    v = np.random.randn(batch_size, num_heads, head_size).astype(np.float32)

    # Pre-allocated KV cache (initially filled with some data)
    cache_k = np.random.randn(
        batch_size,
        num_heads,
        head_size,
        max_model_len,
    ).astype(np.float32)
    cache_v = np.random.randn(
        batch_size,
        max_model_len,
        num_heads,
        head_size,
    ).astype(np.float32)

    print("Test inputs:")
    print(f"Batch size: {batch_size}")
    print(f"Max model length: {max_model_len}")
    print(f"Num heads: {num_heads}")
    print(f"Head size: {head_size}")
    print(f"K shape: {k.shape}")
    print(f"Cache K shape: {cache_k.shape}")
    print(f"V shape: {v.shape}")
    print(f"Cache V shape: {cache_v.shape}")
    print(f"Position IDs: {position_ids}")

    expected_cache_k = cache_k.copy()
    np.put_along_axis(
        expected_cache_k,
        position_ids.reshape(batch_size, 1, 1, 1),
        k.astype(cache_k.dtype).reshape(batch_size, num_heads, head_size, 1),
        axis=3,
    )
    expected_cache_v = cache_v.copy()
    np.put_along_axis(
        expected_cache_v,
        position_ids.reshape(batch_size, 1, 1, 1),
        v.astype(cache_v.dtype).reshape(batch_size, 1, num_heads, head_size),
        axis=1,
    )

    # Run the kernel
    kv_cache_update_kernel = DeviceKernel.compile_and_load(
        decode_update_kv_cache[num_heads],
        name=f"decode_update_kv_cache_{batch_size}",
        position_ids=position_ids,
        k=k,
        v=v,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    position_ids_nkipy = DeviceTensor.from_numpy(position_ids)
    k_nkipy = DeviceTensor.from_numpy(k)
    v_nkipy = DeviceTensor.from_numpy(v)
    cache_k_nkipy = DeviceTensor.from_numpy(cache_k)
    cache_v_nkipy = DeviceTensor.from_numpy(cache_v)
    kv_cache_update_kernel(
        inputs={
            "position_ids": position_ids_nkipy,
            "k": k_nkipy,
            "v": v_nkipy,
            "cache_k.must_alias_input": cache_k_nkipy,
            "cache_v.must_alias_input": cache_v_nkipy,
        },
        outputs={
            "cache_k": cache_k_nkipy,
            "cache_v": cache_v_nkipy,
        },
    )
    updated_cache_k = cache_k_nkipy.numpy()
    updated_cache_v = cache_v_nkipy.numpy()

    # Verify the results
    print("\nVerifying results...")

    # Check if the updates are correct
    k_match = np.allclose(updated_cache_k, expected_cache_k, rtol=1e-3, atol=1e-3)
    v_match = np.allclose(updated_cache_v, expected_cache_v, rtol=1e-3, atol=1e-3)

    print(f"K cache update correct: {k_match}")
    print(f"V cache update correct: {v_match}")
    assert k_match and v_match
