from nkipy.core.nki_op import wrap_nki_kernel

import numpy as np
import nkipy.core.typing as nt
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import math


# TODO: This kernel may be faster than fused_rmsnorm_gemm_v0_nc_transpose, but
# currently there's a compiler error with this kernel.
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_rmsnorm_gemm_v0_dma_transpose(
    x: nl.ndarray,
    weight: nl.ndarray,
    y: nl.ndarray,
    bias: nl.ndarray,
    eps: float = 1e-6,
) -> nl.ndarray:
    """Fused RMSNorm Gemm Kernel with the following key design choices:
    1. Use dma_transpose to transpose A
    2. Use 2-way reduction to calculate sum(A^2)
    3. Do not interleave RMS calculation with Matmul

    Profiling trace can be found in
    https://neuron-profiler.corp.amazon.com/profile/yanmwang_rmsnorm_3

    Args:
        x (nl.ndarray): hidden states
        weight (nl.ndarray): RMSNorm weight
        y (nl.ndarray): gemm weight
        eps (float, optional): RMSNorm eps. Defaults to 1e-6.

    Returns:
        nl.ndarray: hidden states
    """

    # Use float32 to reduce numerical error
    rms_compute_dtype = nl.float32

    if x.ndim == 2:
        B, H = x.shape
        S = 1
    elif x.ndim == 3:
        B, S, H = x.shape
        assert S == 1, "Only support tokengen"
    else:
        raise ValueError(f"Malformed shape of x {x.shape}")

    if weight.ndim == 2:
        assert weight.shape == (1, H), f"Malformed shape of weight {weight.shape}"
    elif weight.ndim == 1:
        assert weight.shape == (H,), f"Malformed shape of weight {weight.shape}"
    else:
        raise ValueError(f"Malformed shape of weight {weight.shape}")

    H_, N = y.shape
    assert H == H_, f"Incompatible matmul shape {x.shape} @ {y.shape}"
    if bias.ndim == 2:
        assert bias.shape == (1, N), f"Malformed shape of bias {bias.shape}"
    elif bias.ndim == 1:
        assert bias.shape == (N,), f"Malformed shape of bias {bias.shape}"
    else:
        raise ValueError(f"Malformed shape of bias {bias.shape}")

    assert B <= 128, "This kernel only support max batch size of 128"
    # Create output tensor with original shape and dtype
    output = nl.ndarray((B, S, N), dtype=x.dtype, buffer=nl.hbm)

    # Unify Shapes
    x = x.reshape((B, H))
    output_reshaped = output.reshape((B, N))
    weight = weight.reshape((H,))
    bias = bias.reshape((1, N))

    H0 = nl.tile_size.pmax  # 128
    H1 = math.ceil(H / H0)

    N0 = 128
    N1 = math.ceil(N / N0)

    input_sb = nl.zeros((H0, H1, B), dtype=x.dtype, buffer=nl.sbuf)
    weight_sb = nl.zeros((H0, H1), dtype=weight.dtype, buffer=nl.sbuf)

    # Load weight (H) to (H0, H1)
    i_p, i_f1 = nl.mgrid[0:H0, 0:H1]
    mask = i_p + i_f1 * H0 < H
    nisa.dma_copy(src=weight[i_p + i_f1 * H0], dst=weight_sb[i_p, i_f1], mask=mask)

    # Load x
    for h in nl.affine_range(H1):
        lhs = input_sb[:, h, :]
        i_b, i_h0 = nl.mgrid[:B, :H0]
        mask = h * H0 + i_h0 < H
        lhs[...] = nisa.dma_transpose(x[i_b, h * H0 + i_h0], mask=mask)

    # Load y (H, N) to sbuf with shape (H0, H1, N0, N1)
    rhs_sb = nl.ndarray((H1, nl.par_dim(H0), N1, N0), dtype=x.dtype, buffer=nl.sbuf)
    for h in nl.affine_range(H1):
        i_h0, i_n1, i_n0 = nl.mgrid[0:H0, 0:N1, 0:N0]
        mask = (h * H0 + i_h0 < H) & (i_n1 * N0 + i_n0 < N)
        nisa.dma_copy(
            src=y[h * H0 + i_h0, i_n1 * N0 + i_n0],
            dst=rhs_sb[h, i_h0, i_n1, i_n0],
            mask=mask,
        )

    # Compute x^2
    i_p, i_f0, i_f1 = nl.mgrid[0:H0, 0:H1, 0:B]
    z = nl.ndarray(input_sb.shape, dtype=rms_compute_dtype, buffer=nl.sbuf)
    zero_bias = nl.ndarray((H0, 1), dtype=rms_compute_dtype, buffer=nl.sbuf)
    zero_bias[...] = 0.0
    z[...] = nisa.activation(nl.square, input_sb, bias=zero_bias)

    # Sum across H dimension
    reduced = nl.ndarray((H0, B), dtype=rms_compute_dtype, buffer=nl.sbuf)
    reduced[...] = nisa.tensor_reduce(nl.add, z, axis=(1))

    # Apply weight to input: x * weight
    input_sb[i_p, i_f0, i_f1] = nisa.tensor_tensor(
        input_sb[i_p, i_f0, i_f1], weight_sb[i_p, i_f0], nl.multiply
    )

    # Reduce across partitions
    rmsnorm_reduction_const = nisa.memset((H0, N0), value=1.0, dtype=rms_compute_dtype)
    final_reduced = nl.ndarray((B, N0), dtype=nl.float32, buffer=nl.psum)
    final_reduced[...] = nisa.nc_matmul(reduced, rmsnorm_reduction_const)

    # Compute 1/sqrt(mean(x^2) + eps)
    eps_loaded = nisa.memset((B, 1), value=eps, dtype=rms_compute_dtype)
    rms_inv = nl.ndarray((B, N0), dtype=rms_compute_dtype, buffer=nl.sbuf)
    rms_inv[...] = nisa.activation(
        nl.rsqrt, final_reduced, scale=(1.0 / H), bias=eps_loaded
    )

    # Load bias
    bias_sb = nl.ndarray((1, N), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=bias, dst=bias_sb)
    bias_sb = bias_sb.broadcast_to((B, N))

    # Compute (x @ y) * rms_inv
    for n in nl.affine_range(N1):
        psum = nl.zeros((B, N0), dtype=nl.float32, buffer=nl.psum)
        for h in nl.affine_range(H1):
            lhs = input_sb[:, h, :]
            rhs = rhs_sb[h, :, n, :]
            i_h0 = nl.arange(H0)[:, None]
            psum += nisa.nc_matmul(lhs[i_h0 < H], rhs[i_h0 < H])
        i_b, i_n0 = nl.mgrid[:B, :N0]
        res = nisa.tensor_tensor(psum, rms_inv[i_b, i_n0], nl.multiply)
        mask = n * N0 + i_n0 < N
        res[...] = nisa.tensor_tensor(
            res[i_b, i_n0], bias_sb[i_b, n * N0 + i_n0], nl.add, mask=mask
        )
        nisa.dma_copy(src=res, dst=output_reshaped[i_b, n * N0 + i_n0], mask=mask)

    return output


@nki.compiler.skip_middle_end_transformations
@nki.jit(debug_kernel=True, show_compiler_tb=True)
def fused_rmsnorm_gemm_v0_nc_transpose(
    x: nl.ndarray,
    weight: nl.ndarray,
    y: nl.ndarray,
    bias: nl.ndarray,
    eps: float = 1e-6,
) -> nl.ndarray:
    """Similar to fused_rmsnorm_gemm_v0_dma_transpose, but use nc_transpose to transpose A

    Args:
        x (nl.ndarray): hidden states
        weight (nl.ndarray): RMSNorm weight
        y (nl.ndarray): gemm weight
        bias (nl.ndarray, optional): bias tensor
        eps (float, optional): RMSNorm eps. Defaults to 1e-6.

    Returns:
        nl.ndarray: output
    """
    # Use float32 to reduce numerical error
    rms_compute_dtype = nl.float32

    if x.ndim == 2:
        B, H = x.shape
        S = 1
    elif x.ndim == 3:
        B, S, H = x.shape
        assert S == 1, "Only support tokengen"
    else:
        raise ValueError(f"Malformed shape of x {x.shape}")

    if weight.ndim == 2:
        assert weight.shape == (1, H), f"Malformed shape of weight {weight.shape}"
    elif weight.ndim == 1:
        assert weight.shape == (H,), f"Malformed shape of weight {weight.shape}"
    else:
        raise ValueError(f"Malformed shape of weight {weight.shape}")

    H_, N = y.shape
    assert H == H_, f"Incompatible matmul shape {x.shape} @ {y.shape}"
    if bias.ndim == 2:
        assert bias.shape == (1, N), f"Malformed shape of bias {bias.shape}"
    elif bias.ndim == 1:
        assert bias.shape == (N,), f"Malformed shape of bias {bias.shape}"
    else:
        raise ValueError(f"Malformed shape of bias {bias.shape}")

    assert B <= 128, "This kernel only support max batch size of 128"
    # Create output tensor with original shape and dtype
    output = nl.ndarray((B, S, N), dtype=x.dtype, buffer=nl.hbm)

    # Unify Shapes
    x = x.reshape((B, H))
    output_reshaped = output.reshape((B, N))
    weight = weight.reshape((H,))
    bias = bias.reshape((1, N))

    H0 = nl.tile_size.pmax  # 128
    H1 = math.ceil(H / H0)

    N0 = 128
    N1 = math.ceil(N / N0)
    input_sb = nl.zeros((H0, H1, B), dtype=x.dtype, buffer=nl.sbuf)
    tmp_input_sb = nl.ndarray((B, H1, H0), dtype=x.dtype, buffer=nl.sbuf)
    weight_sb = nl.zeros((H0, H1), dtype=weight.dtype, buffer=nl.sbuf)

    # Load x
    i_b, i_h1, i_h0 = nl.mgrid[0:B, 0:H1, 0:H0]
    mask = i_h1 * H0 + i_h0 < H
    nisa.dma_copy(
        src=x[i_b, i_h1 * H0 + i_h0], dst=tmp_input_sb[i_b, i_h1, i_h0], mask=mask
    )

    # Load weight (H) to (H0, H1)
    i_p, i_f1 = nl.mgrid[0:H0, 0:H1]
    mask = i_p + i_f1 * H0 < H
    nisa.dma_copy(src=weight[i_p + i_f1 * H0], dst=weight_sb[i_p, i_f1], mask=mask)

    # Load x
    for h in nl.affine_range(H1):
        lhs = input_sb[:, h, :]
        i_b, i_h0 = nl.mgrid[:B, :H0]
        psum = nisa.nc_transpose(tmp_input_sb[i_b, h, i_h0])
        lhs[...] = nisa.tensor_copy(psum, dtype=lhs.dtype)

    # Load y (H, N) to (H0, H1, N0, N1)
    # sbuf may not be able to hold this tensor if N is large, N=640 in gpt-oss
    rhs_sb = nl.ndarray((H1, nl.par_dim(H0), N1, N0), dtype=x.dtype, buffer=nl.sbuf)
    for h in nl.affine_range(H1):
        i_h0, i_n1, i_n0 = nl.mgrid[0:H0, 0:N1, 0:N0]
        mask = (h * H0 + i_h0 < H) & (i_n1 * N0 + i_n0 < N)
        nisa.dma_copy(
            src=y[h * H0 + i_h0, i_n1 * N0 + i_n0],
            dst=rhs_sb[h, i_h0, i_n1, i_n0],
            mask=mask,
        )

    # Compute x^2
    i_p, i_f0, i_f1 = nl.mgrid[0:H0, 0:H1, 0:B]
    z = nl.ndarray(input_sb.shape, dtype=rms_compute_dtype, buffer=nl.sbuf)
    zero_bias = nl.ndarray((H0, 1), dtype=rms_compute_dtype, buffer=nl.sbuf)
    zero_bias[...] = 0.0
    z[...] = nisa.activation(nl.square, input_sb, bias=zero_bias)

    # Sum across H dimension
    reduced = nl.ndarray((H0, B), dtype=rms_compute_dtype, buffer=nl.sbuf)
    reduced[...] = nisa.tensor_reduce(nl.add, z, axis=(1))

    # Apply weight to input: x * weight
    input_sb[i_p, i_f0, i_f1] = nisa.tensor_tensor(
        input_sb[i_p, i_f0, i_f1], weight_sb[i_p, i_f0], nl.multiply
    )

    # Reduce across partitions
    rmsnorm_reduction_const = nisa.memset((H0, N0), value=1.0, dtype=rms_compute_dtype)
    final_reduced = nl.ndarray((B, N0), dtype=nl.float32, buffer=nl.psum)
    final_reduced[...] = nisa.nc_matmul(reduced, rmsnorm_reduction_const)

    # Compute 1/sqrt(mean(x^2) + eps)
    eps_loaded = nisa.memset((B, 1), value=eps, dtype=rms_compute_dtype)
    rms_inv = nl.ndarray((B, N0), dtype=rms_compute_dtype, buffer=nl.sbuf)
    rms_inv[...] = nisa.activation(
        nl.rsqrt, final_reduced, scale=(1.0 / H), bias=eps_loaded
    )

    # Load bias
    bias_sb = nl.ndarray((1, N), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=bias, dst=bias_sb)
    bias_sb = bias_sb.broadcast_to((B, N))

    for n in nl.affine_range(N1):
        psum = nl.zeros((B, N0), dtype=nl.float32, buffer=nl.psum)
        for h in nl.affine_range(H1):
            lhs = input_sb[:, h, :]
            rhs = rhs_sb[h, :, n, :]
            i_h0 = nl.arange(H0)[:, None]
            psum += nisa.nc_matmul(lhs[i_h0 < H], rhs[i_h0 < H])
        i_b, i_n0 = nl.mgrid[:B, :N0]
        res = nisa.tensor_tensor(psum, rms_inv[i_b, i_n0], nl.multiply)
        mask = n * N0 + i_n0 < N
        res[...] = nisa.tensor_tensor(
            res[i_b, i_n0], bias_sb[i_b, n * N0 + i_n0], nl.add, mask=mask
        )
        nisa.dma_copy(src=res, dst=output_reshaped[i_b, n * N0 + i_n0], mask=mask)

    return output


def fused_rmsnorm_gemm(
    x: nt.tensor,
    weight: nt.tensor,
    y: nt.tensor,
    bias: nt.tensor,
    eps: float = 1e-6,
) -> nt.tensor:
    """
    Perform fused RMSNorm + GEMM operation.

    This function applies RMSNorm to the input tensor x, then performs a matrix multiplication
    with tensor y. The operation is fused for better performance.

    Args:
        x: Input tensor to normalize, shape [B, S, H] or [B, H]
        weight: RMSNorm weight tensor, shape [H] or [1, H]
        y: Right-hand side tensor for matrix multiplication, shape [H, N]
        eps: Small epsilon value to avoid division by zero (default: 1e-6)

    Returns:
        Output tensor with shape [B, S, N] or [B, N] depending on input x shape

    Raises:
        ValueError: If input tensor shapes are incompatible
        AssertionError: If batch size exceeds 128 (kernel limitation)
    """
    # Use wrap_nki_kernel to call the NKI kernel from within NeuronPy
    nki_op = wrap_nki_kernel(
        fused_rmsnorm_gemm_v0_nc_transpose,
        [x, weight, y, bias, eps],
    )
    output = nki_op(x, weight, y, bias)
    return output
