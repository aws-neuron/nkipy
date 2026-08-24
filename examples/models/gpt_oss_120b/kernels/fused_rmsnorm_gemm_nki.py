from nkipy.core.nki_op import wrap_nki_kernel

import numpy as np
import nkipy.core.typing as nt
import nki
import nki.language as nl
import nki.isa as nisa
import math


@nki.jit
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

    assert x.ndim in (2, 3), f"Malformed shape of x {x.shape}"
    if x.ndim == 2:
        B, H = x.shape
        S = 1
    else:
        B, S, H = x.shape
        assert S == 1, "Only support tokengen"

    assert weight.ndim in (1, 2), f"Malformed shape of weight {weight.shape}"
    if weight.ndim == 2:
        assert weight.shape == (1, H), f"Malformed shape of weight {weight.shape}"
    else:
        assert weight.shape == (H,), f"Malformed shape of weight {weight.shape}"

    H_, N = y.shape
    assert H == H_, f"Incompatible matmul shape {x.shape} @ {y.shape}"
    assert bias.ndim in (1, 2), f"Malformed shape of bias {bias.shape}"
    if bias.ndim == 2:
        assert bias.shape == (1, N), f"Malformed shape of bias {bias.shape}"
    else:
        assert bias.shape == (N,), f"Malformed shape of bias {bias.shape}"

    assert B <= 128, "This kernel only support max batch size of 128"
    # Create output tensor with original shape and dtype
    output = nl.ndarray((B, S, N), dtype=x.dtype, buffer=nl.shared_hbm)

    # Unify Shapes
    x = x.reshape((B, H))
    output_reshaped = output.reshape((B, N))
    weight_col = weight.reshape((H, 1))
    bias = bias.reshape((1, N))

    H0 = nl.tile_size.pmax  # 128
    H1 = math.ceil(H / H0)
    N_TILE = nl.tile_size.gemm_moving_fmax  # 512
    NT = math.ceil(N / N_TILE)

    # --- RMSNorm with rows (B) in the partition dim ---
    # Load x (B, H).
    x_sb = nl.ndarray((B, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb[0:B, 0:H], src=x[0:B, 0:H])

    # sum(x^2) over the free (H) dim, then rms_inv = 1/sqrt(mean + eps).
    sq = nl.ndarray((B, H), dtype=rms_compute_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x_sb, data2=x_sb, op=nl.multiply)
    ss = nl.ndarray((B, 1), dtype=rms_compute_dtype, buffer=nl.psum)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1)
    rms_inv = nl.ndarray((B, 1), dtype=rms_compute_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=rms_inv, data=ss, op0=nl.multiply, operand0=1.0 / H, op1=nl.add, operand1=eps
    )
    nisa.activation(dst=rms_inv, op=nl.rsqrt, data=rms_inv)

    # normed = x * rms_inv (rms_inv is a per-partition scalar broadcast over H).
    normed = nl.ndarray((B, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=normed, data=x_sb, op0=nl.multiply, operand0=rms_inv)

    # Load bias (1, N) once; broadcast on the partition dim when added.
    bias_sb = nl.ndarray((1, N), dtype=bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sb[0:1, 0:N], src=bias[0:1, 0:N])

    # --- out = (normed * weight) @ y + bias, contracting over H in H0 tiles ---
    for nt_i in range(NT):
        n_start = nt_i * N_TILE
        n_size = min(N_TILE, N - n_start)
        psum = nl.zeros((B, N_TILE), dtype=nl.float32, buffer=nl.psum)
        for h_i in range(H1):
            h_start = h_i * H0
            h_size = min(H0, H - h_start)

            # Transpose normed[:, h_tile] (B, h_size) -> (h_size, B) so the
            # contraction dim H sits on the partition axis for nc_matmul. The
            # Tensor-engine transpose is matmul-based and must land in PSUM.
            normed_T_psum = nl.ndarray((H0, B), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=normed_T_psum[0:h_size, 0:B],
                data=normed[0:B, h_start : h_start + h_size],
                engine=nisa.engine.tensor,
            )
            normed_T = nl.ndarray((H0, B), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=normed_T[0:h_size, 0:B], src=normed_T_psum[0:h_size, 0:B])

            # Fold RMSNorm weight in as a per-partition scalar (weight is [H,1],
            # so weight[h_tile] is [h_size, 1] broadcast over the B free dim).
            # tensor_scalar requires an fp32 scalar operand.
            w_h = nl.ndarray((H0, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_h[0:h_size, 0:1], src=weight_col[h_start : h_start + h_size, 0:1]
            )
            nisa.tensor_scalar(
                dst=normed_T[0:h_size, 0:B],
                data=normed_T[0:h_size, 0:B],
                op0=nl.multiply,
                operand0=w_h[0:h_size, 0:1],
            )

            # y[h_tile, n_tile] -> [h_size, n_size], contraction dim on partition.
            y_h = nl.ndarray((H0, N_TILE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=y_h[0:h_size, 0:n_size],
                src=y[h_start : h_start + h_size, n_start : n_start + n_size],
            )

            # psum[B, n] += normed_T_weighted[h_size, B].T @ y_h[h_size, n].
            # First h-tile initializes the PSUM bank; later tiles accumulate.
            nisa.nc_matmul(
                dst=psum[0:B, 0:n_size],
                stationary=normed_T[0:h_size, 0:B],
                moving=y_h[0:h_size, 0:n_size],
                accumulate=(h_i > 0),
            )

        # Add bias (broadcast the (1, n_size) tile over the B partitions) and store.
        res = nl.ndarray((B, N_TILE), dtype=x.dtype, buffer=nl.sbuf)
        bias_block = nl.broadcast_to(
            bias_sb[0:1, n_start : n_start + n_size], (B, n_size)
        )
        nisa.tensor_tensor(
            dst=res[0:B, 0:n_size],
            data1=psum[0:B, 0:n_size],
            data2=bias_block,
            op=nl.add,
        )
        nisa.dma_copy(
            dst=output_reshaped[0:B, n_start : n_start + n_size],
            src=res[0:B, 0:n_size],
        )

    return output


# Historical alternate implementation. The original dma_transpose variant used
# nl.mgrid/par_dim indexing that the beta-3 NKI frontend no longer supports and
# was already flagged as hitting a compiler error. Alias it to the working
# nc_transpose kernel so both public symbols behave identically.
fused_rmsnorm_gemm_v0_dma_transpose = fused_rmsnorm_gemm_v0_nc_transpose


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
    # Use wrap_nki_kernel to call the NKI kernel from within NKIPy
    nki_op = wrap_nki_kernel(
        fused_rmsnorm_gemm_v0_nc_transpose,
        [x, weight, y, bias],
        is_nki_beta_3_version=True,
        kernel_kwargs={"eps": eps},
    )
    output = nki_op(x, weight, y, bias)
    return output
