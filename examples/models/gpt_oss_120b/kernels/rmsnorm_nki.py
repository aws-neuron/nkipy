import nki
import nki.language as nl
import nki.isa as nisa


def stream_shuffle_broadcast(src, dst):
    """Broadcast src's first partition across all partitions of dst.

    src is [1, F], dst is [P, F] with P a multiple of 32. Uses nc_stream_shuffle
    per 32-partition bank, replacing the removed nl.mgrid broadcast idiom.
    """
    dst_npar = dst.shape[0]
    free_dim = dst.shape[1]
    shuffle_mask = [0] * 32

    assert dst_npar % 32 == 0
    for i in range(dst_npar // 32):
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : (i + 1) * 32, 0:free_dim],
            shuffle_mask=shuffle_mask,
        )


@nki.jit
def rmsnorm(x, weight, eps=1e-6):
    """Perform RMSNorm on input tensor using NKI (beta 3 API).

    :param x: Tensor to normalize, shape [B, H] or [B, 1, H] (tokengen).
    :param weight: RMSNorm weight, shape [H] or [1, H].
    :param eps: Small value to avoid division by zero.
    :return: HBM tensor with rmsnorm applied, same shape as ``x``.
    """
    MAX_P = nl.tile_size.pmax  # 128

    output = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    assert x.ndim in (2, 3), f"Malformed shape of x {x.shape}"
    if x.ndim == 2:
        B, H = x.shape
    else:
        B, S, H = x.shape
        assert S == 1, "Only support tokengen"

    assert weight.ndim in (1, 2), f"Malformed shape of weight {weight.shape}"
    if weight.ndim == 2:
        assert weight.shape == (1, H), f"Malformed shape of weight {weight.shape}"
    else:
        assert weight.shape == (H,), f"Malformed shape of weight {weight.shape}"

    # Flatten to 2D (rows in partitions, H in free dim).
    x = x.reshape((B, H))
    output_reshaped = output.reshape((B, H))
    weight = weight.reshape((1, H))

    # Load RMSNorm weight once into SBUF, reused by all row chunks.
    g_tile = nl.ndarray((1, H), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_tile[0:1, 0:H], src=weight[0:1, 0:H])

    num_chunks = (B + MAX_P - 1) // MAX_P
    for i in nl.affine_range(num_chunks):
        p_start = i * MAX_P
        valid_rows = min(MAX_P, B - p_start)

        # Load valid rows from HBM (padded partitions are unused).
        a = nl.ndarray((MAX_P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=a[0:valid_rows, 0:H],
            src=x[p_start : p_start + valid_rows, 0:H],
        )

        # a^2 -> t (reused below as normalized output).
        t = nl.ndarray((MAX_P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=a, data2=a, op=nl.multiply)

        # sum(a^2) across H.
        sq_sum = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=sq_sum, data=t, op=nl.add, axis=1)

        # rsqrt(mean(a^2) + eps).
        s = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=s,
            data=sq_sum,
            op0=nl.multiply,
            operand0=1.0 / H,
            op1=nl.add,
            operand1=eps,
        )
        nisa.activation(dst=s, data=s, op=nl.rsqrt)

        # a * rsqrt -> t.
        nisa.tensor_scalar(dst=t, data=a, operand0=s, op0=nl.multiply)

        # Broadcast weight across partitions and multiply.
        g_bcast = nl.ndarray((MAX_P, H), dtype=g_tile.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(g_tile, g_bcast)
        nisa.tensor_tensor(dst=t, data1=t, data2=g_bcast, op=nl.multiply)

        # Store only valid rows back to HBM.
        nisa.dma_copy(
            dst=output_reshaped[p_start : p_start + valid_rows, 0:H],
            src=t[0:valid_rows, 0:H],
        )

    return output
