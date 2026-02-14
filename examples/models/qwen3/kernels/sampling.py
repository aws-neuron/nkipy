import nki
import nki.isa as nisa
import nki.language as nl
import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

# Import config from parent directory
from config import Config
from nkipy.core import (
    nki_op,  # noqa: F401, make sure monkey patch is applied
    tensor_apis,
)

# Import kernels from the kernels directory
from .rmsnorm import rmsnorm_kernel


def stream_shuffle_broadcast(src, dst):
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


@nki.jit(platform_target="trn2")
def nki_rmsnorm_kernel(input_tensor, weight, eps):
    """
    RMSNorm NKI kernel - based on AWS official tutorial pattern.
    Migrated to NKI Beta 2 API.

    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]
        weight: RMSNorm weight parameter [hidden_size]
        eps: Small epsilon for numerical stability

    Returns:
        output: Normalized tensor with same shape as input
    """
    MAX_P = 128

    output = nl.ndarray(
        input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm
    )
    assert input_tensor.shape[1] == weight.shape[0]

    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    num_chunks = (num_rows + MAX_P - 1) // MAX_P

    # Load RMSNorm weight once into SBUF, reused by all rows
    g_tile = nl.ndarray((1, hidden_size), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=g_tile[0:1, 0:hidden_size],
        src=weight.reshape((1, hidden_size))[0:1, 0:hidden_size],
    )

    for i in nl.affine_range(num_chunks):
        p_start = i * MAX_P
        valid_rows = min(MAX_P, num_rows - p_start)

        # Load valid rows from HBM (padded partitions are unused)
        a = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=a[0:valid_rows, 0:hidden_size],
            src=input_tensor[p_start : p_start + valid_rows, 0:hidden_size],
        )

        # a^2 -> t (reused below as normalized output)
        t = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=a, data2=a, op=nl.multiply)

        # sum(a^2)
        sq_sum = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=sq_sum, data=t, op=nl.add, axis=1)

        # rsqrt(mean(a^2) + eps), in-place
        s = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=s,
            data=sq_sum,
            op0=nl.multiply,
            operand0=1.0 / hidden_size,
            op1=nl.add,
            operand1=eps,
        )
        nisa.activation(dst=s, data=s, op=nl.rsqrt)

        # a * rsqrt -> t
        nisa.tensor_scalar(dst=t, data=a, operand0=s, op0=nl.multiply)

        # Broadcast weight and multiply
        g_bcast = nl.ndarray((MAX_P, hidden_size), dtype=g_tile.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(g_tile, g_bcast)
        nisa.tensor_tensor(dst=t, data1=t, data2=g_bcast, op=nl.multiply)

        # Store only valid rows back to HBM
        nisa.dma_copy(
            dst=output[p_start : p_start + valid_rows, 0:hidden_size],
            src=t[0:valid_rows, 0:hidden_size],
        )

    return output


def greedy_sampling(
    h, norm_weight, lm_head_weight, configs: Config, use_nki_rmsnorm=False
):
    """Greedy sampling kernel for token generation."""

    B, S, H = h.shape
    # Note: this is just for showing how to use use a NKI kernel inside NKIPy
    if use_nki_rmsnorm:
        h = h.reshape(-1, H)  # batch*seq_len, hidden_size
        h = nki_rmsnorm_kernel(h, norm_weight, configs.norm_eps)
        h = h.reshape(B, S, H)
    else:
        h = rmsnorm_kernel(h, norm_weight, configs.norm_eps)

    logits = h[:, -1, :] @ lm_head_weight

    logits, next_id = tensor_apis.topk(logits, k=1, axis=1)
    logits_all = cc.all_gather(
        logits, all_gather_dim=1, replica_groups=[list(range(dist.get_world_size()))]
    )
    next_id_all = cc.all_gather(
        next_id, all_gather_dim=1, replica_groups=[list(range(dist.get_world_size()))]
    )

    _, top_index = tensor_apis.topk(logits_all, k=1, axis=1)
    final_next_id = np.empty_like(next_id)

    vocab_per_device = lm_head_weight.shape[1]
    for b in range(configs.max_batch_size):
        device_idx = top_index[b]
        local_idx = next_id_all[b, device_idx]
        global_idx = device_idx * vocab_per_device + local_idx
        final_next_id[b] = global_idx

    return final_next_id
