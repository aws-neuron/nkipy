import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
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


@nki.jit
def nki_rmsnorm_kernel(input_tensor, weight, eps):
    """
    RMSNorm NKI kernel - based on AWS official tutorial pattern.

    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]
        weight: RMSNorm weight parameter [hidden_size]
        eps: Small epsilon for numerical stability

    Returns:
        output: Normalized tensor with same shape as input
    """
    # Create output tensor in shared HBM
    output = nl.ndarray(
        input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm
    )

    # Make sure shapes match
    assert input_tensor.shape[1] == weight.shape[0]

    # Generate tensor indices
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(input_tensor.shape[1])[None, :]

    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]

    # Load RMSNorm weight once, reused by all rows
    g_tile = nl.load(weight.reshape((1, weight.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    for i in nl.affine_range(math.ceil(input_tensor.shape[0] / 128)):
        # Load input data from HBM to SBUF
        a_tile = nl.load(input_tensor[i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

        # Compute element-wise square
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements along last dimension
        square_sum = nl.sum(in_square, axis=[1])

        # Compute mean
        mean = square_sum / hidden_size

        # Take reciprocal of sqrt with eps
        rms_reciprocal = nl.rsqrt(mean + eps)

        # Normalize: multiply input by reciprocal of RMS
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        g_bcast = g_tile.broadcast_to((128, hidden_size))

        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

        # Store results back to HBM
        nl.store(
            output[i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows)
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

    logits = h[:, h.shape[1] - 1, :] @ lm_head_weight

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
        device_idx = np.copy(top_index[b])
        local_idx = next_id_all[b, device_idx]
        global_idx = device_idx * vocab_per_device + local_idx
        final_next_id[b] = global_idx

    return final_next_id
