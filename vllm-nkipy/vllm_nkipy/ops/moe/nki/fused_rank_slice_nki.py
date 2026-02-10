# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

from torch_to_nkipy.utils.nki import NKIOpRegistry


# @nki.compiler.skip_middle_end_transformations
# @nki.jit
@NKIOpRegistry.register("mylib::rank_slice_add_kernel_custom_op")
def rank_slice_add_kernel(
    x: nki.tensor, y: nki.tensor, rank: nki.tensor, batch_size_per_rank: int
):
    """
    NKI kernel for element-wise addition with rank-based slicing.

    Args:
        x: Input tensor (batch_size, hidden_size).
        y: Input tensor (batch_size, hidden_size).
        rank: Rank index tensor.
        batch_size_per_rank: Batch elements per rank.

    Returns:
        Output tensor (batch_size_per_rank, hidden_size) with x + y for specified rank.
    """
    assert len(x.shape) == 2, "The x tensor must have shape (batch_size, hidden_size)"
    assert len(y.shape) == 2, "The y tensor must have shape (batch_size, hidden_size)"
    assert x.shape == y.shape, "The x tensor and y tensor must have the same shape"
    B, H = x.shape
    rank_sbuf = nl.load(rank)
    num_ranks = B // batch_size_per_rank
    assert B % batch_size_per_rank == 0, (
        "batch_size must be divisible by batch_size_per_rank"
    )
    x_reshaped = x.reshape((num_ranks, batch_size_per_rank, H))
    y_reshaped = y.reshape((num_ranks, batch_size_per_rank, H))
    output = nl.ndarray((batch_size_per_rank, H), dtype=x.dtype, buffer=nl.hbm)
    x_sbuf = nl.load(src=x_reshaped[rank_sbuf[0, 0], :, :])
    y_sbuf = nl.load(src=y_reshaped[rank_sbuf[0, 0], :, :])
    y = nisa.tensor_tensor(x_sbuf, y_sbuf, nl.add)
    nisa.dma_copy(src=y, dst=output)
    return output
