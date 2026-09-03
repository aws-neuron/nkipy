from nkipy.core.nki_op import wrap_nki_kernel

import nkipy.core.typing as nt
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rank_slice_add_kernel(
    x: nl.NkiTensor, y: nl.NkiTensor, rank: nl.NkiTensor, batch_size_per_rank: int
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
    num_ranks = B // batch_size_per_rank
    assert (
        B % batch_size_per_rank == 0
    ), "batch_size must be divisible by batch_size_per_rank"

    # Load the rank index into SBUF and use it as a scalar_offset for a dynamic
    # (data-dependent) DMA gather. Fancy-indexing an HBM tensor by an SBUF value
    # (x[rank_sbuf[0, 0]]) is not honored by the beta-3 frontend -- it silently
    # pins to index 0 -- so we build an explicit access pattern instead.
    rank_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank_sbuf[0:1, 0:1], src=rank[0:1, 0:1])

    x_reshaped = x.reshape((num_ranks, batch_size_per_rank, H))
    y_reshaped = y.reshape((num_ranks, batch_size_per_rank, H))
    output = nl.ndarray((batch_size_per_rank, H), dtype=x.dtype, buffer=nl.shared_hbm)

    # Gather rows [rank*bspr : (rank+1)*bspr] via scalar_offset on dim 0.
    x_sbuf = nl.ndarray((batch_size_per_rank, H), dtype=x.dtype, buffer=nl.sbuf)
    y_sbuf = nl.ndarray((batch_size_per_rank, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_sbuf[0:batch_size_per_rank, 0:H],
        src=x_reshaped.ap(
            pattern=[[H, batch_size_per_rank], [1, H]],
            offset=0,
            scalar_offset=rank_sbuf,
            indirect_dim=0,
        ),
    )
    nisa.dma_copy(
        dst=y_sbuf[0:batch_size_per_rank, 0:H],
        src=y_reshaped.ap(
            pattern=[[H, batch_size_per_rank], [1, H]],
            offset=0,
            scalar_offset=rank_sbuf,
            indirect_dim=0,
        ),
    )
    out_sbuf = nl.ndarray((batch_size_per_rank, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_sbuf, data1=x_sbuf, data2=y_sbuf, op=nl.add)
    nisa.dma_copy(src=out_sbuf, dst=output)
    return output


def fused_rank_slice_add(
    x: nt.tensor,
    y: nt.tensor,
    rank: nt.tensor,
    batch_size_per_rank: int,
) -> nt.tensor:
    """
    NKIPy wrapper for fused rank-based slicing and element-wise addition.

    Args:
        x: Input tensor (batch_size, hidden_size).
        y: Input tensor (batch_size, hidden_size).
        rank: Rank index tensor.
        batch_size_per_rank: Batch elements per rank.

    Returns:
        Output tensor (batch_size_per_rank, hidden_size) with x + y for specified rank.
    """
    # Use wrap_nki_kernel to call the NKI kernel from within NKIPy
    nki_op = wrap_nki_kernel(
        rank_slice_add_kernel,
        [x, y, rank],
        is_nki_beta_3_version=True,
        kernel_kwargs={"batch_size_per_rank": batch_size_per_rank},
    )
    output = nki_op(x, y, rank)
    return output
