# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch

from vllm_nkipy.compile import local_compile
from vllm_nkipy.ops.moe.nki.fused_rank_slice_nki import rank_slice_add_kernel


@torch.library.custom_op("mylib::rank_slice_add_kernel_custom_op", mutates_args=())
def rank_slice_add_kernel_custom_op(
    x: torch.Tensor,
    y: torch.Tensor,
    rank: torch.Tensor,
    batch_size_per_rank: int,
) -> torch.Tensor:
    output = rank_slice_add_kernel(
        x,
        y,
        rank,
        batch_size_per_rank,
    )
    return output


@rank_slice_add_kernel_custom_op.register_fake
def _(
    x: torch.Tensor,
    y: torch.Tensor,
    rank: torch.Tensor,
    batch_size_per_rank: int,
) -> torch.Tensor:
    _, d = x.shape
    return torch.empty(batch_size_per_rank, d, dtype=x.dtype, device=x.device)


@local_compile(
    backend="nkipy",
    device="nkipy",
    force=True,
    name="rank_slice_add_kernel_custom_op_compiled",
    fullgraph=True,
    dynamic=False,
)
def rank_slice_add_kernel_custom_op_compiled(
    x: torch.Tensor,
    y: torch.Tensor,
    rank: torch.Tensor,
    batch_size_per_rank: int,
) -> torch.Tensor:
    return rank_slice_add_kernel_custom_op(
        x=x,
        y=y,
        rank=rank,
        batch_size_per_rank=batch_size_per_rank,
    )
