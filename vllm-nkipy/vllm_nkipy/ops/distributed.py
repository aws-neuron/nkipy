# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# TODO: complete custom all-gather w/ arg dim
import torch
from torch.distributed import ProcessGroup


# TODO: hint: not use process group, follow torch code
@torch.library.custom_op("vllm_nkipy:custom_all_gather", mutates_args=())
def custom_all_gather_impl(
    data: torch.Tensor, all_gather_dim: int, replica_groups: ProcessGroup
) -> torch.Tensor:
    """Custom op implementation for all_gather using NKIPy backend."""
    pass


@custom_all_gather_impl.register_fake
def custom_all_gather_fake(
    data: torch.Tensor, all_gather_dim: int, replica_groups: ProcessGroup
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    group_size = (
        len(replica_groups[0]) if replica_groups and len(replica_groups) > 0 else 1
    )
    out_shape = list(data.shape)
    if out_shape:
        out_shape[all_gather_dim] *= group_size
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


# def custom_all_gather(data: torch.Tensor, all_gather_dim: int, replica_groups: ProcessGroup) -> torch.Tensor:  # noqa
