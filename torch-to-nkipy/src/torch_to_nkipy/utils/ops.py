# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.library.custom_op("nkipy::mark_subgraph_identity", mutates_args=())
def mark_subgraph_identity(x: torch.Tensor) -> torch.Tensor:
    """
    Mark subgraph boundary with an identity (no-op) operator.

    This operator serves as a graph annotation used to indicate subgraph
    boundaries or partitioning points for custom compilation backends
    (e.g., NKIPy). It performs an identity transformation — returning
    the input tensor unchanged — so that it preserves the original dataflow
    and avoids being removed by dead-code elimination.

    Key properties:
        • No computation is modified (pure identity behavior)
        • No side effects or in-place mutation
        • Exists only to mark graph boundaries during tracing or partitioning

    Example:
        >>> x = mark_subgraph_identity(x)
        # Inserts a boundary marker into the FX/AOT graph while preserving dataflow.
    """
    return x


@torch.library.register_fake("nkipy::mark_subgraph_identity")
def _fake_mark_subgraph_identity(x: torch.Tensor):
    return torch.empty_like(x)


@torch.library.register_kernel("nkipy::mark_subgraph_identity", "cpu")
def _cpu_mark_subgraph_identity(x: torch.Tensor):
    return x.clone()


def _mark_subgraph_identity_backward(ctx, grad_out, x):
    return (grad_out, None)


torch.library.register_autograd(
    "nkipy::mark_subgraph_identity", _mark_subgraph_identity_backward
)
