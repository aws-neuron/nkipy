# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public knob() API for annotating tensors with hardware placement and tiling hints.

Dispatches based on the active tracing backend:
- kernelgen: emits nkipy.AnnotateOp into MLIR
- hlo: warns and ignores
- cpu / no trace: no-op pass-through
"""

from __future__ import annotations

import warnings
from typing import List, Optional


def knob(
    tensor,
    *,
    partition_dim: Optional[int] = None,
    mem_space: Optional[str] = None,
    tile_size: Optional[List[int]] = None,
    reduction_tile: Optional[List[int]] = None,
):
    """Annotate a tensor with hardware placement and tiling hints.

    Only effective when using the kernelgen backend.  When used with the HLO
    backend, issues a warning and returns the tensor unchanged.

    Args:
        tensor: The tensor to annotate.
        partition_dim: Dimension to partition (must be < tensor rank).
        mem_space: Memory space ("Hbm", "Psum", "Sbuf", or "SharedHbm").
        tile_size: Tile sizes for each dimension.
        reduction_tile: Tile sizes for reduction dimensions (e.g., K in matmul).

    Returns:
        The same tensor, unchanged.
    """
    from nkipy.core.backend import get_backend

    backend = get_backend()

    if backend == "kernelgen":
        from nkipy.core.tensor import NKIPyTensorRef

        if not isinstance(tensor, NKIPyTensorRef):
            return tensor

        if mem_space is None and partition_dim is None and tile_size is None and reduction_tile is None:
            return tensor

        import nkipy_kernelgen.builder as B

        B.annotate(
            tensor.backend_tensor.handle,
            partition_dim=partition_dim,
            mem_space=mem_space,
            tile_size=tile_size,
            reduction_tile=reduction_tile,
        )
        return tensor
    elif backend == "hlo":
        warnings.warn(
            "knob() annotations are only effective with backend='kernelgen'. "
            "Ignoring annotation.",
            stacklevel=2,
        )

    return tensor
