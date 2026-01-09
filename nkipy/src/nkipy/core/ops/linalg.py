# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear algebra operations: matmul"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# matmul
# -----------------------------------------------------------------------------
matmul = Op("matmul")


@matmul.impl("hlo")
def _matmul_hlo(x, y, out=None, dtype=None):
    """Matrix multiplication (HLO).

    Supports batched matrix multiplication with broadcasting.
    """
    from nkipy.core.backend.hlo import (
        broadcast_to_shape_hlo,
        find_common_type_hlo,
        get_hlo_context,
    )
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    result_dtype = find_common_type_hlo(x, y)

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor
    if isinstance(y, NKIPyTensorRef):
        y = y.backend_tensor

    # Matmul requires at least 1D arrays
    assert len(x.shape) >= 1 and len(y.shape) >= 1, "matmul requires at least 1D arrays"

    # Determine output shape, contracting dimensions, and batch dimensions
    if len(x.shape) == 1 and len(y.shape) == 1:
        # Vector dot product: contract dimension 0 of both
        assert x.shape[0] == y.shape[0], "Incompatible shapes for dot product"
        result_shape = ()
        lhs_contracting_dims = [0]
        rhs_contracting_dims = [0]
        lhs_batch_dims = []
        rhs_batch_dims = []
    elif len(x.shape) == 2 and len(y.shape) == 1:
        # Matrix-vector multiplication
        assert x.shape[1] == y.shape[0], (
            "Incompatible shapes for matrix-vector multiplication"
        )
        result_shape = (x.shape[0],)
        lhs_contracting_dims = [1]
        rhs_contracting_dims = [0]
        lhs_batch_dims = []
        rhs_batch_dims = []
    elif len(x.shape) == 1 and len(y.shape) == 2:
        # Vector-matrix multiplication
        assert x.shape[0] == y.shape[0], (
            "Incompatible shapes for vector-matrix multiplication"
        )
        result_shape = (y.shape[1],)
        lhs_contracting_dims = [0]
        rhs_contracting_dims = [0]
        lhs_batch_dims = []
        rhs_batch_dims = []
    else:
        # General matrix multiplication (2D or batched)
        assert x.shape[-1] == y.shape[-2], "Incompatible shapes for matmul"

        # Broadcast batch dimensions if needed
        x_batch_shape = x.shape[:-2]
        y_batch_shape = y.shape[:-2]
        batch_shape = tuple(np.broadcast_shapes(x_batch_shape, y_batch_shape))
        result_shape = batch_shape + (x.shape[-2], y.shape[-1])

        # If batch dimensions don't match, broadcast the operands first
        target_x_shape = batch_shape + tuple(x.shape[-2:])
        target_y_shape = batch_shape + tuple(y.shape[-2:])

        if x.shape != target_x_shape:
            x = broadcast_to_shape_hlo(ctx, x, target_x_shape)

        if y.shape != target_y_shape:
            y = broadcast_to_shape_hlo(ctx, y, target_y_shape)

        # Contracting dimensions
        lhs_contracting_dims = [len(target_x_shape) - 1]
        rhs_contracting_dims = [len(target_y_shape) - 2]

        # Batch dimensions
        lhs_batch_dims = list(range(len(batch_shape)))
        rhs_batch_dims = list(range(len(batch_shape)))

    # Build dot operation with dimension numbers
    result_tensor = ctx.build_op(
        "dot",
        [x, y],
        result_shape,
        result_dtype,
        {
            "lhs_contracting_dimensions": lhs_contracting_dims,
            "rhs_contracting_dimensions": rhs_contracting_dims,
            "lhs_batch_dimensions": lhs_batch_dims,
            "rhs_batch_dimensions": rhs_batch_dims,
        },
    )

    return NKIPyTensorRef(result_tensor)
