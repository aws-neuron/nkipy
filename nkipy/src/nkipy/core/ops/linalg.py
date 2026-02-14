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
    Follows numpy semantics for 1D inputs: a 1D left operand is promoted to
    2D by prepending a 1, a 1D right operand by appending a 1, and the extra
    dimension is removed from the result after the dot.
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

    # Handle 1D inputs by promoting to 2D (numpy matmul semantics).
    # The added dimension is stripped from the result after the dot.
    squeeze_lhs = False
    squeeze_rhs = False

    if len(x.shape) == 1 and len(y.shape) == 1:
        # Vector dot product: contract dimension 0 of both
        assert x.shape[0] == y.shape[0], "Incompatible shapes for dot product"
        result_shape = ()
        lhs_contracting_dims = [0]
        rhs_contracting_dims = [0]
        lhs_batch_dims = []
        rhs_batch_dims = []
    else:
        # Promote 1D operands to 2D so the general path handles them.
        if len(x.shape) == 1:
            # (K,) -> (1, K)
            x = ctx.build_op("reshape", [x], (1, x.shape[0]), x.dtype)
            squeeze_lhs = True

        if len(y.shape) == 1:
            # (K,) -> (K, 1)
            y = ctx.build_op("reshape", [y], (y.shape[0], 1), y.dtype)
            squeeze_rhs = True

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

    # Strip the dimensions that were added for 1D promotion.
    if squeeze_lhs or squeeze_rhs:
        final_shape = list(result_shape)
        # squeeze_lhs removes the second-to-last dim (the prepended 1)
        # squeeze_rhs removes the last dim (the appended 1)
        # When both are true the result is already scalar from the 1D x 1D
        # path above, so this branch won't fire for that case.
        if squeeze_lhs:
            final_shape.pop(-2)
        if squeeze_rhs:
            final_shape.pop(-1)
        final_shape = tuple(final_shape)
        result_tensor = ctx.build_op(
            "reshape", [result_tensor], final_shape, result_dtype
        )

    return NKIPyTensorRef(result_tensor)
