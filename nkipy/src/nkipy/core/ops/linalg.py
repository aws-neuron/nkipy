# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear algebra operations: matmul, dot"""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Primitive linalg ops
# -----------------------------------------------------------------------------
matmul = Op("matmul")
dot = Op("dot")

# -----------------------------------------------------------------------------
# Composed linalg ops
# -----------------------------------------------------------------------------

norm = Op("norm")


@norm.composed_impl
def _norm(x, ord=None, axis=None, keepdims=False):
    """L2/Frobenius norm: sqrt(sum(x*x, axis))."""
    from nkipy.core.ops.binary import multiply
    from nkipy.core.ops.reduce import sum
    from nkipy.core.ops.unary import sqrt

    if ord is not None and ord != "fro":
        raise NotImplementedError(
            f"Only L2/Frobenius norm is supported (ord=None or 'fro'), got ord={ord}"
        )

    if ord == "fro":
        if len(x.shape) < 2:
            raise ValueError("Invalid norm order 'fro' for vectors")
        if axis is not None and (not isinstance(axis, (list, tuple)) or len(axis) < 2):
            raise ValueError("Invalid norm order 'fro' for vectors")

    squared = multiply(x, x)
    sum_squared = sum(squared, axis=axis, keepdims=keepdims)
    return sqrt(sum_squared)


outer = Op("outer")


@outer.composed_impl
def _outer(a, b, out=None):
    """Outer product: reshape a to (n, 1), b to (1, m), multiply."""
    from nkipy.core.ops.binary import multiply
    from nkipy.core.ops.transform import reshape

    a_flat = reshape(a, (-1, 1))
    b_flat = reshape(b, (1, -1))
    return multiply(a_flat, b_flat)


trace = Op("trace")
