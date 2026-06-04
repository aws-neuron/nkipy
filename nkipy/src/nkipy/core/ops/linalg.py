# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear algebra operations: matmul, dot"""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Primitive linalg ops
# -----------------------------------------------------------------------------
matmul = Op("matmul")
dot = Op("dot")


@dot.composed_impl
def _dot(a, b, out=None):
    """np.dot: sum product over last axis of a and second-to-last of b.

    For 1D/2D inputs, identical to matmul. For N-D × M-D (M>=2),
    decompose into reshape + matmul + reshape to get outer-product
    batch semantics.
    """
    import numpy as np
    from nkipy.core.ops.transform import reshape

    a_ndim = len(a.shape)
    b_ndim = len(b.shape)

    # Cases that match matmul directly
    if a_ndim <= 2 and b_ndim <= 2:
        return matmul(a, b)
    if b_ndim == 1:
        return matmul(a, b)

    # N-D × M-D (M >= 2): outer product on batch dims
    # a: (...A, K), b: (...B, K, N) → result: (...A, ...B, N)
    K = a.shape[-1]
    a_batch = a.shape[:-1]  # (...A)
    b_batch = b.shape[:-2]  # (...B)
    N = b.shape[-1]

    # Flatten a to (prod(a_batch), K) and b to (prod(b_batch), K, N)
    from math import prod
    a_flat = reshape(a, (prod(a_batch), K))
    b_flat = reshape(b, (prod(b_batch), K, N))

    # For each batch element of b, compute a_flat @ b_batch_i
    # This is a_flat @ b_flat[i] for each i — but we can't loop.
    # Instead: transpose b to (K, prod(b_batch)*N), matmul, reshape
    from nkipy.core.ops.transform import transpose as transpose_op
    # b_flat: (prod(b_batch), K, N) → transpose to (K, prod(b_batch)*N)
    b_t = transpose_op(b_flat, (1, 0, 2))  # (K, prod(b_batch), N)
    b_2d = reshape(b_t, (K, prod(b_batch) * N))  # (K, prod(b_batch)*N)
    # matmul: (prod(a_batch), K) @ (K, prod(b_batch)*N) → (prod(a_batch), prod(b_batch)*N)
    result_2d = matmul(a_flat, b_2d)
    # reshape to (...A, ...B, N)
    result_shape = a_batch + b_batch + (N,)
    return reshape(result_2d, result_shape)

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
