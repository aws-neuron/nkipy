# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Einstein summation (einsum) kernel specifications.

This module provides kernel specifications for testing various einsum patterns:
- Matrix multiplication
- Batch operations
- Reductions
- Transposes
- Outer products
"""

import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec

# =============================================================================
# Matrix Operations
# =============================================================================


def matmul_einsum(A, B):
    """Matrix multiplication using einsum: ij,jk->ik"""
    return np.einsum('ij,jk->ik', A, B)


def batch_matmul_einsum(A, B):
    """Batch matrix multiplication using einsum: bij,bjk->bik"""
    return np.einsum('bij,bjk->bik', A, B)


# =============================================================================
# Transpose and Permutation
# =============================================================================


def transpose_einsum(A):
    """Transpose using einsum: ij->ji"""
    return np.einsum('ij->ji', A)


def permute_dims_einsum(A):
    """Permute dimensions using einsum: ijk->kij"""
    return np.einsum('ijk->kij', A)


# =============================================================================
# Reductions
# =============================================================================


def trace_einsum(A):
    """Matrix trace using einsum: ii->"""
    return np.einsum('ii->', A)


def sum_axis_einsum(A):
    """Sum along axis using einsum: ij->i"""
    return np.einsum('ij->i', A)


# =============================================================================
# Outer Products
# =============================================================================


def outer_product_einsum(a, b):
    """Outer product using einsum: i,j->ij"""
    return np.einsum('i,j->ij', a, b)


# =============================================================================
# Advanced Patterns
# =============================================================================


def dot_product_einsum(a, b):
    """Dot product using einsum: i,i->"""
    return np.einsum('i,i->', a, b)


def bilinear_form_einsum(x, A, y):
    """Bilinear form x^T A y using einsum: i,ij,j->"""
    return np.einsum('i,ij,j->', x, A, y)


# =============================================================================
# Kernel Specifications
# =============================================================================

kernel_specs = [
    # Matrix multiplication
    KernelSpec(
        function=matmul_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None], default=(32, 64)),
                dtype_spec=CommonTypes.FLOATS,
                description="First matrix (M, K)",
            ),
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None], default=(64, 48)),
                dtype_spec=CommonTypes.FLOATS,
                description="Second matrix (K, N)",
            ),
        ],
        is_pure_numpy=True,
        description="Matrix multiplication via einsum (ij,jk->ik)",
    ),
    # Batch matrix multiplication
    KernelSpec(
        function=batch_matmul_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None, None], default=(4, 32, 64)),
                dtype_spec=CommonTypes.FLOATS,
                description="First batched matrix (B, M, K)",
            ),
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None, None], default=(4, 64, 48)),
                dtype_spec=CommonTypes.FLOATS,
                description="Second batched matrix (B, K, N)",
            ),
        ],
        is_pure_numpy=True,
        description="Batch matrix multiplication via einsum (bij,bjk->bik)",
    ),
    # Transpose
    KernelSpec(
        function=transpose_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None], default=(32, 64)),
                dtype_spec=CommonTypes.FLOATS,
                description="Matrix to transpose",
            ),
        ],
        is_pure_numpy=True,
        description="2D transpose via einsum (ij->ji)",
    ),
    # Permute dimensions
    KernelSpec(
        function=permute_dims_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None, None], default=(4, 32, 64)),
                dtype_spec=CommonTypes.FLOATS,
                description="3D tensor to permute",
            ),
        ],
        is_pure_numpy=True,
        description="3D permutation via einsum (ijk->kij)",
    ),
    # Sum along axis
    KernelSpec(
        function=sum_axis_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None, None], default=(32, 64)),
                dtype_spec=CommonTypes.FLOATS,
                description="Matrix to reduce",
            ),
        ],
        is_pure_numpy=True,
        description="Sum along last axis via einsum (ij->i)",
    ),
    # Outer product
    KernelSpec(
        function=outer_product_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None], default=(32,)),
                dtype_spec=CommonTypes.FLOATS,
                description="First vector",
            ),
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None], default=(64,)),
                dtype_spec=CommonTypes.FLOATS,
                description="Second vector",
            ),
        ],
        is_pure_numpy=True,
        description="Outer product via einsum (i,j->ij)",
    ),
    # Dot product
    KernelSpec(
        function=dot_product_einsum,
        inputs=[
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None], default=(128,)),
                dtype_spec=CommonTypes.FLOATS,
                description="First vector",
            ),
            TensorInputSpec(
                shape_spec=ShapeSpec(dims=[None], default=(128,)),
                dtype_spec=CommonTypes.FLOATS,
                description="Second vector",
            ),
        ],
        is_pure_numpy=True,
        description="Dot product via einsum (i,i->)",
    ),
]
