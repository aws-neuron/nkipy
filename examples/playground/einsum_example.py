#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example demonstrating einsum operation in NKIPy.

Einstein summation notation provides a concise and powerful way to
express tensor operations including matrix multiplication, transposes,
reductions, and complex tensor contractions.
"""

import numpy as np

from nkipy.runtime.decorators import simulate_jit


# =============================================================================
# Matrix Operations
# =============================================================================

@simulate_jit
def matmul_einsum(A, B):
    """Matrix multiplication using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('ij,jk->ik', A, B)


@simulate_jit
def batch_matmul_einsum(A, B):
    """Batch matrix multiplication using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('bij,bjk->bik', A, B)


# =============================================================================
# Transpose and Permutation
# =============================================================================

@simulate_jit
def transpose_einsum(A):
    """Transpose using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('ij->ji', A)


@simulate_jit
def permute_dims_einsum(A):
    """Permute dimensions using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('ijk->kij', A)


# =============================================================================
# Reductions
# =============================================================================

@simulate_jit
def trace_einsum(A):
    """Matrix trace using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('ii->', A)


@simulate_jit
def sum_axis_einsum(A):
    """Sum along axis using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('ij->i', A)


# =============================================================================
# Outer Products
# =============================================================================

@simulate_jit
def outer_product_einsum(a, b):
    """Outer product using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('i,j->ij', a, b)


# =============================================================================
# Advanced Patterns
# =============================================================================

@simulate_jit
def dot_product_einsum(a, b):
    """Dot product using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('i,i->', a, b)


@simulate_jit
def bilinear_form_einsum(x, A, y):
    """Bilinear form x^T A y using einsum."""
    import nkipy.core.ops as ops
    return ops.einsum('i,ij,j->', x, A, y)


@simulate_jit
def attention_pattern_einsum(Q, K, V):
    """Simplified attention pattern: Q @ K^T @ V using einsum.
    
    This computes (Q @ K^T) @ V in one operation.
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_v, d_v)
    
    Note: This is simplified - real attention includes scaling and softmax.
    """
    import nkipy.core.ops as ops
    # Q @ K^T: 'bid,bjd->bij' where i=seq_q, j=seq_k
    # (Q @ K^T) @ V: 'bij,bjd->bid' where final d is d_v
    # Combined: 'bik,bjk,bjd->bid'
    return ops.einsum('bik,bjk,bjd->bid', Q, K, V)


def main():
    print("=" * 70)
    print("NKIPy einsum Examples")
    print("=" * 70)
    
    # Example 1: Matrix Multiplication
    print("\n1. Matrix Multiplication (ij,jk->ik):")
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    result = matmul_einsum(A, B)
    print(f"A @ B =\n{result}")
    print(f"NumPy result:\n{np.einsum('ij,jk->ik', A, B)}")
    
    # Example 2: Batch Matrix Multiplication
    print("\n2. Batch Matrix Multiplication (bij,bjk->bik):")
    A_batch = np.random.rand(2, 3, 4).astype(np.float32)
    B_batch = np.random.rand(2, 4, 5).astype(np.float32)
    result_batch = batch_matmul_einsum(A_batch, B_batch)
    print(f"Batch A shape: {A_batch.shape}")
    print(f"Batch B shape: {B_batch.shape}")
    print(f"Result shape: {result_batch.shape}")
    
    # Example 3: Transpose
    print("\n3. Transpose (ij->ji):")
    C = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"C =\n{C}")
    result_transpose = transpose_einsum(C)
    print(f"C^T =\n{result_transpose}")
    
    # Example 4: Trace
    print("\n4. Matrix Trace (ii->):")
    D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    print(f"D =\n{D}")
    trace = trace_einsum(D)
    print(f"Trace(D) = {trace}")
    print(f"NumPy trace: {np.trace(D)}")
    
    # Example 5: Outer Product
    print("\n5. Outer Product (i,j->ij):")
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5], dtype=np.float32)
    print(f"a = {a}")
    print(f"b = {b}")
    result_outer = outer_product_einsum(a, b)
    print(f"Outer product a ⊗ b =\n{result_outer}")
    
    # Example 6: Dot Product
    print("\n6. Dot Product (i,i->):")
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([5, 6, 7, 8], dtype=np.float32)
    print(f"x = {x}")
    print(f"y = {y}")
    dot = dot_product_einsum(x, y)
    print(f"x · y = {dot}")
    print(f"NumPy dot: {np.dot(x, y)}")
    
    # Example 7: Bilinear Form
    print("\n7. Bilinear Form x^T A y (i,ij,j->):")
    x = np.array([1, 2], dtype=np.float32)
    A = np.array([[3, 4], [5, 6]], dtype=np.float32)
    y = np.array([7, 8], dtype=np.float32)
    print(f"x = {x}")
    print(f"A =\n{A}")
    print(f"y = {y}")
    bilinear = bilinear_form_einsum(x, A, y)
    print(f"x^T A y = {bilinear}")
    
    # Example 8: Sum along axis
    print("\n8. Sum Along Axis (ij->i):")
    E = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"E =\n{E}")
    sum_result = sum_axis_einsum(E)
    print(f"Sum along axis 1: {sum_result}")
    
    # Example 9: Attention-like pattern (simplified)
    print("\n9. Simplified Attention Pattern (bik,bjk,bjd->bid):")
    Q = np.random.rand(1, 2, 3).astype(np.float32)  # (batch=1, seq_q=2, d_k=3)
    K = np.random.rand(1, 4, 3).astype(np.float32)  # (batch=1, seq_k=4, d_k=3)
    V = np.random.rand(1, 4, 5).astype(np.float32)  # (batch=1, seq_v=4, d_v=5)
    print(f"Q shape: {Q.shape} (batch, seq_q, d_k)")
    print(f"K shape: {K.shape} (batch, seq_k, d_k)")
    print(f"V shape: {V.shape} (batch, seq_v, d_v)")
    attn_result = attention_pattern_einsum(Q, K, V)
    print(f"Attention output shape: {attn_result.shape} (batch, seq_q, d_v)")
    
    print("\n" + "=" * 70)
    print("All einsum examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
