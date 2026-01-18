# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for einsum operation."""

import numpy as np
import pytest

from nkipy.runtime.decorators import simulate_jit


class TestEinsumMatmul:
    """Test einsum for matrix multiplication patterns."""

    def test_matmul_basic(self):
        """Test basic matrix multiplication: ij,jk->ik"""
        @simulate_jit
        def kernel_matmul(A, B):
            import nkipy.core.ops as ops
            return ops.einsum('ij,jk->ik', A, B)

        A = np.random.rand(3, 4).astype(np.float32)
        B = np.random.rand(4, 5).astype(np.float32)
        
        result = kernel_matmul(A, B)
        expected = np.einsum('ij,jk->ik', A, B)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_matmul_implicit_output(self):
        """Test matrix multiplication with implicit output: ij,jk"""
        @simulate_jit
        def kernel_matmul_implicit(A, B):
            import nkipy.core.ops as ops
            return ops.einsum('ij,jk', A, B)

        A = np.random.rand(3, 4).astype(np.float32)
        B = np.random.rand(4, 5).astype(np.float32)
        
        result = kernel_matmul_implicit(A, B)
        expected = np.einsum('ij,jk', A, B)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch_matmul(self):
        """Test batched matrix multiplication: bij,bjk->bik"""
        @simulate_jit
        def kernel_batch_matmul(A, B):
            import nkipy.core.ops as ops
            return ops.einsum('bij,bjk->bik', A, B)

        A = np.random.rand(2, 3, 4).astype(np.float32)
        B = np.random.rand(2, 4, 5).astype(np.float32)
        
        result = kernel_batch_matmul(A, B)
        expected = np.einsum('bij,bjk->bik', A, B)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_vector_dot_product(self):
        """Test vector dot product: i,i->"""
        @simulate_jit
        def kernel_dot(a, b):
            import nkipy.core.ops as ops
            return ops.einsum('i,i->', a, b)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        
        result = kernel_dot(a, b)
        expected = np.einsum('i,i->', a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiplication: ij,j->i"""
        @simulate_jit
        def kernel_matvec(A, b):
            import nkipy.core.ops as ops
            return ops.einsum('ij,j->i', A, b)

        A = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4).astype(np.float32)
        
        result = kernel_matvec(A, b)
        expected = np.einsum('ij,j->i', A, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumTranspose:
    """Test einsum for transpose operations."""

    def test_transpose_2d(self):
        """Test 2D transpose: ij->ji"""
        @simulate_jit
        def kernel_transpose(A):
            import nkipy.core.ops as ops
            return ops.einsum('ij->ji', A)

        A = np.random.rand(3, 4).astype(np.float32)
        result = kernel_transpose(A)
        expected = np.einsum('ij->ji', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_transpose_3d(self):
        """Test 3D transpose: ijk->kji"""
        @simulate_jit
        def kernel_transpose_3d(A):
            import nkipy.core.ops as ops
            return ops.einsum('ijk->kji', A)

        A = np.random.rand(2, 3, 4).astype(np.float32)
        result = kernel_transpose_3d(A)
        expected = np.einsum('ijk->kji', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_permute_dims(self):
        """Test dimension permutation: ijk->jki"""
        @simulate_jit
        def kernel_permute(A):
            import nkipy.core.ops as ops
            return ops.einsum('ijk->jki', A)

        A = np.random.rand(2, 3, 4).astype(np.float32)
        result = kernel_permute(A)
        expected = np.einsum('ijk->jki', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumReduction:
    """Test einsum for reduction operations."""

    def test_sum_all(self):
        """Test sum of all elements: ij->"""
        @simulate_jit
        def kernel_sum_all(A):
            import nkipy.core.ops as ops
            return ops.einsum('ij->', A)

        A = np.random.rand(3, 4).astype(np.float32)
        result = kernel_sum_all(A)
        expected = np.einsum('ij->', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sum_axis(self):
        """Test sum along axis: ij->i"""
        @simulate_jit
        def kernel_sum_axis(A):
            import nkipy.core.ops as ops
            return ops.einsum('ij->i', A)

        A = np.random.rand(3, 4).astype(np.float32)
        result = kernel_sum_axis(A)
        expected = np.einsum('ij->i', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_trace(self):
        """Test matrix trace: ii->"""
        @simulate_jit
        def kernel_trace(A):
            import nkipy.core.ops as ops
            return ops.einsum('ii->', A)

        A = np.random.rand(4, 4).astype(np.float32)
        result = kernel_trace(A)
        expected = np.einsum('ii->', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_diagonal(self):
        """Test extracting diagonal: ii->i"""
        @simulate_jit
        def kernel_diagonal(A):
            import nkipy.core.ops as ops
            return ops.einsum('ii->i', A)

        A = np.random.rand(4, 4).astype(np.float32)
        result = kernel_diagonal(A)
        expected = np.einsum('ii->i', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumOuterProduct:
    """Test einsum for outer product operations."""

    def test_outer_product(self):
        """Test outer product: i,j->ij"""
        @simulate_jit
        def kernel_outer(a, b):
            import nkipy.core.ops as ops
            return ops.einsum('i,j->ij', a, b)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0], dtype=np.float32)
        
        result = kernel_outer(a, b)
        expected = np.einsum('i,j->ij', a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_outer_product_3d(self):
        """Test 3D outer product: i,j,k->ijk"""
        @simulate_jit
        def kernel_outer_3d(a, b, c):
            import nkipy.core.ops as ops
            return ops.einsum('i,j,k->ijk', a, b, c)

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([5.0, 6.0], dtype=np.float32)
        
        result = kernel_outer_3d(a, b, c)
        expected = np.einsum('i,j,k->ijk', a, b, c)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumBroadcast:
    """Test einsum for broadcasting operations."""

    def test_broadcast_multiply(self):
        """Test element-wise multiply with broadcasting: ij,j->ij"""
        @simulate_jit
        def kernel_broadcast_mul(A, b):
            import nkipy.core.ops as ops
            return ops.einsum('ij,j->ij', A, b)

        A = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4).astype(np.float32)
        
        result = kernel_broadcast_mul(A, b)
        expected = np.einsum('ij,j->ij', A, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch_broadcast(self):
        """Test batched broadcasting: bij,bj->bij"""
        @simulate_jit
        def kernel_batch_broadcast(A, b):
            import nkipy.core.ops as ops
            return ops.einsum('bij,bj->bij', A, b)

        A = np.random.rand(2, 3, 4).astype(np.float32)
        b = np.random.rand(2, 4).astype(np.float32)
        
        result = kernel_batch_broadcast(A, b)
        expected = np.einsum('bij,bj->bij', A, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumComplex:
    """Test complex einsum patterns."""

    def test_bilinear_form(self):
        """Test bilinear form: i,ij,j->"""
        @simulate_jit
        def kernel_bilinear(x, A, y):
            import nkipy.core.ops as ops
            return ops.einsum('i,ij,j->', x, A, y)

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        A = np.random.rand(3, 3).astype(np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        
        result = kernel_bilinear(x, A, y)
        expected = np.einsum('i,ij,j->', x, A, y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_tensor_contraction(self):
        """Test tensor contraction: ijk,jkl->il"""
        @simulate_jit
        def kernel_contraction(A, B):
            import nkipy.core.ops as ops
            return ops.einsum('ijk,jkl->il', A, B)

        A = np.random.rand(2, 3, 4).astype(np.float32)
        B = np.random.rand(3, 4, 5).astype(np.float32)
        
        result = kernel_contraction(A, B)
        expected = np.einsum('ijk,jkl->il', A, B)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumEdgeCases:
    """Test edge cases for einsum."""

    def test_identity(self):
        """Test identity operation: ij->ij"""
        @simulate_jit
        def kernel_identity(A):
            import nkipy.core.ops as ops
            return ops.einsum('ij->ij', A)

        A = np.random.rand(3, 4).astype(np.float32)
        result = kernel_identity(A)
        expected = np.einsum('ij->ij', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar(self):
        """Test scalar operations."""
        @simulate_jit
        def kernel_scalar(A):
            import nkipy.core.ops as ops
            return ops.einsum('->', A)

        A = np.array(5.0, dtype=np.float32)
        result = kernel_scalar(A)
        expected = np.einsum('->', A)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_single_element(self):
        """Test with single element arrays."""
        @simulate_jit
        def kernel_single(a, b):
            import nkipy.core.ops as ops
            return ops.einsum('i,i->', a, b)

        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        
        result = kernel_single(a, b)
        expected = np.einsum('i,i->', a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
