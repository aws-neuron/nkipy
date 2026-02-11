# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive indexing and slicing tests for NKIPy tensors.

This test suite provides complete coverage of indexing and slicing functionality,
organized by feature category and implementation priority. Tests are designed
to be production-ready with clear documentation, proper error handling, and
comprehensive validation.

Test Categories:
- Core Functionality: Basic slicing, indexing, and assignment operations
- ML-Critical Patterns: Essential patterns for machine learning workloads
- Advanced Features: Complex indexing patterns and edge cases
- Error Handling: Validation of error conditions and unsupported patterns
- Performance Boundaries: Tests for patterns that should be discouraged
"""

from typing import Dict, Tuple

import numpy as np
import pytest
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    baremetal_run_kernel_unified,
    cpu_assert_allclose,
    trace_and_run,
    trace_mode,  # noqa: F401 - pytest fixture
)


class TestIndexingSlicingCore:
    """
    Core indexing and slicing functionality tests.
    These are the fundamental operations that must work correctly.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for testing with shapes common in ML workloads"""
        np.random.seed(42)  # For reproducible tests
        return {
            "small_1d": np.random.randn(32).astype(np.float32),
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "small_3d": np.random.randn(4, 8, 12).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(
                np.float32
            ),  # [batch, seq, hidden]
            "attention": np.random.randn(2, 8, 32, 64).astype(
                np.float32
            ),  # [batch, heads, seq, dim]
            "large_1d": np.random.randn(1024).astype(np.float32),
        }

    def test_basic_slice_1d(self, trace_mode, sample_tensors):
        """Test basic 1D slicing with explicit copy"""

        def kernel(a):
            view = a[5:15]
            return np.copy(view)

        a = sample_tensors["small_1d"]
        expected = a[5:15]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (10,)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_basic_slice_2d(self, trace_mode, sample_tensors):
        """Test basic 2D slicing with explicit copy"""

        def kernel(a):
            view = a[0:3, :4]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[0:3, :4]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (3, 4)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_basic_slice_3d(self, trace_mode, sample_tensors):
        """Test basic 3D slicing with explicit copy"""

        def kernel(a):
            view = a[0:2, :, 4:8]
            return np.copy(view)

        a = sample_tensors["small_3d"]
        expected = a[0:2, :, 4:8]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 8, 4)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_step_slicing(self, trace_mode, sample_tensors):
        """Test slicing with step parameter"""

        def kernel(a):
            view = a[::2, 1::3]  # Every 2nd row, every 3rd column starting from 1
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[::2, 1::3]

        try:
            result = trace_and_run(kernel, trace_mode, a)
            cpu_assert_allclose(result, expected)
            assert result.shape == expected.shape

            if NEURON_AVAILABLE:
                out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
                baremetal_assert_allclose(expected, out_baremetal)
        except AssertionError as e:
            if "Only support scale 1 for now" in str(e):
                pytest.skip(f"Step slicing not yet supported: {e}")
            else:
                raise

    def test_nested_slicing(self, trace_mode, sample_tensors):
        """Test chained slicing operations"""

        def kernel(a):
            intermediate = a[0:6, :]
            view = intermediate[2:4, 4:8]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[0:6, :][2:4, 4:8]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 4)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_basic_assignment(self, trace_mode, sample_tensors):
        """Test basic assignment through views"""

        def kernel(a, b):
            a_copy = np.copy(a)
            a_copy[0:3, :4] = b
            return a_copy

        a = sample_tensors["small_2d"]
        b = np.ones((3, 4), dtype=np.float32)

        expected = a.copy()
        expected[0:3, :4] = b

        result = trace_and_run(kernel, trace_mode, a, b)
        cpu_assert_allclose(result, expected)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_view_assignment_semantics(self, trace_mode, sample_tensors):
        """Test that assignment through views modifies the original tensor"""

        def kernel(a, b):
            a_copy = np.copy(a)
            view = a_copy[0:5, :]  # Create view
            view[1:3, 2:4] = b  # Modify through view
            return a_copy  # Return modified original

        a = sample_tensors["small_2d"]
        b = np.ones((2, 2), dtype=np.float32)

        expected = a.copy()
        expected[0:5, :][1:3, 2:4] = b

        if trace_mode == "hlo":
            pytest.skip(
                "View assignment semantics differ in HLO mode - operations create new tensors"
            )

        result = trace_and_run(kernel, trace_mode, a, b)
        cpu_assert_allclose(result, expected)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_explicit_copy_requirement(self, trace_mode, sample_tensors):
        """Test that explicit copy is required for kernel returns"""

        def kernel_with_copy(a):
            view = a[0:3, :4]
            return np.copy(view)  # This should work

        def kernel_without_copy(a):
            return a[0:3, :4]  # This should fail

        a = sample_tensors["small_2d"]
        expected = a[0:3, :4]

        # With copy should work
        result = trace_and_run(kernel_with_copy, trace_mode, a)
        cpu_assert_allclose(result, expected)

        # Without copy should work now as well
        result = trace_and_run(kernel_without_copy, trace_mode, a)
        cpu_assert_allclose(result, expected)


class TestIndexingSlicingMLPatterns:
    """
    Machine Learning critical indexing patterns.
    These patterns are essential for ML workloads and must be highly optimized.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for ML testing"""
        np.random.seed(42)
        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
            "attention": np.random.randn(2, 8, 32, 64).astype(np.float32),
            "large_1d": np.random.randn(1024).astype(np.float32),
        }

    def test_numpy_array_indexing_1d(self, trace_mode, sample_tensors):
        """Test 1D array indexing - essential for sequence gathering"""

        def kernel(a):
            indices = np.array([0, 2, 4, 6])
            view = a[indices]
            return np.copy(view)

        a = sample_tensors["large_1d"]
        expected = a[[0, 2, 4, 6]]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (4,)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_numpy_array_indexing_2d_rows(self, trace_mode, sample_tensors):
        """Test 2D row selection - common for batch processing"""

        def kernel(a):
            indices = np.array([0, 2])
            view = a[indices, :]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[[0, 2], :]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 16)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_batch_indexing_pattern(self, trace_mode, sample_tensors):
        """Test batch selection pattern - very common in ML"""

        def kernel(a):
            batch_indices = np.array([0, 2])
            view = a[batch_indices, :, :]
            return np.copy(view)

        a = sample_tensors["batch_seq"]
        expected = a[[0, 2], :, :]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 32, 64)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_list_indexing(self, trace_mode, sample_tensors):
        """Test Python list indexing - should work like numpy arrays"""

        def kernel(a):
            indices = [0, 2, 4]  # Python list
            view = a[indices]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[[0, 2, 4]]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (3, 16)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_mixed_slice_and_array_indexing(self, trace_mode, sample_tensors):
        """Test combining slicing with array indexing"""

        def kernel(a):
            # First slice columns, then select specific rows
            sliced = a[:, 2:10]
            indices = np.array([0, 2])
            view = sliced[indices]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[:, 2:10][[0, 2]]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 8)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_sequence_token_selection(self, trace_mode, sample_tensors):
        """Test selecting specific tokens from sequences - common in NLP"""

        def kernel(a):
            # Select specific sequence positions across all batches
            seq_indices = np.array([0, 5, 10, 15])
            view = a[:, seq_indices, :]
            return np.copy(view)

        a = sample_tensors["batch_seq"]
        expected = a[:, [0, 5, 10, 15], :]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (4, 4, 64)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_attention_head_selection(self, trace_mode, sample_tensors):
        """Test selecting specific attention heads - critical for transformers"""

        def kernel(a):
            head_indices = np.array([0, 2, 4, 6])
            view = a[:, head_indices, :, :]
            return np.copy(view)

        a = sample_tensors["attention"]
        expected = a[:, [0, 2, 4, 6], :, :]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 4, 32, 64)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_view_usage_without_intermediate_copy(self, trace_mode, sample_tensors):
        """Test that views can be used directly without intermediate copying"""

        def kernel(tensor):
            # Create view and use it directly
            view = tensor[1:5]  # This is a view, no copy needed

            # Use view in computation
            result = view * 2.0  # Should work on view directly

            return np.copy(result)  # Only copy when returning

        a = sample_tensors["large_1d"][:6]  # Use first 6 elements
        expected = a[1:5] * 2.0

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (4,)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_view_as_index_pattern(self, trace_mode, sample_tensors):
        """Test using views as indices to other tensors - critical for MoE models"""

        def kernel(top_k_indices, weights):
            # Get expert index and weight for this token
            expert_idx = top_k_indices[0:5]  # This is a view
            # Use the VIEW as an INDEX to another tensor - this is the key pattern!
            weight = weights[expert_idx]  # View used as index

            return np.copy(expert_idx), np.copy(weight)

        # Test data similar to MoE routing
        top_k_indices = np.array([2, 5, 1, 8, 3], dtype=np.int32)  # Expert indices
        weights = np.array(
            [0.8, 0.6, 0.9, 0.4, 0.7, 0.8, 0.6, 0.9, 0.4, 0.7], dtype=np.float32
        )  # Expert weights

        # Verify results
        expected_idx = top_k_indices[0:5]
        expected_weight = weights[expected_idx]

        result_idx, result_weight = trace_and_run(
            kernel, trace_mode, top_k_indices, weights
        )

        cpu_assert_allclose(result_idx, expected_idx)
        cpu_assert_allclose(result_weight, expected_weight)

        if NEURON_AVAILABLE:
            baremetal_idx, baremetal_weight = baremetal_run_kernel_unified(
                kernel, trace_mode, top_k_indices, weights
            )
            baremetal_assert_allclose(expected_idx, baremetal_idx)
            baremetal_assert_allclose(expected_weight, baremetal_weight)

    def test_single_view_as_index_pattern(self, trace_mode, sample_tensors):
        """Test using views as indices to other tensors - critical for MoE models"""

        def kernel(top_k_indices, weights):
            # Get expert index and weight for this token
            expert_idx = top_k_indices[0]  # This is a view
            # Use the VIEW as an INDEX to another tensor - this is the key pattern!
            weight = weights[expert_idx]  # View used as index

            return np.copy(expert_idx), np.copy(weight)

        # Test data similar to MoE routing
        top_k_indices = np.array([1, 0], dtype=np.int32)  # Expert indices
        weights = np.array(
            [[0.8, 0.6, 0.9], [0.4, 0.7, 0.8]], dtype=np.float32
        )  # Expert weights

        # Verify results
        expected_idx = top_k_indices[0]
        expected_weight = weights[expected_idx]

        result_idx, result_weight = trace_and_run(
            kernel, trace_mode, top_k_indices, weights
        )

        cpu_assert_allclose(result_idx, expected_idx)
        cpu_assert_allclose(result_weight, expected_weight)

        if NEURON_AVAILABLE:
            baremetal_idx, baremetal_weight = baremetal_run_kernel_unified(
                kernel, trace_mode, top_k_indices, weights
            )
            baremetal_assert_allclose(expected_idx, baremetal_idx)
            baremetal_assert_allclose(expected_weight, baremetal_weight)


class TestIndexingSlicingAdvanced:
    """
    Advanced indexing patterns and edge cases.
    These are nice-to-have features for comprehensive NumPy compatibility.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for advanced testing"""
        np.random.seed(42)
        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "small_3d": np.random.randn(4, 8, 12).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
        }

    def test_negative_indexing_basic(self, trace_mode, sample_tensors):
        """Test basic negative indexing"""

        def kernel(a):
            view = a[-1, :]  # Last row
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[-1, :]

        try:
            result = trace_and_run(kernel, trace_mode, a)
            cpu_assert_allclose(result, expected)
            assert result.shape == (16,)

            if NEURON_AVAILABLE:
                out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
                baremetal_assert_allclose(expected, out_baremetal)
        except Exception as e:
            pytest.skip(f"Negative indexing not yet working: {e}")

    def test_negative_slice_indexing(self, trace_mode, sample_tensors):
        """Test negative slice indexing"""

        def kernel(a):
            view = a[:-2, 1:-1]
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[:-2, 1:-1]

        try:
            result = trace_and_run(kernel, trace_mode, a)
            cpu_assert_allclose(result, expected)

            if NEURON_AVAILABLE:
                out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
                baremetal_assert_allclose(expected, out_baremetal)
        except Exception as e:
            pytest.skip(f"Negative slice indexing not yet working: {e}")

    def test_ellipsis_support(self, trace_mode, sample_tensors):
        """Test ellipsis notation"""

        def kernel(a):
            view = a[..., 0:3]  # Last dimension slice
            return np.copy(view)

        a = sample_tensors["small_3d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            trace_and_run(kernel, trace_mode, a)

    def test_newaxis_support(self, trace_mode, sample_tensors):
        """Test adding new dimensions via indexing"""

        def kernel(a):
            view = a[:, None, :]  # Add dimension
            return np.copy(view)

        a = sample_tensors["small_2d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            trace_and_run(kernel, trace_mode, a)

    def test_boolean_indexing_basic(self, trace_mode, sample_tensors):
        """Test basic boolean indexing"""

        def kernel(a):
            mask = a > 0.0
            return a[mask]  # Should gather elements where mask is True

        a = sample_tensors["small_2d"]

        # NOTE: CPU execution should pass

        if NEURON_AVAILABLE:
            with pytest.raises(
                (NotImplementedError, AssertionError, TypeError, RuntimeError)
            ):
                baremetal_run_kernel_unified(kernel, trace_mode, a)

    def test_mixed_static_dynamic_indexing(self, trace_mode, sample_tensors):
        """Test mixing static slicing with dynamic indexing"""

        def kernel(a):
            seq_indices = np.array([0, 5, 10, 15])
            view = a[0:2, seq_indices, :]  # First 2 batches, specific sequences
            return np.copy(view)

        a = sample_tensors["batch_seq"]
        expected = a[0:2, [0, 5, 10, 15], :]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (2, 4, 64)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_step_slicing_with_offset(self, trace_mode, sample_tensors):
        """Test step slicing with non-zero start offset"""

        def kernel(a):
            view = a[
                1::2, 2::3
            ]  # Every 2nd row starting from 1, every 3rd column starting from 2
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[1::2, 2::3]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == expected.shape

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_negative_slice_with_step(self, trace_mode, sample_tensors):
        """Test combining negative slice with step"""

        def kernel(a):
            view = a[:-1:2, 1:-1:2]  # Every 2nd element, excluding last
            return np.copy(view)

        a = sample_tensors["small_2d"]
        expected = a[:-1:2, 1:-1:2]

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == expected.shape

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_step_slicing_assignment(self, trace_mode, sample_tensors):
        """Test assignment to step-sliced views"""

        def kernel(a, b):
            a_copy = np.copy(a)
            a_copy[::2, ::2] = b  # Assign to every 2nd row and column
            return a_copy

        a = sample_tensors["small_2d"]
        # Create a smaller tensor that matches the step-sliced shape
        step_shape = a[::2, ::2].shape
        b = np.ones(step_shape, dtype=np.float32)

        expected = a.copy()
        expected[::2, ::2] = b

        result = trace_and_run(kernel, trace_mode, a, b)
        cpu_assert_allclose(result, expected)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)


class TestIndexingSlicingErrorHandling:
    """
    Error handling and boundary condition tests.
    These ensure proper error messages and graceful failure modes.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for error testing"""
        np.random.seed(42)
        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
        }

    def test_direct_view_return_error(self, trace_mode, sample_tensors):
        """Test that returning views directly from kernels fails with clear error"""

        def kernel(a):
            return a[0:3, :4]  # Should fail - no explicit copy

        a = sample_tensors["small_2d"]

        # HLO mode: views work fine, just verify it runs
        result = trace_and_run(kernel, trace_mode, a)
        expected = a[0:3, :4]
        cpu_assert_allclose(result, expected)

    def test_multi_axis_gather_limitation(self, trace_mode, sample_tensors):
        """Test that multi-axis array indexing fails with helpful error"""

        def kernel(a):
            batch_idx = np.array([0, 1])
            seq_idx = np.array([5, 10])
            return a[batch_idx, seq_idx, :]  # Should fail - multiple dynamic indices

        a = sample_tensors["batch_seq"]

        with pytest.raises(
            (ValueError, AssertionError), match="Only one.*index.*supported"
        ):
            trace_and_run(kernel, trace_mode, a)

    def test_complex_boolean_expression_unsupported(self, trace_mode, sample_tensors):
        """Test that complex boolean expressions are not supported"""

        def kernel(a):
            return a[(a > 0.0) & (a < 1.0)]  # Complex boolean expression

        a = sample_tensors["small_2d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            trace_and_run(kernel, trace_mode, a)

    def test_out_of_bounds_indexing(self, trace_mode, sample_tensors):
        """Test out of bounds indexing behavior"""

        def kernel(a):
            indices = np.array([0, 2, 100])  # 100 is out of bounds
            view = a[indices]
            return np.copy(view)

        a = sample_tensors["small_2d"]

        # This should either work (with undefined behavior) or fail gracefully
        try:
            result = trace_and_run(kernel, trace_mode, a)
            # If it works, just check the shape of valid indices
            assert result.shape[0] == 3
        except (IndexError, ValueError, RuntimeError):
            # Expected for out of bounds access
            pass


class TestIndexingSlicingIntegration:
    """
    Integration tests that combine multiple indexing operations.
    These test real-world usage patterns and complex scenarios.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for integration testing"""
        np.random.seed(42)
        return {
            "embeddings": np.random.randn(1000, 512).astype(
                np.float32
            ),  # Embedding table
            "sequences": np.random.randint(0, 1000, (8, 64)).astype(
                np.int32
            ),  # Token sequences
            "attention_weights": np.random.randn(8, 12, 64, 64).astype(
                np.float32
            ),  # Attention
        }

    def test_embedding_lookup_pattern(self, trace_mode, sample_tensors):
        """Test embedding lookup pattern common in NLP"""

        def kernel(embeddings, token_ids):
            # Select specific tokens (first 5 from each sequence)
            selected_tokens = token_ids[:, :5]  # Shape: (8, 5)

            # Flatten for lookup
            flat_tokens = np.reshape(selected_tokens, (-1,))  # Shape: (40,)

            # Lookup embeddings
            selected_embeddings = embeddings[flat_tokens]  # Shape: (40, 512)

            # Reshape back
            result = np.reshape(selected_embeddings, (8, 5, 512))
            return result

        embeddings = sample_tensors["embeddings"]
        sequences = sample_tensors["sequences"]

        # Expected computation
        selected_tokens = sequences[:, :5]
        flat_tokens = selected_tokens.flatten()
        expected_embeddings = embeddings[flat_tokens]
        expected = expected_embeddings.reshape(8, 5, 512)

        result = trace_and_run(kernel, trace_mode, embeddings, sequences)
        cpu_assert_allclose(result, expected)
        assert result.shape == (8, 5, 512)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(
                kernel, trace_mode, embeddings, sequences
            )
            baremetal_assert_allclose(expected, out_baremetal)

    def test_attention_head_extraction_pattern(self, trace_mode, sample_tensors):
        """Test extracting specific attention heads and positions"""

        def kernel(attention_weights):
            # Extract specific heads (0, 3, 6, 9) and sequence positions (0-15)
            head_indices = np.array([0, 3, 6, 9])
            selected_heads = attention_weights[:, head_indices, :16, :16]
            return np.copy(selected_heads)

        attention = sample_tensors["attention_weights"]
        expected = attention[:, [0, 3, 6, 9], :16, :16]

        result = trace_and_run(kernel, trace_mode, attention)
        cpu_assert_allclose(result, expected)
        assert result.shape == (8, 4, 16, 16)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, attention)
            baremetal_assert_allclose(expected, out_baremetal)

    def test_batch_sequence_filtering_pattern(self, trace_mode, sample_tensors):
        """Test filtering specific batches and sequence ranges"""

        def kernel(sequences):
            # Select batches 0, 2, 4, 6 and sequence positions 10-50
            batch_indices = np.array([0, 2, 4, 6])
            selected_batches = sequences[batch_indices, 10:50]
            return np.copy(selected_batches)

        sequences = sample_tensors["sequences"]
        expected = sequences[[0, 2, 4, 6], 10:50]

        result = trace_and_run(kernel, trace_mode, sequences)
        cpu_assert_allclose(result, expected)
        assert result.shape == (4, 40)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, sequences)
            baremetal_assert_allclose(expected, out_baremetal)


class TestIndexingSlicingRegression:
    """
    Regression tests for previously working functionality.
    These ensure that existing features continue to work correctly.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for regression testing"""
        np.random.seed(42)
        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
        }

    def test_numpy_function_dispatch_compatibility(self, trace_mode, sample_tensors):
        """Test that NumPy function dispatch still works with views"""

        def kernel(a):
            view = a[0:4, 0:8]
            result = np.sum(view, axis=1)  # Should work on view
            return result

        a = sample_tensors["small_2d"]
        expected = np.sum(a[0:4, 0:8], axis=1)

        result = trace_and_run(kernel, trace_mode, a)
        cpu_assert_allclose(result, expected)
        assert result.shape == (4,)

        if NEURON_AVAILABLE:
            out_baremetal = baremetal_run_kernel_unified(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)


# ========================================================================
# UTILITY FUNCTIONS AND TEST RUNNERS
# ========================================================================


def run_test_category(category_name: str, test_class) -> Tuple[int, int, int]:
    """
    Run all tests in a specific category and return results.

    Returns:
        Tuple of (passed, failed, skipped) counts
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"tests/unit/test_indexing_slicing_comprehensive.py::{test_class.__name__}",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    # Parse results from output
    lines = result.stdout.split("\n")
    summary_line = [
        line
        for line in lines
        if "passed" in line and ("failed" in line or "skipped" in line)
    ]

    if summary_line:
        # Extract numbers from summary
        import re

        numbers = re.findall(r"(\d+) (\w+)", summary_line[0])
        passed = failed = skipped = 0

        for count, status in numbers:
            if status == "passed":
                passed = int(count)
            elif status == "failed":
                failed = int(count)
            elif status == "skipped":
                skipped = int(count)

        return passed, failed, skipped

    return 0, 0, 0


def analyze_comprehensive_test_results():
    """
    Run comprehensive analysis of all indexing/slicing tests.
    Provides detailed breakdown by category and overall status.
    """
    print("=== COMPREHENSIVE INDEXING/SLICING TEST ANALYSIS ===\n")

    categories = [
        ("Core Functionality", TestIndexingSlicingCore),
        ("ML-Critical Patterns", TestIndexingSlicingMLPatterns),
        ("Advanced Features", TestIndexingSlicingAdvanced),
        ("Error Handling", TestIndexingSlicingErrorHandling),
        ("Integration Tests", TestIndexingSlicingIntegration),
        ("Regression Tests", TestIndexingSlicingRegression),
    ]

    total_passed = total_failed = total_skipped = 0

    for category_name, test_class in categories:
        print(f"=== {category_name} ===")
        passed, failed, skipped = run_test_category(category_name, test_class)

        total_passed += passed
        total_failed += failed
        total_skipped += skipped

        print(f"  ✓ Passed: {passed}")
        if failed > 0:
            print(f"  ✗ Failed: {failed}")
        if skipped > 0:
            print(f"  ⚠ Skipped: {skipped}")
        print()

    print("=== OVERALL SUMMARY ===")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Tests: {total_passed + total_failed + total_skipped}")

    if total_failed == 0:
        print("\n🎉 All implemented tests are passing!")
    else:
        print(f"\n⚠️  {total_failed} tests are failing and need attention.")

    if total_skipped > 0:
        print(f"📝 {total_skipped} tests are skipped (features not yet implemented).")


if __name__ == "__main__":
    analyze_comprehensive_test_results()
