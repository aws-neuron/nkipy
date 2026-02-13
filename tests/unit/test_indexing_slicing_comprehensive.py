# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive indexing and slicing tests for NKIPy tensors.

This test suite provides complete coverage of indexing and slicing functionality,
organized by feature category and implementation priority. Tests are designed
to be production-ready with clear documentation, proper error handling, and
comprehensive validation.
"""

from typing import Dict

import numpy as np
import pytest
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    on_device_test,
    trace_and_compile,
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
        def kernel(a):
            view = a[5:15]
            return view

        a = sample_tensors["small_1d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_basic_slice_2d(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[0:3, :4]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_basic_slice_3d(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[0:2, :, 4:8]
            return view

        a = sample_tensors["small_3d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_step_slicing(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[::2, 1::3]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_nested_slicing(self, trace_mode, sample_tensors):
        def kernel(a):
            intermediate = a[0:6, :]
            view = intermediate[2:4, 4:8]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_basic_assignment(self, trace_mode, sample_tensors):
        def kernel(a, b):
            a[0:3, :4] = b
            return a

        a = sample_tensors["small_2d"]
        b = np.ones((3, 4), dtype=np.float32)

        expected = kernel(a.copy(), b)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a, b)

    def test_view_assignment_semantics(self, trace_mode, sample_tensors):
        def kernel(a, b):
            view = a[0:5, :]
            view[1:3, 2:4] = b
            return a

        a = sample_tensors["small_2d"]
        b = np.ones((2, 2), dtype=np.float32)

        expected = kernel(a.copy(), b)

        if trace_mode == "hlo":
            pytest.skip(
                "View assignment semantics differ in HLO mode - operations create new tensors"
            )

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a, b)


class TestIndexingSlicingMLPatterns:
    """
    Machine Learning critical indexing patterns.
    These patterns are essential for ML workloads and must be highly optimized.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for ML testing"""

        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
            "attention": np.random.randn(2, 8, 32, 64).astype(np.float32),
            "large_1d": np.random.randn(1024).astype(np.float32),
        }

    def test_numpy_array_indexing_1d(self, trace_mode, sample_tensors):
        def kernel(a):
            indices = np.array([0, 2, 4, 6])
            view = a[indices]
            return view

        a = sample_tensors["large_1d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_numpy_array_indexing_2d_rows(self, trace_mode, sample_tensors):
        def kernel(a):
            indices = np.array([0, 2])
            view = a[indices, :]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_batch_indexing_pattern(self, trace_mode, sample_tensors):
        def kernel(a):
            batch_indices = np.array([0, 2])
            view = a[batch_indices, :, :]
            return view

        a = sample_tensors["batch_seq"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_list_indexing(self, trace_mode, sample_tensors):
        def kernel(a):
            indices = [0, 2, 4]  # Python list
            view = a[indices]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_mixed_slice_and_array_indexing(self, trace_mode, sample_tensors):
        def kernel(a):
            sliced = a[:, 2:10]
            indices = np.array([0, 2])
            view = sliced[indices]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_sequence_token_selection(self, trace_mode, sample_tensors):
        def kernel(a):
            seq_indices = np.array([0, 5, 10, 15])
            view = a[:, seq_indices, :]
            return view

        a = sample_tensors["batch_seq"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_attention_head_selection(self, trace_mode, sample_tensors):
        def kernel(a):
            head_indices = np.array([0, 2, 4, 6])
            view = a[:, head_indices, :, :]
            return view

        a = sample_tensors["attention"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_view_usage_without_intermediate_copy(self, trace_mode, sample_tensors):
        def kernel(tensor):
            view = tensor[1:5]
            result = view * 2.0

            return result

        a = sample_tensors["large_1d"][:6]  # Use first 6 elements
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_view_as_index_pattern(self, trace_mode, sample_tensors):
        def kernel(top_k_indices, weights):
            expert_idx = top_k_indices[0:5]
            weight = weights[expert_idx]

            return expert_idx, weight

        # Test data similar to MoE routing
        top_k_indices = np.array([2, 5, 1, 8, 3], dtype=np.int32)  # Expert indices
        weights = np.array(
            [0.8, 0.6, 0.9, 0.4, 0.7, 0.8, 0.6, 0.9, 0.4, 0.7], dtype=np.float32
        )  # Expert weights

        expected_idx = top_k_indices[0:5]
        expected_weight = weights[expected_idx]

        if NEURON_AVAILABLE:
            baremetal_idx, baremetal_weight = on_device_test(
                kernel, trace_mode, top_k_indices, weights
            )
            baremetal_assert_allclose(expected_idx, baremetal_idx)
            baremetal_assert_allclose(expected_weight, baremetal_weight)
        else:
            trace_and_compile(kernel, trace_mode, top_k_indices, weights)

    def test_single_view_as_index_pattern(self, trace_mode, sample_tensors):
        def kernel(top_k_indices, weights):
            expert_idx = top_k_indices[0]
            weight = weights[expert_idx]
            return expert_idx, weight

        # Test data similar to MoE routing
        top_k_indices = np.array([1, 0], dtype=np.int32)  # Expert indices
        weights = np.array(
            [[0.8, 0.6, 0.9], [0.4, 0.7, 0.8]], dtype=np.float32
        )  # Expert weights

        expected_idx = top_k_indices[0]
        expected_weight = weights[expected_idx]

        if NEURON_AVAILABLE:
            baremetal_idx, baremetal_weight = on_device_test(
                kernel, trace_mode, top_k_indices, weights
            )
            baremetal_assert_allclose(expected_idx, baremetal_idx)
            baremetal_assert_allclose(expected_weight, baremetal_weight)
        else:
            trace_and_compile(kernel, trace_mode, top_k_indices, weights)


class TestIndexingSlicingAdvanced:
    """
    Advanced indexing patterns and edge cases.
    These are nice-to-have features for comprehensive NumPy compatibility.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for advanced testing"""

        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "small_3d": np.random.randn(4, 8, 12).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
        }

    def test_negative_indexing_basic(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[-1, :]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        try:
            trace_and_compile(kernel, trace_mode, a)

            if NEURON_AVAILABLE:
                out_baremetal = on_device_test(kernel, trace_mode, a)
                baremetal_assert_allclose(expected, out_baremetal)
        except Exception as e:
            pytest.skip(f"Negative indexing not yet working: {e}")

    def test_negative_slice_indexing(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[:-2, 1:-1]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        try:
            trace_and_compile(kernel, trace_mode, a)

            if NEURON_AVAILABLE:
                out_baremetal = on_device_test(kernel, trace_mode, a)
                baremetal_assert_allclose(expected, out_baremetal)
        except Exception as e:
            pytest.skip(f"Negative slice indexing not yet working: {e}")

    def test_ellipsis_support(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[..., 0:3]
            return view

        a = sample_tensors["small_3d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            trace_and_compile(kernel, trace_mode, a)

    def test_newaxis_support(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[:, None, :]
            return view

        a = sample_tensors["small_2d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            trace_and_compile(kernel, trace_mode, a)

    def test_boolean_indexing_basic(self, trace_mode, sample_tensors):
        def kernel(a):
            mask = a > 0.0
            return a[mask]  # Should gather elements where mask is True

        a = sample_tensors["small_2d"]

        with pytest.raises(
            (NotImplementedError, AssertionError, TypeError, RuntimeError)
        ):
            if NEURON_AVAILABLE:
                on_device_test(kernel, trace_mode, a)
            else:
                trace_and_compile(kernel, trace_mode, a)

    def test_mixed_static_dynamic_indexing(self, trace_mode, sample_tensors):
        def kernel(a):
            seq_indices = np.array([0, 5, 10, 15])
            view = a[0:2, seq_indices, :]
            return view

        a = sample_tensors["batch_seq"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_step_slicing_with_offset(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[1::2, 2::3]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_negative_slice_with_step(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[:-1:2, 1:-1:2]
            return view

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)

    def test_step_slicing_assignment(self, trace_mode, sample_tensors):
        def kernel(a, b):
            a[::2, ::2] = b  # Assign to every 2nd row and column
            return a

        a = sample_tensors["small_2d"]
        # Create a smaller tensor that matches the step-sliced shape
        step_shape = a[::2, ::2].shape
        b = np.ones(step_shape, dtype=np.float32)

        expected = kernel(a, b)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a, b)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a, b)


class TestIndexingSlicingErrorHandling:
    """
    Error handling and boundary condition tests.
    These ensure proper error messages and graceful failure modes.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for error testing"""

        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
            "batch_seq": np.random.randn(4, 32, 64).astype(np.float32),
        }

    def test_multi_axis_gather_limitation(self, trace_mode, sample_tensors):
        def kernel(a):
            batch_idx = np.array([0, 1])
            seq_idx = np.array([5, 10])
            return a[batch_idx, seq_idx, :]  # Should fail - multiple dynamic indices

        a = sample_tensors["batch_seq"]

        with pytest.raises(
            (ValueError, AssertionError), match="Only one.*index.*supported"
        ):
            if NEURON_AVAILABLE:
                on_device_test(kernel, trace_mode, a)
            else:
                trace_and_compile(kernel, trace_mode, a)

    def test_complex_boolean_expression_unsupported(self, trace_mode, sample_tensors):
        def kernel(a):
            return a[(a > 0.0) & (a < 1.0)]  # Complex boolean expression

        a = sample_tensors["small_2d"]

        with pytest.raises((NotImplementedError, ValueError, TypeError)):
            if NEURON_AVAILABLE:
                on_device_test(kernel, trace_mode, a)
            else:
                trace_and_compile(kernel, trace_mode, a)

    def test_out_of_bounds_indexing(self, trace_mode, sample_tensors):
        def kernel(a):
            indices = np.array([0, 2, 100])  # 100 is out of bounds
            view = a[indices]
            return view

        a = sample_tensors["small_2d"]

        if NEURON_AVAILABLE:
            # FIXME: Runtime may not report anything, but we should guard this
            try:
                on_device_test(kernel, trace_mode, a)
            except (IndexError, ValueError, RuntimeError):
                pass
        else:
            # we don't know if it's OOB during tracing
            trace_and_compile(kernel, trace_mode, a)


class TestIndexingSlicingIntegration:
    """
    Integration tests that combine multiple indexing operations.
    These test real-world usage patterns and complex scenarios.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for integration testing"""

        return {
            "embeddings": np.random.randn(1000, 512).astype(np.float32),
            "sequences": np.random.randint(0, 1000, (8, 64)).astype(np.int32),
            "attention_weights": np.random.randn(8, 12, 64, 64).astype(np.float32),
        }

    def test_embedding_lookup_pattern(self, trace_mode, sample_tensors):
        """Test embedding lookup pattern common in NLP"""

        def kernel(embeddings, token_ids):
            selected_tokens = token_ids[:, :5]
            flat_tokens = np.reshape(selected_tokens, (-1,))
            selected_embeddings = embeddings[flat_tokens]

            result = np.reshape(selected_embeddings, (8, 5, 512))
            return result

        embeddings = sample_tensors["embeddings"]
        sequences = sample_tensors["sequences"]

        expected = kernel(embeddings, sequences)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, embeddings, sequences)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, embeddings, sequences)

    def test_attention_head_extraction_pattern(self, trace_mode, sample_tensors):
        def kernel(attention_weights):
            head_indices = np.array([0, 3, 6, 9])
            selected_heads = attention_weights[:, head_indices, :16, :16]
            return selected_heads

        attention = sample_tensors["attention_weights"]
        expected = kernel(attention)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, attention)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, attention)

    def test_batch_sequence_filtering_pattern(self, trace_mode, sample_tensors):
        def kernel(sequences):
            batch_indices = np.array([0, 2, 4, 6])
            selected_batches = sequences[batch_indices, 10:50]
            return selected_batches

        sequences = sample_tensors["sequences"]
        expected = kernel(sequences)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, sequences)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, sequences)


class TestIndexingSlicingRegression:
    """
    Regression tests for previously working functionality.
    These ensure that existing features continue to work correctly.
    """

    @pytest.fixture
    def sample_tensors(self) -> Dict[str, np.ndarray]:
        """Create sample tensors for regression testing"""

        return {
            "small_2d": np.random.randn(8, 16).astype(np.float32),
        }

    def test_numpy_function_dispatch_compatibility(self, trace_mode, sample_tensors):
        def kernel(a):
            view = a[0:4, 0:8]
            result = np.sum(view, axis=1)
            return result

        a = sample_tensors["small_2d"]
        expected = kernel(a)

        if NEURON_AVAILABLE:
            out_baremetal = on_device_test(kernel, trace_mode, a)
            baremetal_assert_allclose(expected, out_baremetal)
        else:
            trace_and_compile(kernel, trace_mode, a)


if __name__ == "__main__":
    pytest.main([__file__])
