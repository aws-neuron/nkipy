# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenEmbedding(NKIPyTestBase):
    @pytest.mark.parametrize(
        "vocab_size,embed_dim,indices_shape,dtype",
        [
            (128, 64, (16,), torch.float32),  # Basic 1D lookup
            (128, 64, (8, 16), torch.float32),  # 2D batch of indices
            (1024, 128, (8, 16, 32), torch.float32),  # 3D batch of indices
            (1024, 128, (32, 32), torch.float16),  # FP16
            (1024, 128, (128, 32), torch.bfloat16),  # BFloat16
            (128, 128, (1,), torch.float32),  # Single index
        ],
    )
    def test_embedding_basic(self, vocab_size, embed_dim, indices_shape, dtype):
        """Test embedding.default with different shapes and dtypes."""

        def test_func(weight, indices):
            return torch.ops.aten.embedding.default(weight, indices)

        # Create embedding weight matrix
        weight = torch.randn(size=(vocab_size, embed_dim), dtype=dtype)

        # Create indices tensor with values in the valid range [0, vocab_size-1]
        indices = torch.randint(0, vocab_size, size=indices_shape, dtype=torch.int64)

        self.run_test_on_host(test_func, (weight, indices))
        self.run_test_on_device(test_func, (weight, indices))

    def test_embedding_specific_indices(self):
        """Test embedding.default with specific indices."""

        def test_func(weight, indices):
            return torch.ops.aten.embedding.default(weight, indices)

        # Create small embedding matrix with recognizable values
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # Index 0
                [4.0, 5.0, 6.0],  # Index 1
                [7.0, 8.0, 9.0],  # Index 2
                [10.0, 11.0, 12.0],  # Index 3
            ],
            dtype=torch.float32,
        )

        # Create specific indices to verify exact output
        indices = torch.tensor([0, 2, 1, 3, 2, 0], dtype=torch.int64)

        self.run_test_on_host(test_func, (weight, indices))
        self.run_test_on_device(test_func, (weight, indices))

        # Test with batched indices
        batched_indices = torch.tensor([[0, 2, 1], [3, 2, 0]], dtype=torch.int64)

        self.run_test_on_host(test_func, (weight, batched_indices))
        self.run_test_on_device(test_func, (weight, batched_indices))

    def test_embedding_edge_cases(self):
        """Test embedding.default with edge cases."""

        def test_func(weight, indices):
            return torch.ops.aten.embedding.default(weight, indices)

        # Test with a single embedding vector
        single_vector_weight = torch.randn(size=(1, 10), dtype=torch.float32)
        single_indices = torch.zeros(
            size=(5,), dtype=torch.int64
        )  # All zeros to select the only vector

        self.run_test_on_host(test_func, (single_vector_weight, single_indices))
        self.run_test_on_device(test_func, (single_vector_weight, single_indices))

        # Test with a scalar index (0-dimensional tensor)
        scalar_index = torch.tensor(0, dtype=torch.int64)
        weight = torch.randn(size=(5, 10), dtype=torch.float32)

        self.run_test_on_host(test_func, (weight, scalar_index))
        self.run_test_on_device(test_func, (weight, scalar_index))
