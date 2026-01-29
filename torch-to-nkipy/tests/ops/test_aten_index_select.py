# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenIndexSelect(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,indices_shape,dtype",
        [
            ((5, 768, 2048), 0, (1,), torch.float32),  # Single index along dim 0
            ((5, 768, 2048), 0, (3,), torch.float32),  # Multiple indices along dim 0
            ((10, 20, 30), 1, (5,), torch.float32),  # Selection along middle dimension
            ((10, 20, 30), 2, (10,), torch.float32),  # Selection along last dimension
            ((100,), 0, (20,), torch.float32),  # 1D tensor selection
            ((5, 2048, 2048), 0, (1,), torch.float16),  # FP16 dtype
            ((5, 2048, 4096), 0, (1,), torch.bfloat16),  # BF16 dtype
            ((50, 100), 0, (50,), torch.float32),  # Select all elements
        ],
    )
    def test_index_select_shapes_dims(self, shape, dim, indices_shape, dtype):
        """Test aten.index_select.default with different shapes, dims and dtypes."""

        def test_func(tensor, indices):
            return torch.ops.aten.index_select.default(tensor, dim, indices)

        # Create input tensor
        input_tensor = torch.randn(size=shape, dtype=dtype)

        # Create indices tensor (values within valid range for the dimension)
        max_index = shape[dim] - 1
        indices = torch.randint(0, max_index + 1, size=indices_shape, dtype=torch.int64)

        self.run_test_on_host(test_func, (input_tensor, indices))
        self.run_test_on_device(test_func, (input_tensor, indices))

    def test_index_select_repeated_indices(self):
        """Test aten.index_select.default with repeated indices."""

        def test_func(tensor, indices):
            return torch.ops.aten.index_select.default(tensor, 0, indices)

        # Create input tensor
        input_tensor = torch.randn(size=(10, 20), dtype=torch.float32)

        # Create indices with repeated values to test duplication
        indices = torch.tensor([0, 2, 2, 5, 5, 5], dtype=torch.int64)

        self.run_test_on_host(test_func, (input_tensor, indices))
        self.run_test_on_device(test_func, (input_tensor, indices))

    def test_index_select_docstring_example(self):
        """Test the specific example mentioned in the docstring."""

        def test_func(tensor, indices):
            return torch.ops.aten.index_select.default(tensor, 0, indices)

        # Create input tensor of shape [5, 768, 2048]
        input_tensor = torch.randn(size=(5, 768, 2048), dtype=torch.float32)

        # Create indices tensor containing [3]
        indices = torch.tensor([3], dtype=torch.int64)

        self.run_test_on_host(test_func, (input_tensor, indices))
        self.run_test_on_device(test_func, (input_tensor, indices))

    def test_index_select_scalar_index(self):
        """Test aten.index_select.default with a scalar index tensor."""

        def test_func(tensor, index):
            return torch.ops.aten.index_select.default(tensor, 1, index)

        # Create input tensor
        input_tensor = torch.randn(size=(5, 10, 15), dtype=torch.float32)

        # Create scalar index (single value, no dimensions)
        index = torch.tensor(3, dtype=torch.int64)

        self.run_test_on_host(test_func, (input_tensor, index))
        self.run_test_on_device(test_func, (input_tensor, index))

    def test_index_select_specified_content(self):
        """Test with specified content to verify correctness."""

        def test_func(tensor, indices):
            return torch.ops.aten.index_select.default(tensor, 0, indices)

        # Create input tensor with specified content
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
        )

        # Select rows 0 and 2
        indices = torch.tensor([0, 2], dtype=torch.int64)
        # Expected result: tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])

        self.run_test_on_host(test_func, (input_tensor, indices))
        self.run_test_on_device(test_func, (input_tensor, indices))
