# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSelectInt(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,index,dtype",
        [
            ((5, 10, 15), 0, 2, torch.float32),  # Select along first dimension
            ((5, 10, 15), 1, 5, torch.float32),  # Select along middle dimension
            ((5, 10, 15), 2, 7, torch.float32),  # Select along last dimension
            ((5, 10), 0, 0, torch.float32),  # Select first element
            ((5, 10), 0, 4, torch.float32),  # Select last element
            ((5, 16, 2048), 0, 2, torch.float16),  # FP16 dtype
            ((5, 16, 4096), 0, 2, torch.bfloat16),  # BF16 dtype
        ],
    )
    def test_select_int_basic(self, shape, dim, index, dtype):
        """Test with different shapes, dimensions, indices, and dtypes."""

        def test_func(tensor):
            return torch.ops.aten.select.int(tensor, dim, index)

        # Create input tensor
        input_tensor = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    @pytest.mark.parametrize(
        "dim,index",
        [
            (0, -1),  # Last element of first dimension
            (1, -2),  # Second-to-last element of second dimension
            (2, -5),  # Fifth-to-last element of third dimension
        ],
    )
    def test_select_int_negative_indices(self, dim, index):
        """Test aten.select.int with negative indices."""

        def test_func(tensor):
            return torch.ops.aten.select.int(tensor, dim, index)

        # Create input tensor
        input_tensor = torch.randn(size=(10, 12, 15), dtype=torch.float32)

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_select_int_specified_content(self):
        """Test aten.select.int with specified content to verify correctness."""

        def test_func_dim0(tensor):
            # Select row 1 from a 3x4 tensor
            return torch.ops.aten.select.int(tensor, 0, 1)

        def test_func_dim1(tensor):
            # Select column 2 from a 3x4 tensor
            return torch.ops.aten.select.int(tensor, 1, 2)

        # Create input tensor with specified content
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )

        # Expected results:
        # test_func_dim0: tensor([5.0, 6.0, 7.0, 8.0])
        # test_func_dim1: tensor([3.0, 7.0, 11.0])

        self.run_test_on_host(test_func_dim0, (input_tensor,))
        self.run_test_on_device(test_func_dim0, (input_tensor,))

        self.run_test_on_host(test_func_dim1, (input_tensor,))
        self.run_test_on_device(test_func_dim1, (input_tensor,))
