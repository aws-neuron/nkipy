# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenTopk(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,k,dtype",
        [
            ((10, 20), 5, torch.float32),  # Basic 2D case
            ((5, 10, 15), 3, torch.float32),  # 3D tensor
            ((2, 3, 4, 5), 2, torch.float32),  # 4D tensor
            ((100,), 10, torch.float32),  # 1D tensor
            ((10, 20), 1, torch.float32),  # k=1 (minimum)
            ((10, 20), 20, torch.float32),  # k=dim_size (maximum)
            ((10, 2048), 5, torch.float16),  # FP16 dtype
            # Accuracy issue in BF16
            # ((10, 2048), 5, torch.bfloat16),             # BF16 dtype
        ],
    )
    def test_topk_last_dim(self, shape, k, dtype):
        """Test along the last dimension with different shapes, k values, and dtypes."""

        def test_func(tensor):
            return torch.ops.aten.topk.default(tensor, k)

        # Create input tensor with random values
        input_tensor = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_device(test_func, (input_tensor,))

    def test_topk_values_last_dim(self):
        """Test with specified tensor values along the last dimension."""

        def test_func_for_known_values(tensor):
            return torch.ops.aten.topk.default(tensor, 3)

        # Create a tensor with known values where sorting is clear
        # Use values that are clearly different to avoid instability with sorting
        input_tensor = torch.tensor(
            [[1.0, 10.0, 2.0, 20.0, 3.0], [30.0, 40.0, 5.0, 50.0, 6.0]],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func_for_known_values, (input_tensor,))

    @pytest.mark.xfail(reason="largest is not supported")
    def test_topk_largest_kwarg(self):
        """Test that topk ignores the largest kwarg."""

        def test_func_with_largest_kwarg(tensor):
            return torch.ops.aten.topk.default(tensor, 3, largest=False)

        # Create input tensor
        input_tensor = torch.randn(size=(5, 10), dtype=torch.float32)

        self.run_test_on_device(test_func_with_largest_kwarg, (input_tensor,))

    @pytest.mark.xfail(reason="sorted is not supported")
    def test_topk_sorted_kwarg(self):
        """Test that topk ignores the sorted kwarg."""

        def test_func_with_sorted_kwarg(tensor):
            return torch.ops.aten.topk.default(tensor, 3, sorted=False)

        # Create input tensor
        input_tensor = torch.randn(size=(5, 10), dtype=torch.float32)

        self.run_test_on_device(test_func_with_sorted_kwarg, (input_tensor,))

    def test_topk_k_equal_to_dim_size(self):
        """Test topk when k equals the size of the last dimension."""

        def test_func_k_equals_dim(tensor):
            # k equals the last dimension size
            return torch.ops.aten.topk.default(tensor, 5)

        # Create input tensor where last dim has size 5
        input_tensor = torch.randn(size=(3, 5), dtype=torch.float32)

        self.run_test_on_device(test_func_k_equals_dim, (input_tensor,))
