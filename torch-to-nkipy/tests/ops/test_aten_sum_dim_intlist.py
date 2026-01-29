# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSumDimIntList(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            ((5, 10, 15), 0, False, torch.float32),  # Sum along first dimension
            ((5, 10, 15), 1, False, torch.float32),  # Sum along middle dimension
            ((5, 10, 15), 2, False, torch.float32),  # Sum along last dimension
            ((5, 10, 15), 0, True, torch.float32),  # keepdim=True
            (
                (5, 10, 15),
                [0, 2],
                False,
                torch.float32,
            ),  # Sum along multiple dimensions
            ((5, 10, 15), [-1, -2], False, torch.float32),  # Negative dimensions
            ((5, 16, 2048), 0, False, torch.float16),  # FP16 dtype
            ((5, 32, 1024), 0, False, torch.bfloat16),  # BF16 dtype
        ],
    )
    def test_sum_dim_intlist_basic(self, shape, dim, keepdim, dtype):
        """Test aten.sum.dim_IntList with different dimensions and keepdim values."""

        def test_func(tensor):
            return torch.ops.aten.sum.dim_IntList(tensor, dim, keepdim)

        # Create input tensor
        input_tensor = torch.randn(size=shape, dtype=dtype)

        if dtype not in (torch.float16, torch.bfloat16):
            self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_sum_dim_intlist_specified_values(self):
        """Test sum.dim_IntList with specified tensor values to verify correctness."""

        def test_func1(tensor):
            # Sum along dim=0
            return torch.ops.aten.sum.dim_IntList(tensor, 0, False)

        def test_func2(tensor):
            # Sum along dim=1
            return torch.ops.aten.sum.dim_IntList(tensor, 1, False)

        # Create input tensor with specified content
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
        )

        # Expected results:
        # test_func1 (sum dim=0): tensor([12.0, 15.0, 18.0])
        # test_func2 (sum dim=1): tensor([6.0, 15.0, 24.0])

        self.run_test_on_host(test_func1, (input_tensor,))
        self.run_test_on_device(test_func1, (input_tensor,))

        self.run_test_on_host(test_func2, (input_tensor,))
        self.run_test_on_device(test_func2, (input_tensor,))
