# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenRepeat(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,repeat_sizes,dtype",
        [
            ((2, 3), (2, 2), torch.float32),  # Basic 2D repeat
            ((2,), (3,), torch.float32),  # 1D tensor repeat
            ((2, 3, 4), (2, 1, 3), torch.float32),  # 3D with different repeats
            ((1,), (4,), torch.float16),  # 1D with FP16
            ((2, 1, 3), (1, 4, 1), torch.bfloat16),  # 3D with BFloat16
            ((2, 3), (1, 1), torch.float32),  # No actual repeat
            ((1, 1, 1), (2, 2, 2), torch.float32),  # Repeat singleton dimensions
            ((2, 3, 4), (2, 2, 2), torch.float32),  # Uniform repeat
        ],
    )
    def test_repeat_shapes_dtypes(self, shape, repeat_sizes, dtype):
        """Test aten.repeat.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.repeat.default(x, repeat_sizes)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_repeat_special_cases(self):
        """Test repeat.default with special cases."""

        def test_func(x, sizes):
            return torch.ops.aten.repeat.default(x, sizes)

        # Test cases
        test_cases = [
            # Single element tensor with large repeat
            (torch.tensor([1.0]), (1000,)),
            # Repeat with ones
            (torch.randn(2, 3), (1, 1)),
            # Large tensor with small repeat
            (torch.randn(100, 100), (2, 1)),
            # Mixed dimensions repeat
            (torch.randn(2, 3, 4), (3, 1, 2)),
        ]

        for tensor, sizes in test_cases:
            self.run_test_on_host(test_func, (tensor, sizes))
            self.run_test_on_device(test_func, (tensor, sizes))

    def test_repeat_content_verification(self):
        """Test repeat.default with specific content to verify correct repetition."""

        def test_func(x):
            return torch.ops.aten.repeat.default(x, (2, 2))

        # Create input tensor with specific values
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_repeat_edge_cases(self):
        """Test repeat.default with edge cases."""

        def test_func(x, sizes):
            return torch.ops.aten.repeat.default(x, sizes)

        edge_cases = [
            # Single element repeat many times
            (torch.tensor([1.0]), (5,)),
            # Repeat with mixed large and small sizes
            (torch.randn(2, 2), (1000, 1)),
            # Repeat with size 1 in some dimensions
            (torch.randn(2, 1, 2), (1, 5, 1)),
        ]

        for tensor, sizes in edge_cases:
            self.run_test_on_host(test_func, (tensor, sizes))
            self.run_test_on_device(test_func, (tensor, sizes))

    def test_repeat_with_special_values(self):
        """Test repeat.default with special values."""

        def test_func(x):
            return torch.ops.aten.repeat.default(x, (2, 2))

        # Create tensor with special values
        special_values = torch.tensor(
            [[float("inf"), -float("inf")], [0.0, -0.0]], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
