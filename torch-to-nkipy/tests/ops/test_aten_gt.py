# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenGt(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            ((128, 256), torch.float16),  # FP16
            ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_gt_shapes_dtypes(self, shape, dtype):
        """Test aten.gt.Tensor with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.gt.Tensor(x, y)

        # Create two input tensors
        arg_0 = torch.randn(size=shape, dtype=dtype)
        arg_1 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_gt_broadcasting(self):
        """Test gt.Tensor with broadcasting."""

        def test_func(x, y):
            return torch.ops.aten.gt.Tensor(x, y)

        # Test broadcasting with different shapes
        shapes = [
            ((4, 3), (3,)),  # Broadcasting along first dimension
            ((6, 1, 5), (6, 4, 5)),  # Broadcasting middle dimension
            ((1, 8), (7, 8)),  # Broadcasting batch dimension
            ((5, 1, 6), (5, 4, 1)),  # Broadcasting multiple dimensions
        ]

        for shape_x, shape_y in shapes:
            arg_0 = torch.randn(size=shape_x, dtype=torch.float32)
            arg_1 = torch.randn(size=shape_y, dtype=torch.float32)

            self.run_test_on_host(test_func, (arg_0, arg_1))
            self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_gt_special_values(self):
        """Test gt.Tensor with special values."""

        def test_func(x, y):
            return torch.ops.aten.gt.Tensor(x, y)

        # Create tensors with special values
        x = torch.tensor(
            [
                float("inf"),  # Infinity
                -float("inf"),  # Negative infinity
                0.0,  # Zero
                1.0,  # Positive number
                -1.0,  # Negative number
                float("inf"),  # Compare inf with inf
                -float("inf"),  # Compare -inf with -inf
            ],
            dtype=torch.float32,
        )

        y = torch.tensor(
            [
                -float("inf"),  # Compare with negative infinity
                float("inf"),  # Compare with infinity
                0.0,  # Compare with zero
                -1.0,  # Compare with negative number
                1.0,  # Compare with positive number
                float("inf"),  # Compare inf with inf
                -float("inf"),  # Compare -inf with -inf
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

    def test_gt_edge_cases(self):
        """Test gt.Tensor with edge cases."""

        def test_func(x, y):
            return torch.ops.aten.gt.Tensor(x, y)

        # Test cases with very close numbers
        x = torch.tensor(
            [
                1.0,
                1.0000001,
                1e-7,
                1e-8,
                1.0,
                -1.0,
            ],
            dtype=torch.float32,
        )

        y = torch.tensor(
            [
                1.0,
                1.0,
                1e-8,
                1e-7,
                -1.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))
