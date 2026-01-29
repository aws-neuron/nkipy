# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenFloor(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),  # Basic FP32 case
            ((16, 32), torch.float16),  # FP16 case
            ((16, 32), torch.bfloat16),  # BFloat16 case
            ((8, 16, 32), torch.float32),  # 3D tensor
            ((4, 8, 16, 32), torch.float32),  # 4D tensor
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_floor_basic(self, shape, dtype):
        """Test floor.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Generate random tensor with values between -10 and 10
        arg_0 = torch.randn(size=shape, dtype=dtype) * 10

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_floor_specific_values(self):
        """Test floor.default with specific input values."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Test with values that should give predictable floor results
        x = torch.tensor(
            [
                1.0,
                1.1,
                1.5,
                1.9,  # Floor should be 1 for all these
                0.0,
                0.1,
                0.9,  # Floor should be 0 for all these
                -0.1,
                -0.5,
                -0.9,  # Floor should be -1 for all these
                -1.0,
                -1.1,
                -1.9,  # Floor should be -1, -2, -2
                2.0,
                -2.0,  # Values that are already integers
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_floor_edge_cases(self):
        """Test floor.default with edge cases."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Test with special floating point values
        x = torch.tensor(
            [
                1e7,  # Very large value
                -1e7,  # Very large negative value
                1e-7,  # Very small positive value
                -1e-7,  # Very small negative value
                1.999999,  # Very close to 2.0
                1.000001,  # Very close to 1.0
                -0.999999,  # Very close to -1.0
                -1.000001,  # Very close to -1.0
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_floor_range_values(self):
        """Test floor.default with a range of values from -10 to 10."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Test with a range of values
        x = torch.linspace(-10, 10, 100, dtype=torch.float32)

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_floor_multidimensional(self):
        """Test floor.default with structured multidimensional tensors."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Create a 2D tensor with specific patterns
        x = torch.tensor(
            [
                [0.1, 0.5, 0.9, 1.1, 1.5, 1.9],
                [-0.1, -0.5, -0.9, -1.1, -1.5, -1.9],
                [2.1, 2.5, 2.9, -2.1, -2.5, -2.9],
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_floor_dtypes(self):
        """Test floor.default with different floating-point dtypes."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Create the same tensor in different dtypes
        values = [1.1, 1.5, 1.9, -1.1, -1.5, -1.9, 0.1, -0.1]

        # Test with float32
        x_float32 = torch.tensor(values, dtype=torch.float32)
        self.run_test_on_host(test_func, (x_float32,))
        self.run_test_on_device(test_func, (x_float32,))

        # Test with float16
        x_float16 = torch.tensor(values, dtype=torch.float16)
        self.run_test_on_host(test_func, (x_float16,))
        self.run_test_on_device(test_func, (x_float16,))

        # Test with bfloat16
        x_bfloat16 = torch.tensor(values, dtype=torch.bfloat16)
        self.run_test_on_host(test_func, (x_bfloat16,))
        self.run_test_on_device(test_func, (x_bfloat16,))

    def test_floor_zeros_and_integers(self):
        """Test floor.default with zeros and values that are already integers."""

        def test_func(x):
            return torch.ops.aten.floor.default(x)

        # Test with zeros and integers
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32)

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))
