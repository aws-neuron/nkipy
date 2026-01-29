# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSqrt(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16,), torch.float32),  # 1D tensor
            ((4, 8), torch.float32),  # 2D tensor
            ((2, 4, 8), torch.float32),  # 3D tensor
            ((2, 3, 4, 5), torch.float32),  # 4D tensor
            ((32, 32), torch.float32),  # Square matrix
            ((128, 256), torch.float32),  # Larger matrix
            ((1,), torch.float32),  # Single element
            ((), torch.float32),  # Scalar (0D tensor)
            ((4, 8), torch.float16),  # Half precision
            ((4, 8), torch.bfloat16),  # BFloat16
        ],
    )
    def test_sqrt_shapes_dtypes(self, shape, dtype):
        """Test aten.sqrt.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.sqrt.default(x)

        # Create tensor with positive values (sqrt requires non-negative input)
        arg_0 = torch.rand(size=shape, dtype=dtype) * 10.0 + 0.1

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
        ],
    )
    def test_sqrt_edge_cases(self, dtype):
        """Test aten.sqrt.default with edge cases."""

        def test_func(x):
            return torch.ops.aten.sqrt.default(x)

        # Test with zeros
        arg_0 = torch.zeros(size=(4, 8), dtype=dtype)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with ones
        arg_0 = torch.ones(size=(4, 8), dtype=dtype)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with small values near zero
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * 0.01
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with large values
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * 1000.0
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_sqrt_special_values(self):
        """Test aten.sqrt.default with special values."""

        def test_func(x):
            return torch.ops.aten.sqrt.default(x)

        # Test with perfect squares
        arg_0 = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 100.0], dtype=torch.float32)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with fractional values
        arg_0 = torch.tensor([0.25, 0.5, 0.75, 1.5, 2.5], dtype=torch.float32)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
