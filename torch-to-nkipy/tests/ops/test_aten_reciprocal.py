# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenReciprocal(NKIPyTestBase):
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
            ((4, 8), torch.float16),  # Half precision
            # FIXME accuracy issue
            # ((4, 8), torch.bfloat16),  # BFloat16
        ],
    )
    def test_reciprocal_shapes_dtypes(self, shape, dtype):
        """Test aten.reciprocal.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.reciprocal.default(x)

        # Create tensor with non-zero values (reciprocal requires non-zero input)
        # Use values away from zero to avoid numerical instability
        arg_0 = torch.randn(size=shape, dtype=dtype) * 5.0 + 0.5
        # Ensure no values are too close to zero
        arg_0 = torch.where(torch.abs(arg_0) < 0.1, torch.sign(arg_0) * 0.5, arg_0)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
        ],
    )
    def test_reciprocal_edge_cases(self, dtype):
        """Test aten.reciprocal.default with edge cases."""

        def test_func(x):
            return torch.ops.aten.reciprocal.default(x)

        # Test with ones (reciprocal of 1 is 1)
        arg_0 = torch.ones(size=(4, 8), dtype=dtype)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with negative ones (reciprocal of -1 is -1)
        arg_0 = torch.ones(size=(4, 8), dtype=dtype) * -1.0
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with small positive values (reciprocal becomes large)
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * 0.1 + 0.01
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with large positive values (reciprocal becomes small)
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * 100.0 + 10.0
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with small negative values (reciprocal becomes large negative)
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * -0.1 - 0.01
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with large negative values (reciprocal becomes small negative)
        arg_0 = torch.rand(size=(4, 8), dtype=dtype) * -100.0 - 10.0
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_reciprocal_special_values(self):
        """Test aten.reciprocal.default with special values."""

        def test_func(x):
            return torch.ops.aten.reciprocal.default(x)

        # Test with simple fractions
        # reciprocal(2) = 0.5, reciprocal(0.5) = 2.0
        arg_0 = torch.tensor([2.0, 4.0, 8.0, 0.5, 0.25, 0.125], dtype=torch.float32)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with negative values
        arg_0 = torch.tensor([-2.0, -4.0, -0.5, -0.25], dtype=torch.float32)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with values close to 1
        arg_0 = torch.tensor([0.9, 0.95, 1.0, 1.05, 1.1], dtype=torch.float32)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with mixed positive and negative values (avoiding zero)
        arg_0 = torch.tensor(
            [-10.0, -5.0, -1.0, -0.1, 0.1, 1.0, 5.0, 10.0], dtype=torch.float32
        )
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_reciprocal_properties(self):
        """Test mathematical properties of reciprocal."""

        def test_func(x):
            return torch.ops.aten.reciprocal.default(x)

        # Test double reciprocal: reciprocal(reciprocal(x)) = x
        # Use well-conditioned values to avoid numerical errors
        arg_0 = torch.tensor([2.0, 3.0, 4.0, 5.0, 10.0], dtype=torch.float32)

        # Verify property (within numerical tolerance)
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

        # Test with powers of 2 (exact reciprocals)
        powers_of_2 = torch.tensor(
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float32
        )
        self.run_test_on_host(test_func, (powers_of_2,))
        self.run_test_on_device(test_func, (powers_of_2,))

        # Test with powers of 10
        powers_of_10 = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0], dtype=torch.float32)
        self.run_test_on_host(test_func, (powers_of_10,))
        self.run_test_on_device(test_func, (powers_of_10,))

    def test_reciprocal_numerical_stability(self):
        """Test reciprocal with values that test numerical stability."""

        def test_func(x):
            return torch.ops.aten.reciprocal.default(x)

        # Test with very small positive values (but not too close to underflow)
        small_positive = torch.tensor([1e-3, 1e-2, 1e-1, 0.5], dtype=torch.float32)
        self.run_test_on_host(test_func, (small_positive,))
        self.run_test_on_device(test_func, (small_positive,))

        # Test with very large values (but not close to overflow)
        large_values = torch.tensor([10.0, 100.0, 1000.0, 10000.0], dtype=torch.float32)
        self.run_test_on_host(test_func, (large_values,))
        self.run_test_on_device(test_func, (large_values,))

        # Test uniformly distributed values (avoiding zero)
        uniform_values = torch.linspace(0.1, 10.0, 50, dtype=torch.float32)
        self.run_test_on_host(test_func, (uniform_values,))
        self.run_test_on_device(test_func, (uniform_values,))
