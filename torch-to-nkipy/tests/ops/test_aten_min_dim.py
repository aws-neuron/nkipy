# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMinDim(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            # Basic 2D cases
            ((16, 32), 0, False, torch.float32),  # Reduce first dim, don't keep dim
            ((16, 32), 1, False, torch.float32),  # Reduce second dim, don't keep dim
            ((16, 32), 0, True, torch.float32),  # Reduce first dim, keep dim
            ((16, 32), 1, True, torch.float32),  # Reduce second dim, keep dim
            # 3D cases - test transpose logic for non-last dimensions
            ((8, 16, 32), 0, False, torch.float32),  # First dim (needs transpose)
            ((8, 16, 32), 1, False, torch.float32),  # Middle dim (needs transpose)
            ((8, 16, 32), 2, False, torch.float32),  # Last dim (no transpose needed)
            ((8, 16, 32), 1, True, torch.float32),  # Middle dim, keep dim
            ((8, 16, 32), -1, True, torch.float32),  # Negative dim index (last dim)
            ((8, 16, 32), -2, False, torch.float32),  # Negative dim index (middle dim)
            ((8, 16, 32), -3, True, torch.float32),  # Negative dim index (first dim)
            # 4D cases - more complex transpose scenarios
            ((4, 8, 16, 32), 0, False, torch.float32),  # First dim
            ((4, 8, 16, 32), 1, True, torch.float32),  # Second dim, keep dim
            ((4, 8, 16, 32), 2, False, torch.float32),  # Third dim
            ((4, 8, 16, 32), 3, True, torch.float32),  # Last dim (no transpose)
            ((4, 8, 16, 32), -1, False, torch.float32),  # Last dim with negative index
            ((4, 8, 16, 32), -2, True, torch.float32),  # Third dim with negative index
            # Different dtypes
            ((64, 128), 0, False, torch.float16),  # FP16
            # FIXME accuracy issue
            # ((128, 256), 1, False, torch.bfloat16),  # BFloat16
            ((32, 64), 0, True, torch.float16),  # FP16 with keepdim
            # FIXME accuracy issue
            # ((64, 32), 1, True, torch.bfloat16),  # BFloat16 with keepdim
            # Edge cases
            ((1, 32), 0, True, torch.float32),  # Singleton first dimension
            ((1, 32), 0, False, torch.float32),  # Singleton first dimension, no keepdim
            ((32, 1), 1, True, torch.float32),  # Singleton second dimension
            (
                (32, 1),
                1,
                False,
                torch.float32,
            ),  # Singleton second dimension, no keepdim
            ((5,), 0, False, torch.float32),  # 1D tensor
            ((5,), 0, True, torch.float32),  # 1D tensor with keepdim
            ((1,), 0, False, torch.float32),  # Single element tensor
            ((1,), 0, True, torch.float32),  # Single element tensor with keepdim
            # Larger tensors to test performance
            ((512, 256), 0, False, torch.float32),
            ((256, 512), 1, True, torch.float32),
        ],
    )
    def test_min_dim_basic(self, shape, dim, keepdim, dtype):
        """Test min.dim with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten.min.dim(x, dim, keepdim)

        # Create input tensor with varied values to ensure meaningful min operations
        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 1.0)

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            ((10, 20), 0, torch.float32),
            ((5, 10, 15), 1, torch.float32),
            ((4, 6, 8, 10), 2, torch.float32),
        ],
    )
    def test_min_dim_tuple_return(self, shape, dim, dtype):
        """Test that min.dim returns a proper tuple of (values, indices)."""

        def test_func(x):
            values, indices = torch.ops.aten.min.dim(x, dim, False)
            # Test that we can access both parts of the tuple
            return values + indices.float()  # Simple operation to verify both work

        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 1.0)

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            # FIXME accuracy issue
            # torch.bfloat16
        ],
    )
    def test_min_dim_identical_values(self, dtype):
        """Test min.dim behavior when all values along dimension are identical."""

        def test_func(x):
            return torch.ops.aten.min.dim(x, 1, False)

        # Create tensor with identical values along dimension 1
        # Should return the first occurrence (index 0)
        arg_0 = torch.ones((128, 64), dtype=dtype) * 3.14

        self.run_test_on_device(test_func, (arg_0,))

    def test_min_dim_extreme_values(self):
        """Test min.dim with extreme values (very large/small numbers)."""

        def test_func(x):
            return torch.ops.aten.min.dim(x, 0, True)

        # Mix of very large and very small values
        arg_0 = torch.tensor(
            [[1e6, -1e6, 1e-6], [-1e6, 1e6, -1e-6], [1e-6, -1e-6, 1e6]],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func, (arg_0,))

    def test_min_dim_with_negative_values(self):
        """Test min.dim specifically with negative values to ensure min logic
        works correctly."""

        def test_func(x):
            return torch.ops.aten.min.dim(x, 1, False)

        # Create tensor with mix of positive and negative values
        # Min should find the most negative values
        arg_0 = torch.tensor(
            [[5.0, -10.0, 3.0, -2.0], [-1.0, 8.0, -15.0, 4.0], [2.0, -3.0, 9.0, -20.0]],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            # FIXME accuracy issue
            # torch.bfloat16
        ],
    )
    def test_min_dim_numerical_stability(self, dtype):
        """Test min.dim with values that test numerical stability."""

        def test_func(x):
            return torch.ops.aten.min.dim(x, 0, False)

        # Create tensor with very close values to test numerical precision
        base_val = 1e-7 if dtype == torch.float32 else 1e-3
        arg_0 = torch.tensor(
            [
                [base_val, base_val + 1e-8, base_val + 2e-8],
                [base_val - 1e-8, base_val, base_val + 1e-8],
            ],
            dtype=dtype,
        )

        self.run_test_on_device(test_func, (arg_0,))
