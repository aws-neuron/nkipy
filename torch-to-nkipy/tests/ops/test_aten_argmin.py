# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenArgminDefault(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            # Basic 2D cases with specific dim
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
            # FIXME Accuracy issues
            # ((128, 256), 1, False, torch.bfloat16),  # BFloat16
            ((32, 64), 0, True, torch.float16),  # FP16 with keepdim
            # ((64, 32), 1, True, torch.bfloat16),  # BFloat16 with keepdim
            # Edge cases with specific dimensions
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
    def test_argmin_default_with_dim(self, shape, dim, keepdim, dtype):
        """Test argmin.default with specific dimension."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x, dim, keepdim)

        # Create input tensor with varied values to ensure meaningful argmin operations
        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 1.0)

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            # Global argmin tests (no dim specified)
            ((16, 32), torch.float32),  # 2D case
            ((8, 16, 32), torch.float32),  # 3D case
            ((4, 8, 16, 32), torch.float32),  # 4D case
            ((100,), torch.float32),  # 1D case
            ((1,), torch.float32),  # Single element
            ((5, 1, 10), torch.float32),  # Mix of regular and singleton dims
            ((200, 300), torch.float16),  # FP16
            # FIXME Accuracy issues
            # ((150, 250), torch.bfloat16),  # BFloat16
            # Large tensors for global argmin
            ((1000, 500), torch.float32),
            ((100, 100, 10), torch.float32),
        ],
    )
    def test_argmin_default_global(self, shape, dtype):
        """Test argmin.default without dimension (global argmin)."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x)  # No dim specified

        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 1.0)
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dim,keepdim",
        [
            (0, False),
            (1, False),
            (0, True),
            (1, True),
        ],
    )
    def test_argmin_default_known_values(self, dim, keepdim):
        """Test argmin with known values to verify correctness."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x, dim, keepdim)

        # Create tensor where we know the argmin indices
        arg_0 = torch.tensor(
            [
                [5.0, 1.0, 8.0, 3.0],  # argmin along dim=1 should be 1
                [2.0, 9.0, 4.0, 6.0],  # argmin along dim=1 should be 0
                [7.0, 6.0, 1.0, 8.0],  # argmin along dim=1 should be 2
            ],
            dtype=torch.float32,
        )
        # argmin along dim=0: [1, 0, 2, 0] (2.0, 1.0, 1.0, 3.0)

        self.run_test_on_device(test_func, (arg_0,))

    def test_argmin_default_global_known_values(self):
        """Test global argmin with known values."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x)  # Global argmin

        # Create tensor where we know the global argmin
        # Flatten order: [5, 1, 8, 3, 2, 9, 4, 6, 7, 6, 0, 8]
        # Min value 0.0 should be at flattened index 10
        arg_0 = torch.tensor(
            [
                [5.0, 1.0, 8.0, 3.0],
                [2.0, 9.0, 4.0, 6.0],
                [7.0, 6.0, 0.0, 8.0],
            ],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            # FIXME Accuracy issues
            # torch.bfloat16
        ],
    )
    def test_argmin_default_identical_values(self, dtype):
        """Test argmin behavior when all values along dimension are identical."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x, 1, False)

        # Create tensor with identical values along dimension 1
        # Should return the first occurrence (index 0)
        arg_0 = torch.ones((128, 64), dtype=dtype) * 3.14

        self.run_test_on_device(test_func, (arg_0,))

    def test_argmin_default_negative_values(self):
        """Test argmin with negative values to ensure correct minimum finding."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x, 1, False)

        # Create tensor with mix of positive and negative values
        # Argmin should find the most negative values
        arg_0 = torch.tensor(
            [
                [5.0, -10.0, 3.0, -2.0],  # argmin should be 1 (-10.0)
                [-1.0, 8.0, -15.0, 4.0],  # argmin should be 2 (-15.0)
                [2.0, -3.0, 9.0, -20.0],  # argmin should be 3 (-20.0)
            ],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func, (arg_0,))

    def test_argmin_default_positive_values(self):
        """Test argmin with all positive values."""

        def test_func(x):
            return torch.ops.aten.argmin.default(x, 0, True)

        # Mix of positive values - argmin should find the smallest positive
        arg_0 = torch.tensor(
            [[1e6, 1e-6, 1e3], [1e-3, 1e6, 1e-2], [1e-6, 1e-3, 1e6]],
            dtype=torch.float32,
        )

        self.run_test_on_device(test_func, (arg_0,))
