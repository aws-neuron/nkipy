# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLinalgVectorNorm(NKIPyTestBase):
    """Test cases for torch.ops.aten.linalg_vector_norm.default"""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            ((4, 32), torch.float16),  # FP16
            ((8, 16), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_vector_norm_shapes_dtypes(self, shape, dtype):
        """Test linalg_vector_norm.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(x, ord=2)

        # Create input tensor with values spanning a range
        arg_0 = torch.randn(size=shape, dtype=dtype) * 2  # Values roughly in [-4, 4]

        if dtype != torch.bfloat16 and dtype != torch.float16:
            self.run_test_on_host(test_func, (arg_0,))
            self.run_test_on_device(test_func, (arg_0,))
        else:
            self.run_test_on_host(test_func, (arg_0,), rtol=1e-1, atol=1e-1)
            self.run_test_on_device(test_func, (arg_0,), rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize(
        "ord",
        [
            1,  # L1 norm (sum of absolute values)
            2,  # L2 norm (Euclidean)
            float("inf"),  # Maximum absolute value
            float("-inf"),  # Minimum absolute value
            3,  # Other p-norm
            0.5,  # Fractional norm
        ],
    )
    def test_vector_norm_orders(self, ord):
        """Test linalg_vector_norm.default with different norm orders."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(x, ord=ord)

        # Create input with variety of values
        input_tensor = torch.tensor(
            [
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                [0.5, -1.5, 2.5, -3.5, 4.5, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    @pytest.mark.parametrize(
        "dim,keepdim",
        [
            (0, False),  # Norm across first dimension, collapse
            (1, False),  # Norm across second dimension, collapse
            ((0, 1), False),  # Norm across both dimensions, collapse
            (0, True),  # Norm across first dimension, keep dim
            (1, True),  # Norm across second dimension, keep dim
            ((0, 1), True),  # Norm across both dimensions, keep dim
            (None, False),  # Default - flattened norm
        ],
    )
    def test_vector_norm_dimensions(self, dim, keepdim):
        """Test linalg_vector_norm.default with different dimensions and"""
        """keepdim settings."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(
                x, ord=2, dim=dim, keepdim=keepdim
            )

        # Create 3D input tensor
        input_tensor = torch.randn(
            size=(4, 5, 6),
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    @pytest.mark.parametrize(
        "input_dtype,output_dtype",
        [
            (torch.float32, torch.float32),  # Same type
            # Not allowed in torch.ops.aten.linalg_vector_norm.default
            # (torch.float32, torch.float16),  # Downcast.
            # NRT error. runs when there is only one pytest case
            # (torch.float16, torch.float32),  # Upcast
            # (torch.bfloat16, torch.float32),  # BF16 to FP32
        ],
    )
    def test_vector_norm_dtypes(self, input_dtype, output_dtype):
        """Test linalg_vector_norm.default with different input/output dtypes."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(
                x, ord=2, dtype=output_dtype
            )

        # Create input tensor
        input_tensor = torch.randn(
            size=(64, 64),
            dtype=input_dtype,
        )

        if input_dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 1e-1, 1e-1  # Relaxed tolerances for 16-bit types
        else:
            rtol, atol = None, None  # Use default tolerances for other types

        # Run tests with appropriate tolerances
        self.run_test_on_host(test_func, (input_tensor,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (input_tensor,), rtol=rtol, atol=atol)

    def test_vector_norm_zeros(self):
        """Test linalg_vector_norm.default with zero vectors."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(x, ord=2)

        zeros = torch.zeros(10, dtype=torch.float32)
        self.run_test_on_host(test_func, (zeros,))
        self.run_test_on_device(test_func, (zeros,))

    def test_vector_norm_ones(self):
        """Test linalg_vector_norm.default with vectors of ones."""

        def test_func(x):
            return torch.ops.aten.linalg_vector_norm.default(x, ord=2)

        ones = torch.ones(10, dtype=torch.float32)
        self.run_test_on_host(test_func, (ones,))
        self.run_test_on_device(test_func, (ones,))

    def test_vector_norm_negative_values(self):
        """Test linalg_vector_norm.default with negative values."""

        def test_l1_norm(x):
            # L1 norm should sum absolute values
            return torch.ops.aten.linalg_vector_norm.default(x, ord=1)

        def test_l2_norm(x):
            # L2 norm should be the same for positive and negative values
            return torch.ops.aten.linalg_vector_norm.default(x, ord=2)

        # Mix of positive and negative values
        mixed = torch.tensor([-3.0, 4.0, -5.0, 1.0, -2.0], dtype=torch.float32)

        self.run_test_on_host(test_l1_norm, (mixed,))
        self.run_test_on_device(test_l1_norm, (mixed,))

        self.run_test_on_host(test_l2_norm, (mixed,))
        self.run_test_on_device(test_l2_norm, (mixed,))
