# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMul(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape1,shape2,dtype",
        [
            ((16, 32), (16, 32), torch.float32),
            ((8, 16, 32), (8, 16, 32), torch.float32),
            ((4, 1, 32), (4, 8, 32), torch.float32),  # Broadcasting
            ((128, 256), (128, 256), torch.float16),
            ((128, 128), (128, 128), torch.bfloat16),
            ((16, 32), (1,), torch.float32),  # Tensor * scalar-like tensor
        ],
    )
    def test_mul_tensor_shapes_dtypes(self, shape1, shape2, dtype):
        """Test aten.mul.Tensor with different shapes and dtypes."""

        def test_func(a, b):
            return torch.ops.aten.mul.Tensor(a, b)

        arg_0 = torch.randn(size=shape1, dtype=dtype)
        arg_1 = torch.randn(size=shape2, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_mul_tensor_scalar(self):
        """Test aten.mul.Tensor with scalar values."""

        def test_func(a, b):
            return torch.ops.aten.mul.Tensor(a, b)

        arg_0 = torch.randn(size=(16, 32), dtype=torch.float32)
        arg_1 = torch.tensor(5.0)  # Scalar tensor

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    @pytest.mark.parametrize(
        "shape,scalar,dtype",
        [
            ((16, 32), 2.0, torch.float32),
            ((8, 16, 32), 3.5, torch.float32),
            ((4, 8, 16, 32), -1.0, torch.float32),  # Multi-dimensional with negative
            ((128, 256), 0.5, torch.float16),  # FP16 with fraction
            # FIXME accuracy issue
            # ((64, 128), 10.0, torch.bfloat16),  # BFloat16 with larger scalar
            ((16, 32), 0.0, torch.float32),  # Zero scalar
            ((1, 1, 1), 42.0, torch.float32),  # Singleton dimensions
            ((16, 32), 2, torch.int32),  # Integer tensor and scalar
        ],
    )
    def test_mul_scalar_shapes_dtypes(self, shape, scalar, dtype):
        """Test mul.Scalar with different shapes, scalars and dtypes."""

        def test_func(x):
            return torch.ops.aten.mul.Scalar(x, scalar)

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "scalar",
        [
            1.0,  # Identity
            -1.0,  # Negation
            0.0,  # Zero
            2.0,  # Doubling
            0.5,  # Halving
            3.14159,  # Pi
            1e3,  # Large positive
            1e-3,  # Small positive
            -1e3,  # Large negative
            -1e-3,  # Small negative
        ],
    )
    def test_mul_scalar_special_values(self, scalar):
        """Test mul.Scalar with various scalar values."""

        def test_func(x):
            return torch.ops.aten.mul.Scalar(x, scalar)

        # Use a standard tensor
        arg_0 = torch.randn(size=(8, 16), dtype=torch.float32)

        # Regular test for non-NaN values
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
