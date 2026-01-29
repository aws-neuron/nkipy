# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenEq(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,scalar,dtype",
        [
            ((16, 32), 0, torch.float32),  # Zero comparison
            ((16, 32), 1, torch.float32),  # One comparison
            ((16, 32), -1, torch.float32),  # Negative comparison
            ((16, 32), 0.5, torch.float32),  # Fractional comparison
            ((8, 16, 32), 0, torch.float32),  # 3D tensor
            ((4, 8, 16, 32), 1, torch.float32),  # 4D tensor
            ((128, 256), 0, torch.float16),  # FP16
            ((64, 128), 0, torch.bfloat16),  # BFloat16
            ((16, 32), 5, torch.int32),  # Integer tensor
            ((256, 128), 0, torch.bool),  # Boolean tensor
            ((16,), 1, torch.float32),  # 1D tensor
            ((1, 1, 1), 0, torch.float32),  # Singleton dimensions
        ],
    )
    def test_eq_scalar_basic(self, shape, scalar, dtype):
        """Test eq.Scalar with different shapes, scalars and dtypes."""

        def test_func(x):
            return torch.ops.aten.eq.Scalar(x, scalar)

        if dtype == torch.bool:
            arg_0 = torch.zeros(size=shape, dtype=dtype)
        elif dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-10, 10, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "scalar",
        [
            0,
            1,
            -1,
            42,
            -42,
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            float("inf"),
            float("-inf"),
        ],
    )
    def test_eq_scalar_values(self, scalar):
        """Test eq.Scalar with various scalar values."""

        def test_func(x, ref_val):
            # Compare tensor with scalar
            return torch.ops.aten.eq.Scalar(x, scalar)

        # Create tensor with specific values
        x = torch.tensor(
            [scalar, scalar + 1, scalar - 1, 0, 1, -1], dtype=torch.float32
        )

        # Create reference result manually for comparison
        ref_val = torch.tensor([(v == scalar) for v in x], dtype=torch.bool)

        self.run_test_on_host(test_func, (x, ref_val))
        self.run_test_on_device(test_func, (x, ref_val))

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),  # Basic FP32 case
            ((16, 32), torch.float16),  # FP16 case
            ((16, 32), torch.bfloat16),  # BFloat16 case
            ((16, 32), torch.int32),  # Integer tensor
            ((16, 32), torch.bool),  # Boolean tensor
            ((8, 16, 32), torch.float32),  # 3D tensor
            ((4, 8, 16, 32), torch.float32),  # 4D tensor
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_eq_tensor_basic(self, shape, dtype):
        """Test eq.Tensor with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.eq.Tensor(x, y)

        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=dtype)
            arg_1 = torch.randint(0, 2, size=shape, dtype=dtype)
        elif dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-10, 10, size=shape, dtype=dtype)
            arg_1 = torch.randint(-10, 10, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)
            arg_1 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    @pytest.mark.parametrize(
        "shape_a,shape_b,dtype",
        [
            ((16, 32), (32,), torch.float32),  # Broadcasting second tensor
            ((16, 1), (1, 32), torch.float32),  # Broadcasting both tensors
            ((1, 16, 32), (16, 32), torch.float32),  # Different dimensions
            ((16, 32), (1, 16, 32), torch.float32),  # Different dimensions reversed
            ((16, 1, 32), (16, 8, 1), torch.float32),  # Complex broadcasting
            ((1, 1), (5, 5), torch.float32),  # Broadcasting singleton to matrix
            (
                (5, 5),
                (1, 1),
                torch.float32,
            ),  # Broadcasting singleton to matrix reversed
        ],
    )
    def test_eq_tensor_broadcasting(self, shape_a, shape_b, dtype):
        """Test eq.Tensor with different broadcasting scenarios."""

        def test_func(x, y):
            return torch.ops.aten.eq.Tensor(x, y)

        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape_a, dtype=dtype)
            arg_1 = torch.randint(0, 2, size=shape_b, dtype=dtype)
        elif dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-10, 10, size=shape_a, dtype=dtype)
            arg_1 = torch.randint(-10, 10, size=shape_b, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape_a, dtype=dtype)
            arg_1 = torch.randn(size=shape_b, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_eq_tensor_values(self):
        """Test eq.Tensor with specific tensor values to verify correctness."""

        def test_func(x, y):
            return torch.ops.aten.eq.Tensor(x, y)

        # Test with tensors containing specific patterns
        x = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.tensor([0, 1, 0, 3, 0, 5], dtype=torch.float32)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with 2D tensors
        x = torch.tensor([[-1, 0], [1, 2]], dtype=torch.float32)
        y = torch.tensor([[-1, 0], [0, 2]], dtype=torch.float32)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with boolean tensors
        x = torch.tensor([True, False, True], dtype=torch.bool)
        y = torch.tensor([True, False, False], dtype=torch.bool)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

    def test_eq_tensor_edge_cases(self):
        """Test eq.Tensor with edge cases."""

        def test_func(x, y):
            return torch.ops.aten.eq.Tensor(x, y)

        # Test with very large values
        x = torch.tensor([1e5, 1e5, 1e5], dtype=torch.float32)
        y = torch.tensor([1e5, 1e6, 1e5], dtype=torch.float32)
        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with very small values
        x = torch.tensor([1e-5, 1e-6, 1e-5], dtype=torch.float32)
        y = torch.tensor([1e-5, 1e-5, 1e-4], dtype=torch.float32)
        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with all same values
        x = torch.ones(10, dtype=torch.float32)
        y = torch.ones(10, dtype=torch.float32)
        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with all zeros
        x = torch.zeros(10, dtype=torch.float32)
        y = torch.zeros(10, dtype=torch.float32)
        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))
