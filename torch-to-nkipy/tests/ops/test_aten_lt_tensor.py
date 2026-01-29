# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLt(NKIPyTestBase):
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
    def test_lt_tensor_basic(self, shape, dtype):
        """Test lt.Tensor with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.lt.Tensor(x, y)

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
    def test_lt_tensor_broadcasting(self, shape_a, shape_b, dtype):
        """Test lt.Tensor with different broadcasting scenarios."""

        def test_func(x, y):
            return torch.ops.aten.lt.Tensor(x, y)

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

    def test_lt_tensor_values(self):
        """Test lt.Tensor with specific tensor values to verify correctness."""

        def test_func(x, y):
            return torch.ops.aten.lt.Tensor(x, y)

        # Test with tensors containing specific patterns
        x = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.float32)
        y = torch.tensor([-1, -1, 0, 2, 2, 2], dtype=torch.float32)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with 2D tensors
        x = torch.tensor([[-1, 0], [1, 2]], dtype=torch.float32)
        y = torch.tensor([[0, 0], [0, 1]], dtype=torch.float32)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with boolean tensors (False < True)
        x = torch.tensor([False, False, True], dtype=torch.bool)
        y = torch.tensor([False, True, True], dtype=torch.bool)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))
