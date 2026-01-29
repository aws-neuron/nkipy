# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenDiv(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape1,shape2,dtype",
        [
            ((16, 32), (16, 32), torch.float32),
            ((8, 16, 32), (8, 16, 32), torch.float32),
            ((4, 1, 32), (4, 8, 32), torch.float32),  # Broadcasting
            ((128, 256), (128, 256), torch.float16),
            ((128, 128), (128, 128), torch.bfloat16),
            ((16, 32), (1,), torch.float32),  # Tensor / scalar-like tensor
        ],
    )
    def test_div_tensor_shapes_dtypes(self, shape1, shape2, dtype):
        """Test aten.div.Tensor with different shapes and dtypes."""

        def test_func(a, b):
            return torch.ops.aten.div.Tensor(a, b)

        arg_0 = torch.randn(size=shape1, dtype=dtype) + 1.0  # Avoid near-zero values
        arg_1 = torch.randn(size=shape2, dtype=dtype) + 1.0  # Avoid near-zero values

        # Ensure we don't have zeros in the divisor to avoid div-by-zero
        arg_1 = torch.where(arg_1.abs() < 0.1, arg_1 + 0.5, arg_1)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_div_tensor_scalar(self):
        """Test aten.div.Tensor with scalar values."""

        def test_func(a, b):
            return torch.ops.aten.div.Tensor(a, b)

        arg_0 = torch.randn(size=(16, 32), dtype=torch.float32)
        arg_1 = torch.tensor(2.0)  # Scalar tensor

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_div_tensor_by_constant(self):
        """Test division by a constant value."""

        def test_func(a):
            return torch.ops.aten.div.Tensor(a, 2.0)

        arg_0 = torch.randn(size=(16, 32), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
