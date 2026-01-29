# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSub(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape1,shape2,dtype",
        [
            ((16, 32), (16, 32), torch.float32),
            ((8, 16, 32), (8, 16, 32), torch.float32),
            ((4, 1, 32), (4, 8, 32), torch.float32),  # Broadcasting
            ((128, 256), (128, 256), torch.float16),
            ((128, 128), (128, 128), torch.bfloat16),
            ((16, 32), (1,), torch.float32),  # Tensor - scalar-like tensor
        ],
    )
    def test_sub_tensor_shapes_dtypes(self, shape1, shape2, dtype):
        """Test aten.sub.Tensor with different shapes and dtypes."""

        def test_func(a, b):
            return torch.ops.aten.sub.Tensor(a, b)

        arg_0 = torch.randn(size=shape1, dtype=dtype)
        arg_1 = torch.randn(size=shape2, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    @pytest.mark.xfail(reason="kwargs are not supported")
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 0.5, -1.0])
    def test_sub_tensor_alpha(self, alpha):
        """Test aten.sub.Tensor with different alpha values."""

        def test_func(a, b, alpha=alpha):
            return torch.ops.aten.sub.Tensor(a, b, alpha=alpha)

        shape = (16, 32)
        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_sub_tensor_scalar(self):
        """Test aten.sub.Tensor with scalar values."""

        def test_func(a, b):
            return torch.ops.aten.sub.Tensor(a, b)

        arg_0 = torch.randn(size=(16, 32), dtype=torch.float32)
        arg_1 = torch.tensor(5.0)  # Scalar tensor

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))
