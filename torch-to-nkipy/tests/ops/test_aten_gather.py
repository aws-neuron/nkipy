# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenGather(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            ((16, 32), 0, torch.float32),
            ((16, 32), 1, torch.float32),
            ((8, 16, 32), 0, torch.float32),
            ((8, 16, 32), 1, torch.float32),
            ((8, 16, 32), 2, torch.float32),
            ((4, 8, 16, 32), 3, torch.float32),  # Multi-dimensional
            ((128, 256), 0, torch.float16),  # FP16
            ((64, 128), 1, torch.bfloat16),  # BFloat16
        ],
    )
    def test_gather_shapes_dtypes(self, shape, dim, dtype):
        """Test aten.gather.default with different shapes and dtypes."""

        def test_func(x, index):
            return torch.ops.aten.gather.default(x, dim, index)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)
        # Create index tensor with valid indices
        index_shape = list(shape)
        index_shape[dim] = max(1, shape[dim] // 2)  # Reduce size along gather dimension
        index = torch.randint(0, shape[dim], size=index_shape)

        self.run_test_on_host(test_func, (arg_0, index))
        self.run_test_on_device(test_func, (arg_0, index))

    def test_gather_special_values(self):
        """Test gather.default with special values."""

        def test_func(x, index):
            return torch.ops.aten.gather.default(x, 0, index)

        # Create tensor with special values
        input_tensor = torch.tensor(
            [[float("inf"), -float("inf")], [0.0, -0.0], [1.0, -1.0]]
        )

        # Create index tensor
        index = torch.tensor([[0, 1], [2, 0]])

        self.run_test_on_host(test_func, (input_tensor, index))
        self.run_test_on_device(test_func, (input_tensor, index))
