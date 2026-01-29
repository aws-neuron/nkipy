# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenRelu(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            ((128, 256), torch.float16),  # FP16
            ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_relu_shapes_dtypes(self, shape, dtype):
        """Test aten.relu.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.relu.default(x)

        # Create input tensor with both positive and negative values
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_relu_special_values(self):
        """Test relu.default with special values."""

        def test_func(x):
            return torch.ops.aten.relu.default(x)

        special_values = torch.tensor(
            [
                0.0,  # ReLU(0) = 0
                1.0,  # ReLU(1) = 1
                -1.0,  # ReLU(-1) = 0
                0.5,  # ReLU(0.5) = 0.5
                -0.5,  # ReLU(-0.5) = 0
                10.0,  # Large positive
                -10.0,  # Large negative
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
