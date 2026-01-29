# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSin(NKIPyTestBase):
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
    def test_sin_shapes_dtypes(self, shape, dtype):
        """Test aten.sin.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.sin.default(x)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_sin_special_values(self):
        """Test sin.default with special angle values."""

        def test_func(x):
            return torch.ops.aten.sin.default(x)

        # Create tensor with special angles: 0, π/6, π/4, π/3, π/2, π, 3π/2, 2π
        special_angles = torch.tensor(
            [
                0.0,
                math.pi / 6,  # sin(π/6) = 0.5
                math.pi / 4,  # sin(π/4) = √2/2
                math.pi / 3,  # sin(π/3) = √3/2
                math.pi / 2,  # sin(π/2) = 1
                math.pi,  # sin(π) = 0
                3 * math.pi / 2,  # sin(3π/2) = -1
                2 * math.pi,  # sin(2π) = 0
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_angles,))
        self.run_test_on_device(test_func, (special_angles,))
