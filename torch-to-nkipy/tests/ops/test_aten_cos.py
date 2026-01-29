# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from base import NKIPyTestBase


class TestAtenCos(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((1, 1, 1), torch.float32),  # Singleton dimensions
            ((128, 256), torch.float16),
            ((64, 128), torch.bfloat16),
        ],
    )
    def test_cos_shapes_dtypes(self, shape, dtype):
        """Test aten.cos.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.cos.default(x)

        # Create random input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        self.run_test_on_host(test_func, (arg_0,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (arg_0,))

    def test_cos_special_values(self):
        """Test cosine with special angle values."""

        def test_func(x):
            return torch.ops.aten.cos.default(x)

        # Create tensor with special angles: 0, π/2, π, 3π/2, 2π
        special_angles = torch.tensor(
            [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_angles,))
        self.run_test_on_device(test_func, (special_angles,))

    def test_cos_large_values(self):
        """Test cosine with large values."""

        def test_func(x):
            return torch.ops.aten.cos.default(x)

        # Large values test
        large_vals = torch.tensor(
            [1000.0, -1000.0, 10000.0, -10000.0], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (large_vals,))
        self.run_test_on_device(test_func, (large_vals,))
