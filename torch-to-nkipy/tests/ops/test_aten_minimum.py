# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMinimum(NKIPyTestBase):
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
    def test_minimum_shapes_dtypes(self, shape, dtype):
        """Test aten.minimum.default with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.minimum.default(x, y)

        arg_0 = torch.randn(size=shape, dtype=dtype)
        arg_1 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_minimum_broadcasting(self):
        """Test minimum.default with broadcasting."""

        def test_func(x, y):
            return torch.ops.aten.minimum.default(x, y)

        # Test broadcasting
        arg_0 = torch.randn(size=(4, 1, 6), dtype=torch.float32)
        arg_1 = torch.randn(size=(1, 5, 6), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_minimum_special_values(self):
        """Test minimum.default with special values."""

        def test_func(x, y):
            return torch.ops.aten.minimum.default(x, y)

        x = torch.tensor(
            [float("inf"), -float("inf"), 1.0, 0.0, -1.0], dtype=torch.float32
        )
        y = torch.tensor(
            [-float("inf"), float("inf"), 2.0, 0.0, -2.0], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))
