# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenNeg(NKIPyTestBase):
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
    def test_neg_shapes_dtypes(self, shape, dtype):
        """Test neg.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.neg.default(x)

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_neg_special_values(self):
        """Test neg.default with special values."""

        def test_func(x):
            return torch.ops.aten.neg.default(x)

        # Create tensor with special values
        special_values = torch.tensor(
            [
                0,
                1,
                -1,
                42,
                -42,  # Integer values
                0.0,
                1.0,
                -1.0,
                3.14,
                -3.14,  # Float values
                float("inf"),
                float("-inf"),  # Infinity
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
