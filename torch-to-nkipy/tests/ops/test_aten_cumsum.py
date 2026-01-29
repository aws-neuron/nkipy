# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenCumsum(NKIPyTestBase):
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
            ((16,), 0, torch.float32),  # 1D tensor
            ((1, 1, 1), 1, torch.float32),  # Singleton dimensions
        ],
    )
    def test_cumsum_shapes_dtypes(self, shape, dim, dtype):
        """Test aten.cumsum.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.cumsum.default(x, dim)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 0.02)

        if dtype != torch.bfloat16 and dtype != torch.float16:
            self.run_test_on_host(test_func, (arg_0,))
            self.run_test_on_device(test_func, (arg_0,))
        else:
            self.run_test_on_host(test_func, (arg_0,), rtol=1e-1, atol=1e-1)
            self.run_test_on_device(test_func, (arg_0,), rtol=1e-1, atol=1e-1)

    def test_cumsum_special_values(self):
        """Test cumsum.default with special values."""

        def test_func(x):
            return torch.ops.aten.cumsum.default(x, 0)

        special_values = torch.tensor(
            [
                0.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))

    def test_cumsum_edge_cases(self):
        """Test cumsum.default with edge cases."""

        def test_func(x, dim):
            return torch.ops.aten.cumsum.default(x, dim)

        # Test cases
        test_cases = [
            # Single element
            (torch.tensor([1.0]), 0),
            # All zeros
            (torch.zeros((3, 3)), 1),
            # Large values
            (torch.tensor([1e10, 1e10, 1e10]), 0),
            # Small values
            (torch.tensor([1e-10, 1e-10, 1e-10]), 0),
        ]

        for tensor, dim in test_cases:
            self.run_test_on_host(test_func, (tensor, dim))
            self.run_test_on_device(test_func, (tensor, dim))
