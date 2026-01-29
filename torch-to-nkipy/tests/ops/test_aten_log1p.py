# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLog1p(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            ((128, 256), torch.float16),  # FP16
            # FIXME accuracy issue
            # ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_log1p_shapes_dtypes(self, shape, dtype):
        """Test aten.log1p.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.log1p.default(x)

        # Create input tensor with values > -1
        # (since log1p(x) = log(1+x) requires 1+x > 0)
        arg_0 = (
            torch.rand(size=shape, dtype=dtype) * 2.0 - 0.9
        )  # Range: (-0.9, 1.1) to ensure x > -1

        if dtype != torch.bfloat16 and dtype != torch.float16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_log1p_special_values(self):
        """Test log1p.default with special values."""

        def test_func(x):
            return torch.ops.aten.log1p.default(x)

        special_values = torch.tensor(
            [
                0.0,  # log1p(0) = 0
                math.e - 1.0,  # log1p(e-1) = log(e) = 1
                9.0,  # log1p(9) = log(10)
                1.0,  # log1p(1) = log(2)
                -0.5,  # log1p(-0.5) = log(0.5)
                1e-10,  # very small positive number
                -0.9,  # negative value close to -1
                0.1,  # log1p(0.1) = log(1.1)
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))

    def test_log1p_edge_cases(self):
        """Test log1p.default with edge cases."""

        def test_func(x):
            return torch.ops.aten.log1p.default(x)

        # Test with values very close to -1 (but not exactly -1 to avoid -inf)
        edge_values = torch.tensor(
            [
                -0.99999,  # Very close to -1
                -0.999999999,  # Even closer to -1
                1e-8,  # Very small positive
                -1e-8,  # Very small negative
                100.0,  # Large positive value
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (edge_values,))
        self.run_test_on_device(test_func, (edge_values,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            # FIXME accuqracy issue
            # torch.bfloat16
        ],
    )
    def test_log1p_precision(self, dtype):
        """Test log1p.default precision for small values where log1p shines."""

        def test_func(x):
            return torch.ops.aten.log1p.default(x)

        # log1p is more accurate than log(1+x) for small x
        small_values = torch.tensor(
            [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3],
            dtype=dtype,
        )

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16 and dtype != torch.float16:
            self.run_test_on_host(test_func, (small_values,))
        self.run_test_on_device(test_func, (small_values,))
