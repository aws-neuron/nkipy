# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenRsqrt(NKIPyTestBase):
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
    def test_rsqrt_shapes_dtypes(self, shape, dtype):
        """Test rsqrt.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.rsqrt.default(x)

        # Create positive input tensor (rsqrt needs positive values)
        arg_0 = torch.rand(size=shape, dtype=dtype) + 0.1  # Ensure values are positive

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        self.run_test_on_host(test_func, (arg_0,), rtol, atol)
        self.run_test_on_device(test_func, (arg_0,))

    def test_rsqrt_specific_values(self):
        """Test rsqrt.default with specific values for verification."""

        def test_func(x):
            return torch.ops.aten.rsqrt.default(x)

        # Create tensor with specific values for easy verification
        values = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0], dtype=torch.float32)
        # Expected outputs: [1.0, 0.5, 0.3333, 0.25, 0.2]

        self.run_test_on_host(test_func, (values,))
        self.run_test_on_device(test_func, (values,))

    def test_rsqrt_small_values(self):
        """Test rsqrt.default with small positive values."""

        def test_func(x):
            return torch.ops.aten.rsqrt.default(x)

        # Create tensor with small values (should produce large outputs)
        small_values = torch.tensor([1e-2, 1e-4, 1e-6, 1e-8], dtype=torch.float32)

        self.run_test_on_host(test_func, (small_values,))
        self.run_test_on_device(test_func, (small_values,))

    def test_rsqrt_large_values(self):
        """Test rsqrt.default with large values."""

        def test_func(x):
            return torch.ops.aten.rsqrt.default(x)

        # Create tensor with large values (should produce small outputs)
        large_values = torch.tensor([1e2, 1e4, 1e6, 1e8], dtype=torch.float32)

        self.run_test_on_host(test_func, (large_values,))
        self.run_test_on_device(test_func, (large_values,))
