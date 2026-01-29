# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenFullLike(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,fill_value,dtype",
        [
            ((16, 32), 1.0, torch.float32),
            ((8, 16, 32), 0.0, torch.float32),
            ((4, 8, 16, 32), -1.0, torch.float32),  # Multi-dimensional
            ((128, 256), 42.0, torch.float16),  # FP16
            # FIXME accuracy issues
            # ((64, 128), 3.14, torch.bfloat16),  # BFloat16
            ((16,), 2.718, torch.float32),  # 1D tensor
            ((1, 1, 1), 9.99, torch.float32),  # Singleton dimensions
        ],
    )
    def test_full_like_shapes_dtypes(self, shape, fill_value, dtype):
        """Test aten.full_like.default with different shapes, values and dtypes."""

        def test_func(x):
            return torch.ops.aten.full_like.default(x, fill_value)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "fill_value",
        [
            0,  # Integer zero
            1,  # Integer one
            -1,  # Negative integer
            True,  # Boolean True
            False,  # Boolean False
        ],
    )
    def test_full_like_special_values(self, fill_value):
        """Test aten.full_like.default with special fill values."""

        def test_func(x):
            return torch.ops.aten.full_like.default(x, fill_value)

        # Use a standard tensor
        arg_0 = torch.randn(size=(8, 16), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_full_like_scalar_tensor(self):
        """Test full_like with a scalar tensor."""

        def test_func(x):
            return torch.ops.aten.full_like.default(x, 5.0)

        # Scalar tensor (0-dim tensor)
        scalar_tensor = torch.tensor(0.5, dtype=torch.float32)

        self.run_test_on_host(test_func, (scalar_tensor,))
        self.run_test_on_device(test_func, (scalar_tensor,))

    def test_full_like_dtype_preservation(self):
        """Test that full_like preserves the input tensor's dtype."""

        def test_func(x):
            # Should maintain the dtype of x (int32)
            return torch.ops.aten.full_like.default(x, 2.5)  # 2.5 will be cast to int

        # Integer tensor
        arg_0 = torch.ones(size=(4, 8), dtype=torch.int32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
