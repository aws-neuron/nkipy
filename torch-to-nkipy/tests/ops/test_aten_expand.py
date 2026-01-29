# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenExpand(NKIPyTestBase):
    @pytest.mark.parametrize(
        "input_shape,target_shape,dtype",
        [
            ((1, 32), (16, 32), torch.float32),  # Basic expansion
            ((16, 1), (16, 32), torch.float32),  # Expand second dimension
            ((1, 1, 1), (8, 16, 32), torch.float32),  # Expand all dimensions
            ((1, 4, 1), (8, 4, 32), torch.float32),  # Keep middle dimension unchanged
            ((1,), (16,), torch.float32),  # 1D tensor expansion
            ((5, 1, 6), (5, 8, 6), torch.float32),  # Specific sizes
            ((1, 128), (16, 128), torch.float16),  # FP16
            ((1, 128), (128, 128), torch.bfloat16),  # BFloat16
            ((1, 1, 1), (1, 1, 1), torch.float32),  # No expansion (same shape)
        ],
    )
    def test_expand_basic(self, input_shape, target_shape, dtype):
        """Test expand.default with different input/output shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.expand.default(x, target_shape)

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=input_shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=input_shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_expand_with_inference(self):
        """Test expand.default with -1 placeholder for dimension inference."""

        def test_func(x):
            # -1 means "keep this dimension the same"
            return torch.ops.aten.expand.default(x, (-1, 16))

        # Input tensor with shape (3, 1)
        arg_0 = torch.randn(size=(3, 1), dtype=torch.float32)
        # Expected output shape: (3, 16)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_expand_scalar_tensor(self):
        """Test expand.default with scalar (0-dim) tensors."""

        def test_func(x):
            # Expand scalar to multi-dimensional tensor
            return torch.ops.aten.expand.default(x, (8, 16))

        # Scalar tensor (single value)
        scalar_tensor = torch.tensor(3.14, dtype=torch.float32)

        self.run_test_on_host(test_func, (scalar_tensor,))
        self.run_test_on_device(test_func, (scalar_tensor,))

    def test_expand_values(self):
        """Test that expansion correctly repeats values."""

        def test_func(x):
            # Expand tensor with known values to verify repetition
            return torch.ops.aten.expand.default(x, (3, 4))

        # Create tensor with known values
        known_values = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32
        )  # Shape: (1, 4)
        # Expected output:
        # [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]

        self.run_test_on_host(test_func, (known_values,))
        self.run_test_on_device(test_func, (known_values,))
