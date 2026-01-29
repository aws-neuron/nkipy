# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenAlias(NKIPyTestBase):
    """Test suite for torch.ops.aten.alias.default operation."""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),  # Basic FP32 case
            ((16, 32), torch.float16),  # FP16 case
            # FIXME Accuracy issues
            # ((16, 32), torch.bfloat16),  # BFloat16 case
            ((16, 32), torch.int32),  # INT32 case
            ((16, 32), torch.bool),  # Boolean tensor
            ((8, 16, 32), torch.float32),  # 3D tensor
            ((4, 8, 16, 32), torch.float32),  # 4D tensor
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_alias_basic(self, shape, dtype):
        """Test alias.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.alias.default(x + 1)

        # Create tensor with appropriate dtype
        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=dtype)
        elif dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype).normal_(mean=0.0, std=0.02)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_alias_contiguous_and_non_contiguous(self):
        """Test alias.default with contiguous and non-contiguous tensors."""

        def test_func(x):
            return torch.ops.aten.alias.default(x + 1)

        # Test with contiguous tensor
        x_contiguous = torch.randn(4, 4)
        self.run_test_on_host(test_func, (x_contiguous,))
        self.run_test_on_device(test_func, (x_contiguous,))

        # Test with non-contiguous tensor (transpose)
        x_non_contiguous = torch.randn(4, 4).transpose(0, 1)
        self.run_test_on_host(test_func, (x_non_contiguous,))
        self.run_test_on_device(test_func, (x_non_contiguous,))

        # Test with non-contiguous tensor (select/slice)
        x_strided = torch.randn(10, 10)[:, ::2]  # Every other column
        self.run_test_on_host(test_func, (x_strided,))
        self.run_test_on_device(test_func, (x_strided,))

    def test_alias_structured_data(self):
        """Test alias.default with tensors containing structured data."""

        def test_func(x):
            return torch.ops.aten.alias.default(x + 1)

        # Create a tensor with a clear pattern
        x = torch.zeros(5, 5)
        x[0, :] = 1  # First row all ones
        x[:, 0] = 2  # First column all twos
        x.diagonal().fill_(3)  # Diagonal all threes

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Another patterned tensor
        y = torch.zeros(3, 3, 3)
        y[0] = 1  # First 2D slice all ones
        y[:, 1] = 2  # Middle rows all twos
        y[:, :, 2] = 3  # Last channel all threes

        self.run_test_on_host(test_func, (y,))
        self.run_test_on_device(test_func, (y,))
