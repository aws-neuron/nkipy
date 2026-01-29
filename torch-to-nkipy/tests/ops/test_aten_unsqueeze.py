# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenUnsqueeze(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            ((16, 32), 0, torch.float32),  # Add dimension at start
            ((16, 32), 1, torch.float32),  # Add dimension in middle
            ((16, 32), 2, torch.float32),  # Add dimension at end
            ((16, 32), -1, torch.float32),  # Add dimension at end (negative index)
            ((16, 32), -2, torch.float32),  # Add dimension before last (negative index)
            ((16, 32), -3, torch.float32),  # Add dimension at start (negative index)
            ((8, 16, 32), 1, torch.float32),  # 3D tensor
            ((4, 8, 16, 32), 2, torch.float32),  # 4D tensor
            ((128, 32), 0, torch.float16),  # FP16
            ((128, 32), 1, torch.bfloat16),  # BFloat16
            ((128, 32), 2, torch.int32),  # Integer tensor
            ((16,), 0, torch.float32),  # 1D tensor
            ((1, 1, 1), 1, torch.float32),  # Tensor with singleton dimensions
        ],
    )
    def test_unsqueeze_basic(self, shape, dim, dtype):
        """Test unsqueeze.default with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten.unsqueeze.default(x, dim)

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_unsqueeze_scalar_tensor(self):
        """Test unsqueeze with scalar (0-dim) tensors."""

        def test_func(x):
            # For scalar, result should be 1-dim tensor with one element
            return torch.ops.aten.unsqueeze.default(x, 0)

        # Scalar tensor
        scalar_tensor = torch.tensor(3.14, dtype=torch.float32)

        self.run_test_on_host(test_func, (scalar_tensor,))
        self.run_test_on_device(test_func, (scalar_tensor,))
