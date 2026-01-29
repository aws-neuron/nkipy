# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenWhere(NKIPyTestBase):
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
    def test_where_basic(self, shape, dtype):
        """Test where.self with different shapes and dtypes."""

        def test_func(condition, x, y):
            return torch.ops.aten.where.self(condition, x, y)

        # Create input tensors
        if dtype in [torch.int32, torch.int64]:
            x = torch.randint(-100, 100, size=shape, dtype=dtype)
            y = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            x = torch.randn(size=shape, dtype=dtype)
            y = torch.randn(size=shape, dtype=dtype)

        # Create condition tensor (random boolean mask)
        condition = torch.randint(0, 2, size=shape, dtype=torch.bool)

        self.run_test_on_host(test_func, (condition, x, y))
        self.run_test_on_device(test_func, (condition, x, y))
