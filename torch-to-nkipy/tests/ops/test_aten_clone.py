# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenClone(NKIPyTestBase):
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
    def test_clone_basic(self, shape, dtype):
        """Test clone.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.clone.default(x)

        # Create input tensor
        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=dtype)
        elif dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_clone_memory_format_arg(self):
        """Test clone.default with explicit memory_format argument."""

        def test_func(x):
            return torch.ops.aten.clone.default(
                x, memory_format=torch.contiguous_format
            )

        arg_0 = torch.randn(size=(8, 16, 32), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
