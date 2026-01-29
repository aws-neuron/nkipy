# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMeanDefault(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            # Mean over all elements
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # 4D tensor
            ((512, 256), torch.float16),  # FP16
            # FIXME accuracy issue
            # ((128, 512), torch.bfloat16),  # BFloat16
            ((1, 32), torch.float32),  # Singleton dimension
        ],
    )
    def test_mean_default_basic(self, shape, dtype):
        """Test mean.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.mean.default(x)  # No dim or keepdim parameter

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 0.02)

        # Use relaxed tolerances for 16-bit types
        rtol, atol = (
            (1e-1, 1e-1) if dtype in (torch.float16, torch.bfloat16) else (None, None)
        )

        if dtype not in (torch.float16, torch.bfloat16):
            self.run_test_on_host(test_func, (arg_0,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (arg_0,), rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "shape,dtype,out_dtype",
        [
            ((16, 32), torch.float16, torch.float32),  # Upcast from float16
            # ((16, 32), torch.int32, torch.float32),    # Integer to float
        ],
    )
    def test_mean_default_dtype(self, shape, dtype, out_dtype):
        """Test mean.default with dtype conversion."""

        def test_func(x):
            return torch.ops.aten.mean.default(
                x, dtype=out_dtype
            )  # Only tensor and dtype

        # Create input tensor based on dtype
        if dtype in (torch.int32, torch.int64):
            arg_0 = torch.randint(0, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        # Use relaxed tolerances for 16-bit types
        rtol, atol = (
            (1e-1, 1e-1) if dtype in (torch.float16, torch.bfloat16) else (None, None)
        )

        if dtype not in (torch.float16, torch.bfloat16):
            self.run_test_on_host(test_func, (arg_0,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (arg_0,), rtol=rtol, atol=atol)
