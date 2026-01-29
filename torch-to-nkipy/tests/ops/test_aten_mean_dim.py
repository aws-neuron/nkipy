# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMeanDim(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            ((16, 32), 0, False, torch.float32),  # Reduce first dim, don't keep dim
            ((16, 32), 1, False, torch.float32),  # Reduce second dim, don't keep dim
            ((16, 32), 0, True, torch.float32),  # Reduce first dim, keep dim
            ((16, 32), 1, True, torch.float32),  # Reduce second dim, keep dim
            ((8, 16, 32), 1, False, torch.float32),  # 3D tensor, middle dim
            ((8, 16, 32), -1, True, torch.float32),  # Negative dim index (last dim)
            ((4, 8, 16, 32), 2, True, torch.float32),  # 4D tensor
            ((512, 256), 1, False, torch.float16),  # FP16
            # FIXME accuracy issue
            # ((128, 512), 1, False, torch.bfloat16),  # BFloat16
            ((1, 32), 0, True, torch.float32),  # Singleton dimension
        ],
    )
    def test_mean_dim_basic(self, shape, dim, keepdim, dtype):
        """Test mean.dim with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten.mean.dim(x, dim, keepdim)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(0, 0.02)

        if dtype not in (torch.float16, torch.bfloat16):
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
