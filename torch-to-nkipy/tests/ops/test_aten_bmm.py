# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenBmm(NKIPyTestBase):
    @pytest.mark.parametrize(
        "batch_size,m,k,n,dtype",
        [
            (4, 16, 8, 32, torch.float32),  # Basic case
            (1, 16, 8, 32, torch.float32),  # Single batch
            (10, 32, 32, 32, torch.float32),  # Square matrices
            (8, 64, 32, 16, torch.float32),  # Larger matrices
            (4, 128, 256, 512, torch.float16),  # Half precision
            (4, 512, 256, 512, torch.bfloat16),  # BFloat16
        ],
    )
    def test_bmm_shapes_dtypes(self, batch_size, m, k, n, dtype):
        """Test aten.bmm.default with different shapes and dtypes."""

        def test_func(a, b):
            return torch.ops.aten.bmm.default(a, b)

        # Create batch of matrices: a is [batch_size, m, k], b is [batch_size, k, n]
        arg_0 = torch.randn(size=(batch_size, m, k), dtype=dtype).normal_(
            mean=0.0, std=0.02
        )
        arg_1 = torch.randn(size=(batch_size, k, n), dtype=dtype).normal_(
            mean=0.0, std=0.02
        )

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))
