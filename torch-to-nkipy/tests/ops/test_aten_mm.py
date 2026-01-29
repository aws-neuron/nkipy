# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenMm(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape1,shape2,dtype",
        [
            # Standard matrix multiplication shapes
            ((32, 64), (64, 48), torch.float32),  # (m,k) x (k,n)
            ((16, 32), (32, 64), torch.float32),  # Different dimensions
            ((100, 10), (10, 100), torch.float32),  # Square result
            ((1, 64), (64, 1), torch.float32),  # Vector-vector to scalar
            ((128, 256), (256, 512), torch.float16),  # Half precision
            ((512, 256), (256, 512), torch.bfloat16),  # Half precision
        ],
    )
    def test_mm_shapes_dtypes(self, shape1, shape2, dtype):
        """Test aten.mm.default with different shapes and dtypes."""

        def test_func(a, b):
            return torch.ops.aten.mm.default(a, b)

        arg_0 = torch.randn(size=shape1, dtype=dtype).normal_(mean=0.0, std=0.02)
        arg_1 = torch.randn(size=shape2, dtype=dtype).normal_(mean=0.0, std=0.02)

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))
