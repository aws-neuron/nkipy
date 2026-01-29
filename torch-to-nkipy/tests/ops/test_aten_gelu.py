# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenGelu(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            # FIXME Accuracy issues
            # ((4, 8, 16, 32), torch.float16),
            # ((2, 4, 8, 16), torch.bfloat16),
            ((1, 128, 256), torch.float32),
        ],
    )
    def test_gelu_shapes_dtypes(self, shape, dtype):
        """Test aten.gelu.default with different shapes and dtypes."""

        def test_func(input_tensor):
            return torch.ops.aten.gelu.default(input_tensor)

        input_tensor = torch.randn(shape, dtype=dtype).normal_(mean=0.0, std=0.02)

        # Skip BFloat16 test on host due to precision issues
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_gelu_special_values(self):
        """Test GELU with special input values."""

        def test_func(input_tensor):
            return torch.ops.aten.gelu.default(input_tensor)

        # Test with special values: 0, very large, very small, etc.
        special_values = torch.tensor(
            [0.0, 1.0, -1.0, 10.0, -10.0, 1e-6, -1e-6], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (special_values,), rtol=1e-2, atol=1e-2)
        self.run_test_on_device(test_func, (special_values,), rtol=1e-2, atol=1e-2)
