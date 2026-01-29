# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenFull(NKIPyTestBase):
    @pytest.mark.parametrize(
        "size,fill_value,dtype",
        [
            ((16, 32), 1.0, torch.float32),
            ((8, 16, 32), 0.0, torch.float32),
            ((4, 8, 16, 32), -1.0, torch.float32),
            ((128, 256), 0.5, torch.float16),
            # FIXME accuracy issues
            # ((64, 128), 2.0, torch.bfloat16),
            ((16,), 1.0, torch.float32),
            ((1, 1, 1), 3.14, torch.float32),
        ],
    )
    def test_full_shapes_values_dtypes(self, size, fill_value, dtype):
        """Test aten.full.default with different shapes, values, and dtypes."""

        def test_func(t):
            return t + torch.ops.aten.full.default(
                size, fill_value, dtype=dtype, device=t.device
            )

        t = torch.randn(size=size, dtype=dtype)

        self.run_test_on_device(test_func, (t,))

    def test_full_edge_cases(self):
        """Test full.default with edge cases."""

        test_cases = [
            # Empty shape
            ((), 1.0, torch.float32),
            # Very large shape
            ((1000, 1), 0.0, torch.float32),
            # Very small value
            ((2, 2), 1e-10, torch.float32),
            # Very large value
            ((2, 2), 1e10, torch.float32),
        ]

        for size, fill_value, dtype in test_cases:

            def test_func(t):
                return t + torch.ops.aten.full.default(
                    size, fill_value, dtype=dtype, device=t.device
                )

            t = torch.randn(size=size, dtype=dtype)

            self.run_test_on_device(test_func, (t,))
