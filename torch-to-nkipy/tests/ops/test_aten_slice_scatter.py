# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSliceScatter(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,start,end,step,dtype",
        [
            ((16, 32), 0, 2, 10, 1, torch.float32),
            ((8, 16, 32), 1, 0, 8, 1, torch.float32),
            ((4, 8, 16), 2, 5, None, 1, torch.float16),
            ((2, 4, 8), 0, None, 1, 1, torch.bfloat16),
        ],
    )
    def test_slice_scatter_shapes_dtypes(self, shape, dim, start, end, step, dtype):
        """Test aten.slice_scatter.default with different shapes, dims, and dtypes."""

        def test_func(src, input_tensor):  # <-- Reversed order!
            return torch.ops.aten.slice_scatter.default(
                input_tensor, src, dim, start, end, step
            )

        # Prepare values
        if end is None:
            end = shape[dim]
        if start is None:
            start = 0

        slice_len = max(0, (end - start + (step - 1 if step > 0 else step + 1)) // step)
        src_shape = list(shape)
        src_shape[dim] = slice_len

        input_tensor = torch.randn(shape, dtype=dtype)
        src = torch.randn(src_shape, dtype=dtype)

        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (src, input_tensor))
        self.run_test_on_device(test_func, (src, input_tensor))
