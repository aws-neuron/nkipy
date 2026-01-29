# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSigmoid(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # 4D tensor
            ((128, 256), torch.float16),
            # FIXME accuracy issue
            # ((256, 256), torch.bfloat16),
            ((1,), torch.float32),  # Vector
        ],
    )
    def test_sigmoid_shapes_dtypes(self, shape, dtype):
        """Test aten.sigmoid.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.sigmoid.default(x)

        arg_0 = torch.randn(size=shape, dtype=dtype).normal_(mean=0.0, std=0.02)

        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        self.run_test_on_host(test_func, (arg_0,), rtol=rtol, atol=atol)

        self.run_test_on_device(test_func, (arg_0,))

    def test_sigmoid_special_values(self):
        """Test sigmoid with special values: zero, large positive/negative values."""

        def test_func(x):
            return torch.ops.aten.sigmoid.default(x)

        # Create a tensor with specific special values
        x = torch.zeros(10, dtype=torch.float32)
        x[0] = 0.0  # sigmoid(0) = 0.5
        x[1] = 1.0  # sigmoid(1) ≈ 0.731
        x[2] = -1.0  # sigmoid(-1) ≈ 0.269
        x[3] = 10.0  # sigmoid(10) ≈ 0.9999
        x[4] = -10.0  # sigmoid(-10) ≈ 0.0001
        x[5] = 20.0  # Tests saturation close to 1
        x[6] = -20.0  # Tests saturation close to 0
        x[7] = 0.1  # Near zero (positive)
        x[8] = -0.1  # Near zero (negative)
        x[9] = float("inf")  # Infinity should give 1.0

        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))
