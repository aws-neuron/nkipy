# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLog(NKIPyTestBase):
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
    def test_log_shapes_dtypes(self, shape, dtype):
        """Test aten.log.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.log.default(x)

        # Create positive input tensor
        arg_0 = (
            torch.rand(size=shape, dtype=dtype) + 0.1
        )  # Adding 0.1 to ensure positive values

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_log_special_values(self):
        """Test log.default with special values."""

        def test_func(x):
            return torch.ops.aten.log.default(x)

        special_values = torch.tensor(
            [
                1.0,  # log(1) = 0
                math.e,  # log(e) = 1
                10.0,  # log(10)
                2.0,  # log(2)
                0.5,  # log(0.5)
                # FIXME accuracy issue
                # 1e-10,  # very small positive number
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
