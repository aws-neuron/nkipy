# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLtScalar(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype,scalar",
        [
            ((16, 32), torch.float32, 0.0),
            ((8, 16, 32), torch.float32, 1.0),
            ((4, 8, 16), torch.float32, -1.0),
            # FIXME accuracy issue
            # ((128, 256), torch.float16, 0.5),
            # ((64, 128), torch.bfloat16, -0.5),
            ((16,), torch.float32, 2.0),
            ((1, 1, 1), torch.float32, float("inf")),
        ],
    )
    def test_lt_scalar_shapes_dtypes(self, shape, dtype, scalar):
        """Test aten.lt.Scalar with different shapes, dtypes, and scalar values."""

        def test_func(x):
            return torch.ops.aten.lt.Scalar(x, scalar)

        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_lt_scalar_special_values(self):
        """Test lt.Scalar with special values."""

        def test_func(x):
            return torch.ops.aten.lt.Scalar(x, 0.0)

        special_values = torch.tensor(
            [
                -float("inf"),
                -1.0,
                -0.0,
                0.0,
                1.0,
                float("inf"),
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
