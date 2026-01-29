# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLogicalNot(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.bool),
            ((8, 16, 32), torch.bool),
            ((4, 8, 16, 32), torch.bool),  # Multi-dimensional
            ((1, 1, 1), torch.bool),  # Singleton dimensions
            ((16,), torch.bool),  # 1D tensor
            ((128, 256), torch.float32),  # Floating point tensor
        ],
    )
    def test_logical_not_shapes_dtypes(self, shape, dtype):
        """Test logical_not with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.logical_not.default(x)

        # Create tensors with random boolean values
        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=torch.bool)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_logical_not_special_values(self):
        """Test logical_not with special values."""

        def test_func(x):
            return torch.ops.aten.logical_not.default(x)

        # Create tensor with special values (int/float)
        special_values = torch.tensor(
            [
                0,
                1,
                -1,
                2,
                -2,  # Integer values
                0.0,
                1.0,
                -1.0,
                0.5,
                -0.5,  # Float values
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))
