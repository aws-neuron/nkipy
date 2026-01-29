# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenAnyDim(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            ((16, 32), 0, False, torch.bool),  # 2D, reduce first dim
            ((16, 32), 1, False, torch.bool),  # 2D, reduce second dim
            ((8, 16, 32), 1, False, torch.bool),  # 3D, reduce middle dim
            (
                (8, 16, 32),
                -1,
                False,
                torch.bool,
            ),  # 3D, reduce last dim (negative index)
            ((8, 16, 32), 0, True, torch.bool),  # 3D, reduce first dim, keep dim
            ((4, 8, 16, 32), 2, True, torch.bool),  # 4D, reduce dim 2, keep dim
            (
                (16, 32),
                0,
                False,
                torch.float32,
            ),  # Float tensor (will be converted to bool)
            ((16, 32), 1, False, torch.int32),  # Int tensor (will be converted to bool)
            ((1, 32), 0, True, torch.bool),  # Singleton dimension, keep dim
        ],
    )
    def test_any_dim_basic(self, shape, dim, keepdim, dtype):
        """Test any.dim with different shapes, dimensions, keepdim and dtypes."""

        def test_func(x):
            return torch.ops.aten.any.dim(x, dim, keepdim)

        # Create input tensor
        if dtype == torch.bool:
            # Create random boolean tensor (25% True)
            arg_0 = torch.rand(size=shape) < 0.25
        elif dtype in [torch.int32, torch.int64]:
            # Create random integer tensor (some zeros, some non-zeros)
            arg_0 = torch.randint(-5, 5, size=shape, dtype=dtype)
        else:
            # Create random float tensor (some zeros, some non-zeros)
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
