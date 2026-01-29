# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenGeScalar(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,scalar,dtype",
        [
            ((16, 32), 0.0, torch.float32),
            ((8, 16), 1.0, torch.float32),
            ((4, 8, 16), -1.0, torch.float32),
            # FIXME Compiler errors
            # ((128,), 0.5, torch.float16),
            # ((64, 1), -0.5, torch.bfloat16),
            ((1,), float("inf"), torch.float32),
            ((2, 2), -float("inf"), torch.float32),
        ],
    )
    def test_ge_scalar_shapes_dtypes(self, shape, scalar, dtype):
        """Test aten.ge.Scalar with different shapes, scalars, and dtypes."""

        def test_func(x):
            return torch.ops.aten.ge.Scalar(x, scalar)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "tensor,scalar",
        [
            # Compare with zero
            (torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32), 0.0),
            # Compare with infinity
            (torch.tensor([float("inf"), 1.0], dtype=torch.float32), float("inf")),
            # Compare with negative infinity
            (torch.tensor([-float("inf"), -1.0], dtype=torch.float32), -float("inf")),
            # Compare with small numbers
            (torch.tensor([1e-7, 1e-8], dtype=torch.float32), 1e-7),
            # Compare with close numbers
            (torch.tensor([1.0, 1.0000001], dtype=torch.float32), 1.0),
        ],
    )
    def test_ge_scalar_special_values(self, tensor, scalar):
        """Test ge.Scalar with special values."""

        def test_func(x):
            return torch.ops.aten.ge.Scalar(x, scalar)

        self.run_test_on_host(test_func, (tensor,))
        self.run_test_on_device(test_func, (tensor,))

    @pytest.mark.parametrize(
        "tensor,scalar",
        [
            # Test with very large numbers
            (torch.tensor([1e38, -1e38], dtype=torch.float32), 1e37),
            # Test with very small numbers
            (torch.tensor([1e-38, -1e-38], dtype=torch.float32), 1e-39),
            # Test with mixed positive/negative numbers
            (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32), 0.0),
        ],
    )
    def test_ge_scalar_edge_cases(self, tensor, scalar):
        """Test ge.Scalar with edge cases."""

        def test_func(x):
            return torch.ops.aten.ge.Scalar(x, scalar)

        self.run_test_on_host(test_func, (tensor,))
        self.run_test_on_device(test_func, (tensor,))
