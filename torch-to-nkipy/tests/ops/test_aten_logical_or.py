# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLogicalOr(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.bool),
            ((8, 16, 32), torch.bool),
            ((4, 8, 16, 32), torch.bool),  # Multi-dimensional
            ((1, 1, 1), torch.bool),  # Singleton dimensions
            ((16,), torch.bool),  # 1D tensor
            # FIXME compiler error
            # ((128, 256), torch.float32),  # Floating point tensor
        ],
    )
    def test_logical_or_shapes_dtypes(self, shape, dtype):
        """Test logical_or with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.logical_or.default(x, y)

        # Create tensors with random boolean or float values
        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=torch.bool)
            arg_1 = torch.randint(0, 2, size=shape, dtype=torch.bool)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)
            arg_1 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_logical_or_broadcast(self):
        """Test logical_or with broadcasting."""

        def test_func(x, y):
            return torch.ops.aten.logical_or.default(x, y)

        # Test broadcasting with different shapes
        x = torch.randint(0, 2, size=(3, 1, 4), dtype=torch.bool)
        y = torch.randint(0, 2, size=(1, 2, 4), dtype=torch.bool)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with scalar and tensor
        scalar = torch.tensor(True)
        tensor = torch.randint(0, 2, size=(2, 3, 4), dtype=torch.bool)

        self.run_test_on_host(test_func, (scalar, tensor))
        self.run_test_on_device(test_func, (scalar, tensor))

        # Test with tensor and scalar (reverse order)
        self.run_test_on_host(test_func, (tensor, scalar))
        self.run_test_on_device(test_func, (tensor, scalar))

        # Test with False scalar
        scalar_false = torch.tensor(False)
        self.run_test_on_host(test_func, (scalar_false, tensor))
        self.run_test_on_device(test_func, (scalar_false, tensor))
