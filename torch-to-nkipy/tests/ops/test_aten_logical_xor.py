# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLogicalXor(NKIPyTestBase):
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
    def test_logical_xor_shapes_dtypes(self, shape, dtype):
        """Test logical_xor with different shapes and dtypes."""

        def test_func(x, y):
            return torch.ops.aten.logical_xor.default(x, y)

        # Create tensors with random boolean or float values
        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=torch.bool)
            arg_1 = torch.randint(0, 2, size=shape, dtype=torch.bool)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)
            arg_1 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0, arg_1))
        self.run_test_on_device(test_func, (arg_0, arg_1))

    def test_logical_xor_broadcast(self):
        """Test logical_xor with broadcasting."""

        def test_func(x, y):
            return torch.ops.aten.logical_xor.default(x, y)

        # Test broadcasting with different shapes
        x = torch.randint(0, 2, size=(3, 1, 4), dtype=torch.bool)
        y = torch.randint(0, 2, size=(1, 2, 4), dtype=torch.bool)

        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))

        # Test with scalar and tensor
        scalar_true = torch.tensor(True)
        tensor = torch.randint(0, 2, size=(2, 3, 4), dtype=torch.bool)

        self.run_test_on_host(test_func, (scalar_true, tensor))
        self.run_test_on_device(test_func, (scalar_true, tensor))

        # Test with tensor and scalar (reverse order)
        self.run_test_on_host(test_func, (tensor, scalar_true))
        self.run_test_on_device(test_func, (tensor, scalar_true))

        # Test with False scalar
        scalar_false = torch.tensor(False)
        self.run_test_on_host(test_func, (scalar_false, tensor))
        self.run_test_on_device(test_func, (scalar_false, tensor))

    def test_logical_xor_truth_table(self):
        """Test logical_xor basic truth table cases."""

        def test_func(x, y):
            return torch.ops.aten.logical_xor.default(x, y)

        # Create a tensor with all possible combinations for truth table
        x = torch.tensor([False, False, True, True])
        y = torch.tensor([False, True, False, True])

        # Truth table for XOR should be: [False, True, True, False]
        self.run_test_on_host(test_func, (x, y))
        self.run_test_on_device(test_func, (x, y))
