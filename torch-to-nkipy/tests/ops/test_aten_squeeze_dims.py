# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSqueezeDims(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dims,expected_shape",
        [
            ((1, 768, 2048), [0], (768, 2048)),  # Squeeze first dimension
            ((768, 1, 2048), [1], (768, 2048)),  # Squeeze middle dimension
            ((768, 2048, 1), [2], (768, 2048)),  # Squeeze last dimension
            ((1, 768, 1, 2048), [0, 2], (768, 2048)),  # Squeeze multiple dimensions
            ((1, 1, 1, 5), [0, 1, 2], (5,)),  # Squeeze multiple consecutive dimensions
            (
                (1, 5, 1, 10, 1),
                [0, 2, 4],
                (5, 10),
            ),  # Squeeze multiple non-consecutive dimensions
            ((5, 10, 15), [0], (5, 10, 15)),  # No-op (dim is not size 1)
            ((1,), [0], ()),  # Squeeze to scalar (empty shape)
            ((1, 1, 1), [0, 1, 2], ()),  # Squeeze all dimensions to scalar
        ],
    )
    def test_squeeze_dims_shapes(self, shape, dims, expected_shape):
        """Test aten.squeeze.dims with different shapes and dimensions."""

        def test_func(x):
            return torch.ops.aten.squeeze.dims(x, dims)

        # Create input tensor with specified shape
        arg_0 = torch.randn(size=shape, dtype=torch.float32)

        # Run the tests
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
        ],
    )
    def test_squeeze_dims_dtypes(self, dtype):
        """Test aten.squeeze.dims with different data types."""

        def test_func(x):
            return torch.ops.aten.squeeze.dims(x, [0])

        # Create input tensor with size 1 on first dimension
        shape = (1, 2048, 8192)

        # Handle different ways to create tensors of different types
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            arg_0 = torch.randn(size=shape, dtype=dtype)
        else:  # Integer types
            arg_0 = torch.randint(0, 10, size=shape, dtype=dtype)

        # Run the tests
        self.run_test_on_host(test_func, (arg_0,))
        # FIXME compiler bug
        # self.run_test_on_device(test_func, (arg_0,))

    def test_squeeze_dims_values_preserved(self):
        """Test that values are correctly preserved after squeezing."""

        def test_func(x):
            return torch.ops.aten.squeeze.dims(x, [0, 2])

        # Create input tensor with specific values
        arg_0 = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
        )  # Shape: (1, 2, 3)

        # Run the tests
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_squeeze_dims_negative_indices(self):
        """Test aten.squeeze.dims with negative dimension indices."""

        def test_func(x):
            # Using negative indices: -1 refers to the last dimension, etc.
            return torch.ops.aten.squeeze.dims(x, [-3, -1])

        # Create input tensor with size 1 on first and last dimensions
        arg_0 = torch.randn(size=(1, 5, 1), dtype=torch.float32)

        # Run the tests
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
