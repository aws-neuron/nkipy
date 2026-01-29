# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSlice(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,start,end,dtype",
        [
            ((16, 32), 0, 2, 10, torch.float32),  # Slice along first dim
            ((16, 32), 1, 5, 20, torch.float32),  # Slice along second dim
            ((8, 16, 32), 0, 2, 6, torch.float32),  # Slice 3D tensor along first dim
            ((8, 16, 32), 1, 5, 10, torch.float32),  # Slice 3D tensor along second dim
            ((8, 16, 32), 2, 10, 25, torch.float32),  # Slice 3D tensor along third dim
            (
                (4, 8, 16, 32),
                2,
                5,
                10,
                torch.float32,
            ),  # Slice 4D tensor along middle dim
            ((16, 32), 1, 0, 10, torch.float16),  # Slice FP16 tensor
            # FIXME accuracy issue
            # ((16, 32), 0, 5, 10, torch.bfloat16),  # Slice BFloat16 tensor
            ((16, 32), 0, 0, 16, torch.int32),  # Slice integer tensor
            ((16,), 0, 5, 10, torch.float32),  # Slice 1D tensor
            ((1, 32), 1, 10, 20, torch.float32),  # Slice tensor with singleton dim
        ],
    )
    def test_slice_basic(self, shape, dim, start, end, dtype):
        """Test slice.Tensor with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten.slice.Tensor(x, dim, start, end) * 2

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_slice_entire_tensor(self):
        """Test slice.Tensor for the special case of slicing the entire tensor."""

        def test_func(x):
            # This should return a tensor identical to the input
            return torch.ops.aten.slice.Tensor(x, 0, 0, sys.maxsize) * 2

        arg_0 = torch.randn(size=(8, 16), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_slice_specific_values(self):
        """Test slice.Tensor with specific values for verification."""

        def test_func(x):
            # Slice to get a specific subset that's easy to verify
            return torch.ops.aten.slice.Tensor(x, 0, 1, 3) * 2  # Get rows 1,2

        # Create tensor with recognizable pattern
        arg_0 = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32
        )
        # Expected result: [[4,5,6], [7,8,9]]

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_slice_out_of_bounds(self):
        """Test slice.Tensor with out-of-bounds indices."""

        def test_func_0(x):
            # Out-of-bounds indices should be clamped
            return torch.ops.aten.slice.Tensor(x, 0, -10, -1) * 2  # Start too negative

        def test_func_1(x):
            # Out-of-bounds indices should be clamped
            return torch.ops.aten.slice.Tensor(x, 0, 0, 100) * 2  # End too large

        def test_func_2(x):
            # Out-of-bounds indices should be clamped
            return torch.ops.aten.slice.Tensor(x, 0, 20, 30) * 2  # Start beyond size

        arg_0 = torch.randn(size=(5, 10), dtype=torch.float32)

        self.run_test_on_host(test_func_0, (arg_0,))
        # FIXME out of bounds on device returns all zeros
        # self.run_test_on_device(test_func_0, (arg_0,))

        self.run_test_on_host(test_func_1, (arg_0,))
        # FIXME out of bounds on device returns all zeros
        # self.run_test_on_device(test_func_1, (arg_0,))

        self.run_test_on_host(test_func_2, (arg_0,))
        # FIXME out of bounds on device returns all zeros
        # self.run_test_on_device(test_func_2, (arg_0,))

    @pytest.mark.parametrize(
        "shape,dim,start,end,step,dtype",
        [
            # Basic positive step
            ((20, 30), 0, 2, 18, 2, torch.float32),
            # Step on a non-primary dimension
            ((10, 30), 1, 1, 25, 3, torch.float32),
            # Step on the last dimension of a 3D tensor
            ((5, 10, 40), 2, 3, 35, 4, torch.float32),
            # Step larger than the remaining slice
            ((20,), 0, 5, 15, 20, torch.float32),
        ],
    )
    def test_slice_with_step(self, shape, dim, start, end, step, dtype):
        """Test slice.Tensor with a step parameter."""

        def test_func(x):
            return torch.ops.aten.slice.Tensor(x, dim, start, end, step) * 2

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
