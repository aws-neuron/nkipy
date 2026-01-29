# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenAnyDefault(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.bool),
            ((8, 16, 32), torch.bool),
            ((4, 8, 16, 32), torch.bool),  # Multi-dimensional
            ((1, 1, 1), torch.bool),  # Singleton dimensions
            ((16,), torch.bool),  # 1D tensor
            ((128, 256), torch.float32),  # Floating point tensor
            ((64, 128), torch.int32),  # Integer tensor
        ],
    )
    def test_any_default_shapes_dtypes(self, shape, dtype):
        """Test any.default with different shapes and dtypes."""

        def test_func(x):
            # FIXME NEFF returns tensor shape torch.Size([1]) intead of torch.Size([])
            # so we added a view(1) here to work around,
            # need to fix NKIPy lowering in the future
            return torch.ops.aten.any.default(x).view(1)

        # Create tensors with varying values
        if dtype == torch.bool:
            # Create a tensor with mostly False but some True values
            arg_0 = torch.zeros(shape, dtype=torch.bool)
            # Set a few random elements to True
            indices = torch.randint(0, arg_0.numel(), (max(1, arg_0.numel() // 10),))
            arg_0.view(-1)[indices] = True
        elif dtype in [torch.int32, torch.int64]:
            # Create a tensor with mostly zeros but some non-zeros
            arg_0 = torch.zeros(shape, dtype=dtype)
            indices = torch.randint(0, arg_0.numel(), (max(1, arg_0.numel() // 10),))
            arg_0.view(-1)[indices] = torch.randint(1, 10, (len(indices),), dtype=dtype)
        else:
            # Create a tensor with mostly zeros but some non-zeros for float
            arg_0 = torch.zeros(shape, dtype=dtype)
            indices = torch.randint(0, arg_0.numel(), (max(1, arg_0.numel() // 10),))
            arg_0.view(-1)[indices] = torch.rand(len(indices), dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_any_default_special_cases(self):
        """Test any.default with special cases like all-zeros and all-ones."""

        def test_func(x):
            return torch.ops.aten.any.default(x).view(1)

        # All zeros - should return False
        zeros = torch.zeros((3, 4, 5), dtype=torch.bool)
        self.run_test_on_host(test_func, (zeros,))
        self.run_test_on_device(test_func, (zeros,))

        # All ones - should return True
        ones = torch.ones((3, 4, 5), dtype=torch.bool)
        self.run_test_on_host(test_func, (ones,))
        self.run_test_on_device(test_func, (ones,))

        # Single True among many False
        single_true = torch.zeros((10, 10), dtype=torch.bool)
        single_true[5, 5] = True
        self.run_test_on_host(test_func, (single_true,))
        self.run_test_on_device(test_func, (single_true,))

    def test_any_default_non_boolean(self):
        """Test any.default with non-boolean tensors."""

        def test_func(x):
            return torch.ops.aten.any.default(x).view(1)

        # Integer tensor
        int_tensor = torch.zeros((3, 4, 5), dtype=torch.int32)
        int_tensor[1, 2, 3] = 1
        self.run_test_on_host(test_func, (int_tensor,))
        self.run_test_on_device(test_func, (int_tensor,))

        # Float tensor with positive, negative, and zero values
        float_tensor = torch.tensor(
            [[0.0, -1.2, 0.0], [0.0, 0.0, 0.0], [3.14, 0.0, -0.5]]
        )
        self.run_test_on_host(test_func, (float_tensor,))
        self.run_test_on_device(test_func, (float_tensor,))

        # Test with infinity values
        inf_tensor = torch.tensor([0.0, float("inf"), 0.0, float("-inf")])
        self.run_test_on_host(test_func, (inf_tensor,))
        self.run_test_on_device(test_func, (inf_tensor,))

    def test_any_default_mixed_values(self):
        """Test any.default with tensors containing mixed values."""

        def test_func(x):
            return torch.ops.aten.any.default(x).view(1)

        # Test with a tensor containing True and False
        mixed_bool = torch.tensor([False, True, False, False])
        self.run_test_on_host(test_func, (mixed_bool,))
        self.run_test_on_device(test_func, (mixed_bool,))

        # Test with a tensor containing positive, negative, and zero values
        mixed_values = torch.tensor([-5, 0, 3, 0, -1, 0])
        self.run_test_on_host(test_func, (mixed_values,))
        self.run_test_on_device(test_func, (mixed_values,))

        # Test with a tensor containing very small values (close to zero but non-zero)
        small_values = torch.tensor([0.0, 1e-10, 0.0, -1e-10])
        self.run_test_on_host(test_func, (small_values,))
        self.run_test_on_device(test_func, (small_values,))
