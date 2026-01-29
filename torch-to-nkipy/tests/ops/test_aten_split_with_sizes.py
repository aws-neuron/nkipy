# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSplitWithSizes(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,split_sizes,dim,dtype,split_index",
        [
            # Basic cases - return first split
            ((64,), [32, 32], 0, torch.float32, 0),
            # Return second split
            ((128, 64), [16, 48], -1, torch.float32, 1),
            # Different data types and indices
            ((64, 32), [16, 16], -1, torch.float16, 0),
            ((32, 64), [32, 32], 1, torch.bfloat16, 1),
            # More complex splits - return different indices
            ((8, 16, 32), [8, 16, 8], -1, torch.float32, 0),
            ((8, 16, 32), [8, 16, 8], -1, torch.float32, 1),
            ((8, 16, 32), [8, 16, 8], -1, torch.float32, 2),
            # Uneven splits
            ((100,), [30, 50, 20], 0, torch.float32, 1),
        ],
    )
    def test_split_with_sizes_individual_returns(
        self, shape, split_sizes, dim, dtype, split_index
    ):
        """Test individual tensor returns from aten.split_with_sizes.default."""

        def test_func(x):
            splits = torch.ops.aten.split_with_sizes.default(x, split_sizes, dim)
            return splits[split_index]  # Return a specific split result

        # Create a test tensor with the specified shape and dtype
        arg_0 = torch.randn(size=shape, dtype=dtype)

        # Run tests on host and device
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize("split_index", [0, 1, 2])
    def test_split_with_sizes_multiple_operations(self, split_index):
        """Test operations on individual split results."""

        def test_func(x):
            splits = torch.ops.aten.split_with_sizes.default(x, [2, 3, 1], 1)
            # Return a single split after applying an operation
            return splits[split_index] * 2.0  # Scale the split by 2

        # Create a tensor with predictable values
        arg_0 = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )

        # Run tests on host and device
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_split_with_sizes_cat_operation(self):
        """Test concatenating split tensors back together."""

        def test_func(x):
            splits = torch.ops.aten.split_with_sizes.default(x, [2, 2], 1)
            # Process splits separately then concatenate
            return torch.cat([splits[0], splits[1]], dim=1)

        arg_0 = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
        )

        # Run tests on host and device
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_split_with_sizes_equal_parts(self):
        """Test splitting into equal parts for element-wise operations."""

        def test_func(x):
            # Split into equal parts that can be added together
            split0, split1 = torch.ops.aten.split_with_sizes.default(x, [3, 3], 1)
            return split0 + split1  # Now they have the same shape

        arg_0 = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )

        # Run tests on host and device
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
