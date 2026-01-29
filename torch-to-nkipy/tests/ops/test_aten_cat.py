# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenCat(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shapes,dim,dtype",
        [
            # Basic concatenation along dim 0
            ([(2, 3), (3, 3), (1, 3)], 0, torch.float32),
            # Concatenation along dim 1
            ([(3, 2), (3, 3), (3, 1)], 1, torch.float32),
            # Concatenation along dim -1
            ([(3, 2), (3, 3), (3, 1)], -1, torch.float32),
            # Concatenation along middle dimension
            ([(2, 3, 4), (2, 2, 4), (2, 1, 4)], 1, torch.float32),
            # Concatenation along last dimension
            ([(2, 3, 1), (2, 3, 2), (2, 3, 3)], 2, torch.float32),
            # Different dtypes
            ([(3, 4), (2, 4)], 0, torch.float16),
            ([(3, 128), (2, 128)], 0, torch.bfloat16),
            # Higher dimensional tensors
            ([(2, 3, 4, 5), (2, 3, 2, 5)], 2, torch.float32),
        ],
    )
    def test_cat_basic(self, shapes, dim, dtype):
        """Test aten.cat.default with various shapes, dimensions, and dtypes."""

        def test_func(tensors, dim):
            return torch.ops.aten.cat.default(tensors, dim)

        # Create tensors with the specified shapes
        tensors = [torch.randn(size=shape, dtype=dtype) for shape in shapes]
        args = [tensors, dim]

        self.run_test_on_host(test_func, args)
        self.run_test_on_device(test_func, args)

    def test_cat_single_tensor(self):
        """Test concatenation of one tensor (should return the tensor unchanged)."""

        def test_func(tensor):
            return torch.ops.aten.cat.default([tensor])

        arg = torch.randn(size=(3, 4), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg,))
        self.run_test_on_device(test_func, (arg,))
