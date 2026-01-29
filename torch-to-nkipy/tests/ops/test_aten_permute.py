# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenPermute(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dims,dtype",
        [
            # Standard permutations
            ((2, 3, 4), (2, 0, 1), torch.float32),  # Permuting 3D tensor
            ((2, 3, 4, 5), (0, 3, 1, 2), torch.float32),  # Permuting 4D tensor
            ((6, 7), (1, 0), torch.float32),  # Swapping dimensions in 2D tensor
            # Different dtypes
            ((2, 3, 4), (0, 2, 1), torch.float16),
            ((2, 3, 4), (1, 0, 2), torch.bfloat16),
            # Tensors with dimensions of size 1
            ((1, 3, 4), (0, 2, 1), torch.float32),
            ((2, 1, 4), (2, 0, 1), torch.float32),
            # High-dimensional tensors
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 0), torch.float32),  # 5D tensor
        ],
    )
    def test_permute_shapes_dtypes(self, shape, dims, dtype):
        """Test aten.permute.default with different shapes, permutations, and dtypes."""

        def test_func(x):
            return torch.ops.aten.permute.default(x, dims)

        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_permute_identity(self):
        """Test aten.permute.default with identity permutation."""

        def test_func(x):
            return torch.ops.aten.permute.default(x, (0, 1, 2))

        arg_0 = torch.randn(size=(2, 3, 4), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_permute_reverse(self):
        """Test aten.permute.default with reversed dimensions."""

        def test_func(x):
            return torch.ops.aten.permute.default(x, (3, 2, 1, 0))

        arg_0 = torch.randn(size=(2, 3, 4, 5), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
