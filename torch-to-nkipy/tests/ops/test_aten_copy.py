# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenCopy(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((128, 256), torch.float16),
            # FIXME accuracy issues
            # ((128, 128), torch.bfloat16),
            ((1,), torch.float32),  # Scalar-like tensor
        ],
    )
    def test_copy_same_shape_dtype(self, shape, dtype):
        """Test aten.copy.default with same shape and dtype."""

        def test_func(dest, source):
            return torch.ops.aten.copy.default(dest, source) * 2

        # Create destination with values
        dest = torch.randn(size=shape, dtype=dtype)
        # Create source with different values
        source = (
            torch.randn(size=shape, dtype=dtype) * 10
        )  # multiply to ensure different values

        self.run_test_on_host(test_func, (dest, source))
        self.run_test_on_device(test_func, (dest, source))

    @pytest.mark.parametrize(
        "dest_shape,source_shape",
        [
            ((16, 32), (16, 32)),  # Same shape
            ((16, 32), (1, 32)),  # Broadcasting compatible
        ],
    )
    def test_copy_different_shapes(self, dest_shape, source_shape):
        """Test aten.copy.default with different but compatible shapes."""

        def test_func(dest, source):
            return torch.ops.aten.copy.default(dest, source) * 2

        dest = torch.randn(size=dest_shape, dtype=torch.float32)
        source = torch.randn(size=source_shape, dtype=torch.float32)

        self.run_test_on_host(test_func, (dest, source))
        self.run_test_on_device(test_func, (dest, source))

    def test_copy_zeros_and_ones(self):
        """Test aten.copy.default with specific tensor values."""

        def test_func(dest, source):
            return torch.ops.aten.copy.default(dest, source) * 2

        shape = (8, 16)
        dest = torch.zeros(size=shape, dtype=torch.float32)
        source = torch.ones(size=shape, dtype=torch.float32)

        self.run_test_on_host(test_func, (dest, source))
        self.run_test_on_device(test_func, (dest, source))

    def test_copy_different_dtypes(self):
        """Test aten.copy.default with different dtypes."""

        def test_func(dest, source):
            return torch.ops.aten.copy.default(dest, source)

        shape = (16, 32)
        dest = torch.zeros(size=shape, dtype=torch.float32)
        source = torch.ones(size=shape, dtype=torch.float16)

        self.run_test_on_host(test_func, (dest, source))
        self.run_test_on_device(test_func, (dest, source))
