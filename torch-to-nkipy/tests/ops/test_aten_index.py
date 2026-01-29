# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenIndex(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,index_specs,dtype",
        [
            # ((16, 32), [(slice(None), slice(None))], torch.float32),  # Basic slicing
            (
                (8, 16, 32),
                [(slice(None), 0, slice(None))],
                torch.float32,
            ),  # Mixed indexing
            (
                (4, 8, 16),
                [(torch.tensor([0, 1]), slice(None))],
                torch.float32,
            ),  # Tensor indexing
            # ((128, 256), [(slice(None), torch.tensor([0, 1]))], torch.float16),
            # ((64, 128), [(None, slice(None))], torch.bfloat16),  # With None
            # ((16,), [(slice(0, 10),)], torch.float32),  # 1D slicing
        ],
    )
    def test_index_shapes_dtypes(self, shape, index_specs, dtype):
        """Test aten.index.Tensor with different shapes and dtypes."""

        def test_func(x, indices):
            return torch.ops.aten.index.Tensor(x, indices)

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        for indices in index_specs:
            # Convert slice objects to proper indices
            processed_indices = []
            for idx in indices:
                if isinstance(idx, slice):
                    processed_indices.append(None)  # None for slices
                elif isinstance(idx, torch.Tensor):
                    processed_indices.append(idx)
                else:
                    processed_indices.append(torch.tensor(idx))

            self.run_test_on_host(test_func, (arg_0, processed_indices))
            self.run_test_on_device(test_func, (arg_0, processed_indices))
