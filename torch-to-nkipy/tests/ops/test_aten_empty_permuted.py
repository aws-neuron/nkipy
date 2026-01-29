# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenEmptyPermuted(NKIPyTestBase):
    # Dynamo infers incorrect shapes if permutation is enabled.
    @pytest.mark.parametrize(
        "shape,permutation,dtype",
        [
            ([2, 3, 4], [0, 1, 2], torch.float32),  # Basic 3D case
            ([5, 10], [0, 1], torch.float32),  # 2D case with reversed dims
            ([2, 3, 4, 5], [0, 1, 2, 3], torch.float16),  # 4D with mixed permutation
            ([8, 16, 32], [0, 1, 2], torch.float32),  # Identity permutation
            ([3, 4], [0, 1], torch.bfloat16),  # BF16 test
        ],
    )
    def test_empty_permuted_shapes(self, shape, permutation, dtype):
        """Test aten.empty_permuted.default with different shapes and permutations."""

        def test_func():
            t = torch.ops.aten.empty_permuted.default(shape, permutation, dtype=dtype)
            return t.shape

        self.run_test_on_host(test_func, ())
