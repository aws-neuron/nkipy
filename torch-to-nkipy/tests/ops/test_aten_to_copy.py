# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenToCopy(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,src_dtype,target_dtype",
        [
            ((16, 32), torch.float32, torch.float16),  # float32 -> float16
            ((16, 32), torch.float16, torch.float32),  # float16 -> float32
            ((8, 16, 32), torch.float32, torch.bfloat16),  # float32 -> bfloat16
            ((8, 16, 32), torch.bfloat16, torch.float32),  # bfloat16 -> float32
            ((4, 8, 16), torch.int32, torch.float32),  # int32 -> float32
            ((1, 1, 1), torch.float32, torch.float16),  # Singleton dimensions
        ],
    )
    def test_to_copy_basic(self, shape, src_dtype, target_dtype):
        """Test _to_copy.default with different shapes and dtype conversions."""

        def test_func(x):
            return torch.ops.aten._to_copy.default(x, dtype=target_dtype)

        # Create input tensor
        if src_dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=shape, dtype=src_dtype)
        else:
            arg_0 = torch.randn(size=shape, dtype=src_dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_to_copy_same_dtype(self):
        """Test _to_copy.default with same source and target dtype."""

        def test_func(x):
            # Converting to the same dtype should be a no-op in terms of values
            return torch.ops.aten._to_copy.default(x, dtype=torch.float32)

        arg_0 = torch.randn(size=(4, 8), dtype=torch.float32)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
