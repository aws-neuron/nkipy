# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLiftFreshCopy(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),
            ((128, 256), torch.float16),
            ((64, 128), torch.bfloat16),
            ((16,), torch.float32),
            ((1, 1, 1), torch.float32),
        ],
    )
    def test_lift_fresh_copy_shapes_dtypes(self, shape, dtype):
        """Test aten.lift_fresh_copy.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.lift_fresh_copy.default(x)

        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))
