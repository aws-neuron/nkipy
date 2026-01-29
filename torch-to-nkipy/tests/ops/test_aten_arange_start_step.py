# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenArange(NKIPyTestBase):
    @pytest.mark.parametrize(
        "start,end,step,dtype",
        [
            (0, 10, 1, torch.float32),  # Basic case
            (5, 15, 2, torch.float32),  # Different start with step
            (0, 20, 5, torch.int64),  # Integer type
            (-10, 10, 2, torch.float32),  # Negative start
            (10, -10, -2, torch.float32),  # Negative step (decreasing)
            (0, 5, 0.5, torch.float32),  # Fractional step
        ],
    )
    def test_arange_start_step(self, start, end, step, dtype):
        """Test aten.arange.start_step with different parameters."""

        def test_func():
            return torch.ops.aten.arange.start_step(start, end, step, dtype=dtype)

        self.run_test_on_host(test_func, ())
        # FIXME  RuntimeError: Unexpected return value from nki kernel
        # self.run_test_on_device(test_func, ())

    def test_arange_kwarg_step(self):
        """Test aten.arange.start_step with step provided as kwarg."""

        def test_func():
            return torch.ops.aten.arange.start_step(0, 10, step=2, dtype=torch.float32)

        self.run_test_on_host(test_func, ())
        # FIXME  RuntimeError: Unexpected return value from nki kernel [0. 2. 4. 6. 8.]
        # self.run_test_on_device(test_func, ())
