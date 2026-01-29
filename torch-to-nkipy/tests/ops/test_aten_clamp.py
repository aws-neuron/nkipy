# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenClamp(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            # FIXME accuracy issues
            # ((128, 256), torch.float16),  # FP16
            # ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_clamp_shapes_dtypes(self, shape, dtype):
        """Test aten.clamp.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.clamp.default(x, min=-1.0, max=1.0)

        # Create input tensor with values spanning a wide range
        arg_0 = torch.randn(size=shape, dtype=dtype) * 5  # Values roughly in [-10, 10]

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (-1.0, 1.0),  # Standard clamp
            (None, 1.0),  # Only max bound
            (-1.0, None),  # Only min bound
            (0.0, 0.0),  # Min = Max (forces all values to be the same)
            (-5.0, 5.0),  # Wider bounds
        ],
    )
    def test_clamp_bounds(self, min_val, max_val):
        """Test clamp.default with different min/max bound combinations."""

        def test_func(x):
            return torch.ops.aten.clamp.default(x, min=min_val, max=max_val)

        # Create input with wide range of values
        input_tensor = torch.tensor(
            [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_clamp_special_cases(self):
        """Test clamp.default with special values and edge cases."""

        def test_min_only(x):
            return torch.ops.aten.clamp.default(x, min=0.0)

        def test_max_only(x):
            return torch.ops.aten.clamp.default(x, max=0.0)

        def test_equal_bounds(x):
            return torch.ops.aten.clamp.default(x, min=1.0, max=1.0)

        # Create a tensor with positive, negative, and zero values
        input_tensor = torch.tensor(
            [-5.0, -1.0, 0.0, 1.0, 5.0],
            dtype=torch.float32,
        )

        # Test each function
        for func in [test_min_only, test_max_only, test_equal_bounds]:
            self.run_test_on_host(func, (input_tensor,))
            self.run_test_on_device(func, (input_tensor,))
