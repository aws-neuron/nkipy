# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenConstantPadNd(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,pad,value,dtype",
        [
            # 2D tensor padding
            ((16, 32), [1, 1, 2, 2], 0.0, torch.float32),  # pad all sides
            ((8, 16), [0, 1, 1, 0], 0.0, torch.float32),  # asymmetric padding
            ((4, 8), [2, 3, 1, 2], 5.0, torch.float32),  # non-zero padding value
            # 3D tensor padding
            ((4, 8, 16), [1, 1, 2, 2, 0, 1], 0.0, torch.float32),
            ((2, 4, 8), [0, 2, 1, 1, 1, 0], -1.0, torch.float32),
            # 4D tensor padding (common for conv operations)
            ((2, 3, 8, 8), [1, 1, 1, 1, 0, 0, 0, 0], 0.0, torch.float32),
            ((1, 1, 4, 4), [2, 2, 2, 2, 0, 0, 0, 0], 1.0, torch.float32),
            # Different dtypes
            ((16, 32), [1, 1, 1, 1], 0.0, torch.float16),
            ((16, 32), [2, 2, 2, 2], 0.0, torch.bfloat16),
        ],
    )
    def test_constant_pad_nd_shapes_dtypes(self, shape, pad, value, dtype):
        """
        Test aten.constant_pad_nd with different shapes, padding configs, and dtypes.
        """

        def test_func(input_tensor):
            return torch.ops.aten.constant_pad_nd.default(input_tensor, pad, value)

        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "pad_config",
        [
            [0, 0, 0, 0],  # no padding
            [1, 0, 0, 0],  # pad left only
            [0, 1, 0, 0],  # pad right only
            [0, 0, 1, 0],  # pad top only
            [0, 0, 0, 1],  # pad bottom only
            [5, 5, 5, 5],  # large symmetric padding
            [1, 3, 2, 4],  # all different padding values
        ],
    )
    def test_constant_pad_nd_padding_configs(self, pad_config):
        """Test different padding configurations on 2D tensors."""

        def test_func(input_tensor):
            return torch.ops.aten.constant_pad_nd.default(input_tensor, pad_config, 0.0)

        arg_0 = torch.randn(size=(8, 12), dtype=torch.float32)

        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "pad_value",
        [
            0.0,
            1.0,
            -1.0,
            3.14159,
            -2.718,
            100.0,
        ],
    )
    def test_constant_pad_nd_values(self, pad_value):
        """Test different constant padding values."""

        def test_func(input_tensor):
            return torch.ops.aten.constant_pad_nd.default(
                input_tensor, [2, 2, 1, 1], pad_value
            )

        arg_0 = torch.randn(size=(4, 8), dtype=torch.float32)

        self.run_test_on_device(test_func, (arg_0,))

    def test_constant_pad_nd_1d(self):
        """Test padding on 1D tensors."""

        def test_func(input_tensor):
            return torch.ops.aten.constant_pad_nd.default(input_tensor, [2, 3], 0.0)

        arg_0 = torch.randn(size=(10,), dtype=torch.float32)

        self.run_test_on_device(test_func, (arg_0,))

    def test_constant_pad_nd_edge_cases(self):
        """Test edge cases like small tensors and large padding."""

        def test_func(input_tensor):
            return torch.ops.aten.constant_pad_nd.default(
                input_tensor, [10, 10, 10, 10], 42.0
            )

        # Small tensor with large padding
        arg_0 = torch.ones(size=(1, 1), dtype=torch.float32)

        self.run_test_on_device(test_func, (arg_0,))
