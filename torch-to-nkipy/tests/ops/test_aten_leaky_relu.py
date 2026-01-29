# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLeakyRelu(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            ((128, 256), torch.float16),  # FP16
            # FIXME Accuracy issues
            # ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_leaky_relu_shapes_dtypes(self, shape, dtype):
        """Test aten.leaky_relu.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.leaky_relu.default(x, negative_slope=0.01)

        # Create input tensor with both positive and negative values
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_leaky_relu_special_values(self):
        """Test leaky_relu.default with special values."""

        def test_func(x):
            return torch.ops.aten.leaky_relu.default(x, negative_slope=0.01)

        special_values = torch.tensor(
            [
                0.0,  # LeakyReLU(0) = 0
                1.0,  # LeakyReLU(1) = 1
                -1.0,  # LeakyReLU(-1) = -0.01
                0.5,  # LeakyReLU(0.5) = 0.5
                -0.5,  # LeakyReLU(-0.5) = -0.005
                10.0,  # Large positive
                -10.0,  # Large negative = -0.1
                -0.001,  # Very small negative
                100.0,  # Very large positive
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (special_values,))
        self.run_test_on_device(test_func, (special_values,))

    @pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2, 0.05, 0.001])
    def test_leaky_relu_different_slopes(self, negative_slope):
        """Test leaky_relu.default with different negative slope values."""

        def test_func(x):
            return torch.ops.aten.leaky_relu.default(x, negative_slope=negative_slope)

        # Create input with both positive and negative values
        arg_0 = torch.tensor(
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "shape,negative_slope,dtype",
        [
            ((32, 64), 0.01, torch.float32),
            ((64, 32), 0.01, torch.float16),
            # FIXME Accuracy issues
            # ((8, 16, 32), 0.2, torch.bfloat16),
        ],
    )
    def test_leaky_relu_combined_params(self, shape, negative_slope, dtype):
        """Test leaky_relu.default with combined shape, slope, and dtype variations."""

        def test_func(x):
            return torch.ops.aten.leaky_relu.default(x, negative_slope=negative_slope)

        # Create input tensor with both positive and negative values
        arg_0 = torch.randn(size=shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_leaky_relu_edge_cases(self):
        """Test leaky_relu.default with edge cases."""

        def test_func(x):
            return torch.ops.aten.leaky_relu.default(x, negative_slope=0.01)

        # Test with extreme values
        edge_values = torch.tensor(
            [
                float("inf"),  # Positive infinity
                float(
                    "-inf"
                ),  # Negative infinity (should become negative_slope * -inf)
                1e10,  # Very large positive
                -1e10,  # Very large negative
                1e-10,  # Very small positive
                -1e-10,  # Very small negative
            ],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (edge_values,))
        self.run_test_on_device(test_func, (edge_values,))
