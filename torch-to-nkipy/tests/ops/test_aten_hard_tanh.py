# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenHardtanh(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.float32),
            ((8, 16, 32), torch.float32),
            ((4, 8, 16, 32), torch.float32),  # Multi-dimensional
            # FIXME Accuracy issues
            # ((128, 256), torch.float16),  # FP16
            # ((64, 128), torch.bfloat16),  # BFloat16
            ((16,), torch.float32),  # 1D tensor
            ((1, 1, 1), torch.float32),  # Singleton dimensions
        ],
    )
    def test_hardtanh_shapes_dtypes(self, shape, dtype):
        """Test aten.hardtanh.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.hardtanh.default(x, -1.0, 1.0)

        # Create input tensor with values spanning a wide range
        arg_0 = torch.randn(size=shape, dtype=dtype) * 3  # Values roughly in [-6, 6]

        # Use relaxed tolerances for 16-bit types
        if dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 1e-1, 1e-1
        else:
            rtol, atol = None, None

        self.run_test_on_host(test_func, (arg_0,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (arg_0,), rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (-1.0, 1.0),  # Standard hardtanh (default values)
            (-2.0, 2.0),  # Wider bounds
            (0.0, 1.0),  # ReLU-like activation
            (-0.5, 0.5),  # Narrower bounds
            (0.0, 0.0),  # Min = Max (forces all values to be zero)
            (-6.0, 6.0),  # Very wide bounds
        ],
    )
    def test_hardtanh_bounds(self, min_val, max_val):
        """Test hardtanh.default with different min/max bound combinations."""

        def test_func(x):
            return torch.ops.aten.hardtanh.default(x, min_val, max_val)

        # Create input with wide range of values
        input_tensor = torch.tensor(
            [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (input_tensor,))
        self.run_test_on_device(test_func, (input_tensor,))

    def test_hardtanh_default_params(self):
        """Test hardtanh.default with default parameters."""

        def test_default_hardtanh(x):
            return torch.ops.aten.hardtanh.default(
                x
            )  # Should use -1.0, 1.0 as defaults

        def test_partial_params(x):
            return torch.ops.aten.hardtanh.default(
                x, -1.0
            )  # Should use 1.0 as max default

        # Create a tensor with values that will be clipped by default bounds
        input_tensor = torch.tensor(
            [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0],
            dtype=torch.float32,
        )

        # Test functions
        for func in [test_default_hardtanh, test_partial_params]:
            self.run_test_on_host(func, (input_tensor,))
            self.run_test_on_device(func, (input_tensor,))

    def test_hardtanh_special_cases(self):
        """Test hardtanh.default with special values and edge cases."""

        def test_symmetric_bounds(x):
            return torch.ops.aten.hardtanh.default(x, -2.0, 2.0)

        def test_asymmetric_bounds(x):
            return torch.ops.aten.hardtanh.default(x, -1.0, 3.0)

        def test_positive_only(x):
            return torch.ops.aten.hardtanh.default(x, 0.0, 5.0)

        def test_negative_only(x):
            return torch.ops.aten.hardtanh.default(x, -5.0, 0.0)

        # Create a tensor with positive, negative, and zero values
        input_tensor = torch.tensor(
            [-8.0, -2.0, -1.0, 0.0, 1.0, 2.0, 8.0],
            dtype=torch.float32,
        )

        # Test each function
        for func in [
            test_symmetric_bounds,
            test_asymmetric_bounds,
            test_positive_only,
            test_negative_only,
        ]:
            self.run_test_on_host(func, (input_tensor,))
            self.run_test_on_device(func, (input_tensor,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            # FIXME Accuracy issues
            # torch.float16,
            # torch.bfloat16,
        ],
    )
    def test_hardtanh_dtype_consistency(self, dtype):
        """Test hardtanh.default preserves input dtype."""

        def test_func(x):
            return torch.ops.aten.hardtanh.default(x, -1.0, 1.0)

        # Create input tensor
        input_tensor = torch.randn(size=(32, 64), dtype=dtype) * 2

        # Use appropriate tolerances
        if dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 1e-1, 1e-1
        else:
            rtol, atol = None, None

        self.run_test_on_host(test_func, (input_tensor,), rtol=rtol, atol=atol)
        self.run_test_on_device(test_func, (input_tensor,), rtol=rtol, atol=atol)
