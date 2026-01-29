# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenPowTensorTensor(NKIPyTestBase):
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
    def test_pow_shapes_dtypes(self, shape, dtype):
        """Test aten.pow.Tensor_Tensor with different shapes and dtypes."""

        def test_func(base, exponent):
            return torch.ops.aten.pow.Tensor_Tensor(base, exponent)

        base = torch.rand(size=shape, dtype=dtype) + 0.5  # Ensure positive values
        exponent = torch.rand(size=shape, dtype=dtype) * 2  # Keep exponents small

        self.run_test_on_host(test_func, (base, exponent))
        self.run_test_on_device(test_func, (base, exponent))

    def test_pow_broadcasting(self):
        """Test pow.Tensor_Tensor with broadcasting."""

        def test_func(base, exponent):
            return torch.ops.aten.pow.Tensor_Tensor(base, exponent)

        base = torch.rand(16, 32, dtype=torch.float32) + 0.5
        exponent = torch.rand(32, dtype=torch.float32) * 2

        self.run_test_on_host(test_func, (base, exponent))
        self.run_test_on_device(test_func, (base, exponent))
