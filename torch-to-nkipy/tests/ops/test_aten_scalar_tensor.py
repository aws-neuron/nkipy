# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenScalarTensor(NKIPyTestBase):
    @pytest.mark.parametrize(
        "value,dtype",
        [
            (5, torch.float32),  # Integer to float32
            (3.14, torch.float32),  # Float to float32
            (0, torch.float32),  # Zero to float32
            (-42, torch.float32),  # Negative integer to float32
            (1e-5, torch.float32),  # Small float to float32
            (1e5, torch.float32),  # Large float to float32
            (5, torch.float16),  # Integer to float16
            (3.14, torch.float16),  # Float to float16
            (5, torch.bfloat16),  # Integer to bfloat16
            (3.14, torch.bfloat16),  # Float to bfloat16
            (5, torch.int32),  # Integer to int32
            (5, torch.int64),  # Integer to int64
            (True, torch.bool),  # Boolean to bool tensor
            (False, torch.bool),  # Boolean to bool tensor
            (0, torch.bool),  # Zero to bool tensor
            (1, torch.bool),  # One to bool tensor
        ],
    )
    def test_scalar_tensor_values_dtypes(self, value, dtype):
        """Test aten.scalar_tensor.default with different values and dtypes."""

        def test_func():
            return torch.ops.aten.scalar_tensor.default(value, dtype=dtype)

        self.run_test_on_host(test_func, ())
        # FIXME kernel specialize error when running on device
        # self.run_test_on_device(test_func, ())
