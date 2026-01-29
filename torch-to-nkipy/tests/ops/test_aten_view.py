# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenView(NKIPyTestBase):
    @pytest.mark.parametrize(
        "input_shape,output_shape,dtype",
        [
            ((16, 32), (512,), torch.float32),  # Flatten 2D -> 1D
            ((16, 32), (8, 64), torch.float32),  # Basic reshape
            ((16, 32), (16, 4, 8), torch.float32),  # 2D -> 3D
            ((16, 32), (2, 8, 4, 8), torch.float32),  # 2D -> 4D
            ((8, 16, 32), (8, 512), torch.float32),  # 3D -> 2D
            ((8, 16, 32), (4096,), torch.float32),  # Flatten 3D -> 1D
            ((512,), (16, 32), torch.float32),  # 1D -> 2D
            ((16, 32), (16, 32), torch.float32),  # Identity reshape
            ((16, 32), (16, -1), torch.float32),  # Auto-inference with -1
            ((16, 32), (-1, 16), torch.float32),  # Auto-inference with -1
            ((16, 32), (4, -1, 8), torch.float32),  # Auto-inference in middle
            ((32, 64), (2048,), torch.float16),  # FP16
            ((16, 128), (2048,), torch.bfloat16),  # BFloat16
            ((16, 64), (1024,), torch.int32),  # Integer tensor
            ((1, 1, 1), (1,), torch.float32),  # Tensor with singleton dimensions
        ],
    )
    def test_view_basic(self, input_shape, output_shape, dtype):
        """Test view.default with different input/output shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.view.default(x, output_shape)

        # Create input tensor
        if dtype in [torch.int32, torch.int64]:
            arg_0 = torch.randint(-100, 100, size=input_shape, dtype=dtype)
        else:
            arg_0 = torch.randn(size=input_shape, dtype=dtype)

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_view_scalar_to_tensor(self):
        """Test view with scalar (0-dim) tensors."""

        def test_func(x):
            # Scalar to 1-dim tensor with one element
            return torch.ops.aten.view.default(x, (1,))

        # Scalar tensor
        scalar_tensor = torch.tensor(3.14, dtype=torch.float32)

        self.run_test_on_host(test_func, (scalar_tensor,))
        self.run_test_on_device(test_func, (scalar_tensor,))

    def test_view_tensor_to_scalar(self):
        """Test view from 1-element tensor to scalar."""

        def test_func(x):
            # 1-element tensor to scalar (0-dim tensor)
            return torch.ops.aten.view.default(x, ())

        # 1-element tensor
        single_element_tensor = torch.tensor([3.14], dtype=torch.float32)

        self.run_test_on_host(test_func, (single_element_tensor,))
        self.run_test_on_device(test_func, (single_element_tensor,))
