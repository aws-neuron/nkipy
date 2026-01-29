# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenSoftmax(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            ((16, 32), 1, torch.float32),  # 2D tensor, softmax over dim 1
            ((16, 32), 0, torch.float32),  # 2D tensor, softmax over dim 0
            ((8, 16, 32), 1, torch.float32),  # 3D tensor, softmax over middle dim
            (
                (8, 16, 32),
                -1,
                torch.float32,
            ),  # 3D tensor, softmax over last dim (negative index)
            ((4, 8, 16, 32), 2, torch.float32),  # 4D tensor
            ((128, 256), 1, torch.float16),  # FP16 test
            # FIXME accuracy issue
            # ((64, 128), 1, torch.bfloat16),  # BFloat16 test
            ((16,), 0, torch.float32),  # 1D tensor
            ((1, 32), 1, torch.float32),  # Singleton dimension
        ],
    )
    def test_softmax_shapes_dims(self, shape, dim, dtype):
        """Test _softmax.default with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten._softmax.default(
                x, dim, False
            )  # False for half_to_float

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_softmax_numerical_stability(self):
        """Test softmax numerical stability with extreme values."""

        def test_func(x):
            return torch.ops.aten._softmax.default(x, 1, False)

        # Create tensor with extreme values
        large_vals = torch.tensor(
            [[1.0, 50.0, 100.0], [-50.0, 0.0, 50.0], [-100.0, -50.0, 0.0]],
            dtype=torch.float32,
        )

        self.run_test_on_host(test_func, (large_vals,))
        self.run_test_on_device(test_func, (large_vals,))

        # Very large values
        very_large = torch.tensor(
            [[1e3, 1e4, 1e5], [-1e3, 0.0, 1e3], [-1e5, -1e4, -1e3]], dtype=torch.float32
        )

        self.run_test_on_host(test_func, (very_large,))
        self.run_test_on_device(test_func, (very_large,))

    @pytest.mark.xfail(
        reason="softmax with half to float conversion is not supported on CPU"
    )
    def test_softmax_half_to_float_flag(self):
        """Test softmax with half_to_float flag."""

        def test_func_true(x):
            # With half_to_float=True, output should be float32 even with float16 input
            return torch.ops.aten._softmax.default(x, 1, True)

        def test_func_false(x):
            # With half_to_float=False, output should keep input dtype
            return torch.ops.aten._softmax.default(x, 1, False)

        # Create float16 tensor
        tensor_f16 = torch.randn(size=(10, 20), dtype=torch.float16)

        self.run_test_on_host(test_func_true, (tensor_f16,))
        self.run_test_on_device(test_func_true, (tensor_f16,))

        self.run_test_on_host(test_func_false, (tensor_f16,))
        self.run_test_on_device(test_func_false, (tensor_f16,))
