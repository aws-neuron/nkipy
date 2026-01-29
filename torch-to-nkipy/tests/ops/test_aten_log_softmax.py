# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenLogSoftmax(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            ((16, 32), 1, torch.float32),  # 2D tensor, log_softmax over dim 1
            ((16, 32), 0, torch.float32),  # 2D tensor, log_softmax over dim 0
            ((8, 16, 32), 1, torch.float32),  # 3D tensor, log_softmax over middle dim
            (
                (8, 16, 32),
                -1,
                torch.float32,
            ),  # 3D tensor, log_softmax over last dim (negative index)
            ((4, 8, 16, 32), 2, torch.float32),  # 4D tensor
            ((128, 256), 1, torch.float16),  # FP16 test
            # FIXME accuracy issue
            # ((64, 128), 1, torch.bfloat16),  # BFloat16 test
            ((16,), 0, torch.float32),  # 1D tensor
            ((1, 32), 1, torch.float32),  # Singleton dimension
        ],
    )
    def test_log_softmax_shapes_dims(self, shape, dim, dtype):
        """Test _log_softmax.default with different shapes, dimensions and dtypes."""

        def test_func(x):
            return torch.ops.aten._log_softmax.default(
                x, dim, False
            )  # False for half_to_float

        # Create input tensor
        arg_0 = torch.randn(size=shape, dtype=dtype)

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16 and dtype != torch.float16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_log_softmax_numerical_stability(self):
        """Test log_softmax numerical stability with extreme values."""

        def test_func(x):
            return torch.ops.aten._log_softmax.default(x, 1, False)

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

    def test_log_softmax_properties(self):
        """Test log_softmax mathematical properties."""

        def test_func(x):
            return torch.ops.aten._log_softmax.default(x, 1, False)

        # Create test tensor
        test_tensor = torch.randn(size=(8, 16), dtype=torch.float32)

        # Run the test
        result = test_func(test_tensor)

        # Verify that log_softmax values are <= 0 (since probabilities are <= 1)
        assert torch.all(result <= 0.0), "Log-softmax values should be <= 0"

        # Verify that exp(log_softmax) sums to 1 along the specified dimension
        exp_result = torch.exp(result)
        sums = torch.sum(exp_result, dim=1)
        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
            rtol=1e-5,
            atol=1e-6,
            msg="exp(log_softmax) should sum to 1 along the specified dimension",
        )

        self.run_test_on_host(test_func, (test_tensor,))
        self.run_test_on_device(test_func, (test_tensor,))

    def test_log_softmax_vs_log_of_softmax(self):
        """Test that log_softmax matches log(softmax(x)) but is more
        numerically stable."""

        def log_softmax_func(x):
            return torch.ops.aten._log_softmax.default(x, 1, False)

        def log_of_softmax_func(x):
            softmax_result = torch.ops.aten._softmax.default(x, 1, False)
            return torch.log(softmax_result)

        # Test with normal values
        normal_vals = torch.randn(size=(4, 8), dtype=torch.float32)

        log_softmax_result = log_softmax_func(normal_vals)
        log_of_softmax_result = log_of_softmax_func(normal_vals)

        torch.testing.assert_close(
            log_softmax_result,
            log_of_softmax_result,
            rtol=1e-5,
            atol=1e-6,
            msg="log_softmax should match log(softmax(x)) for normal values",
        )

        self.run_test_on_host(log_softmax_func, (normal_vals,))
        self.run_test_on_device(log_softmax_func, (normal_vals,))

    @pytest.mark.xfail(
        reason="log_softmax with half to float conversion is not supported"
    )
    def test_log_softmax_half_to_float_flag(self):
        """Test log_softmax with half_to_float flag."""

        def test_func_true(x):
            # With half_to_float=True, output should be float32 even with float16 input
            return torch.ops.aten._log_softmax.default(x, 1, True)

        def test_func_false(x):
            # With half_to_float=False, output should keep input dtype
            return torch.ops.aten._log_softmax.default(x, 1, False)

        # Create float16 tensor
        tensor_f16 = torch.randn(size=(10, 20), dtype=torch.float16)

        self.run_test_on_host(test_func_true, (tensor_f16,))
        self.run_test_on_device(test_func_true, (tensor_f16,))

        self.run_test_on_host(test_func_false, (tensor_f16,))
        self.run_test_on_device(test_func_false, (tensor_f16,))

    def test_log_softmax_edge_cases(self):
        """Test log_softmax with edge cases."""

        def test_func(x):
            return torch.ops.aten._log_softmax.default(x, -1, False)

        # Test with uniform values
        uniform_vals = torch.ones(size=(3, 5), dtype=torch.float32)
        result = test_func(uniform_vals)
        expected_log_prob = torch.log(torch.tensor(1.0 / 5.0))  # log(1/n) for uniform
        torch.testing.assert_close(
            result,
            expected_log_prob * torch.ones_like(result),
            rtol=1e-5,
            atol=1e-6,
            msg="Uniform input should give uniform log probabilities",
        )

        # Test with single element
        single_element = torch.tensor([[5.0]], dtype=torch.float32)
        result_single = test_func(single_element)
        torch.testing.assert_close(
            result_single,
            torch.tensor([[0.0]]),
            rtol=1e-5,
            atol=1e-6,
            msg="Single element should give log probability of 0",
        )

        self.run_test_on_host(test_func, (uniform_vals,))
        self.run_test_on_device(test_func, (uniform_vals,))
        self.run_test_on_host(test_func, (single_element,))
        self.run_test_on_device(test_func, (single_element,))
