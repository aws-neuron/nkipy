# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenPowTensorScalar(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,exponent,dtype",
        [
            ((16, 32), 2.0, torch.float32),  # Square - float exponent
            ((16, 32), 2, torch.float32),  # Square - int exponent
            ((8, 16, 32), 3.0, torch.float32),  # Cube
            ((4, 8, 16, 32), 0.5, torch.float32),  # Square root
            ((128, 256), 0.0, torch.float16),  # Zero exponent (results in all ones)
            # FIXME accuracy issue
            # ((64, 128), 1.0, torch.bfloat16),  # Identity exponent
            ((16, 32), -1.0, torch.float32),  # Reciprocal
            ((1, 1, 1), 4.0, torch.float32),  # Singleton dimensions
        ],
    )
    def test_pow_tensor_scalar_shapes_exponents(self, shape, exponent, dtype):
        """Test pow.Tensor_Scalar with different shapes, exponents, and dtypes."""

        def test_func(x):
            return torch.ops.aten.pow.Tensor_Scalar(x, exponent)

        # For tests with negative exponents or fractional exponents with negative bases,
        # use positive input values to avoid numerical issues
        if exponent < 0 or (exponent != int(exponent) and exponent > 0):
            arg_0 = (
                torch.abs(torch.randn(size=shape, dtype=dtype)) + 0.1
            )  # Ensure positive values
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype)

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "exponent",
        [
            0.0,  # Zero exponent - results in all ones
            1.0,  # Identity exponent
            2.0,  # Square
            3.0,  # Cube
            0.5,  # Square root
            -1.0,  # Reciprocal
            -0.5,  # Reciprocal of square root
            0.25,  # Fourth root
            4.0,  # Fourth power
            10.0,  # Tenth power
            -2.0,  # Reciprocal of square
            1 / 3,  # Cube root
        ],
    )
    def test_pow_tensor_scalar_exponents(self, exponent):
        """Test pow.Tensor_Scalar with various exponent values."""

        def test_func(x):
            return torch.ops.aten.pow.Tensor_Scalar(x, exponent)

        # Use positive values for all inputs to avoid numerical issues
        arg_0 = torch.abs(torch.randn(size=(8, 16), dtype=torch.float32)) + 0.1

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_pow_tensor_scalar_special_cases(self):
        """Test pow.Tensor_Scalar with special input values."""

        def test_basic_pow(x, exp):
            return torch.ops.aten.pow.Tensor_Scalar(x, exp)

        # Test 0^0 = 1 (by convention)
        zero_tensor = torch.zeros(size=(1,), dtype=torch.float32)
        self.run_test_on_host(lambda x: test_basic_pow(x, 0.0), (zero_tensor,))
        self.run_test_on_device(lambda x: test_basic_pow(x, 0.0), (zero_tensor,))

        # Test case with mixed positive/negative values and integer power
        mixed_tensor = torch.tensor(
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32
        )
        self.run_test_on_host(lambda x: test_basic_pow(x, 2), (mixed_tensor,))
        self.run_test_on_device(lambda x: test_basic_pow(x, 2), (mixed_tensor,))

        # Test case with mixed values and odd integer power (preserves signs)
        self.run_test_on_host(lambda x: test_basic_pow(x, 3), (mixed_tensor,))
        self.run_test_on_device(lambda x: test_basic_pow(x, 3), (mixed_tensor,))

        # Test with large exponent (may cause overflow for large inputs)
        small_values = torch.tensor([0.5, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float32)
        self.run_test_on_host(lambda x: test_basic_pow(x, 30.0), (small_values,))
        self.run_test_on_device(lambda x: test_basic_pow(x, 30.0), (small_values,))


class TestAtenPowScalar(NKIPyTestBase):
    @pytest.mark.parametrize(
        "shape,base,dtype",
        [
            ((16, 32), 2.0, torch.float32),  # Base 2 - float
            ((16, 32), 2, torch.float32),  # Base 2 - int
            ((8, 16, 32), 3.0, torch.float32),  # Base 3
            ((4, 8, 16, 32), 10.0, torch.float32),  # Base 10
            ((128, 256), 0.5, torch.float16),  # Base 0.5 (between 0 and 1)
            # FIXME accuracy issue
            # ((64, 128), 1.0, torch.bfloat16),  # Base 1 (always results in 1)
            ((16, 32), 0.1, torch.float32),  # Small base
            ((1, 1, 1), 5.0, torch.float32),  # Singleton dimensions
        ],
    )
    def test_pow_scalar_shapes_bases(self, shape, base, dtype):
        """Test pow.Scalar with different shapes, base values, and dtypes."""

        def test_func(exponent):
            return torch.ops.aten.pow.Scalar(base, exponent)

        # Generate random exponents
        # For bases < 1, use smaller exponents to avoid underflow
        # For bases > 1, use moderate exponents to avoid overflow
        if base < 1.0 and base > 0:
            arg_0 = torch.randn(size=shape, dtype=dtype) * 2.0  # Range roughly [-2, 2]
        else:
            arg_0 = torch.randn(size=shape, dtype=dtype) * 3.0  # Range roughly [-3, 3]

        # FIXME BFloat16 numpy precision issue when running on host
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    @pytest.mark.parametrize(
        "base",
        [
            0.5,  # Fractional base < 1
            1.0,  # Base 1 - always results in 1
            2.0,  # Base 2
            3.0,  # Base 3
            10.0,  # Base 10
            0.1,  # Small base
            1.5,  # Base between 1 and 2
            4.0,  # Base 4
            0.25,  # Base 1/4
            100.0,  # Large base
        ],
    )
    def test_pow_scalar_bases(self, base):
        """Test pow.Scalar with various base values."""

        def test_func(exponent):
            return torch.ops.aten.pow.Scalar(base, exponent)

        # Generate exponents with appropriate range based on base
        if base < 1.0:
            # For bases < 1, larger exponents drive values toward 0
            arg_0 = torch.randn(size=(8, 16), dtype=torch.float32) * 5.0
        elif base == 1.0:
            # For base 1, any exponent works
            arg_0 = torch.randn(size=(8, 16), dtype=torch.float32) * 10.0
        else:
            # For bases > 1, use smaller exponents to avoid overflow
            arg_0 = torch.randn(size=(8, 16), dtype=torch.float32) * 3.0

        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_pow_scalar_special_cases(self):
        """Test pow.Scalar with special input values."""

        # Test base^0 = 1 for any base (except 0^0)
        def test_zero_exponent(exponent):
            return torch.ops.aten.pow.Scalar(2.0, exponent)

        zero_exp = torch.zeros(size=(4, 8), dtype=torch.float32)
        self.run_test_on_host(test_zero_exponent, (zero_exp,))
        self.run_test_on_device(test_zero_exponent, (zero_exp,))

        # Test base^1 = base
        def test_one_exponent(exponent):
            return torch.ops.aten.pow.Scalar(2.0, exponent)

        one_exp = torch.ones(size=(4, 8), dtype=torch.float32)
        self.run_test_on_host(test_one_exponent, (one_exp,))
        self.run_test_on_device(test_one_exponent, (one_exp,))

        # Test 1^exponent = 1 for any exponent
        def test_base_one(exponent):
            return torch.ops.aten.pow.Scalar(1.0, exponent)

        any_exp = torch.randn(size=(4, 8), dtype=torch.float32) * 10.0
        self.run_test_on_host(test_base_one, (any_exp,))
        self.run_test_on_device(test_base_one, (any_exp,))

        # Test 2^n for specific integer exponents
        def test_powers_of_two(exponent):
            return torch.ops.aten.pow.Scalar(2.0, exponent)

        int_exponents = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0], dtype=torch.float32
        )
        self.run_test_on_host(test_powers_of_two, (int_exponents,))
        self.run_test_on_device(test_powers_of_two, (int_exponents,))

        # Test negative exponents (reciprocal)
        def test_negative_exponents(exponent):
            return torch.ops.aten.pow.Scalar(2.0, exponent)

        neg_exponents = torch.tensor([-1.0, -2.0, -3.0, -0.5], dtype=torch.float32)
        self.run_test_on_host(test_negative_exponents, (neg_exponents,))
        self.run_test_on_device(test_negative_exponents, (neg_exponents,))

        # Test fractional exponents (roots)
        def test_fractional_exponents(exponent):
            return torch.ops.aten.pow.Scalar(4.0, exponent)

        frac_exponents = torch.tensor([0.5, 0.25, 0.75, 1.5, 2.5], dtype=torch.float32)
        self.run_test_on_host(test_fractional_exponents, (frac_exponents,))
        self.run_test_on_device(test_fractional_exponents, (frac_exponents,))

        # Test exponential growth (base > 1)
        def test_exponential_growth(exponent):
            return torch.ops.aten.pow.Scalar(1.5, exponent)

        growth_exponents = torch.linspace(0, 10, 20, dtype=torch.float32)
        self.run_test_on_host(test_exponential_growth, (growth_exponents,))
        self.run_test_on_device(test_exponential_growth, (growth_exponents,))

        # Test exponential decay (base < 1)
        def test_exponential_decay(exponent):
            return torch.ops.aten.pow.Scalar(0.5, exponent)

        decay_exponents = torch.linspace(0, 10, 20, dtype=torch.float32)
        self.run_test_on_host(test_exponential_decay, (decay_exponents,))
        self.run_test_on_device(test_exponential_decay, (decay_exponents,))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
        ],
    )
    def test_pow_scalar_edge_cases(self, dtype):
        """Test pow.Scalar with edge cases."""

        # Test with very small exponents
        def test_small_exp(exponent):
            return torch.ops.aten.pow.Scalar(10.0, exponent)

        small_exp = torch.rand(size=(4, 8), dtype=dtype) * 0.01
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_small_exp, (small_exp,))
        self.run_test_on_device(test_small_exp, (small_exp,))

        # Test with large exponents (use small base to avoid overflow)
        def test_large_exp(exponent):
            return torch.ops.aten.pow.Scalar(1.1, exponent)

        large_exp = torch.rand(size=(4, 8), dtype=dtype) * 50.0
        if dtype != torch.bfloat16:
            self.run_test_on_host(test_large_exp, (large_exp,))
        self.run_test_on_device(test_large_exp, (large_exp,))
