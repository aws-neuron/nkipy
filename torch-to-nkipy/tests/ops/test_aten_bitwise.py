# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenBitwiseOps(NKIPyTestBase):
    """Test suite for PyTorch tensor bitwise operations (AND, OR, XOR)."""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.int8),  # Basic INT8 case
            # FIXME compiler error
            # ((16, 32), torch.int16),  # INT16 case
            ((16, 32), torch.int32),  # INT32 case
            ((16, 32), torch.int64),  # INT64 case
            ((16, 32), torch.uint8),  # UINT8 case
            # ((16, 32), torch.bool),    # FIXME Boolean tensor depenend on np.logical
            ((8, 16, 32), torch.int32),  # 3D tensor
            ((4, 8, 16, 32), torch.int32),  # 4D tensor
            ((16,), torch.int32),  # 1D tensor
            ((1, 1, 1), torch.int32),  # Singleton dimensions
        ],
    )
    def test_bitwise_ops_basic(self, shape, dtype):
        """Test bitwise operations with different shapes and dtypes."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=dtype)
            arg_1 = torch.randint(0, 2, size=shape, dtype=dtype)
        else:
            # For integer types, generate values within the appropriate range
            low = 0 if dtype == torch.uint8 else -100
            high = 100
            arg_0 = torch.randint(low, high, size=shape, dtype=dtype)
            arg_1 = torch.randint(low, high, size=shape, dtype=dtype)

        # Test bitwise AND
        self.run_test_on_host(test_bitwise_and, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_and, (arg_0, arg_1))

        # Test bitwise OR
        self.run_test_on_host(test_bitwise_or, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_or, (arg_0, arg_1))

        # Test bitwise XOR
        self.run_test_on_host(test_bitwise_xor, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_xor, (arg_0, arg_1))

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((16, 32), (32,)),  # Broadcasting second tensor
            ((16, 1), (1, 32)),  # Broadcasting both tensors
            ((1, 16, 32), (16, 32)),  # Different dimensions
            ((16, 32), (1, 16, 32)),  # Different dimensions reversed
            ((16, 1, 32), (16, 8, 1)),  # Complex broadcasting
            ((1, 1), (5, 5)),  # Broadcasting singleton to matrix
            ((5, 5), (1, 1)),  # Broadcasting singleton to matrix reversed
        ],
    )
    def test_bitwise_ops_broadcasting(self, shape_a, shape_b):
        """Test bitwise operations with different broadcasting scenarios."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        # Create tensors with random integer values for broadcasting tests
        arg_0 = torch.randint(0, 256, size=shape_a, dtype=torch.int32)
        arg_1 = torch.randint(0, 256, size=shape_b, dtype=torch.int32)

        # Test bitwise AND with broadcasting
        self.run_test_on_host(test_bitwise_and, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_and, (arg_0, arg_1))

        # Test bitwise OR with broadcasting
        self.run_test_on_host(test_bitwise_or, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_or, (arg_0, arg_1))

        # Test bitwise XOR with broadcasting
        self.run_test_on_host(test_bitwise_xor, (arg_0, arg_1))
        self.run_test_on_device(test_bitwise_xor, (arg_0, arg_1))

    def test_bitwise_ops_truth_table(self):
        """Test bitwise operations with all possible bit combinations."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        # Create all 4 combinations of 0/1 bits for testing bitwise operations
        x = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.uint8)

        # Test bitwise AND truth table
        self.run_test_on_host(test_bitwise_and, (x, y))
        self.run_test_on_device(test_bitwise_and, (x, y))

        # Test bitwise OR truth table
        self.run_test_on_host(test_bitwise_or, (x, y))
        self.run_test_on_device(test_bitwise_or, (x, y))

        # Test bitwise XOR truth table
        self.run_test_on_host(test_bitwise_xor, (x, y))
        self.run_test_on_device(test_bitwise_xor, (x, y))

        # FIXME Boolean tensor depenend on np.logical
        # Test with boolean type as well
        # x_bool = torch.tensor([False, False, True, True])
        # y_bool = torch.tensor([False, True, False, True])

        # self.run_test_on_host(test_bitwise_and, (x_bool, y_bool))
        # self.run_test_on_device(test_bitwise_and, (x_bool, y_bool))
        # self.run_test_on_host(test_bitwise_or, (x_bool, y_bool))
        # self.run_test_on_device(test_bitwise_or, (x_bool, y_bool))
        # self.run_test_on_host(test_bitwise_xor, (x_bool, y_bool))
        # self.run_test_on_device(test_bitwise_xor, (x_bool, y_bool))

    def test_bitwise_ops_bit_patterns(self):
        """Test bitwise operations with various bit patterns."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        # Common bit patterns for testing bitwise operations
        patterns = [
            [0x00, 0xFF],  # All zeros, all ones
            [0x55, 0xAA],  # Alternating bits (01010101, 10101010)
            [0x0F, 0xF0],  # Half and half (00001111, 11110000)
            [0x33, 0xCC],  # Pairs of bits (00110011, 11001100)
            [0x01, 0x80],  # Single bits (00000001, 10000000)
            [0x03, 0xC0],  # Two bits (00000011, 11000000)
            [0x07, 0xE0],  # Three bits (00000111, 11100000)
            [0x3F, 0xC0],  # Mixed patterns (00111111, 11000000)
        ]

        for pattern in patterns:
            x = torch.tensor(pattern[0], dtype=torch.uint8).expand(4)
            y = torch.tensor(pattern[1], dtype=torch.uint8).expand(4)

            # Test bitwise operations with this pattern
            self.run_test_on_host(test_bitwise_and, (x, y))
            self.run_test_on_device(test_bitwise_and, (x, y))
            self.run_test_on_host(test_bitwise_or, (x, y))
            self.run_test_on_device(test_bitwise_or, (x, y))
            self.run_test_on_host(test_bitwise_xor, (x, y))
            self.run_test_on_device(test_bitwise_xor, (x, y))

    def test_bitwise_ops_negative_values(self):
        """Test bitwise operations with negative values."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        # Test with positive and negative values
        x = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=torch.int8)
        y = torch.tensor([5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5], dtype=torch.int8)

        # Test all bitwise operations
        self.run_test_on_host(test_bitwise_and, (x, y))
        self.run_test_on_device(test_bitwise_and, (x, y))
        self.run_test_on_host(test_bitwise_or, (x, y))
        self.run_test_on_device(test_bitwise_or, (x, y))
        self.run_test_on_host(test_bitwise_xor, (x, y))
        self.run_test_on_device(test_bitwise_xor, (x, y))

        # Test with int32 for larger range
        x = torch.tensor(
            [-10000, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000],
            dtype=torch.int32,
        )
        y = torch.tensor(
            [10000, 1000, 100, 10, 1, 0, -1, -10, -100, -1000, -10000],
            dtype=torch.int32,
        )

        self.run_test_on_host(test_bitwise_and, (x, y))
        self.run_test_on_device(test_bitwise_and, (x, y))
        self.run_test_on_host(test_bitwise_or, (x, y))
        self.run_test_on_device(test_bitwise_or, (x, y))
        self.run_test_on_host(test_bitwise_xor, (x, y))
        self.run_test_on_device(test_bitwise_xor, (x, y))

    def test_bitwise_ops_powers_of_two(self):
        """Test bitwise operations with powers of two."""

        def test_bitwise_and(x, y):
            return torch.ops.aten.bitwise_and.Tensor(x, y)

        def test_bitwise_or(x, y):
            return torch.ops.aten.bitwise_or.Tensor(x, y)

        def test_bitwise_xor(x, y):
            return torch.ops.aten.bitwise_xor.Tensor(x, y)

        # Powers of 2 are interesting for bitwise operations
        powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
        x = torch.tensor(powers_of_2, dtype=torch.uint8)
        y = torch.tensor(powers_of_2[::-1], dtype=torch.uint8)  # Reverse order

        self.run_test_on_host(test_bitwise_and, (x, y))
        self.run_test_on_device(test_bitwise_and, (x, y))
        self.run_test_on_host(test_bitwise_or, (x, y))
        self.run_test_on_device(test_bitwise_or, (x, y))
        self.run_test_on_host(test_bitwise_xor, (x, y))
        self.run_test_on_device(test_bitwise_xor, (x, y))

        # Larger powers of 2 for int32
        large_powers = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        x = torch.tensor(large_powers, dtype=torch.int32)
        y = torch.tensor(large_powers[::-1], dtype=torch.int32)  # Reverse order

        self.run_test_on_host(test_bitwise_and, (x, y))
        self.run_test_on_device(test_bitwise_and, (x, y))
        self.run_test_on_host(test_bitwise_or, (x, y))
        self.run_test_on_device(test_bitwise_or, (x, y))
        self.run_test_on_host(test_bitwise_xor, (x, y))
        self.run_test_on_device(test_bitwise_xor, (x, y))
