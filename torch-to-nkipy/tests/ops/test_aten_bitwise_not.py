# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from base import NKIPyTestBase


class TestAtenBitwiseNot(NKIPyTestBase):
    """Test suite for torch.ops.aten.bitwise_not.default operation."""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((16, 32), torch.int8),  # Basic INT8 case
            # FIXME compiler error
            # ((16, 32), torch.int16),  # INT16 case
            ((16, 32), torch.int32),  # INT32 case
            ((16, 32), torch.uint8),  # UINT8 case
            ((16, 32), torch.bool),  # Boolean tensor
            ((8, 16, 32), torch.int32),  # 3D tensor
            ((4, 8, 16, 32), torch.int32),  # 4D tensor
            ((16,), torch.int32),  # 1D tensor
            ((1, 1, 1), torch.int32),  # Singleton dimensions
        ],
    )
    def test_bitwise_not_basic(self, shape, dtype):
        """Test bitwise_not.default with different shapes and dtypes."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        if dtype == torch.bool:
            arg_0 = torch.randint(0, 2, size=shape, dtype=dtype)
        else:
            # For integer types, generate values within the appropriate range
            low = 0 if dtype == torch.uint8 else -100
            high = 100
            arg_0 = torch.randint(low, high, size=shape, dtype=dtype)

        # Test bitwise NOT
        self.run_test_on_host(test_func, (arg_0,))
        self.run_test_on_device(test_func, (arg_0,))

    def test_bitwise_not_truth_table(self):
        """Test bitwise_not.default with 0/1 values."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        # Test with 0 and 1 for uint8 (should get 255 and 254)
        x_uint8 = torch.tensor([0, 1], dtype=torch.uint8)
        self.run_test_on_host(test_func, (x_uint8,))
        self.run_test_on_device(test_func, (x_uint8,))

        # Test with False and True for boolean (should get True and False)
        x_bool = torch.tensor([False, True])
        self.run_test_on_host(test_func, (x_bool,))
        self.run_test_on_device(test_func, (x_bool,))

        # Test with 0 and 1 for int8 (should get -1 and -2)
        x_int8 = torch.tensor([0, 1], dtype=torch.int8)
        self.run_test_on_host(test_func, (x_int8,))
        self.run_test_on_device(test_func, (x_int8,))

    def test_bitwise_not_bit_patterns(self):
        """Test bitwise_not.default with various bit patterns."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        # Common bit patterns for testing bitwise NOT
        patterns = [
            0x00,  # All zeros (00000000)
            0xFF,  # All ones (11111111)
            0x55,  # Alternating bits (01010101)
            0xAA,  # Alternating bits (10101010)
            0x0F,  # Half zeros, half ones (00001111)
            0xF0,  # Half ones, half zeros (11110000)
            0x33,  # Pairs of bits (00110011)
            0xCC,  # Pairs of bits (11001100)
            0x01,  # Single bit (00000001)
            0x80,  # Single bit (10000000)
            0x03,  # Two bits (00000011)
            0xC0,  # Two bits (11000000)
            0x07,  # Three bits (00000111)
            0xE0,  # Three bits (11100000)
            0x3F,  # Mixed pattern (00111111)
        ]

        # Test each pattern with uint8
        for pattern in patterns:
            x = torch.tensor(pattern, dtype=torch.uint8).expand(4)
            self.run_test_on_host(test_func, (x,))
            self.run_test_on_device(test_func, (x,))

    def test_bitwise_not_negative_values(self):
        """Test bitwise_not.default with negative values."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        # Test with range of positive and negative values
        x = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=torch.int8)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with larger range in int32
        x = torch.tensor(
            [-10000, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000],
            dtype=torch.int32,
        )
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_bitwise_not_powers_of_two(self):
        """Test bitwise_not.default with powers of two."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        # Powers of 2 are interesting for bitwise operations
        powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
        x = torch.tensor(powers_of_2, dtype=torch.uint8)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

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
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Negative powers of 2 for signed types
        neg_powers = [-1, -2, -4, -8, -16, -32, -64, -128]
        x = torch.tensor(neg_powers, dtype=torch.int8)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_bitwise_not_boolean_patterns(self):
        """Test bitwise_not.default with boolean patterns."""

        def test_func(x):
            return torch.ops.aten.bitwise_not.default(x)

        # Test with checkerboard pattern
        x = torch.zeros((4, 4), dtype=torch.bool)
        x[::2, ::2] = True
        x[1::2, 1::2] = True
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with striped pattern
        x = torch.zeros((4, 4), dtype=torch.bool)
        x[::2, :] = True
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with all True
        x = torch.ones((4, 4), dtype=torch.bool)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with all False
        x = torch.zeros((4, 4), dtype=torch.bool)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

    def test_bitwise_not_chained_operations(self):
        """Test bitwise_not.default with chained operations."""

        def test_func(x):
            # Double NOT should return the original value
            return torch.ops.aten.bitwise_not.default(
                torch.ops.aten.bitwise_not.default(x)
            )

        # Test with various values
        x = torch.tensor([0, 1, 2, 3, 127, 128, 255], dtype=torch.uint8)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with signed values
        x = torch.tensor([-5, -1, 0, 1, 5], dtype=torch.int8)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))

        # Test with boolean values
        x = torch.tensor([False, True, False, True])
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))
