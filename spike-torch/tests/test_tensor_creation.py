# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor creation on nkipy device."""

import pytest
import torch

import spike_torch


class TestEmptyTensor:
    """Tests for torch.empty on nkipy device."""

    def test_empty_tensor(self):
        """Create empty tensor on nkipy device."""
        x = torch.empty(10, 10, device="nkipy")
        assert x.device.type == "nkipy"
        assert x.shape == (10, 10)

    def test_empty_1d(self):
        """Create 1D empty tensor."""
        x = torch.empty(100, device="nkipy")
        assert x.shape == (100,)
        assert x.device.type == "nkipy"

    def test_empty_3d(self):
        """Create 3D empty tensor."""
        x = torch.empty(2, 3, 4, device="nkipy")
        assert x.shape == (2, 3, 4)
        assert x.device.type == "nkipy"

    def test_empty_scalar(self):
        """Create scalar (0-dim) tensor."""
        x = torch.empty((), device="nkipy")
        assert x.shape == ()
        assert x.device.type == "nkipy"


class TestEmptyStrided:
    """Tests for torch.empty_strided on nkipy device."""

    def test_empty_strided_contiguous(self):
        """Create strided tensor with contiguous layout."""
        x = torch.empty_strided((10, 10), (10, 1), device="nkipy")
        assert x.stride() == (10, 1)
        assert x.device.type == "nkipy"

    def test_empty_strided_fortran(self):
        """Create strided tensor with Fortran layout."""
        x = torch.empty_strided((10, 10), (1, 10), device="nkipy")
        assert x.stride() == (1, 10)


class TestDtypes:
    """Tests for various dtypes on nkipy device."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
        ],
    )
    def test_various_dtypes(self, dtype):
        """Test tensor creation with various dtypes."""
        x = torch.empty(10, device="nkipy", dtype=dtype)
        assert x.dtype == dtype
        assert x.device.type == "nkipy"


class TestZeroSizeTensor:
    """Tests for zero-size tensors."""

    def test_zero_elements(self):
        """Create tensor with zero elements."""
        x = torch.empty(0, device="nkipy")
        assert x.numel() == 0
        assert x.device.type == "nkipy"

    def test_zero_in_shape(self):
        """Create tensor with zero in shape."""
        x = torch.empty(10, 0, 10, device="nkipy")
        assert x.shape == (10, 0, 10)
        assert x.numel() == 0


class TestDeviceIndex:
    """Tests for device index specification."""

    def test_device_index_0(self):
        """Create tensor on device 0."""
        x = torch.empty(10, device="nkipy:0")
        assert x.device.type == "nkipy"
        assert x.device.index == 0

    def test_device_object(self):
        """Create tensor using device object."""
        device = torch.device("nkipy", 0)
        x = torch.empty(10, device=device)
        assert x.device == device


class TestLargeTensors:
    """Tests for large tensor allocation."""

    def test_large_1d(self):
        """Create large 1D tensor."""
        x = torch.empty(1_000_000, device="nkipy")
        assert x.numel() == 1_000_000

    def test_large_2d(self):
        """Create large 2D tensor."""
        x = torch.empty(1000, 1000, device="nkipy")
        assert x.numel() == 1_000_000
