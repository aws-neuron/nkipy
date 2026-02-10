# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor creation on nkipy device."""

import pytest
import torch

import spiky.torch as spiky_torch  # noqa: F401


class TestEmptyTensor:
    def test_empty_tensor(self):
        x = torch.empty(10, 10, device="nkipy")
        assert x.device.type == "nkipy"
        assert x.shape == (10, 10)

    def test_empty_1d(self):
        x = torch.empty(100, device="nkipy")
        assert x.shape == (100,)
        assert x.device.type == "nkipy"

    def test_empty_3d(self):
        x = torch.empty(2, 3, 4, device="nkipy")
        assert x.shape == (2, 3, 4)
        assert x.device.type == "nkipy"

    def test_empty_scalar(self):
        x = torch.empty((), device="nkipy")
        assert x.shape == ()
        assert x.device.type == "nkipy"


class TestEmptyStrided:
    def test_empty_strided_contiguous(self):
        x = torch.empty_strided((10, 10), (10, 1), device="nkipy")
        assert x.stride() == (10, 1)
        assert x.device.type == "nkipy"

    def test_empty_strided_fortran(self):
        x = torch.empty_strided((10, 10), (1, 10), device="nkipy")
        assert x.stride() == (1, 10)


class TestDtypes:
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
        x = torch.empty(10, device="nkipy", dtype=dtype)
        assert x.dtype == dtype
        assert x.device.type == "nkipy"


class TestZeroSizeTensor:
    def test_zero_elements(self):
        x = torch.empty(0, device="nkipy")
        assert x.numel() == 0
        assert x.device.type == "nkipy"

    def test_zero_in_shape(self):
        x = torch.empty(10, 0, 10, device="nkipy")
        assert x.shape == (10, 0, 10)
        assert x.numel() == 0


class TestDeviceIndex:
    def test_device_index_0(self):
        x = torch.empty(10, device="nkipy:0")
        assert x.device.type == "nkipy"
        assert x.device.index == 0

    def test_device_object(self):
        device = torch.device("nkipy", 0)
        x = torch.empty(10, device=device)
        assert x.device == device


class TestLargeTensors:
    def test_large_1d(self):
        x = torch.empty(1_000_000, device="nkipy")
        assert x.numel() == 1_000_000

    def test_large_2d(self):
        x = torch.empty(1000, 1000, device="nkipy")
        assert x.numel() == 1_000_000

