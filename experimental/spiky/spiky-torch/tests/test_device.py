# SPDX-License-Identifier: Apache-2.0

"""Tests for nkipy device registration and properties."""

import pytest
import torch

import spiky.torch as spiky_torch


class TestDeviceAvailability:
    def test_device_available(self):
        assert spiky_torch.device_count() > 0
        assert spiky_torch.is_available()

    def test_device_type_registered(self):
        device = torch.device("nkipy")
        assert device.type == "nkipy"

    def test_device_with_index(self):
        device = torch.device("nkipy:0")
        assert device.type == "nkipy"
        assert device.index == 0


class TestDeviceProperties:
    def test_device_count(self):
        count = spiky_torch.device_count()
        assert count > 0
        assert count <= 128

    def test_current_device(self):
        current = spiky_torch.current_device()
        count = spiky_torch.device_count()
        assert 0 <= current < count

    def test_set_device(self):
        original = spiky_torch.current_device()
        try:
            spiky_torch.set_device(0)
            assert spiky_torch.current_device() == 0
        finally:
            spiky_torch.set_device(original)

    def test_set_invalid_device_raises(self):
        count = spiky_torch.device_count()
        with pytest.raises(Exception):
            spiky_torch.set_device(count + 100)

    def test_set_negative_device_raises(self):
        with pytest.raises(Exception):
            spiky_torch.set_device(-1)


class TestTorchNkipyModule:
    def test_torch_nkipy_exists(self):
        assert hasattr(torch, "nkipy")

    def test_torch_nkipy_device_count(self):
        count = torch.nkipy.device_count()
        assert count == spiky_torch.device_count()

    def test_torch_nkipy_is_available(self):
        assert torch.nkipy.is_available() == spiky_torch.is_available()

    def test_torch_nkipy_current_device(self):
        assert torch.nkipy.current_device() == spiky_torch.current_device()

    def test_torch_nkipy_set_device(self):
        original = spiky_torch.current_device()
        try:
            torch.nkipy.set_device(0)
            assert spiky_torch.current_device() == 0
        finally:
            spiky_torch.set_device(original)

