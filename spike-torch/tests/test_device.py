# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nkipy device registration and properties."""

import pytest
import torch

import spike_torch


class TestDeviceAvailability:
    """Tests for device availability."""

    def test_device_available(self):
        """Device is registered and available."""
        assert spike_torch.device_count() > 0
        assert spike_torch.is_available()

    def test_device_type_registered(self):
        """Spike device type is properly registered."""
        # Should be able to create a device object
        device = torch.device("nkipy")
        assert device.type == "nkipy"

    def test_device_with_index(self):
        """Can specify device index."""
        device = torch.device("nkipy:0")
        assert device.type == "nkipy"
        assert device.index == 0


class TestDeviceProperties:
    """Tests for device properties."""

    def test_device_count(self):
        """Device count is reasonable."""
        count = spike_torch.device_count()
        assert count > 0
        assert count <= 128  # Reasonable upper bound for large machines

    def test_current_device(self):
        """Current device is within valid range."""
        current = spike_torch.current_device()
        count = spike_torch.device_count()
        assert 0 <= current < count

    def test_set_device(self):
        """Can set current device."""
        original = spike_torch.current_device()
        try:
            spike_torch.set_device(0)
            assert spike_torch.current_device() == 0
        finally:
            spike_torch.set_device(original)

    def test_set_invalid_device_raises(self):
        """Setting invalid device raises error."""
        count = spike_torch.device_count()
        with pytest.raises(Exception):  # Could be ValueError or RuntimeError
            spike_torch.set_device(count + 100)

    def test_set_negative_device_raises(self):
        """Setting negative device raises error."""
        with pytest.raises(Exception):
            spike_torch.set_device(-1)


class TestTorchNkipyModule:
    """Tests for torch.nkipy module integration."""

    def test_torch_spike_exists(self):
        """torch.nkipy module is accessible."""
        assert hasattr(torch, "nkipy")

    def test_torch_spike_device_count(self):
        """torch.nkipy.device_count() works."""
        count = torch.nkipy.device_count()
        assert count == spike_torch.device_count()

    def test_torch_spike_is_available(self):
        """torch.nkipy.is_available() works."""
        assert torch.nkipy.is_available() == spike_torch.is_available()

    def test_torch_spike_current_device(self):
        """torch.nkipy.current_device() works."""
        assert torch.nkipy.current_device() == spike_torch.current_device()

    def test_torch_spike_set_device(self):
        """torch.nkipy.set_device() works."""
        original = spike_torch.current_device()
        try:
            torch.nkipy.set_device(0)
            assert spike_torch.current_device() == 0
        finally:
            spike_torch.set_device(original)
