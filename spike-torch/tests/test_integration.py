# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for nkipy-torch with PyTorch."""

import pytest
import torch

import spike_torch


class TestTensorMethods:
    """Tests for PyTorch tensor methods with nkipy device."""

    def test_tensor_nkipy_method(self):
        """tensor.nkipy() method works."""
        x = torch.randn(10, 10)
        y = x.nkipy()
        assert y.device.type == "nkipy"
        torch.testing.assert_close(y.cpu(), x)

    def test_tensor_nkipy_with_device(self):
        """tensor.nkipy(device) method works."""
        x = torch.randn(10, 10)
        y = x.nkipy(0)
        assert y.device.type == "nkipy"
        assert y.device.index == 0


class TestModuleMethods:
    """Tests for PyTorch module methods with nkipy device."""

    def test_module_nkipy_method(self):
        """module.nkipy() method works."""
        model = torch.nn.Linear(10, 10)
        model = model.nkipy()

        # Check parameters are on nkipy device
        for param in model.parameters():
            assert param.device.type == "nkipy"

    def test_module_to_nkipy(self):
        """module.to('nkipy') works."""
        model = torch.nn.Linear(10, 10)
        model = model.to("nkipy")

        for param in model.parameters():
            assert param.device.type == "nkipy"


class TestDeviceGuard:
    """Tests for device guard functionality."""

    def test_device_guard(self):
        """Device guard context manager works."""
        original_device = spike_torch.current_device()
        try:
            with torch.device("nkipy:0"):
                # Operations here should use device 0
                x = torch.randn(10, device="nkipy")
                assert x.device.index == 0
        finally:
            spike_torch.set_device(original_device)


class TestGradient:
    """Tests for gradient computation."""

    def test_requires_grad(self):
        """Tensors can require grad."""
        x = torch.randn(10, 10, device="nkipy", requires_grad=True)
        assert x.requires_grad

    def test_detach(self):
        """Detach works on nkipy tensors."""
        x = torch.randn(10, 10, device="nkipy", requires_grad=True)
        y = x.detach()
        assert not y.requires_grad


class TestContiguity:
    """Tests for contiguity checks."""

    def test_is_contiguous(self):
        """Contiguity check works."""
        x = torch.randn(10, 10, device="nkipy")
        assert x.is_contiguous()

    def test_non_contiguous_transpose(self):
        """Transpose creates non-contiguous tensor."""
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        assert not y.is_contiguous()

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_contiguous_copy(self):
        """contiguous() creates contiguous copy."""
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        z = y.contiguous()
        assert z.is_contiguous()


class TestTensorProperties:
    """Tests for tensor properties."""

    def test_shape(self):
        """Shape property works."""
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.shape == (2, 3, 4)
        assert x.size() == torch.Size([2, 3, 4])

    def test_dim(self):
        """dim() method works."""
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.dim() == 3

    def test_numel(self):
        """numel() method works."""
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.numel() == 24

    def test_stride(self):
        """stride() method works."""
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.stride() == (12, 4, 1)

    def test_storage_offset(self):
        """storage_offset() method works."""
        x = torch.randn(10, device="nkipy")
        assert x.storage_offset() == 0

        y = x[2:]
        assert y.storage_offset() == 2
