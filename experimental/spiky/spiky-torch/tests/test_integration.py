# SPDX-License-Identifier: Apache-2.0

"""Integration tests for nkipy device integration with PyTorch."""

import pytest
import torch

import spiky.torch as spiky_torch


class TestTensorMethods:
    def test_tensor_nkipy_method(self):
        x = torch.randn(10, 10)
        y = x.nkipy()
        assert y.device.type == "nkipy"
        torch.testing.assert_close(y.cpu(), x)

    def test_tensor_nkipy_with_device(self):
        x = torch.randn(10, 10)
        y = x.nkipy(0)
        assert y.device.type == "nkipy"
        assert y.device.index == 0


class TestModuleMethods:
    def test_module_nkipy_method(self):
        model = torch.nn.Linear(10, 10)
        model = model.nkipy()
        for param in model.parameters():
            assert param.device.type == "nkipy"

    def test_module_to_nkipy(self):
        model = torch.nn.Linear(10, 10)
        model = model.to("nkipy")
        for param in model.parameters():
            assert param.device.type == "nkipy"


class TestDeviceGuard:
    def test_device_guard(self):
        original_device = spiky_torch.current_device()
        try:
            with torch.device("nkipy:0"):
                x = torch.randn(10, device="nkipy")
                assert x.device.index == 0
        finally:
            spiky_torch.set_device(original_device)


class TestGradient:
    def test_requires_grad(self):
        x = torch.randn(10, 10, device="nkipy", requires_grad=True)
        assert x.requires_grad

    def test_detach(self):
        x = torch.randn(10, 10, device="nkipy", requires_grad=True)
        y = x.detach()
        assert not y.requires_grad


class TestContiguity:
    def test_is_contiguous(self):
        x = torch.randn(10, 10, device="nkipy")
        assert x.is_contiguous()

    def test_non_contiguous_transpose(self):
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        assert not y.is_contiguous()

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_contiguous_copy(self):
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        z = y.contiguous()
        assert z.is_contiguous()


class TestTensorProperties:
    def test_shape(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.shape == (2, 3, 4)
        assert x.size() == torch.Size([2, 3, 4])

    def test_dim(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.dim() == 3

    def test_numel(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.numel() == 24

    def test_stride(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        assert x.stride() == (12, 4, 1)

    def test_storage_offset(self):
        x = torch.randn(10, device="nkipy")
        assert x.storage_offset() == 0
        y = x[2:]
        assert y.storage_offset() == 2

