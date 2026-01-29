# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for copy operations between CPU and nkipy device."""

import pytest
import torch

import spike_torch


class TestCPUToSpike:
    """Tests for CPU to nkipy copy operations."""

    def test_cpu_to_nkipy(self):
        """Copy tensor from CPU to nkipy."""
        cpu_tensor = torch.randn(10, 10)
        nkipy_tensor = cpu_tensor.to("nkipy")
        assert nkipy_tensor.device.type == "nkipy"

    def test_cpu_to_nkipy_preserves_shape(self):
        """Shape is preserved during copy."""
        cpu_tensor = torch.randn(2, 3, 4)
        nkipy_tensor = cpu_tensor.to("nkipy")
        assert nkipy_tensor.shape == cpu_tensor.shape

    def test_cpu_to_nkipy_various_dtypes(self):
        """Copy works with various dtypes."""
        for dtype in [torch.float32, torch.float16, torch.int32]:
            cpu_tensor = torch.ones(10, dtype=dtype)
            nkipy_tensor = cpu_tensor.to("nkipy")
            assert nkipy_tensor.dtype == dtype


class TestSpikeToCPU:
    """Tests for nkipy to CPU copy operations."""

    def test_nkipy_to_cpu(self):
        """Copy tensor from nkipy to CPU."""
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"

    def test_nkipy_to_cpu_method(self):
        """Use .cpu() method."""
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"

    def test_nkipy_to_cpu_to_method(self):
        """Use .to() method."""
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.to("cpu")
        assert cpu_tensor.device.type == "cpu"


class TestRoundtrip:
    """Tests for data preservation through roundtrip."""

    def test_roundtrip_preserves_data(self):
        """Data preserved through CPU -> nkipy -> CPU."""
        original = torch.randn(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_zeros(self):
        """Zero tensor roundtrip."""
        original = torch.zeros(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_ones(self):
        """Ones tensor roundtrip."""
        original = torch.ones(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_arange(self):
        """Arange tensor roundtrip."""
        original = torch.arange(100).float()
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)


class TestSpikeToDSpikeCopy:
    """Tests for nkipy to nkipy copy operations."""

    def test_nkipy_to_nkipy_clone(self):
        """Clone nkipy tensor."""
        x = torch.randn(10, 10, device="nkipy")
        y = x.clone()
        torch.testing.assert_close(x.cpu(), y.cpu())

    def test_nkipy_to_nkipy_copy_(self):
        """In-place copy between nkipy tensors."""
        x = torch.randn(10, 10, device="nkipy")
        y = torch.empty(10, 10, device="nkipy")
        y.copy_(x)
        torch.testing.assert_close(x.cpu(), y.cpu())


class TestCopyInPlace:
    """Tests for in-place copy operations."""

    def test_copy_inplace_cpu_to_nkipy(self):
        """In-place copy from CPU to nkipy."""
        cpu_tensor = torch.randn(10, 10)
        nkipy_tensor = torch.empty(10, 10, device="nkipy")
        nkipy_tensor.copy_(cpu_tensor)
        torch.testing.assert_close(nkipy_tensor.cpu(), cpu_tensor)

    def test_copy_inplace_nkipy_to_cpu(self):
        """In-place copy from nkipy to CPU."""
        original = torch.randn(10, 10)
        nkipy_tensor = original.to("nkipy")
        cpu_tensor = torch.empty(10, 10)
        cpu_tensor.copy_(nkipy_tensor)
        torch.testing.assert_close(cpu_tensor, original)


class TestDtypeConversion:
    """Tests for dtype conversion during copy."""

    def test_float32_to_float16(self):
        """Convert float32 to float16 during copy."""
        cpu_tensor = torch.randn(10, 10, dtype=torch.float32)
        nkipy_tensor = cpu_tensor.to("nkipy", dtype=torch.float16)
        assert nkipy_tensor.dtype == torch.float16

    def test_float16_to_float32(self):
        """Convert float16 to float32 during copy."""
        nkipy_tensor = torch.randn(10, 10, device="nkipy", dtype=torch.float16)
        cpu_tensor = nkipy_tensor.to("cpu", dtype=torch.float32)
        assert cpu_tensor.dtype == torch.float32
