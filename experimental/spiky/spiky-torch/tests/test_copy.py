# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Tests for copy operations between CPU and nkipy device."""

import spiky.torch as spiky_torch  # noqa: F401
import torch


class TestCPUToNkipy:
    def test_cpu_to_nkipy(self):
        cpu_tensor = torch.randn(10, 10)
        nkipy_tensor = cpu_tensor.to("nkipy")
        assert nkipy_tensor.device.type == "nkipy"

    def test_cpu_to_nkipy_preserves_shape(self):
        cpu_tensor = torch.randn(2, 3, 4)
        nkipy_tensor = cpu_tensor.to("nkipy")
        assert nkipy_tensor.shape == cpu_tensor.shape

    def test_cpu_to_nkipy_various_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.int32]:
            cpu_tensor = torch.ones(10, dtype=dtype)
            nkipy_tensor = cpu_tensor.to("nkipy")
            assert nkipy_tensor.dtype == dtype


class TestNkipyToCPU:
    def test_nkipy_to_cpu(self):
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"

    def test_nkipy_to_cpu_method(self):
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"

    def test_nkipy_to_cpu_to_method(self):
        nkipy_tensor = torch.randn(10, 10, device="nkipy")
        cpu_tensor = nkipy_tensor.to("cpu")
        assert cpu_tensor.device.type == "cpu"


class TestRoundtrip:
    def test_roundtrip_preserves_data(self):
        original = torch.randn(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_zeros(self):
        original = torch.zeros(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_ones(self):
        original = torch.ones(10, 10)
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)

    def test_roundtrip_preserves_arange(self):
        original = torch.arange(100).float()
        nkipy = original.to("nkipy")
        back = nkipy.cpu()
        torch.testing.assert_close(original, back)


class TestNkipyToNkipyCopy:
    def test_nkipy_to_nkipy_clone(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x.clone()
        torch.testing.assert_close(x.cpu(), y.cpu())

    def test_nkipy_to_nkipy_copy_(self):
        x = torch.randn(10, 10, device="nkipy")
        y = torch.empty(10, 10, device="nkipy")
        y.copy_(x)
        torch.testing.assert_close(x.cpu(), y.cpu())


class TestCopyInPlace:
    def test_copy_inplace_cpu_to_nkipy(self):
        cpu_tensor = torch.randn(10, 10)
        nkipy_tensor = torch.empty(10, 10, device="nkipy")
        nkipy_tensor.copy_(cpu_tensor)
        torch.testing.assert_close(nkipy_tensor.cpu(), cpu_tensor)

    def test_copy_inplace_nkipy_to_cpu(self):
        original = torch.randn(10, 10)
        nkipy_tensor = original.to("nkipy")
        cpu_tensor = torch.empty(10, 10)
        cpu_tensor.copy_(nkipy_tensor)
        torch.testing.assert_close(cpu_tensor, original)


class TestDtypeConversion:
    def test_float32_to_float16(self):
        cpu_tensor = torch.randn(10, 10, dtype=torch.float32)
        nkipy_tensor = cpu_tensor.to("nkipy", dtype=torch.float16)
        assert nkipy_tensor.dtype == torch.float16

    def test_float16_to_float32(self):
        nkipy_tensor = torch.randn(10, 10, device="nkipy", dtype=torch.float16)
        cpu_tensor = nkipy_tensor.to("cpu", dtype=torch.float32)
        assert cpu_tensor.dtype == torch.float32
