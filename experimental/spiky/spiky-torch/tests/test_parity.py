# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Parity tests (kept as-is; currently skipped)."""

import pytest
import spiky.torch as spiky_torch  # noqa: F401
import torch

TORCH_TO_NKIPY_AVAILABLE = False


@pytest.mark.skipif(not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available")
class TestTensorCreationParity:
    def test_empty_parity(self):
        x_nkipy = torch.empty(10, 10, device="nkipy")
        assert x_nkipy.shape == (10, 10)
        assert x_nkipy.device.type == "nkipy"

    def test_dtype_parity(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_nkipy = torch.empty(10, device="nkipy", dtype=dtype)
            assert x_nkipy.dtype == dtype


@pytest.mark.skipif(not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available")
class TestCopyParity:
    def test_roundtrip_parity(self):
        original = torch.randn(10, 10)
        x_nkipy = original.to("nkipy").cpu()
        torch.testing.assert_close(x_nkipy, original)


@pytest.mark.skipif(not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available")
class TestViewOpsParity:
    def test_view_parity(self):
        original = torch.arange(24).float().reshape(2, 3, 4)
        x_nkipy = original.to("nkipy")
        y_nkipy = x_nkipy.view(6, 4)
        torch.testing.assert_close(y_nkipy.cpu(), original.view(6, 4))

    def test_transpose_parity(self):
        original = torch.randn(3, 4)
        x_nkipy = original.to("nkipy")
        y_nkipy = x_nkipy.t()
        torch.testing.assert_close(y_nkipy.cpu(), original.t())
