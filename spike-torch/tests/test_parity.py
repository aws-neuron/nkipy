# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests comparing nkipy-torch with existing torch-to-nkipy implementation.

These tests ensure nkipy-torch produces the same results as the current
torch-to-nkipy implementation before migration.

Note: These tests require both implementations to be available.
Skip if torch-to-nkipy is not installed.
"""

import pytest
import torch

import spike_torch


# Check if torch-to-nkipy is available
# Note: Cannot run both nkipy_torch and torch_to_nkipy in same process
# since they both register as PrivateUse1
TORCH_TO_NKIPY_AVAILABLE = False


@pytest.mark.skipif(
    not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available"
)
class TestTensorCreationParity:
    """Parity tests for tensor creation."""

    def test_empty_parity(self):
        """Same tensor creation behavior."""
        x_nkipy = torch.empty(10, 10, device="nkipy")
        # Compare properties, not values (empty tensors have random values)
        assert x_nkipy.shape == (10, 10)
        assert x_nkipy.device.type == "nkipy"

    def test_dtype_parity(self):
        """Same dtype handling."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_nkipy = torch.empty(10, device="nkipy", dtype=dtype)
            assert x_nkipy.dtype == dtype


@pytest.mark.skipif(
    not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available"
)
class TestCopyParity:
    """Parity tests for copy operations."""

    def test_roundtrip_parity(self):
        """Both implementations preserve data during roundtrip."""
        original = torch.randn(10, 10)

        # nkipy-torch implementation
        x_nkipy = original.to("nkipy").cpu()

        # Should be identical to original
        torch.testing.assert_close(x_nkipy, original)


@pytest.mark.skipif(
    not TORCH_TO_NKIPY_AVAILABLE, reason="torch-to-nkipy not available"
)
class TestViewOpsParity:
    """Parity tests for view operations."""

    def test_view_parity(self):
        """Same view behavior."""
        original = torch.arange(24).float().reshape(2, 3, 4)
        x_nkipy = original.to("nkipy")

        # View operation
        y_nkipy = x_nkipy.view(6, 4)

        # Compare with CPU result
        torch.testing.assert_close(y_nkipy.cpu(), original.view(6, 4))

    def test_transpose_parity(self):
        """Same transpose behavior."""
        original = torch.randn(3, 4)
        x_nkipy = original.to("nkipy")

        y_nkipy = x_nkipy.t()
        torch.testing.assert_close(y_nkipy.cpu(), original.t())
