# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for view operations on nkipy tensors."""

import pytest
import torch

import spike_torch


class TestView:
    """Tests for view operation."""

    def test_view_flatten(self):
        """View operation to flatten tensor."""
        x = torch.randn(10, 10, device="nkipy")
        y = x.view(100)
        assert y.shape == (100,)
        assert y.data_ptr() == x.data_ptr()  # Same storage

    def test_view_reshape_2d(self):
        """View operation to reshape 2D tensor."""
        x = torch.randn(10, 10, device="nkipy")
        y = x.view(5, 20)
        assert y.shape == (5, 20)

    def test_view_infer_dim(self):
        """View with -1 to infer dimension."""
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.view(-1, 4)
        assert y.shape == (6, 4)

    def test_view_preserves_data(self):
        """View preserves data."""
        original = torch.arange(12).float()
        x = original.to("nkipy")
        y = x.view(3, 4)
        torch.testing.assert_close(y.cpu(), original.view(3, 4))


class TestReshape:
    """Tests for reshape operation."""

    def test_reshape_contiguous(self):
        """Reshape contiguous tensor."""
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.reshape(6, 4)
        assert y.shape == (6, 4)

    def test_reshape_infer_dim(self):
        """Reshape with -1 to infer dimension."""
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.reshape(-1)
        assert y.shape == (24,)


class TestAsStrided:
    """Tests for as_strided operation."""

    def test_as_strided_basic(self):
        """Basic as_strided operation."""
        x = torch.randn(10, 10, device="nkipy")
        y = x.as_strided((5, 5), (10, 1))
        assert y.shape == (5, 5)

    def test_as_strided_with_offset(self):
        """as_strided with storage offset."""
        x = torch.randn(10, 10, device="nkipy")
        y = x.as_strided((5, 5), (10, 1), 5)
        assert y.shape == (5, 5)
        assert y.storage_offset() == 5


class TestTranspose:
    """Tests for transpose operations."""

    def test_transpose_2d(self):
        """Transpose 2D tensor."""
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        assert y.shape == (4, 3)

    def test_transpose_nd(self):
        """Transpose specific dimensions."""
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.transpose(0, 2)
        assert y.shape == (4, 3, 2)


class TestPermute:
    """Tests for permute operation."""

    def test_permute_3d(self):
        """Permute 3D tensor."""
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.permute(2, 0, 1)
        assert y.shape == (4, 2, 3)

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_permute_preserves_data(self):
        """Permute preserves data."""
        original = torch.arange(24).float().reshape(2, 3, 4)
        x = original.to("nkipy")
        y = x.permute(2, 0, 1)
        torch.testing.assert_close(y.cpu(), original.permute(2, 0, 1))


class TestSlicing:
    """Tests for tensor slicing."""

    def test_slice_1d(self):
        """Slice 1D tensor."""
        x = torch.arange(10, device="nkipy").float()
        y = x[2:5]
        assert y.shape == (3,)

    def test_slice_2d(self):
        """Slice 2D tensor."""
        x = torch.randn(10, 10, device="nkipy")
        y = x[:5, :5]
        assert y.shape == (5, 5)

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_slice_preserves_data(self):
        """Slicing preserves data."""
        original = torch.arange(100).float().reshape(10, 10)
        x = original.to("nkipy")
        y = x[2:5, 3:7]
        torch.testing.assert_close(y.cpu(), original[2:5, 3:7])


class TestUnsqueeze:
    """Tests for unsqueeze operation."""

    def test_unsqueeze_dim0(self):
        """Unsqueeze at dimension 0."""
        x = torch.randn(3, 4, device="nkipy")
        y = x.unsqueeze(0)
        assert y.shape == (1, 3, 4)

    def test_unsqueeze_dim1(self):
        """Unsqueeze at dimension 1."""
        x = torch.randn(3, 4, device="nkipy")
        y = x.unsqueeze(1)
        assert y.shape == (3, 1, 4)


class TestSqueeze:
    """Tests for squeeze operation."""

    def test_squeeze_all(self):
        """Squeeze all dimensions."""
        x = torch.randn(1, 3, 1, 4, 1, device="nkipy")
        y = x.squeeze()
        assert y.shape == (3, 4)

    def test_squeeze_specific_dim(self):
        """Squeeze specific dimension."""
        x = torch.randn(1, 3, 1, 4, device="nkipy")
        y = x.squeeze(0)
        assert y.shape == (3, 1, 4)
