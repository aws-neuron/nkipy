# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resize operations on nkipy tensors."""

import pytest
import torch

import spike_torch


class TestResize:
    """Tests for resize_ operation."""

    def test_resize_grow(self):
        """Resize tensor to larger size."""
        x = torch.empty(10, device="nkipy")
        x.resize_(20)
        assert x.shape == (20,)

    def test_resize_shrink(self):
        """Resize tensor to smaller size."""
        x = torch.empty(20, device="nkipy")
        x.resize_(10)
        assert x.shape == (10,)

    def test_resize_same_size(self):
        """Resize tensor to same size."""
        x = torch.empty(10, device="nkipy")
        x.resize_(10)
        assert x.shape == (10,)

    def test_resize_2d(self):
        """Resize to 2D shape."""
        x = torch.empty(10, device="nkipy")
        x.resize_(5, 5)
        assert x.shape == (5, 5)

    def test_resize_to_zero(self):
        """Resize to zero size."""
        x = torch.empty(10, device="nkipy")
        x.resize_(0)
        assert x.numel() == 0


class TestResizeAs:
    """Tests for resize_as_ operation."""

    def test_resize_as_same_device(self):
        """Resize as another nkipy tensor."""
        x = torch.empty(10, device="nkipy")
        y = torch.empty(5, 5, device="nkipy")
        x.resize_as_(y)
        assert x.shape == y.shape

    def test_resize_as_cpu(self):
        """Resize as a CPU tensor."""
        x = torch.empty(10, device="nkipy")
        y = torch.empty(5, 5)
        x.resize_as_(y)
        assert x.shape == y.shape


class TestResizePreservesData:
    """Tests for data preservation during resize."""

    def test_resize_grow_preserves_existing_data(self):
        """Growing resize preserves existing data."""
        original = torch.arange(10).float()
        x = original.to("nkipy")
        x.resize_(20)
        # First 10 elements should be preserved
        # Note: remaining elements are uninitialized
        torch.testing.assert_close(x.cpu()[:10], original)

    def test_resize_shrink_preserves_data(self):
        """Shrinking resize preserves remaining data."""
        original = torch.arange(20).float()
        x = original.to("nkipy")
        x.resize_(10)
        torch.testing.assert_close(x.cpu(), original[:10])
