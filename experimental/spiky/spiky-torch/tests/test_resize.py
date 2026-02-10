# SPDX-License-Identifier: Apache-2.0

"""Tests for resize operations on nkipy tensors."""

import torch

import spiky.torch as spiky_torch  # noqa: F401


class TestResize:
    def test_resize_grow(self):
        x = torch.empty(10, device="nkipy")
        x.resize_(20)
        assert x.shape == (20,)

    def test_resize_shrink(self):
        x = torch.empty(20, device="nkipy")
        x.resize_(10)
        assert x.shape == (10,)

    def test_resize_same_size(self):
        x = torch.empty(10, device="nkipy")
        x.resize_(10)
        assert x.shape == (10,)

    def test_resize_2d(self):
        x = torch.empty(10, device="nkipy")
        x.resize_(5, 5)
        assert x.shape == (5, 5)

    def test_resize_to_zero(self):
        x = torch.empty(10, device="nkipy")
        x.resize_(0)
        assert x.numel() == 0


class TestResizeAs:
    def test_resize_as_same_device(self):
        x = torch.empty(10, device="nkipy")
        y = torch.empty(5, 5, device="nkipy")
        x.resize_as_(y)
        assert x.shape == y.shape

    def test_resize_as_cpu(self):
        x = torch.empty(10, device="nkipy")
        y = torch.empty(5, 5)
        x.resize_as_(y)
        assert x.shape == y.shape


class TestResizePreservesData:
    def test_resize_grow_preserves_existing_data(self):
        original = torch.arange(10).float()
        x = original.to("nkipy")
        x.resize_(20)
        torch.testing.assert_close(x.cpu()[:10], original)

    def test_resize_shrink_preserves_data(self):
        original = torch.arange(20).float()
        x = original.to("nkipy")
        x.resize_(10)
        torch.testing.assert_close(x.cpu(), original[:10])

