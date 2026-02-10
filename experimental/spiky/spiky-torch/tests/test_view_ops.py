# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Tests for view operations on nkipy tensors."""

import pytest
import spiky.torch as spiky_torch  # noqa: F401
import torch


class TestView:
    def test_view_flatten(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x.view(100)
        assert y.shape == (100,)
        assert y.data_ptr() == x.data_ptr()

    def test_view_reshape_2d(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x.view(5, 20)
        assert y.shape == (5, 20)

    def test_view_infer_dim(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.view(-1, 4)
        assert y.shape == (6, 4)

    def test_view_preserves_data(self):
        original = torch.arange(12).float()
        x = original.to("nkipy")
        y = x.view(3, 4)
        torch.testing.assert_close(y.cpu(), original.view(3, 4))


class TestReshape:
    def test_reshape_contiguous(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.reshape(6, 4)
        assert y.shape == (6, 4)

    def test_reshape_infer_dim(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.reshape(-1)
        assert y.shape == (24,)


class TestAsStrided:
    def test_as_strided_basic(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x.as_strided((5, 5), (10, 1))
        assert y.shape == (5, 5)

    def test_as_strided_with_offset(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x.as_strided((5, 5), (10, 1), 5)
        assert y.shape == (5, 5)
        assert y.storage_offset() == 5


class TestTranspose:
    def test_transpose_2d(self):
        x = torch.randn(3, 4, device="nkipy")
        y = x.t()
        assert y.shape == (4, 3)

    def test_transpose_nd(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.transpose(0, 2)
        assert y.shape == (4, 3, 2)


class TestPermute:
    def test_permute_3d(self):
        x = torch.randn(2, 3, 4, device="nkipy")
        y = x.permute(2, 0, 1)
        assert y.shape == (4, 2, 3)

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_permute_preserves_data(self):
        original = torch.arange(24).float().reshape(2, 3, 4)
        x = original.to("nkipy")
        y = x.permute(2, 0, 1)
        torch.testing.assert_close(y.cpu(), original.permute(2, 0, 1))


class TestSlicing:
    def test_slice_1d(self):
        x = torch.arange(10, device="nkipy").float()
        y = x[2:5]
        assert y.shape == (3,)

    def test_slice_2d(self):
        x = torch.randn(10, 10, device="nkipy")
        y = x[:5, :5]
        assert y.shape == (5, 5)

    @pytest.mark.xfail(
        reason="Non-contiguous tensor copy not yet supported",
        raises=RuntimeError,
    )
    def test_slice_preserves_data(self):
        original = torch.arange(100).float().reshape(10, 10)
        x = original.to("nkipy")
        y = x[2:5, 3:7]
        torch.testing.assert_close(y.cpu(), original[2:5, 3:7])


class TestUnsqueeze:
    def test_unsqueeze_dim0(self):
        x = torch.randn(3, 4, device="nkipy")
        y = x.unsqueeze(0)
        assert y.shape == (1, 3, 4)

    def test_unsqueeze_dim1(self):
        x = torch.randn(3, 4, device="nkipy")
        y = x.unsqueeze(1)
        assert y.shape == (3, 1, 4)


class TestSqueeze:
    def test_squeeze_all(self):
        x = torch.randn(1, 3, 1, 4, 1, device="nkipy")
        y = x.squeeze()
        assert y.shape == (3, 4)

    def test_squeeze_specific_dim(self):
        x = torch.randn(1, 3, 1, 4, device="nkipy")
        y = x.squeeze(0)
        assert y.shape == (3, 1, 4)
