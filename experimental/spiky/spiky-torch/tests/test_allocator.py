# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Tests for nkipy memory allocator."""

import gc

import pytest
import spiky.torch as spiky_torch
import torch


class TestAllocationTracking:
    def test_basic_allocation(self):
        x = torch.randn(1000, device="nkipy")
        ptr = x.data_ptr()
        assert ptr != 0

    def test_multiple_allocations(self):
        x = torch.randn(100, device="nkipy")
        y = torch.randn(100, device="nkipy")
        assert x.data_ptr() != y.data_ptr()

    def test_allocation_after_free(self):
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        y = torch.randn(1000, device="nkipy")
        assert y.data_ptr() != 0


class TestCaching:
    def test_cached_blocks_increases(self):
        spiky_torch.empty_cache()
        initial_cached = spiky_torch.get_cached_blocks()

        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        assert spiky_torch.get_cached_blocks() >= initial_cached


class TestEmptyCache:
    def test_empty_cache_basic(self):
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        spiky_torch.empty_cache()

    def test_empty_cache_clears_cache(self):
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        cached_before = spiky_torch.get_cached_blocks()
        spiky_torch.empty_cache()
        cached_after = spiky_torch.get_cached_blocks()
        assert cached_after <= cached_before


class TestAllocationSizes:
    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000, 100000])
    def test_various_sizes(self, size):
        x = torch.empty(size, device="nkipy")
        assert x.numel() == size

    def test_large_allocation(self):
        x = torch.empty(10 * 1024 * 1024 // 4, dtype=torch.float32, device="nkipy")
        assert x.numel() > 0
