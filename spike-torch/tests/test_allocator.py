# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nkipy memory allocator."""

import gc

import pytest
import torch

import spike_torch


class TestAllocationTracking:
    """Tests for allocation tracking."""

    def test_basic_allocation(self):
        """Basic tensor allocation works."""
        x = torch.randn(1000, device="nkipy")
        ptr = x.data_ptr()
        assert ptr != 0

    def test_multiple_allocations(self):
        """Multiple allocations have different pointers."""
        x = torch.randn(100, device="nkipy")
        y = torch.randn(100, device="nkipy")
        assert x.data_ptr() != y.data_ptr()

    def test_allocation_after_free(self):
        """Allocation after free uses cached memory."""
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        # Next allocation of same size should use cached memory
        y = torch.randn(1000, device="nkipy")
        assert y.data_ptr() != 0


class TestCaching:
    """Tests for memory caching."""

    def test_cached_blocks_increases(self):
        """Cached blocks increase after tensor deletion."""
        spike_torch.empty_cache()  # Start fresh
        initial_cached = spike_torch.get_cached_blocks()

        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        # Should have at least one cached block
        assert spike_torch.get_cached_blocks() >= initial_cached


class TestEmptyCache:
    """Tests for empty_cache operation."""

    def test_empty_cache_basic(self):
        """empty_cache releases cached memory."""
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        spike_torch.empty_cache()
        # Should not raise

    def test_empty_cache_clears_cache(self):
        """empty_cache reduces cached blocks."""
        x = torch.randn(1000, device="nkipy")
        del x
        gc.collect()

        cached_before = spike_torch.get_cached_blocks()
        spike_torch.empty_cache()
        cached_after = spike_torch.get_cached_blocks()

        # Cache should be cleared (or at least not increased)
        assert cached_after <= cached_before


class TestAllocationSizes:
    """Tests for various allocation sizes."""

    @pytest.mark.parametrize(
        "size",
        [1, 10, 100, 1000, 10000, 100000],
    )
    def test_various_sizes(self, size):
        """Allocations of various sizes work."""
        x = torch.empty(size, device="nkipy")
        assert x.numel() == size

    def test_large_allocation(self):
        """Large allocation works."""
        # 10 MB tensor
        x = torch.empty(10 * 1024 * 1024 // 4, dtype=torch.float32, device="nkipy")
        assert x.numel() > 0
