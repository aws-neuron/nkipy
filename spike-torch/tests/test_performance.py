# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance tests for nkipy-torch.

These tests measure allocation and copy performance.
Run with pytest -v -s to see timing output.
"""

import gc
import time

import pytest
import torch

import spike_torch


class TestAllocationPerformance:
    """Performance tests for memory allocation."""

    def test_allocation_performance(self):
        """Allocation should be fast (using caching)."""
        # Warmup
        for _ in range(10):
            x = torch.randn(1000, 1000, device="nkipy")
            del x
        gc.collect()

        # Benchmark
        start = time.time()
        for _ in range(100):
            x = torch.randn(1000, 1000, device="nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n100 allocations (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per allocation")

        # Should benefit from caching - no strict threshold
        # just ensure it doesn't hang

    def test_small_allocation_performance(self):
        """Small allocations should be fast."""
        # Warmup
        for _ in range(100):
            x = torch.randn(100, device="nkipy")
            del x
        gc.collect()

        # Benchmark
        start = time.time()
        for _ in range(1000):
            x = torch.randn(100, device="nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n1000 small allocations (100 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 1000 * 1000:.2f}ms per allocation")


class TestCopyPerformance:
    """Performance tests for copy operations."""

    def test_cpu_to_nkipy_performance(self):
        """CPU to nkipy copy should be fast."""
        cpu_data = torch.randn(1000, 1000)

        # Warmup
        for _ in range(10):
            x = cpu_data.to("nkipy")
            del x
        gc.collect()

        # Benchmark
        start = time.time()
        for _ in range(100):
            x = cpu_data.to("nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n100 CPU->nkipy copies (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per copy")

    def test_nkipy_to_cpu_performance(self):
        """nkipy to CPU copy should be fast."""
        nkipy_data = torch.randn(1000, 1000, device="nkipy")

        # Warmup
        for _ in range(10):
            x = nkipy_data.cpu()
            del x
        gc.collect()

        # Benchmark
        start = time.time()
        for _ in range(100):
            x = nkipy_data.cpu()
            del x
        elapsed = time.time() - start

        print(f"\n100 nkipy->CPU copies (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per copy")

    def test_roundtrip_performance(self):
        """Roundtrip (CPU->nkipy->CPU) performance."""
        cpu_data = torch.randn(1000, 1000)

        # Warmup
        for _ in range(5):
            nkipy_data = cpu_data.to("nkipy")
            back = nkipy_data.cpu()
            del nkipy_data, back
        gc.collect()

        # Benchmark
        start = time.time()
        for _ in range(50):
            nkipy_data = cpu_data.to("nkipy")
            back = nkipy_data.cpu()
            del nkipy_data, back
        elapsed = time.time() - start

        print(f"\n50 roundtrips (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 50 * 1000:.2f}ms per roundtrip")


class TestCachePerformance:
    """Performance tests for cache operations."""

    def test_cache_reuse_speedup(self):
        """Cached allocations should be faster."""
        size = (1000, 1000)

        # Clear cache
        spike_torch.empty_cache()

        # First allocation (no cache)
        start = time.time()
        x = torch.randn(*size, device="nkipy")
        first_alloc_time = time.time() - start
        del x
        gc.collect()

        # Second allocation (should use cache)
        start = time.time()
        y = torch.randn(*size, device="nkipy")
        cached_alloc_time = time.time() - start
        del y

        print(f"\nFirst allocation: {first_alloc_time * 1000:.2f}ms")
        print(f"Cached allocation: {cached_alloc_time * 1000:.2f}ms")

        # Cached should generally be faster, but don't enforce strict threshold
