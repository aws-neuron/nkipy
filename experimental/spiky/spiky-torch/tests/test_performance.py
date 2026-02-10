# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Performance tests for nkipy device integration (slow)."""

import gc
import time

import spiky.torch as spiky_torch
import torch


class TestAllocationPerformance:
    def test_allocation_performance(self):
        for _ in range(10):
            x = torch.randn(1000, 1000, device="nkipy")
            del x
        gc.collect()

        start = time.time()
        for _ in range(100):
            x = torch.randn(1000, 1000, device="nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n100 allocations (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per allocation")

    def test_small_allocation_performance(self):
        for _ in range(100):
            x = torch.randn(100, device="nkipy")
            del x
        gc.collect()

        start = time.time()
        for _ in range(1000):
            x = torch.randn(100, device="nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n1000 small allocations (100 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 1000 * 1000:.2f}ms per allocation")


class TestCopyPerformance:
    def test_cpu_to_nkipy_performance(self):
        cpu_data = torch.randn(1000, 1000)
        for _ in range(10):
            x = cpu_data.to("nkipy")
            del x
        gc.collect()

        start = time.time()
        for _ in range(100):
            x = cpu_data.to("nkipy")
            del x
        elapsed = time.time() - start

        print(f"\n100 CPU->nkipy copies (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per copy")

    def test_nkipy_to_cpu_performance(self):
        nkipy_data = torch.randn(1000, 1000, device="nkipy")
        for _ in range(10):
            x = nkipy_data.cpu()
            del x
        gc.collect()

        start = time.time()
        for _ in range(100):
            x = nkipy_data.cpu()
            del x
        elapsed = time.time() - start

        print(f"\n100 nkipy->CPU copies (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 100 * 1000:.2f}ms per copy")

    def test_roundtrip_performance(self):
        cpu_data = torch.randn(1000, 1000)
        for _ in range(5):
            nkipy_data = cpu_data.to("nkipy")
            back = nkipy_data.cpu()
            del nkipy_data, back
        gc.collect()

        start = time.time()
        for _ in range(50):
            nkipy_data = cpu_data.to("nkipy")
            back = nkipy_data.cpu()
            del nkipy_data, back
        elapsed = time.time() - start

        print(f"\n50 roundtrips (1000x1000 float32): {elapsed:.3f}s")
        print(f"Average: {elapsed / 50 * 1000:.2f}ms per roundtrip")


class TestCachePerformance:
    def test_cache_reuse_speedup(self):
        size = (1000, 1000)
        spiky_torch.empty_cache()

        start = time.time()
        x = torch.randn(*size, device="nkipy")
        first_alloc = time.time() - start
        del x
        gc.collect()

        start = time.time()
        y = torch.randn(*size, device="nkipy")
        cached_alloc = time.time() - start
        del y

        print(f"\nFirst allocation: {first_alloc * 1000:.2f}ms")
        print(f"Cached allocation: {cached_alloc * 1000:.2f}ms")
