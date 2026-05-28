# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-memory staging buffers for RDMA transfers on Trn2.

On Trn2, direct RDMA to NeuronCore HBM achieves only ~3.5 Gbps per rank
(vs. 138 Gbps for host memory). This module provides host staging buffers
so the transfer path becomes:

  Sender:   device --DMA--> host_staging --RDMA--> remote_host_staging
  Receiver: remote_host_staging --DMA--> device

The DMA engine on NeuronCores is fast (~12+ GB/s per device), making the
total transfer limited by RDMA bandwidth rather than the slow device-RDMA path.
"""

import ctypes
import ctypes.util
import mmap
import os

import numpy as np
from typing import List, Tuple

_HUGEPAGE_SIZE = 2 * 1024 * 1024  # 2 MB

_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_POPULATE = 0x8000
_MAP_HUGETLB = 0x40000
_PROT_READ = 0x1
_PROT_WRITE = 0x2

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int


def _available_huge_pages() -> int:
    """Return number of free 2MB huge pages from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("HugePages_Free:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _use_huge_pages(required_bytes: int = 0) -> bool:
    """Decide whether to use huge pages.

    - NKIPY_HUGE_PAGES=1: force on
    - NKIPY_HUGE_PAGES=0: force off
    - Unset: auto-detect based on whether all concurrent workers can fit
    """
    env = os.environ.get("NKIPY_HUGE_PAGES")
    if env is not None:
        return env == "1"
    if required_bytes <= 0:
        return False
    pages_needed = (required_bytes + _HUGEPAGE_SIZE - 1) // _HUGEPAGE_SIZE
    available = _available_huge_pages()
    # Each TP worker allocates independently; need pages_needed * num_workers.
    # Use WORLD_SIZE or tensor-parallel-size if set; otherwise assume 32 (trn2 max).
    tp = int(os.environ.get("WORLD_SIZE",
             os.environ.get("NEURON_RT_NUM_CORES", "32")))
    return available >= pages_needed * max(tp, 1)


def _alloc_huge(size: int) -> tuple:
    """Allocate memory backed by 2MB huge pages.

    Returns (ptr, aligned_size) or raises OSError on failure.
    MAP_POPULATE is omitted: huge pages are physically resident once allocated
    from the pool, so ibv_reg_mr is fast regardless. MAP_POPULATE would serialize
    32 workers on the kernel hugetlb_lock, causing contention.
    """
    aligned = ((size + _HUGEPAGE_SIZE - 1) // _HUGEPAGE_SIZE) * _HUGEPAGE_SIZE
    flags = _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB
    ptr = _libc.mmap(None, aligned, _PROT_READ | _PROT_WRITE, flags, -1, 0)
    if ptr == ctypes.c_void_p(-1).value:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mmap(MAP_HUGETLB) failed for {aligned} bytes: "
                      f"{os.strerror(errno)}")
    return ptr, aligned


class HostStagingBuffer:
    """Page-aligned contiguous host buffer for RDMA staging.

    Allocates a single large buffer that is sliced into sub-regions matching
    individual weight tensors. One ibv_reg_mr call covers the entire buffer.

    When NKIPY_HUGE_PAGES=1, uses 2MB huge pages which dramatically reduce
    ibv_reg_mr latency (fewer page table entries to pin).
    """

    def __init__(self, total_size: int):
        self._size = total_size
        self._huge = _use_huge_pages(total_size)
        if self._huge:
            self._ptr, self._alloc_size = _alloc_huge(total_size)
            self._array = (ctypes.c_char * total_size).from_address(self._ptr)
            self._np_array = np.frombuffer(self._array, dtype=np.uint8)
            self._mmap = None
        else:
            self._mmap = mmap.mmap(-1, total_size)
            self._np_array = np.frombuffer(self._mmap, dtype=np.uint8)
            self._ptr = self._np_array.ctypes.data
            self._alloc_size = total_size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    def slice_as_numpy(self, offset: int, size: int, dtype=np.uint8) -> np.ndarray:
        if self._huge:
            buf = (ctypes.c_char * size).from_address(self._ptr + offset)
            return np.frombuffer(buf, dtype=dtype)
        return np.frombuffer(self._mmap, dtype=dtype,
                             count=size // np.dtype(dtype).itemsize, offset=offset)

    def close(self):
        self._np_array = None
        if self._huge:
            if self._ptr:
                _libc.munmap(ctypes.c_void_p(self._ptr), self._alloc_size)
                self._ptr = 0
            self._array = None
        else:
            self._array = None
            if self._mmap:
                try:
                    self._mmap.close()
                except BufferError:
                    pass
                self._mmap = None

    def __del__(self):
        self.close()


def compute_offsets(sizes: List[int]) -> List[int]:
    """Compute byte offsets for packing buffers contiguously."""
    offsets = []
    offset = 0
    for s in sizes:
        offsets.append(offset)
        offset += s
    return offsets


class PreregisteredStaging:
    """Host staging buffer with pre-registered RDMA MRs.

    Allocated once at engine init; reused across wake/sleep cycles.
    Registers the contiguous buffer as a single MR.
    """

    def __init__(self, sizes: List[int], endpoint):
        self.sizes = sizes
        self.offsets = compute_offsets(sizes)
        self.total_size = sum(sizes)
        self.staging = HostStagingBuffer(self.total_size)

        self.xfer_descs = endpoint.register_contiguous_buffer(
            self.staging.ptr, self.total_size, self.offsets, self.sizes
        )

    def close(self):
        self.staging.close()
        self.xfer_descs = []
