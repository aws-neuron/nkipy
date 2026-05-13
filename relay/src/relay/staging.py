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

import mmap
import numpy as np
from typing import List, Tuple


class HostStagingBuffer:
    """Page-aligned contiguous host buffer for RDMA staging.

    Allocates a single large buffer that is sliced into sub-regions matching
    individual weight tensors. One ibv_reg_mr call covers the entire buffer.
    """

    def __init__(self, total_size: int):
        self._size = total_size
        self._mmap = mmap.mmap(-1, total_size)
        self._array = np.frombuffer(self._mmap, dtype=np.uint8)

    @property
    def ptr(self) -> int:
        return self._array.ctypes.data

    @property
    def size(self) -> int:
        return self._size

    def slice_as_numpy(self, offset: int, size: int, dtype=np.uint8) -> np.ndarray:
        return np.frombuffer(self._mmap, dtype=dtype, count=size // np.dtype(dtype).itemsize, offset=offset)

    def close(self):
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
    Uses single-MR registration for the entire contiguous buffer.
    """

    def __init__(self, sizes: List[int], endpoint):
        self.sizes = sizes
        self.offsets = compute_offsets(sizes)
        self.total_size = sum(sizes)
        self.staging = HostStagingBuffer(self.total_size)

        # Single-MR registration (4 ibv_reg_mr calls instead of 483*4)
        self.xfer_descs = endpoint.register_contiguous_buffer(
            self.staging.ptr, self.total_size, self.offsets, self.sizes)

    def close(self):
        self.staging.close()
        self.xfer_descs = []
