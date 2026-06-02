"""Hardware profiles for Trainium/Inferentia targets.

Each profile captures the timing and bandwidth parameters needed by the cost
model. Add new targets (TRN3, TRN4, ...) as additional instances.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware parameters for cost modeling and tile planning."""

    # DMA engine
    dma_bw_per_engine: float = 23e9       # bytes/sec per engine
    dma_num_engines: int = 16
    dma_transpose_bw_sbuf: float = 215e9  # SBUF→SBUF transpose bandwidth
    dma_transpose_bw_hbm: float = 335e9   # HBM→SBUF transpose bandwidth
    dma_sem_to_start: float = 1300e-9     # seconds

    # Tensor engine
    tensor_freq: float = 2.40e9           # Hz
    tensor_write_drain: float = 150e-9    # seconds
    tensor_sem_to_start: float = 81e-9    # seconds

    # Vector engine
    vector_freq: float = 0.96e9           # Hz
    vector_write_drain: float = 161e-9    # seconds
    vector_sem_to_start: float = 268e-9   # seconds
    vector_min_ii: int = 64               # minimum initiation interval in cycles

    # GpSimd engine (cross-lane operations)
    gpsimd_freq: float = 1.20e9           # Hz
    gpsimd_write_drain: float = 218e-9    # seconds
    gpsimd_sem_to_start: float = 186e-9   # seconds

    # Partition constraints
    partition_max: int = 128

    # Memory capacities
    psum_free_max: int = 512              # max F-elements in PSUM (matmul output)
    matmul_stat_free_max: int = 128
    sbuf_per_partition_bytes: int = 180_224
    psum_per_partition_bytes: int = 16 * 1024

    @property
    def dma_bw(self) -> float:
        return self.dma_bw_per_engine * self.dma_num_engines


# Named targets
TRN2 = HardwareProfile()
