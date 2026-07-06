"""Hardware profile for the TRN2 (gen2) target.

NOT CURRENTLY WIRED UP: no cost model in nkigen-lite consumes these fields
yet, and ``lower_to_nki``'s ``target`` parameter is accepted but never read
— passing a different profile has no effect on lowering today. The
timing/bandwidth numbers below are measured constants for TRN2/gen2 hardware,
not a general spec; there is only one profile because nothing else has been
measured. Extend with additional instances (TRN3, TRN4, ...) once a cost
model or multi-target lowering actually reads ``target``.

The partition/memory-capacity fields mirror the constants in
``nki_ir.ir`` (``PARTITION_MAX``, ``SBUF_PER_PARTITION_BYTES``,
``PSUM_PER_PARTITION_BYTES``), which are what the nki_ir graph verifier
actually enforces — those remain the single source of truth for tile
legality; this profile just exposes them alongside the cost-model fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nkigen_lite.nki_ir.ir import (
    MATMUL_STATIONARY_FREE_MAX,
    PARTITION_MAX,
    PSUM_FREE_MAX,
    PSUM_PER_PARTITION_BYTES,
    SBUF_PER_PARTITION_BYTES,
)


@dataclass(frozen=True)
class HardwareProfile:
    """TRN2/gen2 timing, bandwidth, and tile-capacity parameters."""

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

    # Partition/memory constraints — mirrors nki_ir.ir; see module docstring.
    partition_max: int = field(default=PARTITION_MAX)
    psum_free_max: int = field(default=PSUM_FREE_MAX)
    matmul_stat_free_max: int = field(default=MATMUL_STATIONARY_FREE_MAX)
    sbuf_per_partition_bytes: int = field(default=SBUF_PER_PARTITION_BYTES)
    psum_per_partition_bytes: int = field(default=PSUM_PER_PARTITION_BYTES)

    @property
    def dma_bw(self) -> float:
        return self.dma_bw_per_engine * self.dma_num_engines


# The only measured target so far — see module docstring.
TRN2 = HardwareProfile()
