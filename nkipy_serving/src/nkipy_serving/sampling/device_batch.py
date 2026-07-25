"""Per-batch device sampling metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nkipy_serving.batching.contracts import ForwardBatch
from nkipy_serving.sampling.params import TOP_K_ALL


@dataclass(frozen=True)
class DeviceSamplingBatch:
    use_full_sampler: bool
    temperatures: np.ndarray
    top_ks: np.ndarray
    top_ps: np.ndarray
    min_ps: np.ndarray
    uniform_u: np.ndarray
    needs_logprobs: bool = False
    logprobs_k: int = 0

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch) -> "DeviceSamplingBatch":
        return cls(
            use_full_sampler=bool(forward_batch.use_full_sampler),
            temperatures=np.asarray(forward_batch.temperatures, dtype=np.float32),
            top_ks=np.asarray(forward_batch.top_ks, dtype=np.int32),
            top_ps=np.asarray(forward_batch.top_ps, dtype=np.float32),
            min_ps=np.asarray(forward_batch.min_ps, dtype=np.float32),
            uniform_u=np.asarray(forward_batch.uniform_u, dtype=np.float32),
            needs_logprobs=bool(forward_batch.needs_logprobs),
            logprobs_k=int(forward_batch.logprobs_k),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.use_full_sampler)

    @property
    def needs_filtering(self) -> bool:
        """True if any row requires top-k, top-p, or min-p filtering."""
        if not self.use_full_sampler:
            return False
        return bool(
            np.any(self.top_ks < TOP_K_ALL)
            or np.any(self.top_ps < np.float32(1.0 - 1e-6))
            or np.any(self.min_ps > np.float32(0.0))
        )

    def padded_inputs(self, target_size: int) -> dict[str, np.ndarray]:
        target_size = int(target_size)
        if target_size < 0:
            raise RuntimeError(f"target_size must be >= 0, got {target_size}")

        temperatures = np.ones((target_size,), dtype=np.float32)
        top_ks = np.ones((target_size,), dtype=np.int32)
        top_ps = np.ones((target_size,), dtype=np.float32)
        min_ps = np.zeros((target_size,), dtype=np.float32)
        uniform_u = np.zeros((target_size,), dtype=np.float32)

        copy_len = min(
            target_size,
            int(self.temperatures.shape[0]),
            int(self.top_ks.shape[0]),
            int(self.top_ps.shape[0]),
            int(self.min_ps.shape[0]),
            int(self.uniform_u.shape[0]),
        )
        if copy_len > 0:
            temperatures[:copy_len] = self.temperatures[:copy_len]
            top_ks[:copy_len] = self.top_ks[:copy_len]
            top_ps[:copy_len] = self.top_ps[:copy_len]
            min_ps[:copy_len] = self.min_ps[:copy_len]
            uniform_u[:copy_len] = self.uniform_u[:copy_len]

        return {
            "temperatures": temperatures,
            "top_ks": top_ks,
            "top_ps": top_ps,
            "min_ps": min_ps,
            "uniform_u": uniform_u,
        }
