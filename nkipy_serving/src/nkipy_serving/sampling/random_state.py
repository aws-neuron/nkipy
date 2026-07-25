"""Stateless per-request RNG for device-side sampling.

``draw_uniform(seed, position)`` hashes (seed, position) via MurmurHash3
into a float32 in (ε, 1−ε). No mutable Generator objects needed — any
step is recomputable from (seed, pos) alone.
"""

from __future__ import annotations

import numpy as np

_UNIFORM_EPS = np.float32(1e-7)
_UINT32_MAX = np.float32(np.iinfo(np.uint32).max)


def _fmix32(h: int) -> int:
    """MurmurHash3 32-bit finalizer."""
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def assign_seed(seed: int | None) -> int:
    """Return a concrete uint64 seed.

    If the caller supplied a seed, return it.  Otherwise draw one from
    OS entropy so the request is still reproducible within this process.
    """
    if seed is not None:
        return int(seed) & 0xFFFFFFFFFFFFFFFF
    return int(np.random.default_rng().integers(0, 2**63))


def draw_uniform(seed: int, position: int) -> np.float32:
    """Stateless hash of *(seed, position)* → uniform float32 in (ε, 1−ε).

    Uses MurmurHash3's ``fmix32`` finalizer on a combined key derived from
    the 64-bit seed and the 32-bit decode position.
    """
    # Mix the 64-bit seed into two 32-bit halves, then fold in position.
    lo = int(seed) & 0xFFFFFFFF
    hi = (int(seed) >> 32) & 0xFFFFFFFF
    combined = (lo ^ _fmix32(hi)) & 0xFFFFFFFF
    combined = (combined ^ (int(position) * 0x9E3779B9)) & 0xFFFFFFFF
    h = _fmix32(combined)
    u = np.float32(h) / _UINT32_MAX
    return np.clip(u, _UNIFORM_EPS, np.float32(1.0) - _UNIFORM_EPS)
