"""Reference sampling math — pure-numpy implementations.

Sampling pipeline stages (mirroring vLLM's Sampler layer):

  Stage 1 - Temperature + softmax   ``_compute_probs_from_logits``
  Stage 2 - Top-k filtering         ``_topk_threshold``
  Stage 3 - Top-p filtering         ``_topp_threshold``
  Stage 4 - Min-p filtering         (inline: ``max_prob * min_p``)
  Stage 5 - Combine & guarantee     merge thresholds, force argmax
  Stage 6 - Multinomial sample      CDF prefix-sum, ``count(cdf < target)``

These are reference/test implementations.  The device path uses
NKI kernels in ``nkipy_serving/sampling/nki_kernels.py`` instead.
"""

from __future__ import annotations

import numpy as np

_THRESHOLD_SEARCH_ITERS = 12
_TARGET_EPS = np.float32(1e-7)


# ===================================================================
# Stage 1 - Temperature + softmax
# ===================================================================


def _compute_probs_from_logits(
    logits: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    temp = np.maximum(
        temperatures.astype(np.float32).reshape((-1, 1)),
        np.float32(1e-6),
    )
    scaled_logits = logits.astype(np.float32) / temp
    row_max = np.max(scaled_logits, axis=1, keepdims=True)
    exp_shifted = np.exp(scaled_logits - row_max)
    denom = np.sum(exp_shifted, axis=1, keepdims=True)
    return exp_shifted / np.maximum(denom, np.float32(1e-20))


# ===================================================================
# Stage 2 - Top-k filtering
# ===================================================================


def _topk_threshold(probs: np.ndarray, top_ks: np.ndarray) -> np.ndarray:
    """Binary-search for the highest prob threshold keeping >= top_k tokens."""
    bs, vocab = probs.shape
    targets = top_ks.astype(np.int32).reshape((bs, 1))
    active = targets < int(vocab)
    low = np.zeros((bs, 1), dtype=np.float32)
    high = np.max(probs, axis=1, keepdims=True)
    for _ in range(_THRESHOLD_SEARCH_ITERS):
        mid = (low + high) * np.float32(0.5)
        counts = np.sum(probs >= mid, axis=1, keepdims=True)
        can_raise = counts >= targets
        low = np.where(active, np.where(can_raise, mid, low), low)
        high = np.where(active, np.where(can_raise, high, mid), high)
    return np.where(active, low, np.zeros_like(low))


# ===================================================================
# Stage 3 - Top-p filtering
# ===================================================================


def _topp_threshold(probs: np.ndarray, top_ps: np.ndarray) -> np.ndarray:
    """Binary-search for the highest prob threshold keeping >= top_p mass."""
    bs, _ = probs.shape
    targets = top_ps.astype(np.float32).reshape((bs, 1))
    active = targets < np.float32(1.0 - 1e-6)
    low = np.zeros((bs, 1), dtype=np.float32)
    high = np.max(probs, axis=1, keepdims=True)
    for _ in range(_THRESHOLD_SEARCH_ITERS):
        mid = (low + high) * np.float32(0.5)
        masked = np.where(probs >= mid, probs, np.float32(0.0))
        mass = np.sum(masked, axis=1, keepdims=True)
        can_raise = mass >= targets
        low = np.where(active, np.where(can_raise, mid, low), low)
        high = np.where(active, np.where(can_raise, high, mid), high)
    return np.where(active, low, np.zeros_like(low))


# ===================================================================
# Stages 4-6 - Combine, mask, sample (reference implementation)
# ===================================================================


def _sample_from_probs(
    probs: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
) -> np.ndarray:
    """Reference CPU sampler: threshold -> mask -> CDF -> sample."""
    # Stage 2-4: compute per-filter thresholds.
    max_probs = np.max(probs, axis=1, keepdims=True)
    topk_threshold = _topk_threshold(probs, top_ks)
    topp_threshold = _topp_threshold(probs, top_ps)
    minp_threshold = max_probs * min_ps.astype(np.float32).reshape((-1, 1))

    # Stage 5: combine thresholds, guarantee argmax survives.
    final_threshold = np.maximum(
        np.maximum(topk_threshold, topp_threshold),
        minp_threshold,
    )
    allowed = probs >= final_threshold
    allowed = np.where(probs >= max_probs, True, allowed)

    # Stage 6: CDF-based multinomial sample.
    masked_probs = np.where(allowed, probs, np.float32(0.0))
    cdf = np.cumsum(masked_probs, axis=-1).astype(np.float32)
    total_mass = cdf[:, -1:].astype(np.float32)
    safe_total = np.maximum(total_mass, np.float32(1e-20))
    targets = np.maximum(
        uniform_u.astype(np.float32).reshape((-1, 1)),
        _TARGET_EPS,
    )
    targets = (
        np.minimum(
            targets,
            np.float32(1.0 - _TARGET_EPS),
        )
        * safe_total
    )
    sampled = np.sum(cdf < targets, axis=1).astype(np.int32)
    vocab_last_idx = np.int32(probs.shape[1] - 1)
    return np.minimum(sampled, vocab_last_idx).astype(np.int32)


def _sample_from_logits_reference(
    logits: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
) -> np.ndarray:
    """Full reference pipeline: logits -> probs -> threshold -> sample."""
    probs = _compute_probs_from_logits(logits, temperatures)
    return _sample_from_probs(probs, top_ks, top_ps, min_ps, uniform_u)
