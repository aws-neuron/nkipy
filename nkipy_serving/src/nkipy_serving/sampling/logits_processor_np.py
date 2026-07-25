"""NumPy reference logits processor: sampling + logprobs from raw logits.

Pure-NumPy implementation that mirrors the device-side LogitsProcessor
pipeline.  Used by the numpy execution backend and serves as an accuracy
baseline for the NKI device kernels.

Pipeline:  raw logits → last-token gather → temperature → top-k/top-p/min-p
           → softmax → sample (greedy or stochastic) → logprobs (optional)

Interface parity with ``sampling.logits_processor.LogitsProcessor``:
  - Both return ``LogitsProcessorOutput``
  - Both accept ``DeviceSamplingBatch`` for sampling parameters
  - Both produce top1 candidates (greedy) or next_token_ids + logprobs
  - Difference: device takes hidden states + weights; numpy takes raw logits
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessorOutput


class NumpyLogitsProcessor:
    """CPU logits processor that mirrors the device LogitsProcessor interface.

    Args:
        vocab_offset: Rank-local vocab offset for TP merge (0 for TP=1).
    """

    def __init__(self, *, vocab_offset: int = 0) -> None:
        self._vocab_offset = int(vocab_offset)

    @property
    def vocab_offset(self) -> int:
        return self._vocab_offset

    def forward(
        self,
        logits: np.ndarray,
        sample_mask: np.ndarray,
        query_start_loc: np.ndarray,
        batch_size: int,
        *,
        sampling_batch: DeviceSamplingBatch | None = None,
        needs_logprobs: bool = False,
        logprobs_k: int = 0,
    ) -> LogitsProcessorOutput:
        """Process raw logits into sampled tokens and optional logprobs.

        Args:
            logits: [total_tokens, vocab_size] float32.
            sample_mask: [batch_size] bool — which requests emit a token.
            query_start_loc: [batch_size + 1] cumulative query offsets.
            batch_size: Number of requests in the batch.
            sampling_batch: Sampling parameters (None = greedy).
            needs_logprobs: Whether to compute logprobs.
            logprobs_k: Top-k for logprobs (0 = disabled).

        Returns:
            LogitsProcessorOutput matching the device LogitsProcessor contract.
        """
        bs = int(batch_size)
        mask = np.asarray(sample_mask, dtype=np.bool_)
        query_end_rows = np.asarray(query_start_loc[1:], dtype=np.int64) - 1
        sampled_rows = query_end_rows[mask]
        n_sampled = int(sampled_rows.size)

        if n_sampled == 0:
            return LogitsProcessorOutput(
                top1_values=np.full((bs,), float("-inf"), dtype=np.float32),
                top1_indices=np.zeros((bs,), dtype=np.int32),
            )

        last_logits = np.asarray(logits[sampled_rows], dtype=np.float32)

        # --- Sampling parameters ---
        if sampling_batch is not None:
            temperatures = np.asarray(
                sampling_batch.temperatures[:bs], dtype=np.float32
            )[mask]
            top_ks = np.asarray(sampling_batch.top_ks[:bs], dtype=np.int32)[mask]
            top_ps = np.asarray(sampling_batch.top_ps[:bs], dtype=np.float32)[mask]
            min_ps = np.asarray(sampling_batch.min_ps[:bs], dtype=np.float32)[mask]
            uniform_u = np.asarray(sampling_batch.uniform_u[:bs], dtype=np.float32)[
                mask
            ]
        else:
            temperatures = np.ones(n_sampled, dtype=np.float32)
            top_ks = np.ones(n_sampled, dtype=np.int32)  # top_k=1 → greedy
            top_ps = np.ones(n_sampled, dtype=np.float32)
            min_ps = np.zeros(n_sampled, dtype=np.float32)
            uniform_u = np.zeros(n_sampled, dtype=np.float32)

        is_greedy = top_ks == 1
        needs_sampling = ~is_greedy

        # --- Temperature scaling ---
        scaled_logits = last_logits.copy()
        if np.any(needs_sampling):
            temps = np.maximum(temperatures[needs_sampling], np.float32(1e-6))
            scaled_logits[needs_sampling] /= temps[:, None]

        # --- Filtering ---
        filtered_logits = _apply_filters(
            scaled_logits, top_ks, top_ps, min_ps, needs_sampling
        )

        # --- Softmax + Sampling ---
        probs = _softmax(filtered_logits)
        token_ids_sampled = np.empty(n_sampled, dtype=np.int32)
        if np.any(is_greedy):
            token_ids_sampled[is_greedy] = np.argmax(
                last_logits[is_greedy], axis=1
            ).astype(np.int32)
        if np.any(needs_sampling):
            token_ids_sampled[needs_sampling] = _sample_from_probs(
                probs[needs_sampling], uniform_u[needs_sampling]
            )

        # --- Build output ---
        if not needs_logprobs or logprobs_k <= 0:
            top1_values = np.full((bs,), float("-inf"), dtype=np.float32)
            top1_indices = np.zeros((bs,), dtype=np.int32)
            top1_values[mask] = last_logits[np.arange(n_sampled), token_ids_sampled]
            top1_indices[mask] = token_ids_sampled
            return LogitsProcessorOutput(
                top1_values=top1_values,
                top1_indices=top1_indices,
            )

        # --- With logprobs ---
        next_token_ids = np.zeros(bs, dtype=np.int32)
        next_token_ids[mask] = token_ids_sampled + self._vocab_offset

        log_probs = _log_softmax(last_logits)
        chosen_lp = np.full((bs,), float("-inf"), dtype=np.float32)
        chosen_lp[mask] = log_probs[np.arange(n_sampled), token_ids_sampled]

        k = min(logprobs_k, int(log_probs.shape[1]))
        topk_lp_ids = np.zeros((bs, k), dtype=np.int32)
        topk_lp_vals = np.full((bs, k), float("-inf"), dtype=np.float32)

        topk_idx_sampled = np.argpartition(-log_probs, kth=k - 1, axis=1)[:, :k]
        topk_vals_sampled = np.take_along_axis(log_probs, topk_idx_sampled, axis=1)
        order = np.argsort(-topk_vals_sampled, axis=1)
        topk_idx_sampled = np.take_along_axis(topk_idx_sampled, order, axis=1)
        topk_vals_sampled = np.take_along_axis(topk_vals_sampled, order, axis=1)

        topk_lp_ids[mask, :k] = (topk_idx_sampled + self._vocab_offset).astype(np.int32)
        topk_lp_vals[mask, :k] = topk_vals_sampled.astype(np.float32)

        return LogitsProcessorOutput(
            next_token_ids=next_token_ids,
            chosen_logprobs=chosen_lp,
            topk_logprob_vals=topk_lp_vals,
            topk_logprob_ids=topk_lp_ids,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def _apply_filters(
    logits: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply top-k, top-p, and min-p filtering to logits (in-place for masked rows)."""
    if not np.any(mask):
        return logits
    vocab_size = logits.shape[1]
    neg_inf = np.float32(-np.inf)

    for i in np.where(mask)[0]:
        row = logits[i]
        k = int(top_ks[i])

        # Top-k: keep only top k logits.
        if 0 < k < vocab_size:
            threshold = np.partition(row, -k)[-k]
            row[row < threshold] = neg_inf

        # Top-p (nucleus): keep smallest set with cumulative prob >= top_p.
        p = float(top_ps[i])
        if p < 1.0 - 1e-6:
            sorted_idx = np.argsort(-row)
            sorted_probs = _softmax(row[sorted_idx].reshape(1, -1)).ravel()
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, p) + 1
            if cutoff < vocab_size:
                row[sorted_idx[cutoff:]] = neg_inf

        # Min-p: remove tokens with prob < min_p * max_prob.
        mp = float(min_ps[i])
        if mp > 0.0:
            probs = _softmax(row.reshape(1, -1)).ravel()
            threshold_p = mp * np.max(probs)
            row[probs < threshold_p] = neg_inf

    return logits


def _sample_from_probs(probs: np.ndarray, uniform_u: np.ndarray) -> np.ndarray:
    """CDF-based sampling using pre-generated uniform random values.

    Matches the device-side NKI CDF sampler semantics: for each row,
    find the first index where cumulative probability exceeds u.
    """
    cumsum = np.cumsum(probs, axis=-1)
    u = uniform_u[:, None]
    token_ids = np.argmax(cumsum >= u, axis=-1).astype(np.int32)
    return token_ids
