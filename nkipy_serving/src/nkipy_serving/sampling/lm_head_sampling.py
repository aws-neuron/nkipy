"""LM-head sampling device entry points -- greedy and non-greedy paths.

For the **greedy** path the model returns rank-local top-k candidates
(``lm_head_local_topk``); the scheduler merges them across TP ranks.

For the **non-greedy** path the model all-gathers LM-head logits and
feeds the full vocab into a single NKI sampler kernel that fuses all
six stages (``lm_head_sample_tokens`` -> ``sample_tokens``).

RNG contract: ``uniform_u`` is drawn on the scheduler from a per-request
numpy RNG (see ``nkipy_serving/sampling/random_state.py``).  The device kernel is
deterministic given its inputs.
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.ops.nn import apply_rms_norm
from nkipy_serving.sampling.nki_kernels import sample_tokens
from nkipy_serving.sampling.params import TOP_K_ALL

# ===================================================================
# Internal helpers
# ===================================================================


def _tp_groups(tp_degree: int, tp_replica_groups: tuple) -> list[list[int]]:
    return (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )


def _prepare_hidden_for_lm_head(
    hidden: np.ndarray,
    *,
    gather_hidden: bool,
    hidden_dim: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    if not gather_hidden or int(tp_degree) <= 1:
        return hidden

    import nkipy.distributed.collectives as cc

    hidden_full = cc.all_gather(
        hidden,
        all_gather_dim=0,
        replica_groups=_tp_groups(tp_degree, tp_replica_groups),
    )
    if hidden.ndim == 1:
        hidden_full = hidden_full.reshape((int(tp_degree), int(hidden_dim)))
    return hidden_full


def _compute_local_logits(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    rms_norm_eps: float,
    gather_hidden: bool,
    tp_degree: int,
    tp_replica_groups: tuple,
) -> np.ndarray:
    """Hidden -> select -> RMS norm -> LM-head matmul (rank-local)."""
    hidden_dtype = hidden.dtype
    hidden_ready = _prepare_hidden_for_lm_head(
        hidden,
        gather_hidden=gather_hidden,
        hidden_dim=int(final_norm.shape[0]),
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    selected = hidden_ready[last_token_indices.astype(np.int32)]
    normed = apply_rms_norm(selected, final_norm, eps=rms_norm_eps).astype(hidden_dtype)
    return (normed @ lm_head.transpose(1, 0)).astype(np.float32)


def _compute_gathered_logits(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    rms_norm_eps: float,
    gather_hidden: bool,
    tp_degree: int,
    tp_replica_groups: tuple,
) -> np.ndarray:
    """Hidden -> select -> RMS norm -> LM-head matmul -> TP all-gather."""
    logits_local = _compute_local_logits(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    if int(tp_degree) > 1:
        import nkipy.distributed.collectives as cc

        return cc.all_gather(
            logits_local,
            all_gather_dim=1,
            replica_groups=_tp_groups(tp_degree, tp_replica_groups),
        )
    return logits_local


def _sample_from_gathered_logits(
    logits: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
    *,
    unfiltered: bool,
) -> np.ndarray:
    """Clip top_ks to vocab size, then run the NKI sampler."""
    vocab_size = int(logits.shape[1])
    top_ks = np.minimum(
        top_ks.astype(np.int32),
        np.int32(vocab_size if vocab_size > 0 else TOP_K_ALL),
    )
    return sample_tokens(
        logits.astype(np.float32),
        temperatures.astype(np.float32).reshape((-1, 1)),
        top_ks.astype(np.int32).reshape((-1, 1)),
        top_ps.astype(np.float32).reshape((-1, 1)),
        min_ps.astype(np.float32).reshape((-1, 1)),
        uniform_u.astype(np.float32).reshape((-1, 1)),
        _unfiltered=unfiltered,
    ).astype(np.int32)


def _log_softmax_f32(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax in float32."""
    row_max = np.max(logits, axis=1, keepdims=True).astype(np.float32)
    shifted = (logits - row_max).astype(np.float32)
    log_sum_exp = np.log(
        np.maximum(np.sum(np.exp(shifted), axis=1, keepdims=True), np.float32(1e-20))
    ).astype(np.float32)
    return (shifted - log_sum_exp).astype(np.float32)


# ===================================================================
# Device entry points — greedy (top-k merge) and non-greedy (NKI sampler)
# ===================================================================


def lm_head_local_topk(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    rms_norm_eps: float,
    topk: int = 1,
    gather_hidden: bool = False,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rank-local LM-head top-k candidates for greedy TP merge."""
    from nkipy.core import tensor_apis

    logits = _compute_local_logits(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    topk_vals, topk_idx = tensor_apis.topk(logits, k=int(topk), axis=1)
    return topk_vals.astype(np.float32), topk_idx.astype(np.int32)


def lm_head_sample_tokens_with_logprobs(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
    *,
    rms_norm_eps: float,
    logprobs_k: int = 5,
    gather_hidden: bool = False,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    unfiltered: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Device-side token sampling + logprobs extraction.

    Same as ``lm_head_sample_tokens`` but additionally computes log_softmax
    over the gathered logits and extracts chosen_logprobs and top-k logprobs.

    Returns (next_token_ids, chosen_logprobs, topk_vals, topk_ids).
    """
    from nkipy.core import tensor_apis

    logits = _compute_gathered_logits(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )

    next_token_ids = _sample_from_gathered_logits(
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        uniform_u,
        unfiltered=unfiltered,
    ).reshape((-1,))

    log_probs = _log_softmax_f32(logits)

    k = min(int(logprobs_k), int(logits.shape[1]))  # logprobs_k is compile-time
    topk_vals, topk_ids = tensor_apis.topk(log_probs, k=k, axis=1)

    chosen_logprobs = (
        np.take_along_axis(
            log_probs,
            next_token_ids.reshape((-1, 1)),
            axis=1,
        )
        .reshape((-1,))
        .astype(np.float32)
    )

    return (
        next_token_ids,
        chosen_logprobs,
        topk_vals.astype(np.float32),
        topk_ids.astype(np.int32),
    )


def lm_head_sample_tokens(
    hidden: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    temperatures: np.ndarray,
    top_ks: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    uniform_u: np.ndarray,
    *,
    rms_norm_eps: float,
    gather_hidden: bool = False,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    unfiltered: bool = False,
) -> np.ndarray:
    """Device-side token sampling after TP all-gather of LM-head logits.

    ``unfiltered=True`` selects the 3-pass kernel that skips the threshold
    search.  The tensor interface is identical for both paths.
    """
    logits = _compute_gathered_logits(
        hidden,
        final_norm,
        lm_head,
        last_token_indices,
        rms_norm_eps=rms_norm_eps,
        gather_hidden=gather_hidden,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    return _sample_from_gathered_logits(
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        uniform_u,
        unfiltered=unfiltered,
    )
