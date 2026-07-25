"""CPU reference MoE dispatch (numpy).

Used by ``forward_cpu()`` on MoE eager executors. Not traceable / not
device-compilable — this is a slow reference implementation for accuracy
checks.

Two variants are provided, matching the production fused kernels:

  * ``cpu_moe_dispatch_swish`` — Qwen3 MoE: expert gate_up then SiLU(gate)*up
    then down; weighted by per-token routing affinities; add to residual.
  * ``cpu_moe_dispatch_swiglu_oai`` — GPT-OSS: swigluoai_and_mul with
    (alpha=1.702, clamp) and ``(up+1)*gate`` pattern; bias on up pre-shifted
    by +1 (matches ``gate_up_bias_plus1_T``); per-expert down bias added and
    zeroed outside the rank's hidden shard; weighted by affinities; add
    to residual.
"""

from __future__ import annotations

import numpy as np


def _require_shape(
    name: str, actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if actual != expected:
        raise ValueError(f"{name} shape must be {expected}, got {actual}")


def _require_down_bias_shape(
    actual: tuple[int, ...],
    *,
    num_experts: int,
    hidden_size: int,
) -> None:
    if len(actual) < 2 or actual[0] != num_experts or actual[-1] != hidden_size:
        raise ValueError(
            "down_bias_bc shape must be "
            f"({num_experts}, ..., {hidden_size}), got {actual}"
        )


def softmax_topk_masked(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Top-k expert routing with masked softmax.

    Returns ``affinities`` of shape ``(T, num_experts)`` where only the
    top-k entries per token are non-zero (softmax over top-k logits).
    Mirrors the device router: non-topk experts receive -inf mask, then
    softmax is applied row-wise.
    """
    T = logits.shape[0]
    logits_f32 = logits.astype(np.float32)
    topk_idx = np.argsort(-logits_f32, axis=-1)[:, :top_k]
    masked = np.full_like(logits_f32, -1e5, dtype=np.float32)
    rows = np.arange(T)[:, None]
    masked[rows, topk_idx] = logits_f32[rows, topk_idx]
    masked = masked - np.max(masked, axis=-1, keepdims=True)
    exp_x = np.exp(masked)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float32)


def cpu_moe_dispatch_swish(
    normed: np.ndarray,
    affinities: np.ndarray,
    gup_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """Qwen3-style MoE: per-expert SiLU(gate) * up -> down, weighted by affinities.

    Args:
        normed: [T, H] post-norm hidden states.
        affinities: [T, E_local] routing weights (non-topk entries are ~0).
        gup_w: [E_local, H, 2, I_local] gate/up interleaved weights (fp8 ok).
        down_w: [E_local, I_local, H] down projection weights (fp8 ok).

    Returns:
        [T, H] expert contribution (before residual add).
    """
    T, H = normed.shape
    E_local = gup_w.shape[0]
    I_local = gup_w.shape[-1]
    _require_shape("gup_w", gup_w.shape, (E_local, H, 2, I_local))
    _require_shape("down_w", down_w.shape, (E_local, I_local, H))
    _require_shape("affinities", affinities.shape, (T, E_local))

    hidden_f32 = normed.astype(np.float32)
    out = np.zeros((T, H), dtype=np.float32)
    for e in range(E_local):
        gate_w = gup_w[e, :, 0, :].astype(np.float32)  # [H, I_local]
        up_w = gup_w[e, :, 1, :].astype(np.float32)
        dn_w = down_w[e].astype(np.float32)  # [I_local, H]

        gate = hidden_f32 @ gate_w  # [T, I_local]
        up = hidden_f32 @ up_w
        silu = gate * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate)))
        expert_out = (silu * up) @ dn_w  # [T, H]
        out += affinities[:, e : e + 1].astype(np.float32) * expert_out
    return out.astype(normed.dtype)


def cpu_moe_dispatch_swiglu_oai(
    normed: np.ndarray,
    affinities: np.ndarray,
    gup_w: np.ndarray,
    gup_bias_plus1: np.ndarray,
    down_w: np.ndarray,
    down_bias_bc: np.ndarray,
    *,
    alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> np.ndarray:
    """GPT-OSS swigluoai_and_mul MoE dispatch.

    Matches the NKI kernel's gate/up clamping + bias:
        gate = min(gate_proj + bias_gate, limit)
        up   = clamp(up_proj + bias_up + 1, -limit+1, limit+1) ≈ matches the
               kernel's ``(up + 1) * glu`` form where up is clamped to
               [-limit, limit] (kernel stores bias[...,1] += 1, then clamps
               to [-6, 8] = [-limit+1, limit+1]).
        glu  = gate * sigmoid(gate * alpha)
        out  = up * glu  (i.e. (u+1) after shift)

    Args:
        normed: [T, H] post-norm hidden.
        affinities: [T, E_local] routing weights.
        gup_w: [E_local, H, 2, I_local] interleaved gate/up.
        gup_bias_plus1: [E_local, I_local, 2] — bias[...,1] has +1 baked in.
        down_w: [E_local, I_local, H].
        down_bias_bc: [E_local, BLOCK_SIZE, H] — per-expert down bias
            broadcasted over block slots (all rows identical after the +/-
            zero-out for TP).

    Returns:
        [T, H] expert contribution (before residual add).
    """
    T, H = normed.shape
    E_local = gup_w.shape[0]
    I_local = gup_w.shape[-1]
    _require_shape("gup_w", gup_w.shape, (E_local, H, 2, I_local))
    _require_shape("gup_bias_plus1", gup_bias_plus1.shape, (E_local, I_local, 2))
    _require_shape("down_w", down_w.shape, (E_local, I_local, H))
    _require_down_bias_shape(
        down_bias_bc.shape,
        num_experts=E_local,
        hidden_size=H,
    )
    _require_shape("affinities", affinities.shape, (T, E_local))

    hidden_f32 = normed.astype(np.float32)
    out = np.zeros((T, H), dtype=np.float32)
    lim = np.float32(swiglu_limit)
    a = np.float32(alpha)
    up_hi = lim + np.float32(1.0)  # up bias has +1 pre-shift; kernel clamps to [-6, 8]
    up_lo = -lim + np.float32(1.0)
    for e in range(E_local):
        gate_w = gup_w[e, :, 0, :].astype(np.float32)  # [H, I_local]
        up_w = gup_w[e, :, 1, :].astype(np.float32)
        b_gate = gup_bias_plus1[e, :, 0].astype(np.float32)  # [I_local]
        b_up = gup_bias_plus1[e, :, 1].astype(np.float32)
        dn_w = down_w[e].astype(np.float32)  # [I_local, H]
        dn_b = down_bias_bc[e, 0, :].astype(np.float32)  # [H] (rows identical)

        gate_pre = hidden_f32 @ gate_w + b_gate  # [T, I_local]
        gate = np.minimum(gate_pre, lim)
        up_pre = hidden_f32 @ up_w + b_up
        up = np.clip(up_pre, up_lo, up_hi)
        glu = gate / (np.float32(1.0) + np.exp(-gate * a))  # gate * sigmoid(gate*alpha)
        expert_out = (up * glu) @ dn_w + dn_b  # [T, H]
        out += affinities[:, e : e + 1].astype(np.float32) * expert_out
    return out.astype(normed.dtype)
