"""Shared runtime helpers for model-owned DSV4 attention operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)


def _attention_qkv_quant(
    fns: Dsv4GraphFns,
    attn: Any,
    x: np.ndarray,
    freqs_cis: np.ndarray | None,
    *,
    freqs_cos: Any | None = None,
    freqs_sin: Any | None = None,
    positions: Any | None = None,
    block_size: int = 64,
    fp8_max: float = 240.0,
    q_softmax_scale: float | None = None,
    q_token_bucket: int | None = None,
    kv_token_bucket: int | None = None,
    return_qr: bool = True,
    output_tensors: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Fused QKV projection + RoPE + KV-non-rope FP8 qdq."""
    output_kwargs = (
        {"_nkipy_output_tensors": output_tensors} if output_tensors is not None else {}
    )
    table_fn = fns.get("attention_qkv_quant_from_freq_table")
    fuse_q_scale = (
        bool(fns.get("_attention_qkv_table_fuses_q_scale", False))
        and q_softmax_scale is not None
        and q_token_bucket is not None
    )
    fuse_kv_flat = (
        fuse_q_scale
        and bool(fns.get("_attention_qkv_table_outputs_flat_kv", False))
        and kv_token_bucket is not None
    )
    if (
        callable(table_fn)
        and freqs_cos is not None
        and freqs_sin is not None
        and positions is not None
    ):
        return table_fn(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            freqs_cos,
            freqs_sin,
            positions,
            n_heads=int(attn.n_heads),
            head_dim=int(attn.head_dim),
            rope_head_dim=int(attn.rope_head_dim),
            eps=float(attn.eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            **({"return_qr": False} if not bool(return_qr) else {}),
            **(
                {
                    "q_softmax_scale": float(q_softmax_scale),
                    "q_token_bucket": int(q_token_bucket),
                    **(
                        {"kv_token_bucket": int(kv_token_bucket)}
                        if fuse_kv_flat
                        else {}
                    ),
                }
                if fuse_q_scale
                else {}
            ),
            **output_kwargs,
        )
    if freqs_cis is None:
        raise RuntimeError(
            "DSV4 QKV requires either freqs_cis or device frequency tables"
        )
    cos = freqs_cis.real.astype(np.float32)
    sin = freqs_cis.imag.astype(np.float32)
    return fns["attention_qkv_quant"](
        x,
        attn.wq_a,
        attn.q_norm,
        attn.wq_b,
        attn.wkv,
        attn.kv_norm,
        cos,
        sin,
        n_heads=int(attn.n_heads),
        head_dim=int(attn.head_dim),
        rope_head_dim=int(attn.rope_head_dim),
        eps=float(attn.eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        **output_kwargs,
    )


def _attention_out_flat(
    fns: Dsv4GraphFns,
    attn: Any,
    out: Any,
    *,
    output_tensors: dict[str, Any] | None = None,
) -> Any:
    groups = list(getattr(attn, "tp_replica_groups", ()) or ())
    if not groups:
        groups = [list(range(int(getattr(attn, "tp_degree", 1))))]
    else:
        groups = [list(group) for group in groups]
    output_kwargs = (
        {"_nkipy_output_tensors": output_tensors} if output_tensors is not None else {}
    )
    return fns["attention_out_flat"](
        out,
        attn.wo_a,
        attn.wo_b,
        n_groups=int(attn.n_groups),
        tp_degree=int(getattr(attn, "tp_degree", 1)),
        tp_replica_groups=tuple(tuple(int(rank) for rank in group) for group in groups),
        **output_kwargs,
    )


def _attention_out_flat_hidden(
    fns: Dsv4GraphFns,
    attn: Any,
    out: Any,
    *,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    output_tensors: dict[str, Any] | None = None,
) -> Any:
    groups = list(getattr(attn, "tp_replica_groups", ()) or ())
    if not groups:
        groups = [list(range(int(getattr(attn, "tp_degree", 1))))]
    else:
        groups = [list(group) for group in groups]
    output_kwargs = (
        {"_nkipy_output_tensors": output_tensors} if output_tensors is not None else {}
    )
    return fns["attention_out_flat_hidden"](
        out,
        attn.wo_a,
        attn.wo_b,
        n_groups=int(attn.n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(getattr(attn, "tp_degree", 1)),
        tp_replica_groups=tuple(tuple(int(rank) for rank in group) for group in groups),
        **output_kwargs,
    )


def _attention_inverse_rope_out_flat_hidden_from_freq_table(
    fns: Dsv4GraphFns,
    attn: Any,
    out: Any,
    freqs_cos: Any,
    freqs_sin: Any,
    positions: Any,
    *,
    rope_head_dim: int,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    output_tensors: dict[str, Any] | None = None,
) -> Any:
    groups = list(getattr(attn, "tp_replica_groups", ()) or ())
    if not groups:
        groups = [list(range(int(getattr(attn, "tp_degree", 1))))]
    else:
        groups = [list(group) for group in groups]
    output_kwargs = (
        {"_nkipy_output_tensors": output_tensors} if output_tensors is not None else {}
    )
    return fns["attention_inverse_rope_out_flat_hidden_from_freq_table"](
        out,
        attn.wo_a,
        attn.wo_b,
        freqs_cos,
        freqs_sin,
        positions,
        rope_head_dim=int(rope_head_dim),
        n_groups=int(attn.n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(getattr(attn, "tp_degree", 1)),
        tp_replica_groups=tuple(tuple(int(rank) for rank in group) for group in groups),
        **output_kwargs,
    )


def _padded_positions_for_flat_rows(
    metadata: Any | None,
    *,
    start_pos: int,
    bsz: int,
    seqlen: int,
    rows: int,
) -> np.ndarray:
    if metadata is not None and getattr(metadata, "positions", None) is not None:
        positions = np.asarray(metadata.positions, dtype=np.int64).reshape(-1)
    else:
        positions = (
            np.arange(int(seqlen), dtype=np.int64)[None, :]
            + np.arange(int(bsz), dtype=np.int64)[:, None] * 0
            + int(start_pos)
        ).reshape(-1)
    target = int(rows)
    if positions.shape[0] < target:
        pad = np.full((target - positions.shape[0],), int(start_pos), dtype=np.int64)
        positions = np.concatenate((positions, pad), axis=0)
    elif positions.shape[0] > target:
        positions = positions[:target]
    return np.ascontiguousarray(positions.astype(np.int32))


def _device_positions_for_flat_rows(
    backend: Any | None,
    *,
    rows: int,
) -> Any | None:
    """Return backend-owned device positions when they cover ``rows``."""
    step_inputs = getattr(backend, "step_inputs", None) if backend is not None else None
    positions = (
        getattr(step_inputs, "positions", None) if step_inputs is not None else None
    )
    if positions is None:
        return None
    shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
    if not shape or shape[0] < int(rows):
        return None
    if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
        return positions
    return None


def _inverse_rope_tail_flat(
    fns: Dsv4GraphFns,
    out: Any,
    freqs_cis: np.ndarray | None,
    *,
    rope_head_dim: int,
    freqs_cos: Any | None = None,
    freqs_sin: Any | None = None,
    positions: Any | None = None,
    output_tensors: dict[str, Any] | None = None,
) -> Any:
    output_kwargs = (
        {"_nkipy_output_tensors": output_tensors} if output_tensors is not None else {}
    )
    table_fn = fns.get("inverse_rope_tail_flat_from_freq_table")
    if (
        callable(table_fn)
        and freqs_cos is not None
        and freqs_sin is not None
        and positions is not None
    ):
        return table_fn(
            out,
            freqs_cos,
            freqs_sin,
            positions,
            rope_head_dim=int(rope_head_dim),
            **output_kwargs,
        )
    if freqs_cis is None:
        raise RuntimeError(
            "DSV4 inverse RoPE requires either freqs_cis or device frequency tables"
        )
    cos = freqs_cis.real.astype(np.float32)
    sin = freqs_cis.imag.astype(np.float32)
    return fns["inverse_rope_tail_flat"](
        out,
        cos,
        sin,
        rope_head_dim=int(rope_head_dim),
        **output_kwargs,
    )


def _state_owner_ids_from_batch(
    *,
    bsz: int,
    seqlen: int,
    owner_ids: np.ndarray | None,
) -> np.ndarray:
    """Return flattened per-token DSV4 state owners for rectangular input."""

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    if owner_ids is None:
        return np.repeat(np.arange(bsz_i, dtype=np.int32), seqlen_i)
    owners = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
    expected = bsz_i * seqlen_i
    if owners.shape[0] < expected:
        fill = np.int32(0 if owners.shape[0] == 0 else owners[-1])
        padded = np.full((expected,), fill, dtype=np.int32)
        padded[: owners.shape[0]] = owners
        return padded
    if owners.shape != (expected,):
        raise RuntimeError(
            "DSV4 state_owner_ids must be flattened [bsz * seqlen] for "
            f"the current sampled path, got {owners.shape}, expected ({expected},)"
        )
    return owners


def _state_owner_ids_from_metadata(
    metadata: Any | None,
    *,
    bsz: int,
    seqlen: int,
) -> np.ndarray:
    owners = (
        getattr(metadata, "state_owner_ids", None) if metadata is not None else None
    )
    return _state_owner_ids_from_batch(
        bsz=bsz,
        seqlen=seqlen,
        owner_ids=owners,
    )


def _decode_positions_1d_alias(value: Any | None, *, bsz: int) -> Any | None:
    """Canonicalize decode position DeviceTensor aliases to shape ``(bsz,)``."""

    if value is None:
        return None
    bsz_i = int(bsz)
    target_shape = (bsz_i,)
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    if shape == target_shape:
        return value
    if shape != (bsz_i, 1):
        return value
    return _alias_device_value_shape(value, target_shape)


def _decode_positions_1d_array(positions: Any, *, bsz: int) -> np.ndarray:
    return np.asarray(positions[: int(bsz)], dtype=np.int32).reshape(int(bsz))


__all__ = [
    "_attention_inverse_rope_out_flat_hidden_from_freq_table",
    "_attention_out_flat",
    "_attention_out_flat_hidden",
    "_attention_qkv_quant",
    "_decode_positions_1d_alias",
    "_decode_positions_1d_array",
    "_device_positions_for_flat_rows",
    "_inverse_rope_tail_flat",
    "_padded_positions_for_flat_rows",
    "_state_owner_ids_from_batch",
    "_state_owner_ids_from_metadata",
]
