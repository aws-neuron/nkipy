"""Attention graph execution helpers for the DSV4 NEFF runtime."""

from __future__ import annotations

import time
from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.execution_capabilities import (
    Dsv4ExecutionCapabilities,
)
from nkipy_serving.models.deepseek_v4.graph_types import (
    Dsv4GraphFns,
    Dsv4SampledForwardOptions,
    _sampled_warmup_trace,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _attention_inverse_rope_out_flat_hidden_from_freq_table,
    _attention_out_flat,
    _attention_out_flat_hidden,
    _device_positions_for_flat_rows,
    _inverse_rope_tail_flat,
    _padded_positions_for_flat_rows,
    _state_owner_ids_from_batch,
    _state_owner_ids_from_metadata,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.compressor import (
    Dsv4DeferredIndexerState,
    Dsv4DeferredSwaMirror,
    _run_compressor,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.indexer import (
    _run_indexer,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants import (
    _run_compressed_attention_qkv,
    build_compressed_attention_qkv_setup,
)
from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor_if_changed as _overwrite_device_tensor_if_changed,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    mirror_compressor_input_to_device_state as _mirror_compressor_input_to_device_state,
)
from nkipy_serving.ops.deepseek_v4.swa_state import (
    mirror_swa_kv_to_device_cache as _mirror_swa_kv_to_device_cache,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)


def _uniform_prefill_real_seqlen_from_metadata(
    metadata: Any | None,
    *,
    start_pos: int,
    bsz: int,
    seqlen: int,
) -> int | None:
    if int(start_pos) != 0 or metadata is None:
        return None
    for attr in ("positions", "state_owner_ids"):
        value = getattr(metadata, attr, None)
        if value is None:
            continue
        rows = int(np.asarray(value).reshape(-1).shape[0])
        if rows <= 0 or rows >= int(bsz) * int(seqlen) or rows % int(bsz) != 0:
            continue
        real = rows // int(bsz)
        if 0 < real < int(seqlen):
            return real

    base = getattr(metadata, "base", metadata)
    qsl = getattr(base, "query_start_loc", None) if base is not None else None
    if qsl is not None:
        query_start_loc = np.asarray(qsl, dtype=np.int64).reshape(-1)
        if query_start_loc.shape[0] >= int(bsz) + 1:
            q_lens = query_start_loc[1 : int(bsz) + 1] - query_start_loc[: int(bsz)]
            if q_lens.size and np.all(q_lens == q_lens[0]):
                real = int(q_lens[0])
                if 0 < real < int(seqlen):
                    return real
                return None

    total = getattr(base, "total_tokens", None) if base is not None else None
    if total is None:
        return None
    total_i = int(total)
    if total_i <= 0 or total_i % int(bsz) != 0:
        return None
    real = total_i // int(bsz)
    if 0 < real < int(seqlen):
        return real
    return None


_SWA_SHORT_MIRROR_BUCKET = 32


def _swa_owner_window_write_bucket_rows(
    *,
    live_rows: int,
    window_size: int,
    available_rows: int | None = None,
) -> int:
    live_i = int(live_rows)
    win_i = int(window_size)
    if live_i <= 0:
        return 0
    if available_rows is not None:
        available_i = int(available_rows)
        if available_i >= live_i:
            return available_i if available_i <= win_i else win_i
    if live_i <= int(_SWA_SHORT_MIRROR_BUCKET):
        return min(max(int(_SWA_SHORT_MIRROR_BUCKET), live_i), win_i)
    return min(max(live_i, win_i), win_i)


def _pad_owner_ids_for_bucket(owner_ids: Any, *, rows: int) -> np.ndarray:
    owners = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
    rows_i = int(rows)
    if owners.shape[0] >= rows_i:
        return np.ascontiguousarray(owners[:rows_i])
    fill = np.int32(0 if owners.shape[0] == 0 else owners[-1])
    out = np.full((rows_i,), fill, dtype=np.int32)
    out[: owners.shape[0]] = owners
    return out


def run_dsv4_attention_with_backend(
    fns: Dsv4GraphFns,
    attn: Any,
    x: np.ndarray,
    metadata: Any,
    *,
    layer_id: int,
    backend: Any,
    attention_output: Any | None = None,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None = None,
    attention_profile: Callable[[str, float, dict[str, Any]], None] | None = None,
    attention_hidden_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """SWA-only layer (``compress_ratio == 0``) via the paged scheduler."""

    def _profile_attention(stage: str, elapsed_s: float, **fields: Any) -> None:
        if attention_profile is None:
            return
        attention_profile(str(stage), float(elapsed_s), fields)

    caps = Dsv4ExecutionCapabilities.from_graph_fns(fns)
    setup_t0 = time.perf_counter()
    bsz, seqlen, _ = x.shape
    pos0 = int(metadata.positions[0]) if metadata.positions is not None else 0
    n_tokens = int(bsz) * int(seqlen)
    active_bucket = int(getattr(backend, "active_bucket", backend.token_bucket))
    qkv_positions = _padded_positions_for_flat_rows(
        metadata,
        start_pos=pos0,
        bsz=bsz,
        seqlen=seqlen,
        rows=n_tokens,
    )
    qkv_positions_input = _device_positions_for_flat_rows(backend, rows=n_tokens)
    if qkv_positions_input is None:
        qkv_positions_input = qkv_positions
    table_qkv = callable(fns.get("attention_qkv_quant_from_freq_table"))
    freqs_cos = getattr(attn, "freqs_cos", None)
    freqs_sin = getattr(attn, "freqs_sin", None)
    rd = int(attn.rope_head_dim)
    n_heads = int(attn.n_heads)
    head_dim = int(attn.head_dim)
    qkv_fuses_q_scale = (
        bool(fns.get("_attention_qkv_table_fuses_q_scale", False))
        and table_qkv
        and freqs_cos is not None
        and freqs_sin is not None
        and qkv_positions_input is not None
    )
    if qkv_fuses_q_scale and caps.attention_shape_helpers_alias:
        qkv_rows = max(int(n_tokens), int(active_bucket))
        qkv_positions_input = _device_positions_for_flat_rows(backend, rows=qkv_rows)
        if qkv_positions_input is None:
            qkv_positions_input = _padded_positions_for_flat_rows(
                metadata,
                start_pos=pos0,
                bsz=bsz,
                seqlen=seqlen,
                rows=qkv_rows,
            )
    qkv_outputs_flat_kv = qkv_fuses_q_scale and bool(
        fns.get("_attention_qkv_table_outputs_flat_kv", False)
    )
    fused_qkv_write_fn = fns.get("attention_qkv_write_kv_cache_from_freq_table")
    qkv_writes_kv_cache = False
    qkv_write_cache = None
    qkv_write_slots = None
    if callable(fused_qkv_write_fn) and qkv_outputs_flat_kv:
        kv_cache_fn = getattr(backend, "kv_cache", None)
        step_inputs = getattr(backend, "step_inputs", None)
        slot_mapping = (
            getattr(step_inputs, "slot_mapping", None)
            if step_inputs is not None
            else None
        )
        if callable(kv_cache_fn):
            kv_cache = kv_cache_fn(layer_id)
            if hasattr(kv_cache, "tensor_ref") and hasattr(slot_mapping, "tensor_ref"):
                qkv_writes_kv_cache = True
                qkv_write_cache = kv_cache
                qkv_write_slots = slot_mapping
    _sampled_warmup_trace(
        "swa_attention start "
        f"layer={int(layer_id)} bsz={int(bsz)} seqlen={int(seqlen)} "
        f"tokens={int(n_tokens)} active_bucket={int(active_bucket)} "
        f"backend_bucket={int(getattr(backend, 'active_bucket', active_bucket))} "
        f"qkv_fuses_q_scale={bool(qkv_fuses_q_scale)} "
        f"qkv_outputs_flat_kv={bool(qkv_outputs_flat_kv)} "
        f"qkv_writes_kv_cache={bool(qkv_writes_kv_cache)} "
        f"fused_swa={backend.uses_fused_swa_this_step()}",
    )
    product_shape_aliases = caps.attention_shape_helpers_alias
    if caps.require_fused_attention_qkv_table and not qkv_fuses_q_scale:
        raise RuntimeError(
            "DSV4 product SWA attention requires fused QKV frequency-table "
            "path with q-scale"
        )
    if caps.require_flat_swa_kv and not qkv_outputs_flat_kv:
        raise RuntimeError(
            "DSV4 product SWA attention requires QKV table output to emit flat KV rows"
        )
    if caps.require_fused_swa_kv_write and not qkv_writes_kv_cache:
        raise RuntimeError(
            "DSV4 product SWA attention requires QKV table path to write the "
            "paged KV cache using backend device slot_mapping"
        )
    _profile_attention(
        "swa_setup",
        time.perf_counter() - setup_t0,
        bsz=int(bsz),
        seqlen=int(seqlen),
        start_pos=int(pos0),
        n_tokens=int(n_tokens),
        active_bucket=int(active_bucket),
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        qkv_writes_kv_cache=bool(qkv_writes_kv_cache),
        product_shape_aliases=bool(product_shape_aliases),
        fused_swa=backend.uses_fused_swa_this_step(),
        has_attention_scratch=attention_scratch is not None,
    )

    def _scratch(
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        if attention_scratch is None:
            return None
        return attention_scratch(
            kind,
            tuple(int(dim) for dim in shape),
            dtype,
        )

    def _output_kwargs(name: str, out: Any | None) -> dict[str, Any]:
        if out is None:
            return {}
        return {"_nkipy_output_tensors": {name: out}}

    qkv_outputs = None
    if attention_scratch is not None:
        q_output_kind = (
            "attention_q_scaled_t" if qkv_fuses_q_scale else "attention_qkv_q"
        )
        q_output_shape = (
            (active_bucket, head_dim, n_heads)
            if qkv_fuses_q_scale
            else (int(bsz), int(seqlen), n_heads, head_dim)
        )
        q_output_dtype = ml_dtypes.bfloat16 if qkv_fuses_q_scale else np.float32
        kv_output_kind = (
            "attention_qkv_kv_flat" if qkv_outputs_flat_kv else "attention_qkv_kv"
        )
        kv_output_shape = (
            (active_bucket, head_dim)
            if qkv_outputs_flat_kv
            else (int(bsz), int(seqlen), head_dim)
        )
        qkv_outputs = {
            "output0": _scratch(
                q_output_kind,
                q_output_shape,
                q_output_dtype,
            )
        }
        if not qkv_writes_kv_cache:
            qkv_outputs["output1"] = _scratch(
                kv_output_kind,
                kv_output_shape,
                np.float32,
            )
    qkv_t0 = time.perf_counter()
    # Product SWA attention always takes the fused QKV-table write path
    # (guaranteed by the ``_product_require_fused_swa_kv_write`` raise above);
    # the non-fused qkv_quant + standalone write_kv fallback is not part of any
    # live runtime.
    if not qkv_writes_kv_cache or not callable(fused_qkv_write_fn):
        raise RuntimeError("DSV4 qkv_writes_kv_cache requires a fused_qkv_write_fn")
    _sampled_warmup_trace(
        f"swa_attention qkv_write start layer={int(layer_id)}",
    )
    q_dev = fused_qkv_write_fn(
        x,
        attn.wq_a,
        attn.q_norm,
        attn.wq_b,
        attn.wkv,
        attn.kv_norm,
        qkv_write_cache,
        qkv_write_slots,
        freqs_cos,
        freqs_sin,
        qkv_positions_input,
        n_heads=int(attn.n_heads),
        head_dim=int(attn.head_dim),
        rope_head_dim=int(attn.rope_head_dim),
        eps=float(attn.eps),
        block_size=64,
        fp8_max=240.0,
        q_softmax_scale=float(attn.softmax_scale),
        q_token_bucket=active_bucket,
        kv_token_bucket=active_bucket,
        **({"_nkipy_output_tensors": qkv_outputs} if qkv_outputs is not None else {}),
    )
    _sampled_warmup_trace(
        f"swa_attention qkv_write done layer={int(layer_id)}",
    )
    _profile_attention(
        "swa_qkv_prepare",
        time.perf_counter() - qkv_t0,
        qkv_path="qkv_write_kv_cache",
        qkv_writes_kv_cache=True,
    )
    q_scale_t0 = time.perf_counter()
    # ``qkv_fuses_q_scale`` is required by the product graph, so the q-scale is
    # already folded into the QKV-table kernel output.
    q_scaled_t = q_dev
    _profile_attention(
        "swa_q_scale",
        time.perf_counter() - q_scale_t0,
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
    )
    output_prepare_t0 = time.perf_counter()
    # ``backend.attention`` with ``q_scaled_t`` requires a preallocated
    # ``output`` DeviceTensor of shape ``[N_q, h, d]`` fp32. Product mode
    # passes bucket-owned output; JIT mode allocates on demand.
    if attention_output is None:
        out_dev = _get_device_tensor_cls().from_numpy(
            np.zeros(
                (active_bucket, attn.n_heads, attn.head_dim),
                dtype=np.float32,
            ),
            name="paged_out",
        )
    else:
        output_shape = tuple(int(dim) for dim in getattr(attention_output, "shape", ()))
        expected_shape = (active_bucket, n_heads, head_dim)
        if output_shape != expected_shape:
            raise RuntimeError(
                "DSV4 SWA attention output scratch shape mismatch: "
                f"got {output_shape}, expected {expected_shape}"
            )
        out_dev = attention_output
    _profile_attention(
        "swa_output_prepare",
        time.perf_counter() - output_prepare_t0,
        preallocated=attention_output is not None,
        active_bucket=int(active_bucket),
    )
    sink_out = (
        None
        if product_shape_aliases
        else _scratch("attention_sink_2d", (1, n_heads), np.float32)
    )
    sink_kwargs = {} if product_shape_aliases else _output_kwargs("output0", sink_out)
    _sampled_warmup_trace(
        f"swa_attention sink start layer={int(layer_id)}",
    )
    sink_t0 = time.perf_counter()
    sink = fns["attention_sink_2d"](
        attn.attn_sink,
        n_heads=n_heads,
        **sink_kwargs,
    )
    _sampled_warmup_trace(
        f"swa_attention sink done layer={int(layer_id)}",
    )
    _profile_attention(
        "swa_sink",
        time.perf_counter() - sink_t0,
        product_shape_aliases=bool(product_shape_aliases),
    )
    _sampled_warmup_trace(
        f"swa_attention backend_attention start layer={int(layer_id)}",
    )
    attention_kernel_t0 = time.perf_counter()
    out_dev = backend.attention(
        layer_id,
        q_scaled_t=q_scaled_t,
        sink=sink,
        metadata=metadata,
        softmax_scale=float(attn.softmax_scale),
        output=out_dev,
    )
    _sampled_warmup_trace(
        f"swa_attention backend_attention done layer={int(layer_id)}",
    )
    _profile_attention(
        "swa_attention_kernel",
        time.perf_counter() - attention_kernel_t0,
        active_bucket=int(active_bucket),
    )
    postprocess_t0 = time.perf_counter()
    freq_positions = _padded_positions_for_flat_rows(
        metadata,
        start_pos=pos0,
        bsz=bsz,
        seqlen=seqlen,
        rows=active_bucket,
    )
    freq_positions_input = _device_positions_for_flat_rows(
        backend,
        rows=active_bucket,
    )
    if freq_positions_input is None:
        freq_positions_input = freq_positions
    freqs_cos = getattr(attn, "freqs_cos", None)
    freqs_sin = getattr(attn, "freqs_sin", None)
    fused_inverse_out = fns.get(
        "attention_inverse_rope_out_flat_hidden_from_freq_table"
    )
    if (
        attention_hidden_shape is not None
        and callable(fused_inverse_out)
        and freqs_cos is not None
        and freqs_sin is not None
        and freq_positions_input is not None
    ):
        owns_outputs = bool(
            fns.get("_attention_inverse_rope_out_flat_hidden_owns_outputs", False)
        )
        out_bsz, out_seqlen, out_hidden = (
            int(attention_hidden_shape[0]),
            int(attention_hidden_shape[1]),
            int(attention_hidden_shape[2]),
        )
        _sampled_warmup_trace(
            f"swa_attention inverse_out start layer={int(layer_id)}",
        )
        result = _attention_inverse_rope_out_flat_hidden_from_freq_table(
            fns,
            attn,
            out_dev,
            freqs_cos,
            freqs_sin,
            freq_positions_input,
            rope_head_dim=rd,
            bsz=out_bsz,
            seqlen=out_seqlen,
            hidden_size=out_hidden,
            output_tensors={
                "output0": _scratch(
                    "attention_inverse_rope_out_hidden",
                    (out_bsz, out_seqlen, out_hidden),
                    np.float32,
                )
            }
            if attention_scratch is not None and not owns_outputs
            else None,
        )
        _sampled_warmup_trace(
            f"swa_attention inverse_out done layer={int(layer_id)}",
        )
        _profile_attention(
            "swa_postprocess",
            time.perf_counter() - postprocess_t0,
            path="fused_inverse_rope_out_flat_hidden",
            output_shape=(out_bsz, out_seqlen, out_hidden),
            owns_outputs=bool(owns_outputs),
        )
        return result
    # The product graph always provides the fused inverse-RoPE/output-projection
    # kernel and a 3D ``attention_hidden_shape`` (see the early return above), so
    # the standalone inverse-rope-tail + out_flat/out_flat_hidden fallback is
    # unreachable in any live runtime.
    raise RuntimeError(
        "DSV4 product attention requires fused inverse-RoPE/output projection"
    )


def _run_compressed_attention_postprocess(
    *,
    fns: Dsv4GraphFns,
    attn: Any,
    metadata: Any | None,
    backend: Any,
    caps: Dsv4ExecutionCapabilities,
    out_flat: Any,
    start_pos: int,
    bsz: int,
    seqlen: int,
    active_bucket: int,
    n_heads: int,
    head_dim: int,
    rd: int,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None,
    attention_hidden_shape: tuple[int, int, int] | None,
    attention_postprocess_output: Any | None,
    profile_attention: Callable[..., None],
    layer_id: int | None = None,
) -> Any:
    """Run inverse-RoPE and output projection for compressed attention."""
    trace_layer = -1 if layer_id is None else int(layer_id)

    def _scratch(
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        if attention_scratch is None:
            return None
        return attention_scratch(
            kind,
            tuple(int(dim) for dim in shape),
            dtype,
        )

    postprocess_t0 = time.perf_counter()
    postprocess_rows = int(active_bucket)
    if attention_postprocess_output is not None:
        postprocess_shape = tuple(
            int(dim) for dim in getattr(attention_postprocess_output, "shape", ())
        )
        if (
            len(postprocess_shape) != 3
            or postprocess_shape[1] != int(n_heads)
            or postprocess_shape[2] != int(head_dim)
            or postprocess_shape[0] < int(active_bucket)
        ):
            raise RuntimeError(
                "DSV4 compressed attention postprocess output shape mismatch: "
                f"got {postprocess_shape}, expected rows >= {int(active_bucket)} "
                f"and tail {(int(n_heads), int(head_dim))}"
            )
        postprocess_rows = int(postprocess_shape[0])
    out_for_postprocess = (
        attention_postprocess_output
        if attention_postprocess_output is not None
        else out_flat
    )
    freq_positions = _padded_positions_for_flat_rows(
        metadata,
        start_pos=int(start_pos),
        bsz=bsz,
        seqlen=seqlen,
        rows=postprocess_rows,
    )
    freq_positions_input = _device_positions_for_flat_rows(
        backend,
        rows=postprocess_rows,
    )
    if freq_positions_input is None:
        freq_positions_input = freq_positions
    table_inverse_rope = callable(fns.get("inverse_rope_tail_flat_from_freq_table"))
    freqs_cos = getattr(attn, "freqs_cos", None)
    freqs_sin = getattr(attn, "freqs_sin", None)
    freqs_flat = (
        None
        if table_inverse_rope and freqs_cos is not None and freqs_sin is not None
        else attn.freqs_cis[freq_positions.astype(np.int64)]
    )
    fused_inverse_out = fns.get(
        "attention_inverse_rope_out_flat_hidden_from_freq_table"
    )
    if (
        attention_hidden_shape is not None
        and callable(fused_inverse_out)
        and freqs_cos is not None
        and freqs_sin is not None
        and freq_positions_input is not None
    ):
        owns_outputs = bool(
            fns.get("_attention_inverse_rope_out_flat_hidden_owns_outputs", False)
        )
        out_bsz, out_seqlen, out_hidden = (
            int(attention_hidden_shape[0]),
            int(attention_hidden_shape[1]),
            int(attention_hidden_shape[2]),
        )
        _sampled_warmup_trace(
            "compressed_attention postprocess fused_inverse_out start "
            f"layer={trace_layer} rows={int(postprocess_rows)} "
            f"hidden_shape={(out_bsz, out_seqlen, out_hidden)}",
        )
        result = _attention_inverse_rope_out_flat_hidden_from_freq_table(
            fns,
            attn,
            out_for_postprocess,
            freqs_cos,
            freqs_sin,
            freq_positions_input,
            rope_head_dim=rd,
            bsz=out_bsz,
            seqlen=out_seqlen,
            hidden_size=out_hidden,
            output_tensors={
                "output0": _scratch(
                    "attention_inverse_rope_out_hidden",
                    (out_bsz, out_seqlen, out_hidden),
                    np.float32,
                )
            }
            if attention_scratch is not None and not owns_outputs
            else None,
        )
        _sampled_warmup_trace(
            "compressed_attention postprocess fused_inverse_out done "
            f"layer={trace_layer}",
        )
        profile_attention(
            "postprocess",
            time.perf_counter() - postprocess_t0,
            path="fused_inverse_rope_out_flat_hidden",
            output_shape=(out_bsz, out_seqlen, out_hidden),
            owns_outputs=bool(owns_outputs),
        )
        return result
    if caps.require_fused_inverse_rope_out:
        raise RuntimeError(
            "DSV4 product compressed attention requires fused inverse-RoPE/"
            "output projection"
        )
    inverse_t0 = time.perf_counter()
    _sampled_warmup_trace(
        "compressed_attention postprocess inverse_rope start "
        f"layer={trace_layer} rows={int(postprocess_rows)}",
    )
    out_flat = _inverse_rope_tail_flat(
        fns,
        out_for_postprocess,
        freqs_flat,
        rope_head_dim=rd,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        positions=freq_positions_input,
        output_tensors={
            "output0": _scratch(
                "attention_inverse_rope_tail_flat",
                (active_bucket, n_heads, head_dim),
                np.float32,
            )
        }
        if attention_scratch is not None
        else None,
    )
    _sampled_warmup_trace(
        f"compressed_attention postprocess inverse_rope done layer={trace_layer}",
    )
    profile_attention(
        "postprocess_inverse_rope",
        time.perf_counter() - inverse_t0,
        postprocess_rows=int(postprocess_rows),
        table_inverse_rope=bool(table_inverse_rope),
    )
    hidden_size = int(getattr(attn.wo_b, "shape", (0, 0))[0])
    if attention_hidden_shape is not None and "attention_out_flat_hidden" in fns:
        owns_outputs = bool(fns.get("_attention_out_flat_hidden_owns_outputs", False))
        out_bsz, out_seqlen, out_hidden = (
            int(attention_hidden_shape[0]),
            int(attention_hidden_shape[1]),
            int(attention_hidden_shape[2]),
        )
        _sampled_warmup_trace(
            "compressed_attention postprocess out_flat_hidden start "
            f"layer={trace_layer} hidden_shape={(out_bsz, out_seqlen, out_hidden)}",
        )
        result = _attention_out_flat_hidden(
            fns,
            attn,
            out_flat,
            bsz=out_bsz,
            seqlen=out_seqlen,
            hidden_size=out_hidden,
            output_tensors={
                "output0": _scratch(
                    "attention_out_hidden",
                    (out_bsz, out_seqlen, out_hidden),
                    np.float32,
                )
            }
            if attention_scratch is not None and not owns_outputs
            else None,
        )
        _sampled_warmup_trace(
            "compressed_attention postprocess out_flat_hidden done "
            f"layer={trace_layer}",
        )
        profile_attention(
            "postprocess",
            time.perf_counter() - postprocess_t0,
            path="out_flat_hidden",
            output_shape=(out_bsz, out_seqlen, out_hidden),
            owns_outputs=bool(owns_outputs),
        )
        return result
    _sampled_warmup_trace(
        "compressed_attention postprocess out_flat start "
        f"layer={trace_layer} active_bucket={int(active_bucket)}",
    )
    result = _attention_out_flat(
        fns,
        attn,
        out_flat,
        output_tensors={
            "output0": _scratch(
                "attention_out_flat",
                (active_bucket, hidden_size),
                np.float32,
            )
        }
        if attention_scratch is not None
        else None,
    )
    _sampled_warmup_trace(
        f"compressed_attention postprocess out_flat done layer={trace_layer}",
    )
    profile_attention(
        "postprocess",
        time.perf_counter() - postprocess_t0,
        path="out_flat",
        active_bucket=int(active_bucket),
        hidden_size=int(hidden_size),
    )
    return result


def run_dsv4_attention(
    fns: Dsv4GraphFns,
    attn: Any,
    x: np.ndarray,
    start_pos: int,
    *,
    options: Dsv4SampledForwardOptions,
    build_dir: str | None,
    backend: Any,
    device_layer_state: Any,
    metadata: Any | None = None,
    token_bucket: int | None = None,
    attention_output: Any | None = None,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None = None,
    attention_profile: Callable[[str, float, dict[str, Any]], None] | None = None,
    owner_ids_host: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    primary_owner_ids_host: np.ndarray | None = None,
    primary_owner_ids_dev: Any | None = None,
    attention_hidden_shape: tuple[int, int, int] | None = None,
    attention_postprocess_output: Any | None = None,
    layer_id: int | None = None,
) -> np.ndarray:
    """Compressed-layer attention using two-source device attention.

    Requires a backend that exposes ``attention_ephemeral_paged_two_source``
    and a ``device_layer_state`` with SWA ring cache + compressor state.
    Primary reads come from the device SWA cache (or fresh KV for prefill
    fitting one window); secondary reads come from the persistent
    compressed-KV cache.
    """
    if not hasattr(backend, "attention_ephemeral_paged_two_source"):
        raise RuntimeError(
            "run_dsv4_attention requires a backend with "
            "attention_ephemeral_paged_two_source"
        )
    if device_layer_state is None or device_layer_state.compressor is None:
        raise RuntimeError(
            "run_dsv4_attention requires device_layer_state with compressor"
        )

    def _profile_attention(stage: str, elapsed_s: float, **fields: Any) -> None:
        if attention_profile is None:
            return
        attention_profile(str(stage), float(elapsed_s), fields)

    caps = Dsv4ExecutionCapabilities.from_graph_fns(fns)
    setup_t0 = time.perf_counter()

    bsz, seqlen, _ = x.shape
    trace_layer = -1 if layer_id is None else int(layer_id)
    owner_ids = _state_owner_ids_from_metadata(
        metadata,
        bsz=bsz,
        seqlen=seqlen,
    )
    real_prefill_seqlen = _uniform_prefill_real_seqlen_from_metadata(
        metadata,
        start_pos=int(start_pos),
        bsz=int(bsz),
        seqlen=int(seqlen),
    )
    n_tokens = int(bsz * seqlen)
    if token_bucket is None:
        active_bucket = max(
            n_tokens,
            int(getattr(backend, "active_bucket", n_tokens)),
        )
    else:
        active_bucket = max(n_tokens, int(token_bucket))
    qkv_positions = _padded_positions_for_flat_rows(
        metadata,
        start_pos=int(start_pos),
        bsz=bsz,
        seqlen=seqlen,
        rows=n_tokens,
    )
    device_token_positions = _device_positions_for_flat_rows(backend, rows=n_tokens)
    qkv_positions_input = device_token_positions
    if qkv_positions_input is None:
        qkv_positions_input = qkv_positions
    table_qkv = callable(fns.get("attention_qkv_quant_from_freq_table"))
    freqs_cos = getattr(attn, "freqs_cos", None)
    freqs_sin = getattr(attn, "freqs_sin", None)
    freqs = (
        None
        if table_qkv and freqs_cos is not None and freqs_sin is not None
        else attn.freqs_cis[qkv_positions.astype(np.int64)].reshape(
            int(bsz),
            int(seqlen),
            -1,
        )[0 if int(bsz) == 1 else slice(None)]
    )
    win = int(attn.window_size)
    ratio = int(attn.compress_ratio)
    rd = int(attn.rope_head_dim)
    n_heads = int(attn.n_heads)
    head_dim = int(attn.head_dim)
    qkv_fuses_q_scale = (
        bool(fns.get("_attention_qkv_table_fuses_q_scale", False))
        and table_qkv
        and freqs_cos is not None
        and freqs_sin is not None
        and qkv_positions_input is not None
    )
    product_shape_aliases = caps.attention_shape_helpers_alias
    _sampled_warmup_trace(
        "compressed_attention start "
        f"layer={trace_layer} bsz={int(bsz)} seqlen={int(seqlen)} "
        f"start_pos={int(start_pos)} n_tokens={int(n_tokens)} "
        f"active_bucket={int(active_bucket)} ratio={int(ratio)} "
        f"qkv_fuses_q_scale={bool(qkv_fuses_q_scale)}",
    )
    if caps.require_fused_attention_qkv_table and not qkv_fuses_q_scale:
        raise RuntimeError(
            "DSV4 product compressed attention requires fused QKV "
            "frequency-table path with q-scale"
        )
    q_low_dim = int(
        getattr(
            getattr(attn, "q_norm", None),
            "shape",
            getattr(getattr(attn, "wq_a", None), "shape", (0, 0))[:1],
        )[0]
    )
    prefill_device_primary = bool(start_pos == 0 and seqlen <= win)
    topk_t_dev = None
    mask_dev = None

    def _scratch(
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        if attention_scratch is None:
            return None
        return attention_scratch(
            kind,
            tuple(int(dim) for dim in shape),
            dtype,
        )

    def _output_kwargs(name: str, out: Any | None) -> dict[str, Any]:
        if out is None:
            return {}
        return {"_nkipy_output_tensors": {name: out}}

    def _can_defer_indexer_state_to_main_swa(
        idx_comp_kv_dev: Any,
        idx_comp_score_dev: Any,
    ) -> bool:
        return bool(
            int(start_pos) != 0
            and int(seqlen) == 1
            and int(ratio) > 0
            and (int(start_pos) + 1) % int(ratio) != 0
            and not isinstance(idx_comp_kv_dev, np.ndarray)
            and not isinstance(idx_comp_score_dev, np.ndarray)
            and hasattr(device_layer_state.indexer.kv_score_state, "tensor_ref")
        )

    def _defer_indexer_state_to_main_swa(
        idx_comp_kv_dev: Any,
        idx_comp_score_dev: Any,
    ) -> None:
        nonlocal deferred_indexer_state
        deferred_indexer_state = Dsv4DeferredIndexerState(
            compressor=indexer_obj.compressor,
            kv=idx_comp_kv_dev,
            score=idx_comp_score_dev,
            device_state=device_layer_state.indexer,
        )

    _sampled_warmup_trace(
        f"compressed_attention qkv_setup start layer={trace_layer}",
    )
    qkv_setup = build_compressed_attention_qkv_setup(
        fns=fns,
        x=x,
        attn=attn,
        options=options,
        device_layer_state=device_layer_state,
        owner_ids=owner_ids,
        qkv_positions=qkv_positions,
        qkv_positions_input=qkv_positions_input,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        product_shape_aliases=bool(product_shape_aliases),
        attention_scratch=attention_scratch,
        active_bucket=int(active_bucket),
        q_low_dim=int(q_low_dim),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        bsz=int(bsz),
        seqlen=int(seqlen),
        start_pos=int(start_pos),
        win=int(win),
        ratio=int(ratio),
        prefill_device_primary=bool(prefill_device_primary),
    )
    qkv_outputs_flat_kv = bool(qkv_setup.qkv_outputs_flat_kv)
    qkv_path = qkv_setup.path
    indexer_obj = qkv_setup.indexer_obj
    _sampled_warmup_trace(
        "compressed_attention qkv_setup done "
        f"layer={trace_layer} path={qkv_path} "
        f"flat_kv={bool(qkv_outputs_flat_kv)}",
    )

    _profile_attention(
        "setup",
        time.perf_counter() - setup_t0,
        bsz=int(bsz),
        seqlen=int(seqlen),
        start_pos=int(start_pos),
        n_tokens=int(n_tokens),
        active_bucket=int(active_bucket),
        ratio=int(ratio),
        qkv_path=qkv_path,
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        product_shape_aliases=bool(product_shape_aliases),
        prefill_device_primary=bool(prefill_device_primary),
        has_attention_scratch=attention_scratch is not None,
    )

    qkv_t0 = time.perf_counter()
    _sampled_warmup_trace(
        f"compressed_attention qkv_run start layer={trace_layer} path={qkv_path}",
    )
    qkv_result = _run_compressed_attention_qkv(
        qkv_setup=qkv_setup,
        fns=fns,
        x=x,
        attn=attn,
        build_dir=build_dir,
        device_layer_state=device_layer_state,
        owner_ids=owner_ids,
        owner_ids_dev=owner_ids_dev,
        device_token_positions=device_token_positions,
        qkv_positions=qkv_positions,
        qkv_positions_input=qkv_positions_input,
        freqs=freqs,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        active_bucket=int(active_bucket),
        win=int(win),
        ratio=int(ratio),
        start_pos=int(start_pos),
        bsz=int(bsz),
        seqlen=int(seqlen),
        prefill_device_primary=bool(prefill_device_primary),
        attention_scratch=attention_scratch,
    )
    q_dev = qkv_result.q_dev
    kv_dev = qkv_result.kv_dev
    qr_dev = qkv_result.qr_dev
    topk_t_dev = qkv_result.topk_t_dev
    mask_dev = qkv_result.mask_dev
    precomputed_compressor_kv_score = qkv_result.precomputed_compressor_kv_score
    precomputed_compressor_prefill_scatter_rows = (
        qkv_result.precomputed_compressor_prefill_scatter_rows
    )
    precomputed_compressor_decode_scatter_rows = (
        qkv_result.precomputed_compressor_decode_scatter_rows
    )
    indexer_precomputed_compressor_kv_score = (
        qkv_result.indexer_precomputed_compressor_kv_score
    )
    indexer_precomputed_compressor_decode_scatter_rows = (
        qkv_result.indexer_precomputed_compressor_decode_scatter_rows
    )
    indexer_precomputed_compressor_state_written = bool(
        qkv_result.indexer_precomputed_compressor_state_written
    )
    indexer_precomputed_qw = qkv_result.indexer_precomputed_qw
    indexer_precomputed_empty_topk = qkv_result.indexer_precomputed_empty_topk
    deferred_indexer_state = qkv_result.deferred_indexer_state
    compressor_state_swa_write_fused = bool(qkv_result.compressor_state_swa_write_fused)
    bucketed_prefill_done = bool(qkv_result.bucketed_prefill_done)
    bucketed_kv_primary = qkv_result.bucketed_kv_primary
    token_topk_offset = int(qkv_result.token_topk_offset)
    all_kv_offset = int(qkv_result.all_kv_offset)
    attention_rows = int(qkv_result.attention_rows or active_bucket)
    _sampled_warmup_trace(
        "compressed_attention qkv_run done "
        f"layer={trace_layer} path={qkv_path} "
        f"attention_rows={int(attention_rows)} "
        f"bucketed={bool(bucketed_prefill_done)}",
    )
    _profile_attention(
        "qkv_prepare",
        time.perf_counter() - qkv_t0,
        qkv_path=qkv_path,
        topk_ready=topk_t_dev is not None and mask_dev is not None,
        compressor_state_swa_write_fused=bool(compressor_state_swa_write_fused),
        precomputed_compressor=precomputed_compressor_kv_score is not None,
        precomputed_indexer=indexer_precomputed_compressor_kv_score is not None,
        attention_rows=int(attention_rows),
    )
    qkv_flat_kv_dev = kv_dev
    bucketed_prefill_swa_rows = None
    if int(start_pos) == 0 and qkv_outputs_flat_kv:
        candidate = (
            bucketed_kv_primary if bucketed_kv_primary is not None else qkv_flat_kv_dev
        )
        candidate_shape = tuple(int(dim) for dim in getattr(candidate, "shape", ()))
        if (
            len(candidate_shape) == 2
            and int(candidate_shape[0]) > int(n_tokens)
            and int(candidate_shape[0]) % int(bsz) == 0
        ):
            bucketed_prefill_swa_rows = candidate
    if qkv_outputs_flat_kv:
        active_kv = _alias_device_value_first_dim_slice(
            kv_dev,
            start=0,
            size=n_tokens,
        )
        kv_alias = (
            None
            if active_kv is None
            else _alias_device_value_shape(
                active_kv,
                (int(bsz), int(seqlen), int(head_dim)),
            )
        )
        if kv_alias is None:
            raise RuntimeError(
                "DSV4 product compressed attention flat-KV QKV output "
                "requires an NRT alias back to [batch, seqlen, head_dim]"
            )
        kv_dev = kv_alias
    _sampled_warmup_trace(
        "compressed_attention q_scale start "
        f"layer={trace_layer} fused={bool(qkv_fuses_q_scale)} "
        f"attention_rows={int(attention_rows)}",
    )
    if qkv_fuses_q_scale:
        q_scaled_t = q_dev
    else:
        # Fold q's pre-scale + bf16 cast + transpose into the fragment so the
        # attention kernel consumes the DeviceTensor directly (no download).
        q_scaled_out = _scratch(
            "attention_q_scaled_t",
            (attention_rows, head_dim, n_heads),
            ml_dtypes.bfloat16,
        )
        q_scaled_t = fns["q_scale_transpose"](
            q_dev,
            softmax_scale=float(attn.softmax_scale),
            token_bucket=attention_rows,
            **_output_kwargs("output0", q_scaled_out),
        )
    _sampled_warmup_trace(
        f"compressed_attention q_scale done layer={trace_layer}",
    )
    qr = qr_dev
    kv = kv_dev

    # Mirror SWA KV into the device ring cache.
    kv_state_t0 = time.perf_counter()
    _sampled_warmup_trace(
        "compressed_attention kv_state start "
        f"layer={trace_layer} fused={bool(compressor_state_swa_write_fused)}",
    )
    deferred_swa_mirror: Dsv4DeferredSwaMirror | None = None
    mirror_owner_ids = owner_ids
    mirror_start_pos = int(start_pos)
    mirror_seqlen = int(seqlen)
    if int(seqlen) > win:
        owners_rect = np.asarray(owner_ids, dtype=np.int32).reshape(bsz, seqlen)
        mirror_owner_ids = owners_rect[:, seqlen - win :].reshape(-1)
        mirror_start_pos = int(start_pos) + int(seqlen) - win
        mirror_seqlen = win
    use_per_request_mirror = int(mirror_seqlen) <= win and (
        int(bsz) * int(mirror_seqlen) > 128
        or (product_shape_aliases and int(seqlen) > win and int(bsz) > 1)
    )

    def _mirror_device_owner_pos_aliases(
        *,
        source_start: int,
        size: int,
    ) -> tuple[Any | None, Any | None]:
        if owner_ids_dev is None or device_token_positions is None:
            return None, None
        owner_alias = _alias_device_value_first_dim_slice(
            owner_ids_dev,
            start=int(source_start),
            size=int(size),
        )
        pos_alias = _alias_device_value_first_dim_slice(
            device_token_positions,
            start=int(source_start),
            size=int(size),
        )
        if owner_alias is None or pos_alias is None:
            return None, None
        return owner_alias, pos_alias

    if not compressor_state_swa_write_fused and use_per_request_mirror:
        # The aliased SWA scatter kernel is capped at 128 row writes. Product
        # also uses per-request tail aliases for multi-request long prefill
        # because the all-request tail window is not globally contiguous.
        owners_rect = np.asarray(mirror_owner_ids, dtype=np.int32).reshape(
            int(bsz),
            int(mirror_seqlen),
        )
        for req_idx in range(int(bsz)):
            req_tail_kwargs = (
                {}
                if product_shape_aliases
                else _output_kwargs(
                    "output0",
                    _scratch(
                        "attention_kv_request_tail_window",
                        (int(mirror_seqlen), head_dim),
                        np.float32,
                    ),
                )
            )
            req_rows = fns["attention_kv_request_tail_window"](
                kv,
                request_index=req_idx,
                window_size=int(mirror_seqlen),
                head_dim=int(attn.head_dim),
                **req_tail_kwargs,
            )
            req_source_start = req_idx * int(seqlen) + (
                int(seqlen) - int(mirror_seqlen)
            )
            req_owner_ids_dev, req_positions_dev = _mirror_device_owner_pos_aliases(
                source_start=req_source_start,
                size=int(mirror_seqlen),
            )
            _mirror_swa_kv_to_device_cache(
                req_rows,
                mirror_start_pos,
                window_size=win,
                device_layer_state=device_layer_state,
                build_dir=build_dir,
                owner_ids=owners_rect[req_idx],
                owner_ids_dev=req_owner_ids_dev,
                positions_dev=req_positions_dev,
                bsz=1,
                seqlen=mirror_seqlen,
            )
    elif not compressor_state_swa_write_fused:
        if int(seqlen) > win:
            tail_kwargs = (
                {}
                if product_shape_aliases
                else _output_kwargs(
                    "output0",
                    _scratch(
                        "attention_kv_tail_window",
                        (int(bsz) * int(win), head_dim),
                        np.float32,
                    ),
                )
            )
            mirror_rows = fns["attention_kv_tail_window"](
                kv,
                window_size=win,
                head_dim=int(attn.head_dim),
                **tail_kwargs,
            )
        else:
            flatten_kwargs = (
                {}
                if product_shape_aliases
                else _output_kwargs(
                    "output0",
                    _scratch(
                        "attention_kv_flatten",
                        (int(bsz) * int(seqlen), head_dim),
                        np.float32,
                    ),
                )
            )
            mirror_rows = fns["attention_kv_flatten"](
                kv,
                total_tokens=bsz * seqlen,
                head_dim=int(attn.head_dim),
                **flatten_kwargs,
            )
        mirror_source_start = 0
        if int(seqlen) > win and int(bsz) == 1:
            mirror_source_start = int(seqlen) - int(win)
        mirror_owner_ids_dev, mirror_positions_dev = (
            _mirror_device_owner_pos_aliases(
                source_start=mirror_source_start,
                size=int(bsz) * int(mirror_seqlen),
            )
            if not (int(seqlen) > win and int(bsz) > 1)
            else (None, None)
        )
        direct_mirror_rows = mirror_rows
        direct_mirror_owner_ids = mirror_owner_ids
        direct_mirror_owner_ids_dev = mirror_owner_ids_dev
        direct_mirror_positions_dev = mirror_positions_dev
        direct_mirror_bsz = int(bsz)
        direct_mirror_seqlen = int(mirror_seqlen)
        direct_mirror_live_rows: int | None = None
        live_mirror_rows = int(bsz) * int(mirror_seqlen)
        if (
            int(start_pos) == 0
            and bucketed_prefill_swa_rows is not None
            and not isinstance(mirror_rows, np.ndarray)
        ):
            bucketed_shape = tuple(
                int(dim) for dim in getattr(bucketed_prefill_swa_rows, "shape", ())
            )
            bucket_rows = _swa_owner_window_write_bucket_rows(
                live_rows=live_mirror_rows,
                window_size=win,
                available_rows=(int(bucketed_shape[0]) if bucketed_shape else None),
            )
            if (
                bucket_rows > live_mirror_rows
                and bucketed_shape
                and int(bucketed_shape[0]) >= int(bucket_rows)
            ):
                bucket_rows_dev = _alias_device_value_first_dim_slice(
                    bucketed_prefill_swa_rows,
                    start=0,
                    size=int(bucket_rows),
                )
                bucket_owner_ids_dev, bucket_positions_dev = (
                    _mirror_device_owner_pos_aliases(
                        source_start=0,
                        size=int(bucket_rows),
                    )
                )
                if (
                    bucket_rows_dev is not None
                    and bucket_owner_ids_dev is not None
                    and bucket_positions_dev is not None
                ):
                    direct_mirror_rows = bucket_rows_dev
                    direct_mirror_owner_ids = _pad_owner_ids_for_bucket(
                        mirror_owner_ids,
                        rows=int(bucket_rows),
                    )
                    direct_mirror_owner_ids_dev = bucket_owner_ids_dev
                    direct_mirror_positions_dev = bucket_positions_dev
                    direct_mirror_bsz = 1
                    direct_mirror_seqlen = int(bucket_rows)
                    direct_mirror_live_rows = int(live_mirror_rows)
        decode_compression_boundary_for_swa = bool(ratio) and (
            (int(start_pos) + 1) % int(ratio) == 0
        )
        compressor_state_for_swa = getattr(device_layer_state, "compressor", None)
        compressor_kv_score_state_for_swa = getattr(
            compressor_state_for_swa,
            "kv_score_state",
            None,
        )
        compressor_cache_for_swa = getattr(
            compressor_state_for_swa,
            "compressed_kv_cache",
            None,
        )
        can_defer_decode_swa_mirror = (
            bool(ratio)
            and int(start_pos) != 0
            and int(seqlen) == 1
            and int(mirror_seqlen) == 1
            and mirror_start_pos == int(start_pos)
            and (
                not decode_compression_boundary_for_swa
                or (
                    precomputed_compressor_decode_scatter_rows is not None
                    and not isinstance(
                        precomputed_compressor_decode_scatter_rows,
                        np.ndarray,
                    )
                    and hasattr(compressor_cache_for_swa, "tensor_ref")
                )
            )
            and hasattr(compressor_kv_score_state_for_swa, "tensor_ref")
            and hasattr(device_layer_state.swa_kv_cache, "tensor_ref")
            and not isinstance(mirror_rows, np.ndarray)
        )
        comp_spec_for_swa = getattr(compressor_state_for_swa, "spec", None)
        comp_overlap_for_swa = bool(
            getattr(
                comp_spec_for_swa,
                "overlap",
                getattr(getattr(attn, "compressor", None), "overlap", False),
            )
        )
        if comp_overlap_for_swa:
            prefill_swa_tail_len = min(
                int(seqlen),
                int(ratio) + int(seqlen) % int(ratio),
            )
        else:
            prefill_swa_tail_len = int(seqlen) % int(ratio) if bool(ratio) else 0
        can_defer_prefill_swa_mirror = (
            bool(ratio)
            and int(start_pos) == 0
            and int(seqlen) >= int(ratio)
            and int(prefill_swa_tail_len) > 0
            and precomputed_compressor_kv_score is not None
            and not isinstance(precomputed_compressor_kv_score[0], np.ndarray)
            and not isinstance(precomputed_compressor_kv_score[1], np.ndarray)
            and precomputed_compressor_prefill_scatter_rows is not None
            and not isinstance(precomputed_compressor_prefill_scatter_rows, np.ndarray)
            and hasattr(compressor_kv_score_state_for_swa, "tensor_ref")
            and hasattr(compressor_cache_for_swa, "tensor_ref")
            and hasattr(device_layer_state.swa_kv_cache, "tensor_ref")
            and not isinstance(mirror_rows, np.ndarray)
        )
        if can_defer_decode_swa_mirror or can_defer_prefill_swa_mirror:
            deferred_mirror_rows = (
                bucketed_prefill_swa_rows
                if can_defer_prefill_swa_mirror
                and bucketed_prefill_swa_rows is not None
                else mirror_rows
            )
            deferred_swa_mirror = Dsv4DeferredSwaMirror(
                swa_kv_cache=device_layer_state.swa_kv_cache,
                swa_rows=deferred_mirror_rows,
                window_size=int(win),
                start_pos=int(mirror_start_pos),
                bsz=int(bsz),
                seqlen=int(mirror_seqlen),
                owner_ids=mirror_owner_ids,
                owner_ids_dev=mirror_owner_ids_dev,
                positions_dev=mirror_positions_dev,
            )
        else:
            _mirror_swa_kv_to_device_cache(
                direct_mirror_rows,
                mirror_start_pos,
                window_size=win,
                device_layer_state=device_layer_state,
                build_dir=build_dir,
                owner_ids=direct_mirror_owner_ids,
                owner_ids_dev=direct_mirror_owner_ids_dev,
                positions_dev=direct_mirror_positions_dev,
                live_rows=direct_mirror_live_rows,
                bsz=direct_mirror_bsz,
                seqlen=direct_mirror_seqlen,
            )
    _profile_attention(
        "kv_state",
        time.perf_counter() - kv_state_t0,
        compressor_state_swa_write_fused=bool(compressor_state_swa_write_fused),
        deferred_swa_mirror=deferred_swa_mirror is not None,
        mirror_seqlen=int(mirror_seqlen),
        use_per_request_mirror=bool(use_per_request_mirror),
    )
    _sampled_warmup_trace(
        "compressed_attention kv_state done "
        f"layer={trace_layer} deferred_swa_mirror={deferred_swa_mirror is not None}",
    )

    comp_device_state = device_layer_state.compressor

    sparse_t0 = time.perf_counter()
    _sampled_warmup_trace(
        "compressed_attention sparse_prep start "
        f"layer={trace_layer} ratio={int(ratio)}",
    )
    if ratio:
        # When prefill primary is the device SWA cache (stride=window),
        # compressed-topk offsets must start at ``window`` so that
        # indices in ``[seqlen, window)`` don't clash with the primary
        # index range of the two-source kernel.
        if start_pos == 0 and prefill_device_primary:
            offset = win
        elif start_pos == 0:
            offset = (
                int(all_kv_offset)
                or int(token_topk_offset)
                or int(getattr(kv, "shape")[1])
            )
        else:
            offset = win

        require_fused_sparse_prep = caps.require_fused_sparse_attention_prep
        token_topk_prep = fns.get("topk_tokens_concat_pad_sparse_attention_prep")
        if (
            (topk_t_dev is None or mask_dev is None)
            and attn.indexer is not None
            and indexer_precomputed_empty_topk is not None
            and indexer_precomputed_compressor_kv_score is not None
        ):
            topk_t_dev, mask_dev = indexer_precomputed_empty_topk
            idx_comp_kv_dev, idx_comp_score_dev = (
                indexer_precomputed_compressor_kv_score
            )
            if _can_defer_indexer_state_to_main_swa(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ):
                _defer_indexer_state_to_main_swa(
                    idx_comp_kv_dev,
                    idx_comp_score_dev,
                )
            else:
                token_owner_ids = _state_owner_ids_from_batch(
                    bsz=bsz,
                    seqlen=seqlen,
                    owner_ids=owner_ids,
                )
                request_owner_ids_dev = None
                if owner_ids_dev is not None and int(seqlen) == 1:
                    request_owner_ids_dev = _alias_device_value_first_dim_slice(
                        owner_ids_dev,
                        start=0,
                        size=int(bsz),
                    )
                request_positions_dev = None
                if device_token_positions is not None and int(seqlen) == 1:
                    request_positions_dev = _alias_device_value_first_dim_slice(
                        device_token_positions,
                        start=0,
                        size=int(bsz),
                    )
                _mirror_compressor_input_to_device_state(
                    indexer_obj.compressor,
                    idx_comp_kv_dev,
                    idx_comp_score_dev,
                    start_pos,
                    bsz=bsz,
                    seqlen=seqlen,
                    device_state=device_layer_state.indexer,
                    build_dir=build_dir,
                    owner_ids=token_owner_ids,
                    owner_ids_dev=request_owner_ids_dev,
                    positions_dev=request_positions_dev,
                )
        elif (
            (topk_t_dev is None or mask_dev is None)
            and attn.indexer is None
            and token_topk_prep is not None
        ):
            max_c_len = int(qkv_setup.token_topk_max_c_len)
            win_width = int(win) if int(start_pos) > 0 else min(int(seqlen), int(win))
            comp_width = (
                1
                if int(start_pos) == 0 and int(seqlen) // int(ratio) == 0
                else (
                    int(max_c_len) if int(start_pos) > 0 else int(seqlen) // int(ratio)
                )
            )
            k_raw = int(win_width) + int(comp_width)
            n_q = int(attention_rows)
            k_padded = ((k_raw + int(K_TILE) - 1) // int(K_TILE)) * int(K_TILE)
            output_kwargs = (
                {}
                if attention_scratch is None
                else {
                    "_nkipy_output_tensors": {
                        "output0": _scratch(
                            "attention_topk_t",
                            (k_padded, n_q),
                            np.int32,
                        ),
                        "output1": _scratch(
                            "attention_topk_mask",
                            (n_q, k_padded),
                            ml_dtypes.bfloat16,
                        ),
                    }
                }
            )
            topk_t_dev, mask_dev = token_topk_prep(
                x,
                window_size=int(win),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=int(max_c_len),
                rows=int(attention_rows),
                k_tile=int(K_TILE),
                **output_kwargs,
            )
        elif topk_t_dev is None or mask_dev is None:
            indexer_sparse_attention_prep = fns.get(
                "indexer_sparse_attention_prep_static"
            )
            if attn.indexer is not None and callable(indexer_sparse_attention_prep):
                topk_t_dev, mask_dev = _run_indexer(
                    fns,
                    attn.indexer,
                    x,
                    qr,
                    start_pos,
                    offset,
                    build_dir=build_dir,
                    device_state=device_layer_state.indexer,
                    owner_ids=owner_ids,
                    owner_ids_dev=owner_ids_dev,
                    token_positions=device_token_positions,
                    attention_scratch=attention_scratch,
                    sparse_attention_rows=int(attention_rows),
                    sparse_attention_k_tile=int(K_TILE),
                    sparse_attention_window_size=int(win),
                    precomputed_compressor_kv_score=(
                        indexer_precomputed_compressor_kv_score
                    ),
                    precomputed_compressor_decode_scatter_rows=(
                        indexer_precomputed_compressor_decode_scatter_rows
                    ),
                    precomputed_compressor_state_written=(
                        indexer_precomputed_compressor_state_written
                    ),
                    precomputed_qw=indexer_precomputed_qw,
                    precomputed_empty_topk=indexer_precomputed_empty_topk,
                )
            else:
                if require_fused_sparse_prep:
                    prep_kind = "indexer" if attn.indexer is not None else "token"
                    raise RuntimeError(
                        "DSV4 product compressed attention requires fused "
                        f"{prep_kind} sparse-attention prep"
                    )
                topk_win = fns["window_topk_from_tokens"](
                    x,
                    window_size=win,
                    start_pos=start_pos,
                    **_output_kwargs(
                        "output0",
                        _scratch(
                            "attention_window_topk",
                            (
                                int(bsz),
                                1 if int(start_pos) > 0 else int(seqlen),
                                int(win)
                                if int(start_pos) > 0
                                else min(int(seqlen), int(win)),
                            ),
                            np.int32,
                        ),
                    ),
                )
                if attn.indexer is not None:
                    topk_comp = _run_indexer(
                        fns,
                        attn.indexer,
                        x,
                        qr,
                        start_pos,
                        offset,
                        build_dir=build_dir,
                        device_state=device_layer_state.indexer,
                        owner_ids=owner_ids,
                        owner_ids_dev=owner_ids_dev,
                        token_positions=device_token_positions,
                        attention_scratch=attention_scratch,
                        precomputed_compressor_kv_score=(
                            indexer_precomputed_compressor_kv_score
                        ),
                        precomputed_compressor_decode_scatter_rows=(
                            indexer_precomputed_compressor_decode_scatter_rows
                        ),
                        precomputed_compressor_state_written=(
                            indexer_precomputed_compressor_state_written
                        ),
                        precomputed_qw=indexer_precomputed_qw,
                        precomputed_empty_topk=indexer_precomputed_empty_topk,
                    )
                else:
                    if start_pos == 0 and seqlen // ratio == 0:
                        topk_comp = fns["invalid_topk_from_tokens"](
                            x,
                            k=1,
                            **_output_kwargs(
                                "output0",
                                _scratch(
                                    "attention_invalid_topk",
                                    (int(bsz), int(seqlen), 1),
                                    np.int32,
                                ),
                            ),
                        )
                    else:
                        max_c_len = int(qkv_setup.token_topk_max_c_len)
                        topk_comp = fns["compressed_topk_no_indexer_from_tokens"](
                            x,
                            ratio=ratio,
                            offset=offset,
                            start_pos=start_pos,
                            max_c_len=max_c_len,
                            **_output_kwargs(
                                "output0",
                                _scratch(
                                    "attention_compressed_topk",
                                    (
                                        int(bsz),
                                        1 if int(start_pos) > 0 else int(seqlen),
                                        int(max_c_len)
                                        if int(start_pos) > 0
                                        else int(seqlen) // int(ratio),
                                    ),
                                    np.int32,
                                ),
                            ),
                        )

                # ``topk_concat_fn`` casts both args to int32 internally; its
                # output chains into ``topk_sparse_attention_prep`` which emits
                # the ``topk_T [K, N_q]`` + ``mask [N_q, K]`` pair the attention
                # kernel consumes — all on device, no host round-trip.
                win_shape = tuple(int(dim) for dim in getattr(topk_win, "shape", ()))
                comp_shape = tuple(int(dim) for dim in getattr(topk_comp, "shape", ()))
                k_raw = int(win_shape[-1]) + int(comp_shape[-1])
                n_q = int(attention_rows)
                k_padded = ((k_raw + int(K_TILE) - 1) // int(K_TILE)) * int(K_TILE)
                output_kwargs = (
                    {}
                    if attention_scratch is None
                    else {
                        "_nkipy_output_tensors": {
                            "output0": _scratch(
                                "attention_topk_t",
                                (k_padded, n_q),
                                np.int32,
                            ),
                            "output1": _scratch(
                                "attention_topk_mask",
                                (n_q, k_padded),
                                ml_dtypes.bfloat16,
                            ),
                        }
                    }
                )
                fused_topk_prep = fns.get("topk_concat_pad_sparse_attention_prep")
                if fused_topk_prep is not None:
                    topk_t_dev, mask_dev = fused_topk_prep(
                        topk_win,
                        topk_comp,
                        rows=int(attention_rows),
                        k_tile=int(K_TILE),
                        **output_kwargs,
                    )
                else:
                    topk_concat_shape = (*win_shape[:-1], int(k_raw))
                    topk_concat_dev = fns["topk_concat"](
                        topk_win,
                        topk_comp,
                        **_output_kwargs(
                            "output0",
                            _scratch(
                                "attention_topk_concat",
                                topk_concat_shape,
                                np.int32,
                            ),
                        ),
                    )
                    if n_tokens != attention_rows:
                        topk_concat_dev = fns["pad_topk_rows"](
                            topk_concat_dev,
                            rows=attention_rows,
                        )
                    topk_t_dev, mask_dev = fns["topk_sparse_attention_prep"](
                        topk_concat_dev,
                        k_tile=int(K_TILE),
                        **output_kwargs,
                    )
    else:
        topk_t_dev = None
        mask_dev = None

    _profile_attention(
        "sparse_prep",
        time.perf_counter() - sparse_t0,
        ratio=int(ratio),
        topk_ready=topk_t_dev is not None and mask_dev is not None,
        has_indexer=attn.indexer is not None,
        deferred_indexer_state=deferred_indexer_state is not None,
    )
    _sampled_warmup_trace(
        "compressed_attention sparse_prep done "
        f"layer={trace_layer} topk_ready={topk_t_dev is not None and mask_dev is not None}",
    )

    compressor_t0 = time.perf_counter()
    _sampled_warmup_trace(
        "compressed_attention compressor_state start "
        f"layer={trace_layer} fused={bool(compressor_state_swa_write_fused)}",
    )
    compressor_ran = False
    if ratio and not compressor_state_swa_write_fused:
        compressor_ran = True
        _run_compressor(
            fns,
            attn.compressor,
            x,
            start_pos,
            build_dir=build_dir,
            device_state=comp_device_state,
            owner_ids=owner_ids,
            owner_ids_dev=owner_ids_dev,
            token_positions=device_token_positions,
            attention_scratch=attention_scratch,
            precomputed_kv_score=precomputed_compressor_kv_score,
            precomputed_prefill_scatter_rows=(
                precomputed_compressor_prefill_scatter_rows
            ),
            precomputed_decode_scatter_rows=(
                precomputed_compressor_decode_scatter_rows
            ),
            deferred_swa_mirror=deferred_swa_mirror,
            deferred_indexer_state=deferred_indexer_state,
            real_prefill_seqlen=real_prefill_seqlen,
        )
        if deferred_indexer_state is not None and not deferred_indexer_state.consumed:
            _run_compressor(
                fns,
                deferred_indexer_state.compressor,
                x,
                start_pos,
                build_dir=build_dir,
                device_state=deferred_indexer_state.device_state,
                owner_ids=owner_ids,
                owner_ids_dev=owner_ids_dev,
                token_positions=device_token_positions,
                attention_scratch=attention_scratch,
                precomputed_kv_score=(
                    deferred_indexer_state.kv,
                    deferred_indexer_state.score,
                ),
                precomputed_prefill_scatter_rows=(
                    deferred_indexer_state.prefill_scatter_rows
                ),
                precomputed_decode_scatter_rows=(
                    deferred_indexer_state.decode_scatter_rows
                ),
                real_prefill_seqlen=real_prefill_seqlen,
            )
    _profile_attention(
        "compressor_state",
        time.perf_counter() - compressor_t0,
        ran=bool(compressor_ran),
        fused=bool(compressor_state_swa_write_fused),
        deferred_swa_mirror=deferred_swa_mirror is not None,
        deferred_indexer_state=deferred_indexer_state is not None,
    )
    _sampled_warmup_trace(
        "compressed_attention compressor_state done "
        f"layer={trace_layer} ran={bool(compressor_ran)}",
    )

    owner_sync_t0 = time.perf_counter()
    _sampled_warmup_trace(
        "compressed_attention owner_sync start "
        f"layer={trace_layer} attention_rows={int(attention_rows)}",
    )
    # INVARIANT: every top-k entry lives in one flat per-owner index space;
    # compressed entries are rebased by the COMPILED offset (all_kv_offset /
    # token_topk_offset). primary_len below must equal that offset or the
    # two-source gather routes secondary entries into the primary range.
    compiled_topk_offset = int(all_kv_offset) or int(token_topk_offset) or None
    primary_owner_ids_attn: np.ndarray | None = None
    if start_pos == 0:
        if compiled_topk_offset is not None:
            # A prologue produced a top-k tensor: its compiled offset (the value
            # the runner baked == published) IS the two-source primary_len, so
            # they are equal by construction. The offset value also selects the
            # primary BUFFER: offset==win means the window top-k entries index
            # the SWA cache (win rows/owner) — true for ALL short prefill
            # (seqlen<=win), bucketed or not, since the runner bakes offset=win
            # whenever compile_seqlen<=win. offset>win means long prefill whose
            # primary is the flat KV at the compiled rows (bucketed) or the real
            # rows (unbucketed).
            primary_len_attn = int(compiled_topk_offset)
            if int(compiled_topk_offset) <= int(win):
                # Short prefill: SWA cache is the primary (the bucketed scatter
                # populated its tail-window rows just like the legacy write).
                kv_primary = device_layer_state.swa_kv_cache
            elif bucketed_prefill_done:
                # Long bucketed prefill: the prologue emitted the flat KV at the
                # compiled bucket rows (offset == compile_seqlen == its rows);
                # window top-k entries are real positions < offset.
                kv_primary = bucketed_kv_primary
                primary_owner_ids_attn = np.repeat(
                    np.arange(int(bsz), dtype=np.int32),
                    int(seqlen),
                )
            else:
                primary_kv_kwargs = (
                    {}
                    if product_shape_aliases
                    else _output_kwargs(
                        "output0",
                        _scratch(
                            "attention_kv_flatten",
                            (int(bsz) * int(seqlen), head_dim),
                            np.float32,
                        ),
                    )
                )
                kv_primary = fns["attention_kv_flatten"](
                    kv,
                    total_tokens=bsz * seqlen,
                    head_dim=int(attn.head_dim),
                    **primary_kv_kwargs,
                )
                # Fresh long-prefill KV rows are lane-local [local_batch*seqlen];
                # compressed/SWA state stays keyed by global request owner IDs,
                # so the two-source kernel needs distinct owner vectors.
                primary_owner_ids_attn = np.repeat(
                    np.arange(int(bsz), dtype=np.int32),
                    int(seqlen),
                )
        elif prefill_device_primary:
            kv_primary = device_layer_state.swa_kv_cache
            primary_len_attn = int(win)
        else:
            primary_kv_kwargs = (
                {}
                if product_shape_aliases
                else _output_kwargs(
                    "output0",
                    _scratch(
                        "attention_kv_flatten",
                        (int(bsz) * int(seqlen), head_dim),
                        np.float32,
                    ),
                )
            )
            kv_primary = fns["attention_kv_flatten"](
                kv,
                total_tokens=bsz * seqlen,
                head_dim=int(attn.head_dim),
                **primary_kv_kwargs,
            )
            primary_owner_ids_attn = np.repeat(
                np.arange(int(bsz), dtype=np.int32),
                int(seqlen),
            )
            primary_len_attn = int(seqlen)
    else:
        kv_primary = device_layer_state.swa_kv_cache
        primary_len_attn = int(win)
    sink_kwargs = (
        {}
        if product_shape_aliases
        else _output_kwargs(
            "output0",
            _scratch("attention_sink_2d", (1, n_heads), np.float32),
        )
    )
    sink_2d = fns["attention_sink_2d"](
        attn.attn_sink,
        n_heads=int(attn.n_heads),
        **sink_kwargs,
    )

    def _fit_owner_rows(values: np.ndarray, rows: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if int(values.shape[0]) == int(rows):
            return values
        if int(values.shape[0]) > int(rows):
            return values[: int(rows)]
        return np.pad(
            values,
            (0, int(rows) - int(values.shape[0])),
            mode="constant",
            constant_values=0,
        )

    owner_ids_attn = _fit_owner_rows(owner_ids, int(attention_rows))
    if (
        primary_owner_ids_attn is not None
        and int(
            primary_owner_ids_attn.shape[0],
        )
        != attention_rows
    ):
        primary_owner_ids_attn = _fit_owner_rows(
            primary_owner_ids_attn,
            int(attention_rows),
        )

    owner_ids_dev_arg = None
    primary_owner_ids_dev_arg = None
    if owner_ids_dev is not None:
        owner_ids_dev_shape = tuple(
            int(dim) for dim in getattr(owner_ids_dev, "shape", ())
        )
        if owner_ids_dev_shape == (int(attention_rows),):
            if owner_ids_host is None:
                raise RuntimeError("owner_ids_host is required with owner_ids_dev")
            if tuple(owner_ids_host.shape) != (int(attention_rows),):
                raise RuntimeError(
                    "owner_ids_host shape mismatch for DSV4 product attention: "
                    f"got {owner_ids_host.shape}, expected ({int(attention_rows)},)"
                )
            _overwrite_device_tensor_if_changed(
                owner_ids_dev,
                owner_ids_host,
                owner_ids_attn,
                error_context="DSV4 sampled metadata sync",
            )
            owner_ids_dev_arg = owner_ids_dev
    if primary_owner_ids_attn is None:
        primary_owner_ids_dev_arg = None
    elif primary_owner_ids_dev is not None:
        primary_owner_ids_dev_shape = tuple(
            int(dim) for dim in getattr(primary_owner_ids_dev, "shape", ())
        )
        if primary_owner_ids_dev_shape == (int(attention_rows),):
            if primary_owner_ids_host is None:
                raise RuntimeError(
                    "primary_owner_ids_host is required with primary_owner_ids_dev"
                )
            if tuple(primary_owner_ids_host.shape) != (int(attention_rows),):
                raise RuntimeError(
                    "primary_owner_ids_host shape mismatch for DSV4 product attention: "
                    f"got {primary_owner_ids_host.shape}, expected ({int(attention_rows)},)"
                )
            _overwrite_device_tensor_if_changed(
                primary_owner_ids_dev,
                primary_owner_ids_host,
                owner_ids_attn
                if primary_owner_ids_attn is None
                else primary_owner_ids_attn,
                error_context="DSV4 sampled metadata sync",
            )
            primary_owner_ids_dev_arg = primary_owner_ids_dev

    _profile_attention(
        "owner_sync",
        time.perf_counter() - owner_sync_t0,
        owner_ids_dev=owner_ids_dev is not None,
        primary_owner_ids=primary_owner_ids_attn is not None,
        primary_owner_ids_dev=primary_owner_ids_dev_arg is not None,
        active_bucket=int(active_bucket),
        attention_rows=int(attention_rows),
    )
    _sampled_warmup_trace(
        "compressed_attention owner_sync done "
        f"layer={trace_layer} primary_len={int(primary_len_attn)}",
    )

    if (
        topk_t_dev is not None
        and compiled_topk_offset is not None
        and int(compiled_topk_offset) != int(primary_len_attn)
    ):
        raise RuntimeError(
            "DSV4 two-source attention requires primary_len == the compiled "
            f"top-k offset; got primary_len={int(primary_len_attn)} "
            f"offset={int(compiled_topk_offset)} (start_pos={int(start_pos)}, "
            f"seqlen={int(seqlen)}, bucketed={bool(bucketed_prefill_done)}, "
            f"all_kv={int(all_kv_offset)}, tt={int(token_topk_offset)}, "
            f"ratio={int(ratio)}, qkv_path={qkv_path})"
        )
    attention_kernel_t0 = time.perf_counter()
    attention_output_arg = attention_output
    attention_postprocess_output_arg = attention_postprocess_output
    attention_output_shape = None
    expected_attention_shape = (int(attention_rows), int(n_heads), int(head_dim))
    dropped_attention_output_shape = None
    if attention_output_arg is not None:
        attention_output_shape = tuple(
            int(dim) for dim in getattr(attention_output_arg, "shape", ())
        )
        if attention_output_shape == expected_attention_shape:
            pass
        elif (
            len(attention_output_shape) == 3
            and int(attention_output_shape[0]) > int(attention_rows)
            and attention_output_shape[1:] == (int(n_heads), int(head_dim))
        ):
            canonical_attention_output = attention_output_arg
            attention_output_arg = _alias_device_value_first_dim_slice(
                canonical_attention_output,
                start=0,
                size=int(attention_rows),
            )
            if attention_output_arg is None:
                raise RuntimeError(
                    "DSV4 compressed attention requires a first-row alias from "
                    "the bucket attention output buffer"
                )
            attention_postprocess_output_arg = canonical_attention_output
        else:
            dropped_attention_output_shape = attention_output_shape
            attention_output_arg = None
            attention_postprocess_output_arg = None
            _sampled_warmup_trace(
                "compressed_attention backend_attention prealloc_drop "
                f"layer={trace_layer} rows={int(attention_rows)} "
                f"shape={attention_output_shape} "
                f"expected={expected_attention_shape}",
            )
    if attention_output_arg is None:
        fallback_output = _scratch(
            "attention_backend_output",
            expected_attention_shape,
            np.float32,
        )
        if fallback_output is not None:
            fallback_shape = tuple(
                int(dim) for dim in getattr(fallback_output, "shape", ())
            )
            if fallback_shape == expected_attention_shape:
                attention_output_arg = fallback_output
                attention_postprocess_output_arg = None
                attention_output_shape = fallback_shape
                _sampled_warmup_trace(
                    "compressed_attention backend_attention prealloc_fallback "
                    f"layer={trace_layer} rows={int(attention_rows)} "
                    f"dropped_shape={dropped_attention_output_shape} "
                    f"shape={fallback_shape}",
                )
            else:
                _sampled_warmup_trace(
                    "compressed_attention backend_attention fallback_drop "
                    f"layer={trace_layer} rows={int(attention_rows)} "
                    f"shape={fallback_shape} "
                    f"expected={expected_attention_shape}",
                )
    _sampled_warmup_trace(
        "compressed_attention backend_attention start "
        f"layer={trace_layer} rows={int(attention_rows)} "
        f"primary_len={int(primary_len_attn)} "
        f"secondary_stride={int(comp_device_state.spec.max_compressed_len)} "
        f"preallocated={attention_output_arg is not None} "
        f"shape={attention_output_shape} expected={expected_attention_shape}",
    )
    out_flat = backend.attention_ephemeral_paged_two_source(
        q_scaled_t=q_scaled_t,
        q_shape=(attention_rows, n_heads, head_dim),
        kv_primary=kv_primary,
        kv_secondary=comp_device_state.compressed_kv_cache,
        topk_t_dev=topk_t_dev,
        mask_dev=mask_dev,
        owner_ids=owner_ids_attn,
        owner_ids_dev=owner_ids_dev_arg,
        primary_owner_ids=primary_owner_ids_attn,
        primary_owner_ids_dev=primary_owner_ids_dev_arg,
        primary_len=primary_len_attn,
        secondary_stride=int(comp_device_state.spec.max_compressed_len),
        primary_prefix_len=(
            int(win) if int(start_pos) > 0 else min(int(seqlen), int(win))
        ),
        sink=sink_2d,
        softmax_scale=float(attn.softmax_scale),
        output=attention_output_arg,
        return_device=True,
    )
    _sampled_warmup_trace(
        f"compressed_attention backend_attention done layer={trace_layer}",
    )
    _profile_attention(
        "attention_kernel",
        time.perf_counter() - attention_kernel_t0,
        active_bucket=int(active_bucket),
        attention_rows=int(attention_rows),
        preallocated_output=attention_output_arg is not None,
        primary_len=int(primary_len_attn),
        secondary_stride=int(comp_device_state.spec.max_compressed_len),
        has_topk=topk_t_dev is not None and mask_dev is not None,
    )
    _sampled_warmup_trace(
        f"compressed_attention postprocess start layer={trace_layer}",
    )
    result = _run_compressed_attention_postprocess(
        fns=fns,
        attn=attn,
        metadata=metadata,
        backend=backend,
        caps=caps,
        out_flat=out_flat,
        start_pos=int(start_pos),
        bsz=int(bsz),
        seqlen=int(seqlen),
        active_bucket=int(attention_rows),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rd=int(rd),
        attention_scratch=attention_scratch,
        attention_hidden_shape=attention_hidden_shape,
        attention_postprocess_output=attention_postprocess_output_arg,
        profile_attention=_profile_attention,
        layer_id=layer_id,
    )
    _sampled_warmup_trace(
        f"compressed_attention postprocess done layer={trace_layer}",
    )
    return result
