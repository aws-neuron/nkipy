"""Base QKV product kernels for DSV4 product execution."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _compile_product_kernel,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import attention as graph_attention
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _as_product_device_input,
    _is_device_value,
    _require_product_device_value,
    _sample_array,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _TensorSpec,
)
from nkipy_serving.models.deepseek_v4.variants import (
    QkvVariantName,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)


def _qkv_write_shape_candidates(
    *,
    batch_size: int,
    seqlen: int,
    canonical_shape: tuple[int, int],
    candidate_buckets: tuple[int, ...],
    is_decode: bool,
) -> tuple[tuple[int, int], ...]:
    bsz = int(batch_size)
    seq = int(seqlen)
    shapes: list[tuple[int, int]] = [(bsz, seq)]
    if canonical_shape not in shapes:
        shapes.append(canonical_shape)
    if not bool(is_decode):
        single_query = (1, 1)
        if single_query not in shapes:
            shapes.append(single_query)
    for attention_bucket in candidate_buckets:
        attention_bucket_i = int(attention_bucket)
        if attention_bucket_i <= 0:
            continue
        candidate = (1, attention_bucket_i)
        if candidate not in shapes:
            shapes.append(candidate)
    return tuple(shapes)


class Dsv4ProductQkvBaseMixin:
    def precompile_lane_dp_attention_helpers(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile lane-local DP-attention indexer/top-k helpers."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product lane DP-attention precompile could not infer hidden size"
            )
        if not bool(is_decode):
            self._precompile_lane_sequence_hidden_pad(
                bucket,
                batch_size=bsz,
                seqlen=seq,
                hidden_size=hidden_size,
            )
            self._precompile_lane_sequence_hidden_pad_request_buckets(
                bucket,
                hidden_size=hidden_size,
            )
        # Decode helper precompile uses `seqlen` as cached history length so
        # compressed indexer paths can warm boundary-dependent shapes.  The SWA
        # QKV+KV-write helper still sees only the live query token.
        swa_query_seqlen = 1 if bool(is_decode) else seq
        self._precompile_lane_swa_attention_qkv_write_helpers(
            bucket,
            batch_size=bsz,
            seqlen=int(swa_query_seqlen),
            is_decode=bool(is_decode),
        )
        self._precompile_lane_attention_indexer_helpers(
            bucket,
            batch_size=bsz,
            seqlen=seq,
            is_decode=bool(is_decode),
        )

    def _precompile_lane_sequence_hidden_pad(
        self,
        bucket: Dsv4ProductBucket,
        *,
        batch_size: int,
        seqlen: int,
        hidden_size: int,
    ) -> None:
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz != 1 or seq <= 0:
            return
        compile_bsz, compile_seq = self._product_compile_attention_qkv_shape(
            bucket,
            bsz=bsz,
            seqlen=seq,
        )
        if int(compile_bsz) != 1 or int(compile_seq) <= seq:
            return
        kernel_for = getattr(self, "_sequence_hidden_pad_kernel_for", None)
        if not callable(kernel_for):
            return
        x = _TensorSpec(
            (1, seq, int(hidden_size)),
            np.dtype(ml_dtypes.bfloat16),
        )
        kernel_for(
            bucket,
            x,
            rows=int(compile_seq),
            hidden_size=int(hidden_size),
        )

    def _precompile_lane_sequence_hidden_pad_request_buckets(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_size: int,
    ) -> None:
        kernel_for = getattr(self, "_sequence_hidden_pad_kernel_for", None)
        if not callable(kernel_for):
            return
        for request_bucket in self._configured_product_decode_buckets():
            request_bucket_i = int(request_bucket)
            if request_bucket_i <= 1 or request_bucket_i > int(bucket.token_bucket):
                continue
            for seq in range(1, request_bucket_i):
                x = _TensorSpec(
                    (1, int(seq), int(hidden_size)),
                    np.dtype(ml_dtypes.bfloat16),
                )
                kernel_for(
                    bucket,
                    x,
                    rows=int(request_bucket_i),
                    hidden_size=int(hidden_size),
                )
            short_prefill_bucket = self._compressed_attention_bucket_for_tokens(
                int(request_bucket_i),
                int(bucket.token_bucket),
            )
            short_prefill_bucket_i = int(short_prefill_bucket)
            if (
                short_prefill_bucket_i <= request_bucket_i
                or short_prefill_bucket_i > int(bucket.token_bucket)
            ):
                continue
            for seq in range(1, request_bucket_i + 1):
                x = _TensorSpec(
                    (1, int(seq), int(hidden_size)),
                    np.dtype(ml_dtypes.bfloat16),
                )
                kernel_for(
                    bucket,
                    x,
                    rows=short_prefill_bucket_i,
                    hidden_size=int(hidden_size),
                )

    def precompile_lane_dp_attention_decode_continuation_helpers(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
    ) -> None:
        """Precompile first decode-after-prefill compressed-attention shapes."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        self._precompile_lane_attention_indexer_helpers(
            bucket,
            batch_size=bsz,
            seqlen=seq,
            is_decode=True,
        )

    def _attention_backend_step_inputs_for_bucket(
        self, token_bucket: int
    ) -> Any | None:
        backend = getattr(self, "attention_backend", None)
        if backend is None:
            return None
        lookup = getattr(backend, "step_inputs_for_bucket", None)
        if callable(lookup):
            step_inputs = lookup(int(token_bucket))
            if step_inputs is not None:
                return step_inputs
        return getattr(backend, "step_inputs", None)

    def _precompile_lane_swa_attention_qkv_write_helpers(
        self,
        bucket: Dsv4ProductBucket,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile SWA QKV+KV-write kernels for lane-local prefill shapes."""
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        backend = getattr(self, "attention_backend", None)
        kv_cache_fn = getattr(backend, "kv_cache", None)
        if not callable(kv_cache_fn):
            return
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            return
        canonical_shape = self._product_compile_attention_qkv_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seq),
        )
        # Serve and warmup both bucket (warmup slot_mapping/positions are
        # bucket-padded), so single-lane prefill only needs the canonical
        # shape.  Larger product buckets can still execute a lane under a
        # smaller attention bucket (for example live 200 rows under product
        # bucket 1024 but attention bucket 256), so warm those sub-bucket
        # QKV-write keys as canonical bucket shapes too.  Decode can also
        # promote batch kernels and then clamp them back to the active request
        # bucket, so include those `(1, request_bucket)` shapes explicitly.
        candidate_buckets = (
            self._configured_product_decode_buckets()
            if bool(is_decode)
            else self._configured_product_token_buckets()
        )
        shape_candidates = _qkv_write_shape_candidates(
            batch_size=int(bsz),
            seqlen=int(seq),
            canonical_shape=canonical_shape,
            candidate_buckets=tuple(
                int(attention_bucket)
                for attention_bucket in candidate_buckets
                if 0 < int(attention_bucket) <= int(bucket.token_bucket)
            ),
            is_decode=bool(is_decode),
        )
        for layer_id, block in enumerate(
            tuple(getattr(self.runtime_surface, "blocks", ()) or ())
        ):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            ratio = int(
                getattr(getattr(attn, "indexer", None), "compress_ratio", 0)
                or getattr(attn, "compress_ratio", 0)
                or 0
            )
            if ratio > 0:
                continue
            freqs_cos = getattr(attn, "freqs_cos", None)
            freqs_sin = getattr(attn, "freqs_sin", None)
            required = (
                getattr(attn, "wq_a", None),
                getattr(attn, "q_norm", None),
                getattr(attn, "wq_b", None),
                getattr(attn, "wkv", None),
                getattr(attn, "kv_norm", None),
                freqs_cos,
                freqs_sin,
            )
            if any(value is None for value in required):
                continue
            kv_cache = kv_cache_fn(int(layer_id))
            cos_table, sin_table = self._product_freq_tables_for(
                freqs_cos,
                freqs_sin,
                name="attention",
            )
            for compile_bsz, compile_seq in shape_candidates:
                compile_tokens = int(compile_bsz) * int(compile_seq)
                attention_bucket = self._attention_backend_bucket_for_tokens(
                    int(compile_tokens),
                    int(bucket.token_bucket),
                    is_decode=bool(is_decode),
                )
                step_inputs = self._attention_backend_step_inputs_for_bucket(
                    attention_bucket
                )
                slot_mapping = (
                    getattr(step_inputs, "slot_mapping", None)
                    if step_inputs is not None
                    else _TensorSpec((int(attention_bucket),), np.dtype(np.int32))
                )
                positions = (
                    _alias_device_value_shape(
                        getattr(step_inputs, "positions", None),
                        (int(compile_tokens),),
                    )
                    if step_inputs is not None
                    and getattr(step_inputs, "positions", None) is not None
                    else None
                )
                if positions is None:
                    positions = _TensorSpec((int(compile_tokens),), np.dtype(np.int32))
                x = _TensorSpec(
                    (int(compile_bsz), int(compile_seq), int(hidden_size)),
                    np.dtype(ml_dtypes.bfloat16),
                )
                pos = _TensorSpec(
                    (int(compile_tokens),),
                    np.dtype(np.int32),
                )
                self._attention_qkv_kv_cache_write_table_kernel_for(
                    bucket,
                    x,
                    attn.wq_a,
                    attn.q_norm,
                    attn.wq_b,
                    attn.wkv,
                    attn.kv_norm,
                    kv_cache,
                    slot_mapping,
                    cos_table,
                    sin_table,
                    pos,
                    n_heads=int(attn.n_heads),
                    head_dim=int(attn.head_dim),
                    rope_head_dim=int(attn.rope_head_dim),
                    eps=float(attn.eps),
                    block_size=64,
                    fp8_max=240.0,
                    q_softmax_scale=float(attn.softmax_scale),
                    q_token_bucket=int(attention_bucket),
                    kv_token_bucket=int(attention_bucket),
                )

    def _run_product_attention_qkv_quant_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        eps: float,
        block_size: int,
        fp8_max: float,
        q_softmax_scale: float | None = None,
        q_token_bucket: int | None = None,
        kv_token_bucket: int | None = None,
        return_qr: bool = True,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any | None]:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_quant_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(kv_norm, name="dsv4_attention_qkv_kv_norm")
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 3:
            raise RuntimeError(
                "DSV4 product QKV table path expects x [batch, seqlen, hidden], "
                f"got {x_shape}"
            )
        bsz, seqlen, _ = x_shape
        fuse_q_scale = q_softmax_scale is not None and q_token_bucket is not None
        fuse_kv_flat = kv_token_bucket is not None
        if fuse_kv_flat and not fuse_q_scale:
            raise RuntimeError(
                "DSV4 product QKV flat-KV fusion currently requires scaled-Q fusion"
            )
        if not bool(return_qr) and not fuse_kv_flat:
            raise RuntimeError(
                "DSV4 product QKV no-QR path requires flat-KV scaled-Q fusion"
            )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        kernel = self._attention_qkv_table_kernel_for(
            bucket,
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=(float(q_softmax_scale) if fuse_q_scale else None),
            q_token_bucket=int(q_token_bucket) if fuse_q_scale else None,
            kv_token_bucket=int(kv_token_bucket) if fuse_kv_flat else None,
            return_qr=bool(return_qr),
        )
        q_low_dim = int(getattr(q_norm, "shape", (0,))[0])
        outputs = dict(_nkipy_output_tensors or {})
        q = outputs.get("output0")
        if q is None:
            q_shape = (
                (int(q_token_bucket), int(head_dim), int(n_heads))
                if fuse_q_scale
                else (int(bsz), int(seqlen), int(n_heads), int(head_dim))
            )
            q_dtype = ml_dtypes.bfloat16 if fuse_q_scale else np.float32
            q = self._bucket_scratch(
                bucket,
                "attention_q_scaled_t" if fuse_q_scale else "attention_qkv_q",
                q_shape,
                q_dtype,
            )
        kv = outputs.get("output1")
        if kv is None:
            kv_shape = (
                (int(kv_token_bucket), int(head_dim))
                if fuse_kv_flat
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(
                bucket,
                "attention_qkv_kv_flat" if fuse_kv_flat else "attention_qkv_kv",
                kv_shape,
                np.float32,
            )
        qr = None
        if bool(return_qr):
            qr = outputs.get("output2")
            if qr is None:
                qr = self._bucket_scratch(
                    bucket,
                    "attention_qkv_qr",
                    (int(bsz), int(seqlen), q_low_dim),
                    np.float32,
                )
        else:
            outputs.pop("output2", None)
        kernel_outputs = {"output0": q, "output1": kv}
        if bool(return_qr):
            kernel_outputs["output2"] = qr
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "x": x,
                "wq_a": wq_a,
                "q_norm": q_norm,
                "wq_b": wq_b,
                "wkv": wkv,
                "kv_norm": kv_norm,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs=kernel_outputs,
            unload_after_call=False,
        )
        return q, kv, qr

    def _run_product_attention_qkv_write_kv_cache_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        kv_cache: Any,
        slot_mapping: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        eps: float,
        block_size: int,
        fp8_max: float,
        q_softmax_scale: float,
        q_token_bucket: int,
        kv_token_bucket: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_write_kv_cache_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(kv_norm, name="dsv4_attention_qkv_kv_norm")
        kv_cache = _require_product_device_value(
            kv_cache,
            where="attention_qkv_write_kv_cache_from_freq_table/kv_cache",
        )
        slot_mapping = _as_product_device_input(
            slot_mapping,
            name="dsv4_attention_qkv_slot_mapping",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 3:
            raise RuntimeError(
                "DSV4 product QKV+KV-write table path expects x "
                f"[batch, seqlen, hidden], got {x_shape}"
            )
        bsz, seqlen, hidden_size = x_shape
        compile_bsz, compile_seqlen = self._product_compile_attention_qkv_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
        )
        x_kernel = x
        slot_mapping_kernel = slot_mapping
        compile_tokens = int(compile_bsz) * int(compile_seqlen)
        qkv_capacity = min(int(q_token_bucket), int(kv_token_bucket))
        slot_shape = tuple(int(dim) for dim in getattr(slot_mapping, "shape", ()))
        if slot_shape:
            slot_rows = int(slot_shape[0])
            if slot_rows < int(compile_tokens):
                full_slot_mapping = self._product_full_value_for(
                    slot_mapping,
                    (int(compile_tokens),),
                )
                if full_slot_mapping is not None:
                    slot_mapping_kernel = full_slot_mapping
                    slot_rows = int(compile_tokens)
            qkv_capacity = min(int(qkv_capacity), int(slot_rows))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        if _is_device_value(positions) and pos_shape:
            pos_rows = int(np.prod(pos_shape))
            if (
                pos_rows < int(compile_tokens)
                and self._product_full_value_for(positions, (int(compile_tokens),))
                is None
            ):
                qkv_capacity = min(int(qkv_capacity), int(pos_rows))
        live_tokens = int(bsz) * int(seqlen)
        if int(compile_tokens) > int(qkv_capacity):
            if int(qkv_capacity) >= int(live_tokens):
                compile_bsz, compile_seqlen = 1, int(qkv_capacity)
            else:
                compile_bsz, compile_seqlen = int(bsz), int(seqlen)
            compile_tokens = int(compile_bsz) * int(compile_seqlen)
        if int(compile_bsz) != int(bsz) or int(compile_seqlen) != int(seqlen):
            full = self._product_full_value_for(
                x,
                (int(compile_bsz), int(compile_seqlen), int(hidden_size)),
            )
            if full is not None:
                x_kernel = full
            else:
                compile_bsz = int(bsz)
                compile_seqlen = int(seqlen)
                compile_tokens = int(compile_bsz) * int(compile_seqlen)
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(compile_tokens),
        )
        kernel = self._attention_qkv_kv_cache_write_table_kernel_for(
            bucket,
            x_kernel,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            kv_cache,
            slot_mapping_kernel,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            kv_token_bucket=int(kv_token_bucket),
        )
        outputs = dict(_nkipy_output_tensors or {})
        q = outputs.get("output0")
        if q is None:
            q = self._bucket_scratch(
                bucket,
                "attention_q_scaled_t",
                (int(q_token_bucket), int(head_dim), int(n_heads)),
                ml_dtypes.bfloat16,
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "x": x_kernel,
                "wq_a": wq_a,
                "q_norm": q_norm,
                "wq_b": wq_b,
                "wkv": wkv,
                "kv_norm": kv_norm,
                "kv_cache.must_alias_input": kv_cache,
                "slot_mapping": slot_mapping_kernel,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs={"output0": q, "kv_cache": kv_cache},
            unload_after_call=False,
        )
        return q

    def _attention_qkv_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        eps: float,
        block_size: int,
        fp8_max: float,
        q_softmax_scale: float | None = None,
        q_token_bucket: int | None = None,
        kv_token_bucket: int | None = None,
        return_qr: bool = True,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        fuse_q_scale = q_softmax_scale is not None and q_token_bucket is not None
        fuse_kv_flat = kv_token_bucket is not None
        if fuse_kv_flat and not fuse_q_scale:
            raise RuntimeError(
                "DSV4 product QKV flat-KV fusion currently requires scaled-Q fusion"
            )
        if not bool(return_qr) and not fuse_kv_flat:
            raise RuntimeError(
                "DSV4 product QKV no-QR path requires flat-KV scaled-Q fusion"
            )
        key = (
            x_shape,
            str(getattr(x, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wq_a, "shape", ())),
            str(getattr(wq_a, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(q_norm, "shape", ())),
            str(getattr(q_norm, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wq_b, "shape", ())),
            str(getattr(wq_b, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wkv, "shape", ())),
            str(getattr(wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(kv_norm, "shape", ())),
            str(getattr(kv_norm, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            pos_shape,
            str(getattr(positions, "dtype", "unknown")),
            int(n_heads),
            int(head_dim),
            int(rope_head_dim),
            float(eps),
            int(block_size),
            float(fp8_max),
            "scaled_q" if fuse_q_scale else "q",
            float(q_softmax_scale) if fuse_q_scale else None,
            int(q_token_bucket) if fuse_q_scale else None,
            "flat_kv" if fuse_kv_flat else "kv",
            int(kv_token_bucket) if fuse_kv_flat else None,
            bool(return_qr),
        )
        if fuse_kv_flat and not bool(return_qr):
            qkv_name_kind = "scaled_kvflat_noqr_"
        elif fuse_kv_flat:
            qkv_name_kind = "scaled_kvflat_"
        elif fuse_q_scale:
            qkv_name_kind = "scaled_"
        else:
            qkv_name_kind = ""
        name = (
            "dsv4_product_attention_qkv_table_"
            f"{qkv_name_kind}"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}"
        )
        if fuse_kv_flat and not bool(return_qr):
            fn = graph_attention.attention_qkv_quant_scaled_kvflat_no_qr_from_freq_table_fn
        elif fuse_kv_flat:
            fn = graph_attention.attention_qkv_quant_scaled_kvflat_from_freq_table_fn
        elif fuse_q_scale:
            fn = graph_attention.attention_qkv_quant_scaled_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_quant_from_freq_table_fn
        extra_kwargs = (
            {
                "q_softmax_scale": float(q_softmax_scale),
                "q_token_bucket": int(q_token_bucket),
                **({"kv_token_bucket": int(kv_token_bucket)} if fuse_kv_flat else {}),
            }
            if fuse_q_scale
            else {}
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=QkvVariantName.QKV_QUANT,
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(wq_a, fallback_dtype=np.float32),
                _sample_array(q_norm, fallback_dtype=np.float32),
                _sample_array(wq_b, fallback_dtype=np.float32),
                _sample_array(wkv, fallback_dtype=np.float32),
                _sample_array(kv_norm, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                n_heads=int(n_heads),
                head_dim=int(head_dim),
                rope_head_dim=int(rope_head_dim),
                eps=float(eps),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
                **extra_kwargs,
            ),
        )

    def _attention_qkv_kv_cache_write_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        kv_cache: Any,
        slot_mapping: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        eps: float,
        block_size: int,
        fp8_max: float,
        q_softmax_scale: float,
        q_token_bucket: int,
        kv_token_bucket: int,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        key = (
            "kv_cache_write_noqr",
            x_shape,
            str(getattr(x, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wq_a, "shape", ())),
            str(getattr(wq_a, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(q_norm, "shape", ())),
            str(getattr(q_norm, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wq_b, "shape", ())),
            str(getattr(wq_b, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(wkv, "shape", ())),
            str(getattr(wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(kv_norm, "shape", ())),
            str(getattr(kv_norm, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(kv_cache, "shape", ())),
            str(getattr(kv_cache, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(slot_mapping, "shape", ())),
            str(getattr(slot_mapping, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            pos_shape,
            str(getattr(positions, "dtype", "unknown")),
            int(n_heads),
            int(head_dim),
            int(rope_head_dim),
            float(eps),
            int(block_size),
            float(fp8_max),
            float(q_softmax_scale),
            int(q_token_bucket),
            int(kv_token_bucket),
        )
        name = (
            "dsv4_product_attention_qkv_table_scaled_kvwrite_noqr_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"kv{'x'.join(str(v) for v in getattr(kv_cache, 'shape', ())) or '0'}_"
            f"s{'x'.join(str(v) for v in getattr(slot_mapping, 'shape', ())) or '0'}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=QkvVariantName.QKV_WRITE_KV_CACHE,
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.attention_qkv_quant_scaled_kv_cache_write_no_qr_from_freq_table_fn,
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(wq_a, fallback_dtype=np.float32),
                _sample_array(q_norm, fallback_dtype=np.float32),
                _sample_array(wq_b, fallback_dtype=np.float32),
                _sample_array(wkv, fallback_dtype=np.float32),
                _sample_array(kv_norm, fallback_dtype=np.float32),
                _sample_array(kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(slot_mapping, fallback_dtype=np.int32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                n_heads=int(n_heads),
                head_dim=int(head_dim),
                rope_head_dim=int(rope_head_dim),
                eps=float(eps),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                q_softmax_scale=float(q_softmax_scale),
                q_token_bucket=int(q_token_bucket),
                kv_token_bucket=int(kv_token_bucket),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )
