"""Token-top-k QKV product runtime runners."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _as_product_device_input,
    _require_product_device_value,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    bucketed_prefill_token_topk_compile_shape as _bucketed_prefill_token_topk_compile_shape,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    token_topk_prep_widths as _token_topk_prep_widths,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape", ()))


def _output_shape_matches(
    outputs: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
) -> bool:
    value = outputs.get(name)
    if value is None:
        return True
    return _tensor_shape(value) == tuple(int(dim) for dim in shape)


class Dsv4ProductQkvTokenTopkRuntimeMixin:
    def _run_product_attention_qkv_compressor_prefill_post_qdq_token_topk_prep_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        compressor_ape: Any,
        compressor_norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        compressor_cos_table: Any,
        compressor_sin_table: Any,
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
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int | None = None,
        compressor_head_dim: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        return_qr: bool = False,
        write_swa_state_cache: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        compressed_kv_cache: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ring_size: int = 0,
        compressor_state_tail_len: int = 0,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        if bool(return_qr):
            raise RuntimeError(
                "DSV4 product prefill compressor post-QDQ fusion does not "
                "materialize QR"
            )
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_compressor_prefill_post_qdq_token_topk_prep_"
                "from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_prefill_comp_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(kv_norm, name="dsv4_attention_qkv_kv_norm")
        _require_product_device_value(
            compressor_wkv,
            where="attention_qkv_prefill_comp/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_prefill_comp/compressor_wgate",
        )
        _require_product_device_value(
            compressor_ape,
            where="attention_qkv_prefill_comp/compressor_ape",
        )
        _require_product_device_value(
            compressor_norm_weight,
            where="attention_qkv_prefill_comp/compressor_norm_weight",
        )
        if bool(write_swa_state_cache):
            for value, where in (
                (swa_kv_cache, "swa_kv_cache"),
                (kv_score_state, "kv_score_state"),
                (compressed_kv_cache, "compressed_kv_cache"),
            ):
                _require_product_device_value(
                    value,
                    where=f"attention_qkv_prefill_comp/{where}",
                )
            owner_ids = _as_product_device_input(
                owner_ids,
                name="dsv4_product_prefill_comp_owner_ids",
            )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        compressor_cos_table, compressor_sin_table = self._product_freq_tables_for(
            compressor_cos_table,
            compressor_sin_table,
            name="compressor",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        if len(x_shape) != 3 or not comp_shape:
            raise RuntimeError(
                "DSV4 product prefill compressor fusion expects x "
                f"[batch, seqlen, hidden] and compressor_wkv [width, hidden], "
                f"got {x_shape}/{comp_shape}"
            )
        bsz, seqlen, hidden_size = x_shape
        real_seqlen = int(seqlen)
        kv_bucket = int(kv_token_bucket or 0)
        # Bucketed prefill: compile the prologue at the token bucket and mask
        # the real length downstream (mirrors the all-KV indexer prologue).
        bucketed = int(start_pos) == 0 and not bool(write_swa_state_cache)
        # Single source of truth: publish (offset, did_realias, compiled_seqlen)
        # on the executor; the host drives the scatter geometry and the
        # two-source primary_len from it. Default to the passed offset.
        x, seqlen, realiased, offset = self._product_bucketed_prefill_offset(
            bucket,
            x,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            window_size=int(window_size),
            offset=int(offset),
            bucketed=bool(bucketed),
            max_compile_tokens=int(rows),
        )
        if realiased:
            x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
            if len(x_shape) != 3:
                raise RuntimeError(
                    "DSV4 product bucketed token-topk QKV expected x "
                    f"[batch, seqlen, hidden], got {x_shape}"
                )
            bsz, seqlen, hidden_size = x_shape
        self._product_last_qkv_compiled_offset = (
            int(offset),
            bool(realiased),
            int(seqlen),
        )
        _, _, _, k_padded = _token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
        clen = int(seqlen) // int(ratio)
        if int(start_pos) != 0 or clen <= 0:
            raise RuntimeError(
                "DSV4 product prefill compressor fusion requires a compressible "
                f"prefill bucket, got start_pos={int(start_pos)} seqlen={int(seqlen)} "
                f"ratio={int(ratio)}"
            )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        kernel = (
            self._attention_qkv_compressor_prefill_post_qdq_token_topk_prep_kernel_for(
                bucket,
                x,
                wq_a,
                q_norm,
                wq_b,
                wkv,
                kv_norm,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                compressor_cos_table,
                compressor_sin_table,
                positions,
                n_heads=int(n_heads),
                head_dim=int(head_dim),
                rope_head_dim=int(rope_head_dim),
                eps=float(eps),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                q_softmax_scale=float(q_softmax_scale),
                q_token_bucket=int(q_token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=int(max_c_len),
                rows=int(rows),
                k_tile=int(k_tile),
                kv_token_bucket=kv_bucket,
                compressor_head_dim=int(compressor_head_dim),
                compressor_rope_head_dim=int(compressor_rope_head_dim),
                compressor_block_size=int(compressor_block_size),
                compressor_fp8_max=float(compressor_fp8_max),
                compressor_rotate=bool(compressor_rotate),
                compressor_overlap=bool(compressor_overlap),
                compressor_eps=float(compressor_eps),
                write_swa_state_cache=bool(write_swa_state_cache),
                swa_kv_cache=swa_kv_cache,
                kv_score_state=kv_score_state,
                compressed_kv_cache=compressed_kv_cache,
                owner_ids=owner_ids,
                compressor_ring_size=int(compressor_ring_size),
                compressor_state_tail_len=int(compressor_state_tail_len),
            )
        )
        outputs = dict(_nkipy_output_tensors or {})
        if bucketed and int(seqlen) > real_seqlen:
            # Host pre-allocated compressor outputs at the REAL seqlen; the
            # bucketed prologue emits bucket rows — drop and re-alloc.
            for _k in ("output2", "output3", "output4", "output5", "output6"):
                outputs.pop(_k, None)
        q = outputs.get("output0")
        if q is None:
            q = self._bucket_scratch(
                bucket,
                "attention_q_scaled_t",
                (int(q_token_bucket), int(head_dim), int(n_heads)),
                ml_dtypes.bfloat16,
            )
        kv = outputs.get("output1")
        if kv is None:
            kv_kind = "attention_qkv_kv_flat" if kv_bucket else "attention_qkv_kv"
            kv_shape = (
                (kv_bucket, int(head_dim))
                if kv_bucket
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(bucket, kv_kind, kv_shape, np.float32)
        topk_t = outputs.get("output2")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get("output3")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
        n_tokens = int(bsz) * int(seqlen)
        comp_kv = outputs.get("output4")
        if comp_kv is None:
            comp_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        comp_score = outputs.get("output5")
        if comp_score is None:
            comp_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        comp_rows = outputs.get("output6")
        if comp_rows is None:
            comp_rows = self._bucket_scratch(
                bucket,
                "compressor_post_qdq_bf16",
                (int(bsz) * int(clen), int(compressor_head_dim)),
                ml_dtypes.bfloat16,
            )
        kernel_inputs = {
            "x": x,
            "wq_a": wq_a,
            "q_norm": q_norm,
            "wq_b": wq_b,
            "wkv": wkv,
            "kv_norm": kv_norm,
            "compressor_wkv": compressor_wkv,
            "compressor_wgate": compressor_wgate,
            "compressor_ape": compressor_ape,
            "compressor_norm_weight": compressor_norm_weight,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "compressor_cos_table": compressor_cos_table,
            "compressor_sin_table": compressor_sin_table,
            "positions": positions,
        }
        kernel_outputs = {
            "output0": q,
            "output1": kv,
            "output2": topk_t,
            "output3": mask,
            "output4": comp_kv,
            "output5": comp_score,
            "output6": comp_rows,
        }
        if bool(write_swa_state_cache):
            kernel_inputs.update(
                {
                    "swa_kv_cache.must_alias_input": swa_kv_cache,
                    "kv_score_state.must_alias_input": kv_score_state,
                    "compressed_kv_cache.must_alias_input": compressed_kv_cache,
                    "owner_ids": owner_ids,
                }
            )
            kernel_outputs.update(
                {
                    "swa_kv_cache": swa_kv_cache,
                    "kv_score_state": kv_score_state,
                    "compressed_kv_cache": compressed_kv_cache,
                }
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=kernel_inputs,
            outputs=kernel_outputs,
            unload_after_call=False,
        )
        return q, kv, topk_t, mask, comp_kv, comp_score, comp_rows

    def _run_product_attention_qkv_compressor_decode_post_qdq_token_topk_prep_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        kv_score_state: Any,
        owner_ids: Any,
        end_positions: Any,
        compressor_ape: Any,
        compressor_norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        compressor_cos_table: Any,
        compressor_sin_table: Any,
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
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int | None = None,
        compressor_head_dim: int,
        compressor_state_width: int,
        compressor_ring_size: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        return_qr: bool = False,
        write_swa_state_cache: bool = False,
        compressed_cache_stride: int = 0,
        swa_kv_cache: Any | None = None,
        compressed_kv_cache: Any | None = None,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        if bool(return_qr):
            raise RuntimeError(
                "DSV4 product decode compressor post-QDQ fusion does not materialize QR"
            )
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_compressor_decode_post_qdq_token_topk_prep_"
                "from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_decode_comp_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(kv_norm, name="dsv4_attention_qkv_kv_norm")
        for value, where in (
            (compressor_wkv, "compressor_wkv"),
            (compressor_wgate, "compressor_wgate"),
            (kv_score_state, "kv_score_state"),
            (compressor_ape, "compressor_ape"),
            (compressor_norm_weight, "compressor_norm_weight"),
        ):
            _require_product_device_value(
                value,
                where=f"attention_qkv_decode_comp/{where}",
            )
        if bool(write_swa_state_cache):
            for value, where in (
                (swa_kv_cache, "swa_kv_cache"),
                (compressed_kv_cache, "compressed_kv_cache"),
            ):
                _require_product_device_value(
                    value,
                    where=f"attention_qkv_decode_comp/{where}",
                )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_decode_comp_owner_ids",
        )
        end_positions = _as_product_device_input(
            end_positions,
            name="dsv4_product_decode_comp_end_positions",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        compressor_cos_table, compressor_sin_table = self._product_freq_tables_for(
            compressor_cos_table,
            compressor_sin_table,
            name="compressor",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        if len(x_shape) != 3 or not comp_shape:
            raise RuntimeError(
                "DSV4 product decode compressor fusion expects x "
                f"[batch, seqlen, hidden] and compressor_wkv [width, hidden], "
                f"got {x_shape}/{comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        if int(seqlen) != 1 or int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product decode compressor fusion requires decode shape, "
                f"got seqlen={int(seqlen)} start_pos={int(start_pos)}"
            )
        kv_bucket = int(kv_token_bucket or 0)
        _, _, _, k_padded = _token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        kernel = (
            self._attention_qkv_compressor_decode_post_qdq_token_topk_prep_kernel_for(
                bucket,
                x,
                wq_a,
                q_norm,
                wq_b,
                wkv,
                kv_norm,
                compressor_wkv,
                compressor_wgate,
                kv_score_state,
                owner_ids,
                end_positions,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                compressor_cos_table,
                compressor_sin_table,
                positions,
                n_heads=int(n_heads),
                head_dim=int(head_dim),
                rope_head_dim=int(rope_head_dim),
                eps=float(eps),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                q_softmax_scale=float(q_softmax_scale),
                q_token_bucket=int(q_token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=int(max_c_len),
                rows=int(rows),
                k_tile=int(k_tile),
                kv_token_bucket=kv_bucket,
                compressor_head_dim=int(compressor_head_dim),
                compressor_state_width=int(compressor_state_width),
                compressor_ring_size=int(compressor_ring_size),
                compressor_rope_head_dim=int(compressor_rope_head_dim),
                compressor_block_size=int(compressor_block_size),
                compressor_fp8_max=float(compressor_fp8_max),
                compressor_rotate=bool(compressor_rotate),
                compressor_overlap=bool(compressor_overlap),
                compressor_eps=float(compressor_eps),
                write_swa_state_cache=bool(write_swa_state_cache),
                compressed_cache_stride=int(compressed_cache_stride),
                swa_kv_cache=swa_kv_cache,
                compressed_kv_cache=compressed_kv_cache,
            )
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
        kv = outputs.get("output1")
        if kv is None:
            kv_kind = "attention_qkv_kv_flat" if kv_bucket else "attention_qkv_kv"
            kv_shape = (
                (kv_bucket, int(head_dim))
                if kv_bucket
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(bucket, kv_kind, kv_shape, np.float32)
        topk_t = outputs.get("output2")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get("output3")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
        n_tokens = int(bsz) * int(seqlen)
        comp_kv = None
        comp_score = None
        comp_rows = None
        if not bool(write_swa_state_cache):
            comp_kv = outputs.get("output4")
            if comp_kv is None:
                comp_kv = self._bucket_scratch(
                    bucket,
                    "compressor_kv_bf16",
                    (n_tokens, int(comp_shape[0])),
                    ml_dtypes.bfloat16,
                )
            comp_score = outputs.get("output5")
            if comp_score is None:
                comp_score = self._bucket_scratch(
                    bucket,
                    "compressor_score_bf16",
                    (n_tokens, int(comp_shape[0])),
                    ml_dtypes.bfloat16,
                )
            comp_rows = outputs.get("output6")
            if comp_rows is None:
                comp_rows = self._bucket_scratch(
                    bucket,
                    "compressor_decode_post_qdq_bf16",
                    (int(bsz), int(compressor_head_dim)),
                    ml_dtypes.bfloat16,
                )
        inputs = {
            "x": x,
            "wq_a": wq_a,
            "q_norm": q_norm,
            "wq_b": wq_b,
            "wkv": wkv,
            "kv_norm": kv_norm,
            "compressor_wkv": compressor_wkv,
            "compressor_wgate": compressor_wgate,
            "owner_ids": owner_ids,
            "end_positions": end_positions,
            "compressor_ape": compressor_ape,
            "compressor_norm_weight": compressor_norm_weight,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "compressor_cos_table": compressor_cos_table,
            "compressor_sin_table": compressor_sin_table,
            "positions": positions,
        }
        outputs_map = {
            "output0": q,
            "output1": kv,
            "output2": topk_t,
            "output3": mask,
        }
        if bool(write_swa_state_cache):
            inputs.update(
                {
                    "swa_kv_cache.must_alias_input": swa_kv_cache,
                    "kv_score_state.must_alias_input": kv_score_state,
                    "compressed_kv_cache.must_alias_input": compressed_kv_cache,
                }
            )
            outputs_map.update(
                {
                    "swa_kv_cache": swa_kv_cache,
                    "kv_score_state": kv_score_state,
                    "compressed_kv_cache": compressed_kv_cache,
                }
            )
        else:
            inputs["kv_score_state"] = kv_score_state
            outputs_map.update(
                {
                    "output4": comp_kv,
                    "output5": comp_score,
                    "output6": comp_rows,
                }
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=inputs,
            outputs=outputs_map,
            unload_after_call=False,
        )
        if bool(write_swa_state_cache):
            return q, kv, topk_t, mask
        return q, kv, topk_t, mask, comp_kv, comp_score, comp_rows

    def _run_product_attention_qkv_compressor_token_topk_prep_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
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
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int | None = None,
        return_qr: bool = False,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        if bool(return_qr):
            raise RuntimeError(
                "DSV4 product fused QKV/compressor/token-topk path does not "
                "materialize QR"
            )
        bucket = self._require_active_product_bucket(
            where="attention_qkv_compressor_token_topk_prep_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_topk_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_topk_wq_a")
        q_norm = _as_product_device_input(
            q_norm,
            name="dsv4_attention_qkv_topk_q_norm",
        )
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_topk_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_topk_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_topk_kv_norm",
        )
        _require_product_device_value(
            compressor_wkv,
            where="attention_qkv_compressor_token_topk/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_compressor_token_topk/compressor_wgate",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        if len(x_shape) != 3 or not comp_shape:
            raise RuntimeError(
                "DSV4 product fused QKV/compressor/token-topk expects x "
                f"[batch, seqlen, hidden] and compressor_wkv [width, hidden], "
                f"got {x_shape}/{comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        kv_bucket = int(kv_token_bucket or 0)
        _, _, _, k_padded = _token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
        n_tokens = int(bsz) * int(seqlen)
        outputs = dict(_nkipy_output_tensors or {})
        kernel_x = x
        kernel_positions = None
        kernel_shape = x_shape
        kernel_n_tokens = n_tokens
        kernel_offset = int(offset)
        bucketed_shape = _bucketed_prefill_token_topk_compile_shape(
            x_shape,
            canonical_rows=int(rows),
            q_token_bucket=int(q_token_bucket),
            kv_token_bucket=kv_bucket,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            k_tile=int(k_tile),
        )
        if bucketed_shape is not None:
            full_bsz, full_seqlen, full_rows, full_offset = bucketed_shape
            full_shape = (int(full_bsz), int(full_seqlen), int(x_shape[2]))
            full_x = self._product_full_value_for(x, full_shape)
            if (
                full_x is None
                and int(start_pos) == 0
                and int(full_bsz) == 1
                and int(full_seqlen) > int(seqlen)
            ):
                pad_hidden = getattr(self, "_run_product_sequence_hidden_pad", None)
                if callable(pad_hidden):
                    full_x = pad_hidden(
                        bucket,
                        x,
                        rows=int(full_seqlen),
                        hidden_size=int(x_shape[2]),
                    )

            full_positions = self._product_freq_positions_for(
                bucket,
                positions,
                rows=int(full_rows),
            )
            if (
                _tensor_shape(full_positions) != (int(full_rows),)
                and int(start_pos) == 0
                and int(full_bsz) == 1
            ):
                full_positions = self._product_freq_positions_for(
                    bucket,
                    np.arange(int(full_rows), dtype=np.int32),
                    rows=int(full_rows),
                )
            if (
                full_x is not None
                and _tensor_shape(full_positions) == (int(full_rows),)
                and _output_shape_matches(
                    outputs,
                    "output0",
                    (int(q_token_bucket), int(head_dim), int(n_heads)),
                )
                and _output_shape_matches(
                    outputs, "output1", (kv_bucket, int(head_dim))
                )
                and _output_shape_matches(
                    outputs, "output2", (int(k_padded), int(rows))
                )
                and _output_shape_matches(
                    outputs, "output3", (int(rows), int(k_padded))
                )
                and _output_shape_matches(
                    outputs,
                    "output4",
                    (int(full_rows), int(comp_shape[0])),
                )
                and _output_shape_matches(
                    outputs,
                    "output5",
                    (int(full_rows), int(comp_shape[0])),
                )
            ):
                kernel_x = full_x
                kernel_positions = full_positions
                kernel_shape = full_shape
                kernel_n_tokens = int(full_rows)
                kernel_offset = int(full_offset)
        if kernel_positions is None:
            kernel_positions = self._product_freq_positions_for(
                bucket,
                positions,
                rows=n_tokens,
            )
        dynamic_decode_start_pos = int(start_pos) > 0 and int(seqlen) == 1
        kernel = self._attention_qkv_compressor_token_topk_prep_kernel_for(
            bucket,
            kernel_x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            kernel_positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(kernel_offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=kv_bucket,
            dynamic_decode_start_pos=bool(dynamic_decode_start_pos),
        )
        q = outputs.get("output0")
        if q is None:
            q = self._bucket_scratch(
                bucket,
                "attention_q_scaled_t",
                (int(q_token_bucket), int(head_dim), int(n_heads)),
                ml_dtypes.bfloat16,
            )
        kv = outputs.get("output1")
        if kv is None:
            kv_kind = "attention_qkv_kv_flat" if kv_bucket else "attention_qkv_kv"
            kv_shape = (
                (kv_bucket, int(head_dim))
                if kv_bucket
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(bucket, kv_kind, kv_shape, np.float32)
        topk_t = outputs.get("output2")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get("output3")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
        comp_kv = outputs.get("output4")
        if comp_kv is not None and tuple(getattr(comp_kv, "shape", ())) != (
            int(kernel_n_tokens),
            int(comp_shape[0]),
        ):
            # Host pre-allocated for a different token count (e.g. decode pad
            # batch) than the kernel's compile shape — let the runner alloc.
            comp_kv = None
        if comp_kv is None:
            comp_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (int(kernel_n_tokens), int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        comp_score = outputs.get("output5")
        if comp_score is not None and tuple(getattr(comp_score, "shape", ())) != (
            int(kernel_n_tokens),
            int(comp_shape[0]),
        ):
            comp_score = None
        if comp_score is None:
            comp_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (int(kernel_n_tokens), int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "x": kernel_x,
                "wq_a": wq_a,
                "q_norm": q_norm,
                "wq_b": wq_b,
                "wkv": wkv,
                "kv_norm": kv_norm,
                "compressor_wkv": compressor_wkv,
                "compressor_wgate": compressor_wgate,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": kernel_positions,
            },
            outputs={
                "output0": q,
                "output1": kv,
                "output2": topk_t,
                "output3": mask,
                "output4": comp_kv,
                "output5": comp_score,
            },
            unload_after_call=False,
        )
        if int(kernel_n_tokens) != n_tokens:
            active_shape = (n_tokens, int(comp_shape[0]))
            active_alias = getattr(self, "_product_active_alias", None)
            comp_kv_active = (
                active_alias(comp_kv, active_shape) if callable(active_alias) else None
            )
            comp_score_active = (
                active_alias(comp_score, active_shape)
                if callable(active_alias)
                else None
            )
            if _tensor_shape(comp_kv_active) != active_shape:
                comp_kv_active = _alias_device_value_first_dim_slice(
                    comp_kv,
                    start=0,
                    size=n_tokens,
                )
            if _tensor_shape(comp_score_active) != active_shape:
                comp_score_active = _alias_device_value_first_dim_slice(
                    comp_score,
                    start=0,
                    size=n_tokens,
                )
            if (
                comp_kv_active is None
                or comp_score_active is None
                or _tensor_shape(comp_kv_active) != active_shape
                or _tensor_shape(comp_score_active) != active_shape
            ):
                raise RuntimeError(
                    "DSV4 product fused QKV/compressor/token-topk could not "
                    f"alias bucketed compressor outputs back to {active_shape}; "
                    f"kernel_x_shape={kernel_shape}"
                )
            comp_kv = comp_kv_active
            comp_score = comp_score_active
        return q, kv, topk_t, mask, comp_kv, comp_score

    def _run_product_attention_qkv_compressor_token_topk_prep_write_swa_state_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        swa_kv_cache: Any,
        kv_score_state: Any,
        owner_ids: Any,
        compressor_ape: Any,
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
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int | None = None,
        compressor_ring_size: int,
        return_qr: bool = False,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        if bool(return_qr):
            raise RuntimeError(
                "DSV4 product fused QKV/compressor/SWA-state path does not "
                "materialize QR"
            )
        bucket = self._require_active_product_bucket(
            where="attention_qkv_compressor_token_topk_prep_write_swa_state_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_topk_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_topk_wq_a")
        q_norm = _as_product_device_input(
            q_norm,
            name="dsv4_attention_qkv_topk_q_norm",
        )
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_topk_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_topk_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_topk_kv_norm",
        )
        _require_product_device_value(
            compressor_wkv,
            where="attention_qkv_compressor_token_topk_write/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_compressor_token_topk_write/compressor_wgate",
        )
        _require_product_device_value(
            swa_kv_cache,
            where="attention_qkv_compressor_token_topk_write/swa_kv_cache",
        )
        _require_product_device_value(
            kv_score_state,
            where="attention_qkv_compressor_token_topk_write/kv_score_state",
        )
        _require_product_device_value(
            compressor_ape,
            where="attention_qkv_compressor_token_topk_write/compressor_ape",
        )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_compressor_state_owner_ids",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 3 or int(x_shape[1]) != 1 or int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product fused QKV/compressor/SWA-state path is decode-only; "
                f"got x_shape={x_shape}, start_pos={int(start_pos)}"
            )
        bsz, seqlen, _ = x_shape
        n_tokens = int(bsz) * int(seqlen)
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        if not comp_shape:
            raise RuntimeError("compressor_wkv must have a non-empty shape")
        kv_bucket = int(kv_token_bucket or 0)
        _, _, _, k_padded = _token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=n_tokens,
        )
        kernel = self._attention_qkv_compressor_token_topk_prep_kernel_for(
            bucket,
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
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
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=kv_bucket,
            dynamic_decode_start_pos=True,
            write_swa_state=True,
            swa_kv_cache=swa_kv_cache,
            kv_score_state=kv_score_state,
            owner_ids=owner_ids,
            compressor_ape=compressor_ape,
            compressor_ring_size=int(compressor_ring_size),
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
        kv = outputs.get("output1")
        if kv is None:
            kv_kind = "attention_qkv_kv_flat" if kv_bucket else "attention_qkv_kv"
            kv_shape = (
                (kv_bucket, int(head_dim))
                if kv_bucket
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(bucket, kv_kind, kv_shape, np.float32)
        topk_t = outputs.get("output2")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get("output3")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
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
                "compressor_wkv": compressor_wkv,
                "compressor_wgate": compressor_wgate,
                "swa_kv_cache.must_alias_input": swa_kv_cache,
                "kv_score_state.must_alias_input": kv_score_state,
                "owner_ids": owner_ids,
                "compressor_ape": compressor_ape,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs={
                "output0": q,
                "output1": kv,
                "output2": topk_t,
                "output3": mask,
                "swa_kv_cache": swa_kv_cache,
                "kv_score_state": kv_score_state,
            },
            unload_after_call=False,
        )
        return q, kv, topk_t, mask

    def _run_product_attention_qkv_token_topk_prep_from_freq_table(
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
        q_softmax_scale: float,
        q_token_bucket: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int | None = None,
        return_qr: bool = True,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_token_topk_prep_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_topk_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_topk_wq_a")
        q_norm = _as_product_device_input(
            q_norm,
            name="dsv4_attention_qkv_topk_q_norm",
        )
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_topk_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_topk_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_topk_kv_norm",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 3:
            raise RuntimeError(
                "DSV4 product QKV/top-k table path expects x "
                f"[batch, seqlen, hidden], got {x_shape}"
            )
        bsz, seqlen, _ = x_shape
        kv_bucket = int(kv_token_bucket or 0)
        _, _, _, k_padded = _token_topk_prep_widths(
            x_shape,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        dynamic_decode_start_pos = (
            int(start_pos) > 0 and int(seqlen) == 1 and not bool(return_qr)
        )
        kernel = self._attention_qkv_token_topk_prep_kernel_for(
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
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=kv_bucket,
            return_qr=bool(return_qr),
            dynamic_decode_start_pos=bool(dynamic_decode_start_pos),
        )
        q_low_dim = int(getattr(q_norm, "shape", (0,))[0])
        outputs = dict(_nkipy_output_tensors or {})
        q = outputs.get("output0")
        if q is None:
            q = self._bucket_scratch(
                bucket,
                "attention_q_scaled_t",
                (int(q_token_bucket), int(head_dim), int(n_heads)),
                ml_dtypes.bfloat16,
            )
        kv = outputs.get("output1")
        if kv is None:
            kv_kind = "attention_qkv_kv_flat" if kv_bucket else "attention_qkv_kv"
            kv_shape = (
                (kv_bucket, int(head_dim))
                if kv_bucket
                else (int(bsz), int(seqlen), int(head_dim))
            )
            kv = self._bucket_scratch(
                bucket,
                kv_kind,
                kv_shape,
                np.float32,
            )
        qr = outputs.get("output2") if bool(return_qr) else None
        if bool(return_qr) and qr is None:
            qr = self._bucket_scratch(
                bucket,
                "attention_qkv_qr",
                (int(bsz), int(seqlen), q_low_dim),
                np.float32,
            )
        topk_output_name = "output3" if bool(return_qr) else "output2"
        mask_output_name = "output4" if bool(return_qr) else "output3"
        topk_t = outputs.get(topk_output_name)
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get(mask_output_name)
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
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
            outputs=(
                {
                    "output0": q,
                    "output1": kv,
                    "output2": qr,
                    "output3": topk_t,
                    "output4": mask,
                }
                if bool(return_qr)
                else {
                    "output0": q,
                    "output1": kv,
                    "output2": topk_t,
                    "output3": mask,
                }
            ),
            unload_after_call=False,
        )
        return q, kv, qr, topk_t, mask
