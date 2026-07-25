"""QKV/indexer product runtime runners."""

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
    token_topk_prep_widths as _token_topk_prep_widths,
)


class Dsv4ProductQkvIndexerRuntimeMixin:
    def _run_product_attention_qkv_indexer_compressor_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        indexer_compressor_wkv: Any,
        indexer_compressor_wgate: Any,
        indexer_wq_b: Any,
        indexer_weights_proj: Any,
        cos_table: Any,
        sin_table: Any,
        indexer_cos_table: Any,
        indexer_sin_table: Any,
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
        indexer_score_scale: float,
        indexer_n_heads: int,
        indexer_head_dim: int,
        indexer_rope_head_dim: int,
        indexer_block_size: int,
        indexer_fp8_max: float,
        window_size: int = 0,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_indexer_compressor_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_kv_norm",
        )
        _require_product_device_value(
            compressor_wkv,
            where="attention_qkv_indexer_compressor/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_indexer_compressor/compressor_wgate",
        )
        _require_product_device_value(
            indexer_compressor_wkv,
            where="attention_qkv_indexer_compressor/indexer_compressor_wkv",
        )
        _require_product_device_value(
            indexer_compressor_wgate,
            where="attention_qkv_indexer_compressor/indexer_compressor_wgate",
        )
        indexer_wq_b = _as_product_device_input(
            indexer_wq_b,
            name="dsv4_indexer_wq_b",
        )
        indexer_weights_proj = _as_product_device_input(
            indexer_weights_proj,
            name="dsv4_indexer_weights_proj",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        indexer_cos_table, indexer_sin_table = self._product_freq_tables_for(
            indexer_cos_table,
            indexer_sin_table,
            name="indexer",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        idx_comp_shape = tuple(
            int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())
        )
        if len(x_shape) != 3 or not comp_shape or not idx_comp_shape:
            raise RuntimeError(
                "DSV4 product fused attention/indexer prologue expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        bsz, seqlen, hidden_size = x_shape
        compiled_offset = int(seqlen)
        table_bucketed = False
        if int(seqlen) > 1 and int(bsz) * int(seqlen) <= int(bucket.token_bucket):
            x, seqlen, realiased, _offset = self._product_bucketed_prefill_offset(
                bucket,
                x,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                window_size=int(window_size),
                offset=int(seqlen),
                bucketed=True,
                max_compile_tokens=int(q_token_bucket),
            )
            compiled_offset = int(_offset)
            table_bucketed = bool(realiased)
            if realiased:
                x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
                if len(x_shape) != 3:
                    raise RuntimeError(
                        "DSV4 product bucketed indexer-table QKV expected x "
                        f"[batch, seqlen, hidden], got {x_shape}"
                    )
                bsz, seqlen, hidden_size = x_shape
                _nkipy_output_tensors = None
                q_token_bucket = int(bsz) * int(seqlen)
                kv_token_bucket = int(bsz) * int(seqlen)
        self._product_last_qkv_compiled_offset = (
            int(compiled_offset),
            bool(table_bucketed),
            int(seqlen),
        )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        kernel = self._attention_qkv_indexer_compressor_table_kernel_for(
            bucket,
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_wq_b,
            indexer_weights_proj,
            cos_table,
            sin_table,
            indexer_cos_table,
            indexer_sin_table,
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
            indexer_score_scale=float(indexer_score_scale),
            indexer_n_heads=int(indexer_n_heads),
            indexer_head_dim=int(indexer_head_dim),
            indexer_rope_head_dim=int(indexer_rope_head_dim),
            indexer_block_size=int(indexer_block_size),
            indexer_fp8_max=float(indexer_fp8_max),
        )
        n_tokens = int(bsz) * int(seqlen)
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
            kv = self._bucket_scratch(
                bucket,
                "attention_qkv_kv_flat",
                (int(kv_token_bucket), int(head_dim)),
                np.float32,
            )
        comp_kv = outputs.get("output2")
        if comp_kv is None:
            comp_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        comp_score = outputs.get("output3")
        if comp_score is None:
            comp_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        idx_kv = outputs.get("output4")
        if idx_kv is None:
            idx_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (n_tokens, int(idx_comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        idx_score = outputs.get("output5")
        if idx_score is None:
            idx_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (n_tokens, int(idx_comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        idx_q_t = outputs.get("output6")
        if idx_q_t is None:
            idx_q_t = self._bucket_scratch(
                bucket,
                "indexer_score_q_t",
                (n_tokens, int(indexer_head_dim), int(indexer_n_heads)),
                ml_dtypes.bfloat16,
            )
        idx_w_flat = outputs.get("output7")
        if idx_w_flat is None:
            idx_w_flat = self._bucket_scratch(
                bucket,
                "indexer_score_weights",
                (n_tokens, int(indexer_n_heads)),
                np.float32,
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
                "indexer_compressor_wkv": indexer_compressor_wkv,
                "indexer_compressor_wgate": indexer_compressor_wgate,
                "indexer_wq_b": indexer_wq_b,
                "indexer_weights_proj": indexer_weights_proj,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "indexer_cos_table": indexer_cos_table,
                "indexer_sin_table": indexer_sin_table,
                "positions": positions,
            },
            outputs={
                "output0": q,
                "output1": kv,
                "output2": comp_kv,
                "output3": comp_score,
                "output4": idx_kv,
                "output5": idx_score,
                "output6": idx_q_t,
                "output7": idx_w_flat,
            },
            unload_after_call=False,
        )
        return q, kv, comp_kv, comp_score, idx_kv, idx_score, idx_q_t, idx_w_flat

    def _run_product_attention_qkv_indexer_compressor_table_write_swa_state_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        indexer_compressor_wkv: Any,
        indexer_compressor_wgate: Any,
        indexer_wq_b: Any,
        indexer_weights_proj: Any,
        swa_kv_cache: Any,
        kv_score_state: Any,
        indexer_kv_score_state: Any,
        owner_ids: Any,
        compressor_ape: Any,
        indexer_compressor_ape: Any,
        cos_table: Any,
        sin_table: Any,
        indexer_cos_table: Any,
        indexer_sin_table: Any,
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
        indexer_score_scale: float,
        indexer_n_heads: int,
        indexer_head_dim: int,
        indexer_rope_head_dim: int,
        indexer_block_size: int,
        indexer_fp8_max: float,
        window_size: int,
        ratio: int,
        start_pos: int,
        compressor_ring_size: int,
        indexer_compressor_ring_size: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_indexer_compressor_table_write_swa_state_from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_indexer_table_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_kv_norm",
        )
        for value, where in (
            (compressor_wkv, "compressor_wkv"),
            (compressor_wgate, "compressor_wgate"),
            (indexer_compressor_wkv, "indexer_compressor_wkv"),
            (indexer_compressor_wgate, "indexer_compressor_wgate"),
            (swa_kv_cache, "swa_kv_cache"),
            (kv_score_state, "kv_score_state"),
            (indexer_kv_score_state, "indexer_kv_score_state"),
            (compressor_ape, "compressor_ape"),
            (indexer_compressor_ape, "indexer_compressor_ape"),
        ):
            _require_product_device_value(
                value,
                where=f"attention_qkv_table_write_swa_state/{where}",
            )
        indexer_wq_b = _as_product_device_input(
            indexer_wq_b,
            name="dsv4_indexer_wq_b",
        )
        indexer_weights_proj = _as_product_device_input(
            indexer_weights_proj,
            name="dsv4_indexer_weights_proj",
        )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_table_write_swa_state_owner_ids",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        indexer_cos_table, indexer_sin_table = self._product_freq_tables_for(
            indexer_cos_table,
            indexer_sin_table,
            name="indexer",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        idx_comp_shape = tuple(
            int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())
        )
        if len(x_shape) != 3 or not comp_shape or not idx_comp_shape:
            raise RuntimeError(
                "DSV4 product table dual SWA/state fusion expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        if int(seqlen) != 1 or int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product table dual SWA/state fusion requires decode, "
                f"got seqlen={int(seqlen)} start_pos={int(start_pos)}"
            )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        kernel = self._attention_qkv_indexer_compressor_table_kernel_for(
            bucket,
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_wq_b,
            indexer_weights_proj,
            cos_table,
            sin_table,
            indexer_cos_table,
            indexer_sin_table,
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
            indexer_score_scale=float(indexer_score_scale),
            indexer_n_heads=int(indexer_n_heads),
            indexer_head_dim=int(indexer_head_dim),
            indexer_rope_head_dim=int(indexer_rope_head_dim),
            indexer_block_size=int(indexer_block_size),
            indexer_fp8_max=float(indexer_fp8_max),
            dynamic_decode_start_pos=True,
            write_swa_dual_state=True,
            swa_kv_cache=swa_kv_cache,
            kv_score_state=kv_score_state,
            indexer_kv_score_state=indexer_kv_score_state,
            owner_ids=owner_ids,
            compressor_ape=compressor_ape,
            indexer_compressor_ape=indexer_compressor_ape,
            window_size=int(window_size),
            ratio=int(ratio),
            start_pos=int(start_pos),
            compressor_ring_size=int(compressor_ring_size),
            indexer_compressor_ring_size=int(indexer_compressor_ring_size),
        )
        n_tokens = int(bsz) * int(seqlen)
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
            kv = self._bucket_scratch(
                bucket,
                "attention_qkv_kv_flat",
                (int(kv_token_bucket), int(head_dim)),
                np.float32,
            )
        idx_q_t = outputs.get("output2")
        if idx_q_t is None:
            idx_q_t = self._bucket_scratch(
                bucket,
                "indexer_score_q_t",
                (n_tokens, int(indexer_head_dim), int(indexer_n_heads)),
                ml_dtypes.bfloat16,
            )
        idx_w_flat = outputs.get("output3")
        if idx_w_flat is None:
            idx_w_flat = self._bucket_scratch(
                bucket,
                "indexer_score_weights",
                (n_tokens, int(indexer_n_heads)),
                np.float32,
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
                "indexer_compressor_wkv": indexer_compressor_wkv,
                "indexer_compressor_wgate": indexer_compressor_wgate,
                "indexer_wq_b": indexer_wq_b,
                "indexer_weights_proj": indexer_weights_proj,
                "swa_kv_cache.must_alias_input": swa_kv_cache,
                "kv_score_state.must_alias_input": kv_score_state,
                "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
                "owner_ids": owner_ids,
                "compressor_ape": compressor_ape,
                "indexer_compressor_ape": indexer_compressor_ape,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "indexer_cos_table": indexer_cos_table,
                "indexer_sin_table": indexer_sin_table,
                "positions": positions,
            },
            outputs={
                "output0": q,
                "output1": kv,
                "output2": idx_q_t,
                "output3": idx_w_flat,
                "swa_kv_cache": swa_kv_cache,
                "kv_score_state": kv_score_state,
                "indexer_kv_score_state": indexer_kv_score_state,
            },
            unload_after_call=False,
        )
        return q, kv, idx_q_t, idx_w_flat

    def _run_product_attention_qkv_empty_indexer_compressor_topk_from_freq_table(
        self,
        x: Any,
        wq_a: Any,
        q_norm: Any,
        wq_b: Any,
        wkv: Any,
        kv_norm: Any,
        compressor_wkv: Any,
        compressor_wgate: Any,
        indexer_compressor_wkv: Any,
        indexer_compressor_wgate: Any,
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
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_empty_indexer_compressor_topk_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_empty_indexer_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(
            kv_norm,
            name="dsv4_attention_qkv_kv_norm",
        )
        _require_product_device_value(
            compressor_wkv,
            where="attention_qkv_empty_indexer/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_empty_indexer/compressor_wgate",
        )
        _require_product_device_value(
            indexer_compressor_wkv,
            where="attention_qkv_empty_indexer/indexer_compressor_wkv",
        )
        _require_product_device_value(
            indexer_compressor_wgate,
            where="attention_qkv_empty_indexer/indexer_compressor_wgate",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        idx_comp_shape = tuple(
            int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())
        )
        if len(x_shape) != 3 or not comp_shape or not idx_comp_shape:
            raise RuntimeError(
                "DSV4 product fused empty-indexer prologue expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        dynamic_decode_start_pos = int(start_pos) > 0 and int(seqlen) == 1
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
        kernel = (
            self._attention_qkv_empty_indexer_compressor_token_topk_prep_kernel_for(
                bucket,
                x,
                wq_a,
                q_norm,
                wq_b,
                wkv,
                kv_norm,
                compressor_wkv,
                compressor_wgate,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
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
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=int(max_c_len),
                rows=int(rows),
                k_tile=int(k_tile),
                dynamic_decode_start_pos=bool(dynamic_decode_start_pos),
            )
        )
        n_tokens = int(bsz) * int(seqlen)
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
            kv = self._bucket_scratch(
                bucket,
                "attention_qkv_kv_flat",
                (int(kv_token_bucket), int(head_dim)),
                np.float32,
            )
        comp_kv = outputs.get("output2")
        if comp_kv is None:
            comp_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        comp_score = outputs.get("output3")
        if comp_score is None:
            comp_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (n_tokens, int(comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        idx_kv = outputs.get("output4")
        if idx_kv is None:
            idx_kv = self._bucket_scratch(
                bucket,
                "compressor_kv_bf16",
                (n_tokens, int(idx_comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        idx_score = outputs.get("output5")
        if idx_score is None:
            idx_score = self._bucket_scratch(
                bucket,
                "compressor_score_bf16",
                (n_tokens, int(idx_comp_shape[0])),
                ml_dtypes.bfloat16,
            )
        topk_t = outputs.get("output6")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get("output7")
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
                "indexer_compressor_wkv": indexer_compressor_wkv,
                "indexer_compressor_wgate": indexer_compressor_wgate,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs={
                "output0": q,
                "output1": kv,
                "output2": comp_kv,
                "output3": comp_score,
                "output4": idx_kv,
                "output5": idx_score,
                "output6": topk_t,
                "output7": mask,
            },
            unload_after_call=False,
        )
        return q, kv, comp_kv, comp_score, idx_kv, idx_score, topk_t, mask
