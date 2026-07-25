"""All-KV QKV/indexer product runtime runners."""

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


class Dsv4ProductQkvIndexerAllKvRuntimeMixin:
    def _run_product_attention_qkv_indexer_compressor_all_kv_topk_from_freq_table(
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
        kv_len: int,
        k: int,
        rows: int,
        k_tile: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where="attention_qkv_indexer_compressor_all_kv_topk_from_freq_table"
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_indexer_allkv_x")
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
            where="attention_qkv_indexer_allkv/compressor_wkv",
        )
        _require_product_device_value(
            compressor_wgate,
            where="attention_qkv_indexer_allkv/compressor_wgate",
        )
        _require_product_device_value(
            indexer_compressor_wkv,
            where="attention_qkv_indexer_allkv/indexer_compressor_wkv",
        )
        _require_product_device_value(
            indexer_compressor_wgate,
            where="attention_qkv_indexer_allkv/indexer_compressor_wgate",
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
                "DSV4 product fused all-KV indexer prologue expects x "
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
        if int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV indexer prologue requires k == kv_len, "
                f"got k={int(k)} kv_len={int(kv_len)}"
            )
        if bool(dynamic_decode_start_pos):
            kv_len = self._product_all_kv_decode_compile_kv_len(
                kv_len=int(kv_len),
                seqlen=int(seqlen),
                window_size=int(window_size),
                k_tile=int(k_tile),
            )
            k = int(kv_len)
        win_width = (
            int(window_size)
            if int(start_pos) > 0
            else min(
                int(seqlen),
                int(window_size),
            )
        )
        k_raw = int(win_width) + int(k)
        k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
        kernel = self._attention_qkv_indexer_compressor_all_kv_topk_prep_kernel_for(
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
            kv_len=int(kv_len),
            k=int(k),
            rows=int(rows),
            k_tile=int(k_tile),
            dynamic_decode_start_pos=bool(dynamic_decode_start_pos),
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

    def _run_product_attention_qkv_indexer_compressor_all_kv_topk_write_swa_state_from_freq_table(
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
        swa_kv_cache: Any,
        kv_score_state: Any,
        indexer_kv_score_state: Any,
        owner_ids: Any,
        compressor_ape: Any,
        indexer_compressor_ape: Any,
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
        kv_len: int,
        k: int,
        rows: int,
        k_tile: int,
        compressor_ring_size: int,
        indexer_compressor_ring_size: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_indexer_compressor_all_kv_topk_write_swa_state_"
                "from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_indexer_allkv_x")
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
                where=f"attention_qkv_allkv_write_swa_state/{where}",
            )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_allkv_write_swa_state_owner_ids",
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
                "DSV4 product all-KV dual SWA/state fusion expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        if int(seqlen) != 1 or int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product all-KV dual SWA/state fusion requires decode, "
                f"got seqlen={int(seqlen)} start_pos={int(start_pos)}"
            )
        if int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV dual SWA/state fusion requires k == kv_len, "
                f"got k={int(k)} kv_len={int(kv_len)}"
            )
        compile_kv_len = self._product_all_kv_decode_compile_kv_len(
            kv_len=int(kv_len),
            seqlen=int(seqlen),
            window_size=int(window_size),
            k_tile=int(k_tile),
        )
        compile_k = int(compile_kv_len)
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        win_width = int(window_size)
        k_raw = int(win_width) + int(compile_k)
        k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
        kernel = self._attention_qkv_indexer_compressor_all_kv_topk_prep_kernel_for(
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
            kv_len=int(compile_kv_len),
            k=int(compile_k),
            rows=int(rows),
            k_tile=int(k_tile),
            dynamic_decode_start_pos=True,
            write_swa_dual_state=True,
            swa_kv_cache=swa_kv_cache,
            kv_score_state=kv_score_state,
            indexer_kv_score_state=indexer_kv_score_state,
            owner_ids=owner_ids,
            compressor_ape=compressor_ape,
            indexer_compressor_ape=indexer_compressor_ape,
            compressor_ring_size=int(compressor_ring_size),
            indexer_compressor_ring_size=int(indexer_compressor_ring_size),
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
            kv = self._bucket_scratch(
                bucket,
                "attention_qkv_kv_flat",
                (int(kv_token_bucket), int(head_dim)),
                np.float32,
            )
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
                "indexer_compressor_wkv": indexer_compressor_wkv,
                "indexer_compressor_wgate": indexer_compressor_wgate,
                "swa_kv_cache.must_alias_input": swa_kv_cache,
                "kv_score_state.must_alias_input": kv_score_state,
                "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
                "owner_ids": owner_ids,
                "compressor_ape": compressor_ape,
                "indexer_compressor_ape": indexer_compressor_ape,
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
                "indexer_kv_score_state": indexer_kv_score_state,
            },
            unload_after_call=False,
        )
        return q, kv, topk_t, mask

    def _run_product_attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_from_freq_table(
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
        indexer_compressor_wkv: Any,
        indexer_compressor_wgate: Any,
        indexer_compressor_ape: Any,
        indexer_compressor_norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        compressor_cos_table: Any,
        compressor_sin_table: Any,
        indexer_compressor_cos_table: Any,
        indexer_compressor_sin_table: Any,
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
        kv_len: int,
        k: int,
        rows: int,
        k_tile: int,
        compressor_head_dim: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        indexer_compressor_head_dim: int,
        indexer_compressor_rope_head_dim: int,
        indexer_compressor_block_size: int,
        indexer_compressor_fp8_max: float,
        indexer_compressor_rotate: bool,
        indexer_compressor_overlap: bool,
        indexer_compressor_eps: float,
        write_swa_state_cache: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        compressed_kv_cache: Any | None = None,
        indexer_kv_score_state: Any | None = None,
        indexer_compressed_kv_cache: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ring_size: int = 0,
        compressor_state_tail_len: int = 0,
        indexer_compressor_ring_size: int = 0,
        indexer_compressor_state_tail_len: int = 0,
        max_c_len: int | None = None,
        indexer_max_c_len: int | None = None,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_"
                "from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_allkv_prefill_x")
        wq_a = _as_product_device_input(wq_a, name="dsv4_attention_qkv_wq_a")
        q_norm = _as_product_device_input(q_norm, name="dsv4_attention_qkv_q_norm")
        wq_b = _as_product_device_input(wq_b, name="dsv4_attention_qkv_wq_b")
        wkv = _as_product_device_input(wkv, name="dsv4_attention_qkv_wkv")
        kv_norm = _as_product_device_input(kv_norm, name="dsv4_attention_qkv_kv_norm")
        for value, where in (
            (compressor_wkv, "compressor_wkv"),
            (compressor_wgate, "compressor_wgate"),
            (compressor_ape, "compressor_ape"),
            (compressor_norm_weight, "compressor_norm_weight"),
            (indexer_compressor_wkv, "indexer_compressor_wkv"),
            (indexer_compressor_wgate, "indexer_compressor_wgate"),
            (indexer_compressor_ape, "indexer_compressor_ape"),
            (indexer_compressor_norm_weight, "indexer_compressor_norm_weight"),
        ):
            _require_product_device_value(
                value,
                where=f"attention_qkv_allkv_prefill/{where}",
            )
        if bool(write_swa_state_cache):
            for value, where in (
                (swa_kv_cache, "swa_kv_cache"),
                (kv_score_state, "kv_score_state"),
                (compressed_kv_cache, "compressed_kv_cache"),
                (indexer_kv_score_state, "indexer_kv_score_state"),
                (indexer_compressed_kv_cache, "indexer_compressed_kv_cache"),
            ):
                _require_product_device_value(
                    value,
                    where=f"attention_qkv_allkv_prefill/{where}",
                )
            owner_ids = _as_product_device_input(
                owner_ids,
                name="dsv4_product_allkv_prefill_owner_ids",
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
        (
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
        ) = self._product_freq_tables_for(
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
            name="indexer_compressor",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        idx_comp_shape = tuple(
            int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())
        )
        if len(x_shape) != 3 or not comp_shape or not idx_comp_shape:
            raise RuntimeError(
                "DSV4 product all-KV prefill compressor fusion expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        if int(start_pos) != 0 or int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV prefill compressor fusion requires "
                f"start_pos=0 and k==kv_len, got start_pos={int(start_pos)} "
                f"k={int(k)} kv_len={int(kv_len)}"
            )
        bsz, seqlen, hidden_size = x_shape
        # Bucketed prefill: compile the prologue at the token bucket (one NEFF
        # per bucket) instead of per exact prompt length. Re-alias x up to the
        # bucket seqlen and re-derive every length-dependent scalar from the
        # bucket; the real seqlen survives only as the runtime cache-write mask
        # (cache_real_clen) applied later by the fused NKI scatter. The sparse
        # top-k is position-causal so the selected KV set is identical for real
        # rows (host-proven). See dsv4_prefill_bucket_integration.
        bucketed = int(start_pos) == 0
        # Single source of truth for the compiled top-k offset: whatever this
        # runner bakes into the NEFF is published on the executor so the host
        # drives BOTH the scatter geometry and the two-source primary_len from
        # it (see sampled_attention run_dsv4_attention). Default: the offset
        # passed in (no re-alias). (offset, did_realias, compiled_seqlen).
        x, seqlen, realiased, offset = self._product_bucketed_prefill_offset(
            bucket,
            x,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            window_size=int(window_size),
            offset=int(offset),
            bucketed=bool(bucketed),
        )
        if bucketed:
            compiled_tokens = int(bsz) * int(seqlen)
            q_token_bucket = int(compiled_tokens)
            kv_token_bucket = int(compiled_tokens)
            rows = int(compiled_tokens)
        if realiased:
            x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
            if len(x_shape) != 3:
                raise RuntimeError(
                    "DSV4 product bucketed all-KV QKV expected x "
                    f"[batch, seqlen, hidden], got {x_shape}"
                )
            bsz, seqlen, hidden_size = x_shape
            # Re-derive every length-dependent scalar from the bucket seqlen.
            # State tails size the prologue's returned state rows and are baked
            # into the NEFF key; the bucketed scatter reads only the real tail
            # (overlap tail = ratio + seqlen%ratio).
            kv_len = int(seqlen) // int(ratio)
            k = int(kv_len)
            if compressor_state_tail_len > 0:
                compressor_state_tail_len = int(ratio) + int(seqlen) % int(ratio)
            if indexer_compressor_state_tail_len > 0:
                indexer_compressor_state_tail_len = int(ratio) + int(seqlen) % int(
                    ratio
                )
        self._product_last_qkv_compiled_offset = (
            int(offset),
            bool(realiased),
            int(seqlen),
        )
        clen = int(seqlen) // int(ratio)
        if clen <= 0:
            raise RuntimeError(
                "DSV4 product all-KV prefill compressor fusion requires a "
                f"compressible prefill bucket, got seqlen={int(seqlen)} "
                f"ratio={int(ratio)}"
            )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        win_width = min(int(seqlen), int(window_size))
        k_raw = int(win_width) + int(k)
        k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
        kernel = self._attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_kernel_for(
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
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            cos_table,
            sin_table,
            compressor_cos_table,
            compressor_sin_table,
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
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
            kv_len=int(kv_len),
            k=int(k),
            rows=int(rows),
            k_tile=int(k_tile),
            compressor_head_dim=int(compressor_head_dim),
            compressor_rope_head_dim=int(compressor_rope_head_dim),
            compressor_block_size=int(compressor_block_size),
            compressor_fp8_max=float(compressor_fp8_max),
            compressor_rotate=bool(compressor_rotate),
            compressor_overlap=bool(compressor_overlap),
            compressor_eps=float(compressor_eps),
            indexer_compressor_head_dim=int(indexer_compressor_head_dim),
            indexer_compressor_rope_head_dim=int(indexer_compressor_rope_head_dim),
            indexer_compressor_block_size=int(indexer_compressor_block_size),
            indexer_compressor_fp8_max=float(indexer_compressor_fp8_max),
            indexer_compressor_rotate=bool(indexer_compressor_rotate),
            indexer_compressor_overlap=bool(indexer_compressor_overlap),
            indexer_compressor_eps=float(indexer_compressor_eps),
            write_swa_state_cache=bool(write_swa_state_cache),
            swa_kv_cache=swa_kv_cache,
            kv_score_state=kv_score_state,
            compressed_kv_cache=compressed_kv_cache,
            indexer_kv_score_state=indexer_kv_score_state,
            indexer_compressed_kv_cache=indexer_compressed_kv_cache,
            owner_ids=owner_ids,
            compressor_ring_size=int(compressor_ring_size),
            compressor_state_tail_len=int(compressor_state_tail_len),
            indexer_compressor_ring_size=int(indexer_compressor_ring_size),
            indexer_compressor_state_tail_len=int(indexer_compressor_state_tail_len),
            max_c_len=int(max_c_len or 0),
            indexer_max_c_len=int(indexer_max_c_len or 0),
        )
        n_tokens = int(bsz) * int(seqlen)
        outputs = dict(_nkipy_output_tensors or {})
        if bucketed:
            # Host pre-allocates outputs before this runner canonicalizes x to
            # the product bucket shape. Keep caller q/kv only when their row
            # count matches the compiled QKV token bucket; compressor outputs
            # are always rebuilt at the compiled bucket shape.
            for _k, _rows in (
                ("output0", int(q_token_bucket)),
                ("output1", int(kv_token_bucket)),
            ):
                _out = outputs.get(_k)
                _shape = tuple(int(dim) for dim in getattr(_out, "shape", ()))
                if not _shape or int(_shape[0]) != int(_rows):
                    outputs.pop(_k, None)
            for _k in (
                "output2",
                "output3",
                "output4",
                "output5",
                "output6",
                "output7",
                "output8",
                "output9",
            ):
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
        comp_rows = outputs.get("output8")
        if comp_rows is None:
            comp_rows = self._bucket_scratch(
                bucket,
                "compressor_post_qdq_bf16",
                (int(bsz) * int(clen), int(compressor_head_dim)),
                ml_dtypes.bfloat16,
            )
        idx_rows = outputs.get("output9")
        if idx_rows is None:
            idx_rows = self._bucket_scratch(
                bucket,
                "compressor_post_qdq_bf16",
                (int(bsz) * int(clen), int(indexer_compressor_head_dim)),
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
            "indexer_compressor_wkv": indexer_compressor_wkv,
            "indexer_compressor_wgate": indexer_compressor_wgate,
            "indexer_compressor_ape": indexer_compressor_ape,
            "indexer_compressor_norm_weight": indexer_compressor_norm_weight,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "compressor_cos_table": compressor_cos_table,
            "compressor_sin_table": compressor_sin_table,
            "indexer_compressor_cos_table": indexer_compressor_cos_table,
            "indexer_compressor_sin_table": indexer_compressor_sin_table,
            "positions": positions,
        }
        kernel_outputs = {
            "output0": q,
            "output1": kv,
            "output2": comp_kv,
            "output3": comp_score,
            "output4": idx_kv,
            "output5": idx_score,
            "output6": topk_t,
            "output7": mask,
            "output8": comp_rows,
            "output9": idx_rows,
        }
        if bool(write_swa_state_cache):
            kernel_inputs.update(
                {
                    "swa_kv_cache.must_alias_input": swa_kv_cache,
                    "kv_score_state.must_alias_input": kv_score_state,
                    "compressed_kv_cache.must_alias_input": compressed_kv_cache,
                    "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
                    "indexer_compressed_kv_cache.must_alias_input": (
                        indexer_compressed_kv_cache
                    ),
                    "owner_ids": owner_ids,
                }
            )
            kernel_outputs.update(
                {
                    "swa_kv_cache": swa_kv_cache,
                    "kv_score_state": kv_score_state,
                    "compressed_kv_cache": compressed_kv_cache,
                    "indexer_kv_score_state": indexer_kv_score_state,
                    "indexer_compressed_kv_cache": indexer_compressed_kv_cache,
                }
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=kernel_inputs,
            outputs=kernel_outputs,
            unload_after_call=False,
        )
        return (
            q,
            kv,
            comp_kv,
            comp_score,
            idx_kv,
            idx_score,
            topk_t,
            mask,
            comp_rows,
            idx_rows,
        )

    def _run_product_attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_from_freq_table(
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
        indexer_compressor_wkv: Any,
        indexer_compressor_wgate: Any,
        indexer_kv_score_state: Any,
        indexer_compressor_ape: Any,
        indexer_compressor_norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        compressor_cos_table: Any,
        compressor_sin_table: Any,
        indexer_compressor_cos_table: Any,
        indexer_compressor_sin_table: Any,
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
        kv_len: int,
        k: int,
        rows: int,
        k_tile: int,
        compressor_head_dim: int,
        compressor_state_width: int,
        compressor_ring_size: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        indexer_compressor_head_dim: int,
        indexer_compressor_state_width: int,
        indexer_compressor_ring_size: int,
        indexer_compressor_rope_head_dim: int,
        indexer_compressor_block_size: int,
        indexer_compressor_fp8_max: float,
        indexer_compressor_rotate: bool,
        indexer_compressor_overlap: bool,
        indexer_compressor_eps: float,
        write_swa_state_cache: bool = False,
        swa_kv_cache: Any | None = None,
        compressed_kv_cache: Any | None = None,
        indexer_compressed_kv_cache: Any | None = None,
        max_c_len: int | None = None,
        indexer_max_c_len: int | None = None,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        bucket = self._require_active_product_bucket(
            where=(
                "attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_"
                "from_freq_table"
            )
        )
        x = _as_product_device_input(x, name="dsv4_attention_qkv_allkv_decode_x")
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
            (indexer_compressor_wkv, "indexer_compressor_wkv"),
            (indexer_compressor_wgate, "indexer_compressor_wgate"),
            (indexer_kv_score_state, "indexer_kv_score_state"),
            (indexer_compressor_ape, "indexer_compressor_ape"),
            (indexer_compressor_norm_weight, "indexer_compressor_norm_weight"),
        ):
            _require_product_device_value(
                value,
                where=f"attention_qkv_allkv_decode/{where}",
            )
        if bool(write_swa_state_cache):
            for value, where in (
                (swa_kv_cache, "swa_kv_cache"),
                (compressed_kv_cache, "compressed_kv_cache"),
                (indexer_compressed_kv_cache, "indexer_compressed_kv_cache"),
            ):
                _require_product_device_value(
                    value,
                    where=f"attention_qkv_allkv_decode/{where}",
                )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_allkv_decode_owner_ids",
        )
        end_positions = _as_product_device_input(
            end_positions,
            name="dsv4_product_allkv_decode_end_positions",
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
        (
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
        ) = self._product_freq_tables_for(
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
            name="indexer_compressor",
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        comp_shape = tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ()))
        idx_comp_shape = tuple(
            int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())
        )
        if len(x_shape) != 3 or not comp_shape or not idx_comp_shape:
            raise RuntimeError(
                "DSV4 product all-KV decode compressor fusion expects x "
                "[batch, seqlen, hidden] and compressor weights with width, got "
                f"{x_shape}/{comp_shape}/{idx_comp_shape}"
            )
        bsz, seqlen, _ = x_shape
        if int(seqlen) != 1 or int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product all-KV decode compressor fusion requires decode, "
                f"got seqlen={int(seqlen)} start_pos={int(start_pos)}"
            )
        if int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV decode compressor fusion requires "
                f"k==kv_len, got k={int(k)} kv_len={int(kv_len)}"
            )
        compile_kv_len = self._product_all_kv_decode_compile_kv_len(
            kv_len=int(kv_len),
            seqlen=int(seqlen),
            window_size=int(window_size),
            k_tile=int(k_tile),
        )
        compile_k = int(compile_kv_len)
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(bsz) * int(seqlen),
        )
        win_width = int(window_size)
        k_raw = int(win_width) + int(compile_k)
        k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
        kernel = self._attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_kernel_for(
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
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_kv_score_state,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            cos_table,
            sin_table,
            compressor_cos_table,
            compressor_sin_table,
            indexer_compressor_cos_table,
            indexer_compressor_sin_table,
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
            kv_len=int(compile_kv_len),
            k=int(compile_k),
            rows=int(rows),
            k_tile=int(k_tile),
            compressor_head_dim=int(compressor_head_dim),
            compressor_state_width=int(compressor_state_width),
            compressor_ring_size=int(compressor_ring_size),
            compressor_rope_head_dim=int(compressor_rope_head_dim),
            compressor_block_size=int(compressor_block_size),
            compressor_fp8_max=float(compressor_fp8_max),
            compressor_rotate=bool(compressor_rotate),
            compressor_overlap=bool(compressor_overlap),
            compressor_eps=float(compressor_eps),
            indexer_compressor_head_dim=int(indexer_compressor_head_dim),
            indexer_compressor_state_width=int(indexer_compressor_state_width),
            indexer_compressor_ring_size=int(indexer_compressor_ring_size),
            indexer_compressor_rope_head_dim=int(indexer_compressor_rope_head_dim),
            indexer_compressor_block_size=int(indexer_compressor_block_size),
            indexer_compressor_fp8_max=float(indexer_compressor_fp8_max),
            indexer_compressor_rotate=bool(indexer_compressor_rotate),
            indexer_compressor_overlap=bool(indexer_compressor_overlap),
            indexer_compressor_eps=float(indexer_compressor_eps),
            write_swa_state_cache=bool(write_swa_state_cache),
            swa_kv_cache=swa_kv_cache,
            compressed_kv_cache=compressed_kv_cache,
            indexer_compressed_kv_cache=indexer_compressed_kv_cache,
            max_c_len=int(max_c_len or 0),
            indexer_max_c_len=int(indexer_max_c_len or 0),
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
        comp_kv = None
        comp_score = None
        idx_kv = None
        idx_score = None
        output_shift = 2 if bool(write_swa_state_cache) else 6
        if not bool(write_swa_state_cache):
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
        topk_t = outputs.get(f"output{output_shift}")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (int(k_padded), int(rows)),
                np.int32,
            )
        mask = outputs.get(f"output{output_shift + 1}")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), int(k_padded)),
                ml_dtypes.bfloat16,
            )
        comp_rows = None
        idx_rows = None
        if not bool(write_swa_state_cache):
            comp_rows = outputs.get("output8")
            if comp_rows is None:
                comp_rows = self._bucket_scratch(
                    bucket,
                    "compressor_decode_post_qdq_bf16",
                    (int(bsz), int(compressor_head_dim)),
                    ml_dtypes.bfloat16,
                )
            idx_rows = outputs.get("output9")
            if idx_rows is None:
                idx_rows = self._bucket_scratch(
                    bucket,
                    "compressor_decode_post_qdq_bf16",
                    (int(bsz), int(indexer_compressor_head_dim)),
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
            "indexer_compressor_wkv": indexer_compressor_wkv,
            "indexer_compressor_wgate": indexer_compressor_wgate,
            "indexer_compressor_ape": indexer_compressor_ape,
            "indexer_compressor_norm_weight": indexer_compressor_norm_weight,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "compressor_cos_table": compressor_cos_table,
            "compressor_sin_table": compressor_sin_table,
            "indexer_compressor_cos_table": indexer_compressor_cos_table,
            "indexer_compressor_sin_table": indexer_compressor_sin_table,
            "positions": positions,
        }
        outputs_map = {
            "output0": q,
            "output1": kv,
            f"output{output_shift}": topk_t,
            f"output{output_shift + 1}": mask,
        }
        if bool(write_swa_state_cache):
            inputs.update(
                {
                    "swa_kv_cache.must_alias_input": swa_kv_cache,
                    "kv_score_state.must_alias_input": kv_score_state,
                    "compressed_kv_cache.must_alias_input": compressed_kv_cache,
                    "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
                    "indexer_compressed_kv_cache.must_alias_input": (
                        indexer_compressed_kv_cache
                    ),
                }
            )
            outputs_map.update(
                {
                    "swa_kv_cache": swa_kv_cache,
                    "kv_score_state": kv_score_state,
                    "compressed_kv_cache": compressed_kv_cache,
                    "indexer_kv_score_state": indexer_kv_score_state,
                    "indexer_compressed_kv_cache": indexer_compressed_kv_cache,
                }
            )
        else:
            inputs.update(
                {
                    "kv_score_state": kv_score_state,
                    "indexer_kv_score_state": indexer_kv_score_state,
                }
            )
            outputs_map.update(
                {
                    "output2": comp_kv,
                    "output3": comp_score,
                    "output4": idx_kv,
                    "output5": idx_score,
                    "output8": comp_rows,
                    "output9": idx_rows,
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
        return (
            q,
            kv,
            comp_kv,
            comp_score,
            idx_kv,
            idx_score,
            topk_t,
            mask,
            comp_rows,
            idx_rows,
        )
