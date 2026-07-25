"""Attention graph adapters for DSV4 product execution."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import _compile_product_kernel
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _require_product_device_value,
    _sample_array,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)
from nkipy_serving.runtime.device_tensor import (
    get_device_tensor_cls as _get_device_tensor_cls,
)

_DYNAMIC_OFFSET_I32_CACHE: dict[tuple[int, str], Any] = {}


def _device_scalar_i32(value: int, *, name: str) -> Any:
    key = (int(value), str(name))
    cached = _DYNAMIC_OFFSET_I32_CACHE.get(key)
    if cached is not None:
        return cached
    arr = np.asarray([[int(value)]], dtype=np.int32)
    dev = _get_device_tensor_cls().from_numpy(np.ascontiguousarray(arr), name=name)
    _DYNAMIC_OFFSET_I32_CACHE[key] = dev
    return dev


class Dsv4ProductAttentionGraphMixin:
    def _indexer_sparse_attention_prep_static_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        score: Any,
        x: Any,
        positions: Any | None = None,
        offset_tensor: Any | None = None,
        *,
        bsz: int,
        seqlen: int,
        kv_len: int,
        k: int,
        ratio: int,
        offset: int,
        prefill: bool,
        window_size: int,
        start_pos: int,
        rows: int,
        k_tile: int,
        dynamic_decode_start_pos: bool = False,
        dynamic_prefill_offset: bool = False,
    ) -> Any:
        if bool(dynamic_decode_start_pos) and bool(dynamic_prefill_offset):
            raise RuntimeError(
                "DSV4 product indexer sparse-prep cannot use both dynamic "
                "decode positions and dynamic prefill offset"
            )
        score_shape = tuple(int(dim) for dim in getattr(score, "shape", ()))
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        offset_shape = tuple(int(dim) for dim in getattr(offset_tensor, "shape", ()))
        if bool(dynamic_decode_start_pos):
            if positions is None:
                raise RuntimeError(
                    "DSV4 product indexer sparse-prep decode fusion requires "
                    "device positions"
                )
            if bool(prefill) or int(seqlen) != 1:
                raise RuntimeError(
                    "DSV4 product indexer sparse-prep dynamic fusion is decode-only"
                )
        if bool(dynamic_prefill_offset):
            if offset_tensor is None:
                raise RuntimeError(
                    "DSV4 product indexer sparse-prep dynamic prefill offset "
                    "requires device offset"
                )
            if not bool(prefill) or int(start_pos) != 0:
                raise RuntimeError(
                    "DSV4 product indexer sparse-prep dynamic offset is prefill-only"
                )
        start_pos_key = -1 if bool(dynamic_decode_start_pos) else int(start_pos)
        offset_key = -1 if bool(dynamic_prefill_offset) else int(offset)
        key = (
            score_shape,
            str(getattr(score, "dtype", "unknown")),
            x_shape,
            str(getattr(x, "dtype", "unknown")),
            pos_shape if bool(dynamic_decode_start_pos) else (),
            (
                str(getattr(positions, "dtype", "unknown"))
                if bool(dynamic_decode_start_pos)
                else ""
            ),
            int(bsz),
            int(seqlen),
            int(kv_len),
            int(k),
            int(ratio),
            int(offset_key),
            bool(prefill),
            int(window_size),
            int(start_pos_key),
            int(rows),
            int(k_tile),
            bool(dynamic_decode_start_pos),
            offset_shape if bool(dynamic_prefill_offset) else (),
            (
                str(getattr(offset_tensor, "dtype", "unknown"))
                if bool(dynamic_prefill_offset)
                else ""
            ),
            bool(dynamic_prefill_offset),
        )
        offset_name = "dyn" if bool(dynamic_prefill_offset) else str(int(offset))
        name = (
            "dsv4_product_indexer_sparse_attention_prep_static_"
            f"t{int(bucket.token_bucket)}_"
            f"s{'x'.join(str(v) for v in score_shape)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"b{int(bsz)}_q{int(seqlen)}_kv{int(kv_len)}_k{int(k)}_"
            f"r{int(ratio)}_o{offset_name}_p{int(bool(prefill))}_"
            f"w{int(window_size)}_"
            f"sp{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}_"
            f"rows{int(rows)}_"
            f"kt{int(k_tile)}"
        )
        if bool(dynamic_decode_start_pos):
            fn = graph_moe.indexer_sparse_attention_prep_decode_from_positions_fn
            compile_args = (
                _sample_array(score, fallback_dtype=np.float32),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(positions, fallback_dtype=np.int32),
            )
        elif bool(dynamic_prefill_offset):
            fn = graph_moe.indexer_sparse_attention_prep_static_dynamic_offset_fn
            compile_args = (
                _sample_array(score, fallback_dtype=np.float32),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(offset_tensor, fallback_dtype=np.int32),
            )
        else:
            fn = graph_moe.indexer_sparse_attention_prep_static_fn
            compile_args = (
                _sample_array(score, fallback_dtype=np.float32),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
            )
        compile_kwargs = dict(
            bsz=int(bsz),
            seqlen=int(seqlen),
            kv_len=int(kv_len),
            k=int(k),
            ratio=int(ratio),
            prefill=bool(prefill),
            window_size=int(window_size),
            start_pos=(1 if bool(dynamic_decode_start_pos) else int(start_pos)),
            rows=int(rows),
            k_tile=int(k_tile),
            name=name,
            additional_compiler_args=getattr(self, "compiler_args", ""),
            build_dir=self.build_dir,
        )
        if not bool(dynamic_prefill_offset):
            compile_kwargs["offset"] = int(offset)
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="indexer_sparse_attention_prep_static_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_args,
                **compile_kwargs,
            ),
        )

    def _run_product_prefix_two_token_flats(
        self,
        kv: Any,
        score: Any,
        *,
        bsz: int,
        seqlen: int,
        cutoff: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        self._require_active_product_bucket(where="prefix_two_token_flats")
        _require_product_device_value(kv, where="prefix_two_token_flats/kv")
        _require_product_device_value(score, where="prefix_two_token_flats/score")
        kv_shape = tuple(int(dim) for dim in getattr(kv, "shape", ()))
        if not kv_shape:
            raise RuntimeError("DSV4 product prefix flats expects non-scalar kv")
        out_shape = (int(bsz) * int(cutoff), int(kv_shape[-1]))
        outputs = dict(_nkipy_output_tensors or {})
        if not outputs and (int(bsz) == 1 or int(cutoff) == int(seqlen)):
            kv_alias = _alias_device_value_shape(kv, out_shape)
            score_alias = _alias_device_value_shape(score, out_shape)
            if kv_alias is not None and score_alias is not None:
                return kv_alias, score_alias
        raise RuntimeError(
            "DSV4 product prefix compressor pool expects prefix-contiguous "
            "device tensors that can be represented as zero-copy aliases; "
            "standalone prefix_two_token_flats kernels are not part of the "
            "product path"
        )

    def _run_product_attention_kv_flatten(
        self,
        kv: Any,
        *,
        total_tokens: int,
        head_dim: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        self._require_active_product_bucket(where="attention_kv_flatten")
        _require_product_device_value(kv, where="attention_kv_flatten")
        target_shape = (int(total_tokens), int(head_dim))
        alias = _alias_device_value_shape(kv, target_shape)
        if alias is not None:
            return alias
        raise RuntimeError(
            "DSV4 product attention KV flatten requires a zero-copy "
            "DeviceTensor shape alias; standalone flatten kernels are not "
            "part of the product path"
        )

    def _alias_product_attention_kv_tail_rows(
        self,
        kv: Any,
        *,
        request_index: int | None,
        window_size: int,
        head_dim: int,
        where: str,
    ) -> Any | None:
        _require_product_device_value(kv, where=where)
        kv_shape = tuple(int(dim) for dim in getattr(kv, "shape", ()))
        if len(kv_shape) != 3:
            raise RuntimeError(
                f"DSV4 product {where} expects [batch, seqlen, dim], got {kv_shape}"
            )
        bsz, seqlen, dim = kv_shape
        if int(dim) != int(head_dim):
            raise RuntimeError(
                f"DSV4 product {where} head_dim mismatch: kv dim={int(dim)}, "
                f"head_dim={int(head_dim)}"
            )
        tail = min(int(seqlen), int(window_size))
        flat = _alias_device_value_shape(kv, (int(bsz) * int(seqlen), int(dim)))
        if flat is None:
            return None
        if request_index is None:
            if int(tail) == int(seqlen):
                return _alias_device_value_shape(
                    flat,
                    (int(bsz) * int(tail), int(dim)),
                )
            if int(bsz) != 1:
                return None
            start = int(seqlen) - int(tail)
        else:
            req = int(request_index)
            if req < 0 or req >= int(bsz):
                raise RuntimeError(
                    "DSV4 product request KV tail window request_index outside "
                    f"batch: request_index={req}, batch={int(bsz)}"
                )
            start = int(req) * int(seqlen) + int(seqlen) - int(tail)
        return _alias_device_value_first_dim_slice(
            flat,
            start=int(start),
            size=int(tail),
        )

    def _run_product_attention_kv_tail_window(
        self,
        kv: Any,
        *,
        window_size: int,
        head_dim: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        self._require_active_product_bucket(where="attention_kv_tail_window")
        alias = self._alias_product_attention_kv_tail_rows(
            kv,
            request_index=None,
            window_size=int(window_size),
            head_dim=int(head_dim),
            where="attention_kv_tail_window",
        )
        if alias is None:
            raise RuntimeError(
                "DSV4 product KV tail window requires a contiguous NRT tensor "
                "alias; use per-request tail aliases for multi-request long "
                "prefill"
            )
        return alias

    def _run_product_attention_kv_request_tail_window(
        self,
        kv: Any,
        *,
        request_index: int,
        window_size: int,
        head_dim: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        self._require_active_product_bucket(where="attention_kv_request_tail_window")
        alias = self._alias_product_attention_kv_tail_rows(
            kv,
            request_index=int(request_index),
            window_size=int(window_size),
            head_dim=int(head_dim),
            where="attention_kv_request_tail_window",
        )
        if alias is None:
            raise RuntimeError(
                "DSV4 product request KV tail window requires a contiguous "
                "NRT tensor alias; standalone request-tail kernels are not "
                "part of the product path"
            )
        return alias

    def _run_product_attention_sink_2d(
        self,
        sink: Any,
        *,
        n_heads: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        self._require_active_product_bucket(where="attention_sink_2d")
        _require_product_device_value(sink, where="attention_sink_2d")
        target_shape = (1, int(n_heads))
        alias = _alias_device_value_shape(sink, target_shape)
        if alias is not None:
            return alias
        raise RuntimeError(
            "DSV4 product attention sink reshape requires a zero-copy "
            "DeviceTensor shape alias; standalone sink reshape kernels are "
            "not part of the product path"
        )

    def _run_product_indexer_sparse_attention_prep_static(
        self,
        score: Any,
        x: Any,
        positions: Any | None = None,
        *,
        bsz: int,
        seqlen: int,
        kv_len: int,
        k: int,
        ratio: int,
        offset: int,
        prefill: bool,
        window_size: int,
        start_pos: int,
        rows: int,
        k_tile: int,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        bucket = self._require_active_product_bucket(
            where="indexer_sparse_attention_prep_static"
        )
        _require_product_device_value(
            score,
            where="indexer_sparse_attention_prep_static/score",
        )
        _require_product_device_value(
            x,
            where="indexer_sparse_attention_prep_static/x",
        )
        dynamic_decode_start_pos = (
            int(start_pos) > 0
            and int(seqlen) == 1
            and not bool(prefill)
            and positions is not None
        )
        if bool(dynamic_decode_start_pos):
            _require_product_device_value(
                positions,
                where="indexer_sparse_attention_prep_static/positions",
            )
        dynamic_prefill_offset = (
            bool(prefill) and int(start_pos) == 0 and not bool(dynamic_decode_start_pos)
        )
        offset_tensor = None
        if bool(dynamic_prefill_offset):
            offset_tensor = _device_scalar_i32(
                int(offset),
                name="dsv4_indexer_sparse_prep_offset",
            )
            _require_product_device_value(
                offset_tensor,
                where="indexer_sparse_attention_prep_static/offset",
            )
        kernel = self._indexer_sparse_attention_prep_static_kernel_for(
            bucket,
            score,
            x,
            positions,
            offset_tensor=offset_tensor,
            bsz=int(bsz),
            seqlen=int(seqlen),
            kv_len=int(kv_len),
            k=int(k),
            ratio=int(ratio),
            offset=int(offset),
            prefill=bool(prefill),
            window_size=int(window_size),
            start_pos=int(start_pos),
            rows=int(rows),
            k_tile=int(k_tile),
            dynamic_decode_start_pos=bool(dynamic_decode_start_pos),
            dynamic_prefill_offset=bool(dynamic_prefill_offset),
        )
        win_width = (
            int(window_size)
            if int(start_pos) > 0
            else min(int(seqlen), int(window_size))
        )
        k_raw = int(win_width) + int(k)
        k_padded = ((k_raw + int(k_tile) - 1) // int(k_tile)) * int(k_tile)
        outputs = dict(_nkipy_output_tensors or {})
        topk_t = outputs.get("output0")
        if topk_t is None:
            topk_t = self._bucket_scratch(
                bucket,
                "attention_topk_t",
                (k_padded, int(rows)),
                np.int32,
            )
        mask = outputs.get("output1")
        if mask is None:
            mask = self._bucket_scratch(
                bucket,
                "attention_topk_mask",
                (int(rows), k_padded),
                ml_dtypes.bfloat16,
            )
        inputs = {"score": score, "x": x}
        if bool(dynamic_decode_start_pos):
            inputs["positions"] = positions
        if bool(dynamic_prefill_offset):
            inputs["offset"] = offset_tensor
        kernel(
            inputs=inputs,
            outputs={"output0": topk_t, "output1": mask},
        )
        return topk_t, mask

    def _attention_graph(self) -> dict[str, Any]:
        graph = dict(getattr(self, "graph", {}))
        for stale_key in (
            "attention_zero_like",
            "attention_qkv_compressor_kv_score_from_freq_table",
            "cast_bf16",
            "compressor_kv_score_bf16",
            "compressor_kv_score_token_topk_prep",
            "compressor_norm_2d",
            "compressor_post_pool_freqs_from_table",
            "compressor_qdq_bf16",
            "dp_attention_flatten_pad",
            "dp_attention_lane_scatter",
            "dp_attention_unpad_reshape",
            "head_hidden_flatten",
            "head_hidden_flatten_pad",
            "hc_head",
            "mhc_post",
            "pad_flat_rows",
            "pad_topk_rows",
            "q_scale_transpose",
            "topk_concat",
            "topk_sparse_attention_prep",
            "topk_concat_pad_sparse_attention_prep",
            "topk_tokens_concat_pad_sparse_attention_prep",
            "window_topk_from_tokens",
            "compressed_topk_no_indexer_from_tokens",
            "invalid_topk_from_tokens",
            "indexer_score_reshape",
            "indexer_project_qw_prep",
            "indexer_project_qw_prep_from_freq_table",
            "indexer_compressor_kv_score_project_qw_prep_from_freq_table",
            "indexer_topk_static",
            "inverse_rope_tail_flat_from_freq_table",
            "causal_mask_add",
            "topk_rebase_static",
            "topk_idx",
        ):
            graph.pop(stale_key, None)
        graph["_attention_out_flat_hidden_owns_outputs"] = True
        graph["_attention_qkv_table_fuses_q_scale"] = True
        graph["_attention_qkv_table_outputs_flat_kv"] = True
        graph["_product_require_fused_attention_qkv_table"] = True
        graph["_product_require_flat_swa_kv"] = True
        graph["_product_require_fused_compressor_post_qdq"] = True
        graph["_product_require_fused_sparse_attention_prep"] = True
        graph["_product_require_precomputed_compressor_kv_score"] = True
        graph["_product_require_precomputed_empty_indexer_topk"] = True
        graph["_product_require_precomputed_indexer_qw"] = True
        graph["_product_require_fused_inverse_rope_out"] = True
        graph["_product_require_fused_swa_kv_write"] = True
        graph["_product_prefix_two_token_flats_aliases_prefix"] = True
        graph["_product_attention_shape_helpers_alias"] = True
        graph["attention_out_flat"] = self._run_product_attention_out_flat_unavailable
        graph["attention_out_flat_hidden"] = self._run_product_attention_out_flat_hidden
        graph["attention_qkv_quant_from_freq_table"] = (
            self._run_product_attention_qkv_quant_from_freq_table
        )
        graph["attention_qkv_write_kv_cache_from_freq_table"] = (
            self._run_product_attention_qkv_write_kv_cache_from_freq_table
        )
        graph["attention_qkv_indexer_compressor_qw_prep_from_freq_table"] = (
            self._run_product_attention_qkv_indexer_compressor_from_freq_table
        )
        graph[
            "attention_qkv_indexer_compressor_qw_prep_write_swa_state_from_freq_table"
        ] = self._run_product_attention_qkv_indexer_compressor_table_write_swa_state_from_freq_table
        graph["attention_qkv_indexer_compressor_all_kv_topk_from_freq_table"] = (
            self._run_product_attention_qkv_indexer_compressor_all_kv_topk_from_freq_table
        )
        graph[
            "attention_qkv_indexer_compressor_all_kv_topk_write_swa_state_from_freq_table"
        ] = self._run_product_attention_qkv_indexer_compressor_all_kv_topk_write_swa_state_from_freq_table
        graph[
            "attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_from_freq_table"
        ] = self._run_product_attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_from_freq_table
        graph[
            "attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_from_freq_table"
        ] = self._run_product_attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_from_freq_table
        graph["attention_qkv_empty_indexer_compressor_topk_from_freq_table"] = (
            self._run_product_attention_qkv_empty_indexer_compressor_topk_from_freq_table
        )
        graph["attention_qkv_token_topk_prep_from_freq_table"] = (
            self._run_product_attention_qkv_token_topk_prep_from_freq_table
        )
        graph["attention_qkv_compressor_kv_score_token_topk_prep_from_freq_table"] = (
            self._run_product_attention_qkv_compressor_token_topk_prep_from_freq_table
        )
        graph[
            "attention_qkv_compressor_kv_score_token_topk_prep_write_swa_state_from_freq_table"
        ] = self._run_product_attention_qkv_compressor_token_topk_prep_write_swa_state_from_freq_table
        graph[
            "attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_from_freq_table"
        ] = self._run_product_attention_qkv_compressor_prefill_post_qdq_token_topk_prep_from_freq_table
        graph[
            "attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_from_freq_table"
        ] = self._run_product_attention_qkv_compressor_decode_post_qdq_token_topk_prep_from_freq_table
        graph["attention_kv_flatten"] = self._run_product_attention_kv_flatten
        graph["attention_kv_tail_window"] = self._run_product_attention_kv_tail_window
        graph["attention_kv_request_tail_window"] = (
            self._run_product_attention_kv_request_tail_window
        )
        graph["attention_sink_2d"] = self._run_product_attention_sink_2d
        graph["attention_inverse_rope_out_flat_hidden_from_freq_table"] = (
            self._run_product_attention_inverse_rope_out_flat_hidden_from_freq_table
        )
        graph["_attention_inverse_rope_out_flat_hidden_owns_outputs"] = True
        graph["compressor_post_qdq_from_freq_table"] = (
            self._run_product_compressor_post_qdq_from_freq_table
        )
        graph["compressor_decode_pool_post_qdq_from_state_freq_table"] = (
            self._run_product_compressor_decode_pool_post_qdq_from_state_freq_table
        )
        graph["prefix_two_token_flats"] = self._run_product_prefix_two_token_flats
        graph["indexer_sparse_attention_prep_static"] = (
            self._run_product_indexer_sparse_attention_prep_static
        )
        graph["_product_indexer_sparse_prep_accepts_positions"] = True
        return graph

    def _run_product_attention_out_flat_unavailable(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        del args, kwargs
        raise RuntimeError(
            "DSV4 product attention_out_flat is unsupported; product "
            "attention must request attention_out_flat_hidden so projection "
            "is fused with post/pre or DP-flat restore"
        )
