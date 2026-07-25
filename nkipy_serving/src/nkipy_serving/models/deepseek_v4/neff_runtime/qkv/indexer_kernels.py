"""QKV indexer/compressor product kernel cache helpers."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import _compile_product_kernel
from nkipy_serving.models.deepseek_v4.neff_graphs import attention as graph_attention
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import _sample_array
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
)
from nkipy_serving.models.deepseek_v4.variants import (
    QkvVariantName,
)


class Dsv4ProductQkvIndexerKernelsMixin:
    def _attention_qkv_indexer_compressor_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
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
        dynamic_decode_start_pos: bool = False,
        write_swa_dual_state: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        indexer_kv_score_state: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ape: Any | None = None,
        indexer_compressor_ape: Any | None = None,
        window_size: int = 0,
        ratio: int = 0,
        start_pos: int = 0,
        compressor_ring_size: int = 0,
        indexer_compressor_ring_size: int = 0,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        if bool(dynamic_decode_start_pos):
            if len(x_shape) < 2 or int(x_shape[1]) != 1:
                raise RuntimeError(
                    "DSV4 product table QKV/indexer fusion is decode-only "
                    f"for dynamic start positions, got x={x_shape}"
                )
        start_pos_key = -1 if bool(dynamic_decode_start_pos) else int(start_pos)
        write_dual_key: tuple[Any, ...] = ()
        if bool(write_swa_dual_state):
            write_dual_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(kv_score_state, "shape", ())),
                str(getattr(kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(indexer_kv_score_state, "shape", ())),
                str(getattr(indexer_kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(owner_ids, "shape", ())),
                str(getattr(owner_ids, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressor_ape, "shape", ())),
                str(getattr(compressor_ape, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(indexer_compressor_ape, "shape", ())),
                str(getattr(indexer_compressor_ape, "dtype", "unknown")),
                int(window_size),
                int(ratio),
                int(compressor_ring_size),
                int(indexer_compressor_ring_size),
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
            tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ())),
            str(getattr(compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_wgate, "shape", ())),
            str(getattr(compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())),
            str(getattr(indexer_compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wgate, "shape", ())),
            str(getattr(indexer_compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_wq_b, "shape", ())),
            str(getattr(indexer_wq_b, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_weights_proj, "shape", ())),
            str(getattr(indexer_weights_proj, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_cos_table, "shape", ())),
            str(getattr(indexer_cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_sin_table, "shape", ())),
            str(getattr(indexer_sin_table, "dtype", "unknown")),
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
            float(indexer_score_scale),
            int(indexer_n_heads),
            int(indexer_head_dim),
            int(indexer_rope_head_dim),
            int(indexer_block_size),
            float(indexer_fp8_max),
            bool(dynamic_decode_start_pos),
            bool(write_swa_dual_state),
            int(start_pos_key),
            write_dual_key,
        )
        name_prefix = "dsv4_product_attention_qkv_indexer_compressor_table"
        if bool(write_swa_dual_state):
            if not bool(dynamic_decode_start_pos):
                raise RuntimeError(
                    "DSV4 product table dual SWA/state write is decode-only"
                )
            name_prefix = (
                "dsv4_product_attention_qkv_indexer_compressor_table_"
                "write_swa_dual_state"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"ih{int(indexer_n_heads)}_id{int(indexer_head_dim)}_"
            f"kvflat{int(kv_token_bucket)}"
        )
        if bool(write_swa_dual_state):
            name = (
                f"{name}_w{int(window_size)}_r{int(ratio)}_"
                f"s{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}"
            )
            fn = graph_attention.attention_qkv_indexer_compressor_qw_prep_write_swa_dual_state_decode_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_indexer_compressor_qw_prep_from_freq_table_fn
        compile_start_pos = 1 if bool(dynamic_decode_start_pos) else int(start_pos)
        compile_inputs = [
            _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(wq_a, fallback_dtype=np.float32),
            _sample_array(q_norm, fallback_dtype=np.float32),
            _sample_array(wq_b, fallback_dtype=np.float32),
            _sample_array(wkv, fallback_dtype=np.float32),
            _sample_array(kv_norm, fallback_dtype=np.float32),
            _sample_array(compressor_wkv, fallback_dtype=np.float32),
            _sample_array(compressor_wgate, fallback_dtype=np.float32),
            _sample_array(indexer_compressor_wkv, fallback_dtype=np.float32),
            _sample_array(indexer_compressor_wgate, fallback_dtype=np.float32),
            _sample_array(indexer_wq_b, fallback_dtype=np.float32),
            _sample_array(indexer_weights_proj, fallback_dtype=np.float32),
        ]
        if bool(write_swa_dual_state):
            if (
                swa_kv_cache is None
                or kv_score_state is None
                or indexer_kv_score_state is None
                or owner_ids is None
                or compressor_ape is None
                or indexer_compressor_ape is None
            ):
                raise RuntimeError(
                    "DSV4 product table dual SWA/state write kernel requires "
                    "SWA, state, owner, and APE inputs"
                )
            compile_inputs.extend(
                [
                    _sample_array(swa_kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(kv_score_state, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(
                        indexer_kv_score_state,
                        fallback_dtype=ml_dtypes.bfloat16,
                    ),
                    _sample_array(owner_ids, fallback_dtype=np.int32),
                    _sample_array(compressor_ape, fallback_dtype=np.float32),
                    _sample_array(indexer_compressor_ape, fallback_dtype=np.float32),
                ]
            )
        compile_inputs.extend(
            [
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(indexer_cos_table, fallback_dtype=np.float32),
                _sample_array(indexer_sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
            ]
        )
        compile_kwargs: dict[str, Any] = {
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "rope_head_dim": int(rope_head_dim),
            "eps": float(eps),
            "block_size": int(block_size),
            "fp8_max": float(fp8_max),
            "q_softmax_scale": float(q_softmax_scale),
            "q_token_bucket": int(q_token_bucket),
            "kv_token_bucket": int(kv_token_bucket),
            "indexer_score_scale": float(indexer_score_scale),
            "indexer_n_heads": int(indexer_n_heads),
            "indexer_head_dim": int(indexer_head_dim),
            "indexer_rope_head_dim": int(indexer_rope_head_dim),
            "indexer_block_size": int(indexer_block_size),
            "indexer_fp8_max": float(indexer_fp8_max),
        }
        if bool(write_swa_dual_state):
            compile_kwargs.update(
                {
                    "window_size": int(window_size),
                    "ratio": int(ratio),
                    "start_pos": int(compile_start_pos),
                    "compressor_ring_size": int(compressor_ring_size),
                    "indexer_compressor_ring_size": int(indexer_compressor_ring_size),
                }
            )
        compile_kwargs.update(
            {
                "name": name,
                "additional_compiler_args": getattr(self, "compiler_args", ""),
                "build_dir": self.build_dir,
            }
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE
                if bool(write_swa_dual_state)
                else QkvVariantName.INDEXER_COMPRESSOR_TABLE
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )

    def _attention_qkv_empty_indexer_compressor_token_topk_prep_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
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
        dynamic_decode_start_pos: bool = False,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        if bool(dynamic_decode_start_pos):
            if len(x_shape) < 2 or int(x_shape[1]) != 1:
                raise RuntimeError(
                    "DSV4 product empty-indexer fused QKV/top-k is decode-only "
                    f"for dynamic start positions, got x={x_shape}"
                )
        start_pos_key = -1 if bool(dynamic_decode_start_pos) else int(start_pos)
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
            tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ())),
            str(getattr(compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_wgate, "shape", ())),
            str(getattr(compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())),
            str(getattr(indexer_compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wgate, "shape", ())),
            str(getattr(indexer_compressor_wgate, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos_key),
            int(max_c_len),
            int(rows),
            int(k_tile),
            bool(dynamic_decode_start_pos),
        )
        name = (
            "dsv4_product_attention_qkv_empty_indexer_compressor_topk_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"s{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}_"
            f"c{int(max_c_len)}_rows{int(rows)}_k{int(k_tile)}_"
            f"kvflat{int(kv_token_bucket)}"
        )
        fn = (
            graph_attention.attention_qkv_empty_indexer_compressor_token_topk_prep_decode_from_freq_table_fn
            if bool(dynamic_decode_start_pos)
            else graph_attention.attention_qkv_empty_indexer_compressor_token_topk_prep_from_freq_table_fn
        )
        compile_start_pos = 1 if bool(dynamic_decode_start_pos) else int(start_pos)
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=QkvVariantName.EMPTY_INDEXER_COMPRESSOR_TOPK,
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(wq_a, fallback_dtype=np.float32),
                _sample_array(q_norm, fallback_dtype=np.float32),
                _sample_array(wq_b, fallback_dtype=np.float32),
                _sample_array(wkv, fallback_dtype=np.float32),
                _sample_array(kv_norm, fallback_dtype=np.float32),
                _sample_array(compressor_wkv, fallback_dtype=np.float32),
                _sample_array(compressor_wgate, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wkv, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wgate, fallback_dtype=np.float32),
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
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(compile_start_pos),
                max_c_len=int(max_c_len),
                rows=int(rows),
                k_tile=int(k_tile),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )

    def _attention_qkv_indexer_compressor_all_kv_topk_prep_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
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
        dynamic_decode_start_pos: bool = False,
        write_swa_dual_state: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        indexer_kv_score_state: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ape: Any | None = None,
        indexer_compressor_ape: Any | None = None,
        compressor_ring_size: int = 0,
        indexer_compressor_ring_size: int = 0,
    ) -> Any:
        if int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV indexer QKV/top-k fusion requires "
                f"k == kv_len, got k={int(k)} kv_len={int(kv_len)}"
            )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        if bool(dynamic_decode_start_pos):
            if len(x_shape) < 2 or int(x_shape[1]) != 1:
                raise RuntimeError(
                    "DSV4 product all-KV indexer fused QKV/top-k is "
                    f"decode-only for dynamic start positions, got x={x_shape}"
                )
        start_pos_key = -1 if bool(dynamic_decode_start_pos) else int(start_pos)
        write_dual_key: tuple[Any, ...] = ()
        if bool(write_swa_dual_state):
            write_dual_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(kv_score_state, "shape", ())),
                str(getattr(kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(indexer_kv_score_state, "shape", ())),
                str(getattr(indexer_kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(owner_ids, "shape", ())),
                str(getattr(owner_ids, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressor_ape, "shape", ())),
                str(getattr(compressor_ape, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(indexer_compressor_ape, "shape", ())),
                str(getattr(indexer_compressor_ape, "dtype", "unknown")),
                int(compressor_ring_size),
                int(indexer_compressor_ring_size),
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
            tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ())),
            str(getattr(compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_wgate, "shape", ())),
            str(getattr(compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())),
            str(getattr(indexer_compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wgate, "shape", ())),
            str(getattr(indexer_compressor_wgate, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos_key),
            int(kv_len),
            int(k),
            int(rows),
            int(k_tile),
            bool(dynamic_decode_start_pos),
            bool(write_swa_dual_state),
            write_dual_key,
        )
        name_prefix = "dsv4_product_attention_qkv_indexer_compressor_allkv_topk"
        if bool(write_swa_dual_state):
            if not bool(dynamic_decode_start_pos):
                raise RuntimeError(
                    "DSV4 product all-KV dual SWA/state write is decode-only"
                )
            name_prefix = (
                "dsv4_product_attention_qkv_indexer_compressor_allkv_topk_"
                "write_swa_dual_state"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"s{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}_"
            f"kv{int(kv_len)}_k{int(k)}_rows{int(rows)}_kt{int(k_tile)}_"
            f"kvflat{int(kv_token_bucket)}"
        )
        if bool(write_swa_dual_state):
            fn = graph_attention.attention_qkv_indexer_compressor_all_kv_topk_write_swa_dual_state_decode_from_freq_table_fn
        else:
            fn = (
                graph_attention.attention_qkv_indexer_compressor_all_kv_topk_prep_decode_from_freq_table_fn
                if bool(dynamic_decode_start_pos)
                else graph_attention.attention_qkv_indexer_compressor_all_kv_topk_prep_from_freq_table_fn
            )
        compile_start_pos = 1 if bool(dynamic_decode_start_pos) else int(start_pos)
        compile_inputs = [
            _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(wq_a, fallback_dtype=np.float32),
            _sample_array(q_norm, fallback_dtype=np.float32),
            _sample_array(wq_b, fallback_dtype=np.float32),
            _sample_array(wkv, fallback_dtype=np.float32),
            _sample_array(kv_norm, fallback_dtype=np.float32),
            _sample_array(compressor_wkv, fallback_dtype=np.float32),
            _sample_array(compressor_wgate, fallback_dtype=np.float32),
            _sample_array(indexer_compressor_wkv, fallback_dtype=np.float32),
            _sample_array(indexer_compressor_wgate, fallback_dtype=np.float32),
        ]
        if bool(write_swa_dual_state):
            if (
                swa_kv_cache is None
                or kv_score_state is None
                or indexer_kv_score_state is None
                or owner_ids is None
                or compressor_ape is None
                or indexer_compressor_ape is None
            ):
                raise RuntimeError(
                    "DSV4 product all-KV dual SWA/state write kernel requires "
                    "SWA, state, owner, and APE inputs"
                )
            compile_inputs.extend(
                [
                    _sample_array(swa_kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(kv_score_state, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(
                        indexer_kv_score_state,
                        fallback_dtype=ml_dtypes.bfloat16,
                    ),
                    _sample_array(owner_ids, fallback_dtype=np.int32),
                    _sample_array(compressor_ape, fallback_dtype=np.float32),
                    _sample_array(indexer_compressor_ape, fallback_dtype=np.float32),
                ]
            )
        compile_inputs.extend(
            [
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
            ]
        )
        compile_kwargs: dict[str, Any] = {
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "rope_head_dim": int(rope_head_dim),
            "eps": float(eps),
            "block_size": int(block_size),
            "fp8_max": float(fp8_max),
            "q_softmax_scale": float(q_softmax_scale),
            "q_token_bucket": int(q_token_bucket),
            "kv_token_bucket": int(kv_token_bucket),
            "window_size": int(window_size),
            "ratio": int(ratio),
            "offset": int(offset),
            "start_pos": int(compile_start_pos),
            "kv_len": int(kv_len),
            "k": int(k),
            "rows": int(rows),
            "k_tile": int(k_tile),
        }
        if bool(write_swa_dual_state):
            compile_kwargs["compressor_ring_size"] = int(compressor_ring_size)
            compile_kwargs["indexer_compressor_ring_size"] = int(
                indexer_compressor_ring_size
            )
        compile_kwargs.update(
            {
                "name": name,
                "additional_compiler_args": getattr(self, "compiler_args", ""),
                "build_dir": self.build_dir,
            }
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE
                if bool(write_swa_dual_state)
                else QkvVariantName.INDEXER_ALL_KV_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )

    def _attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
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
        max_c_len: int = 0,
        indexer_max_c_len: int = 0,
    ) -> Any:
        if int(start_pos) != 0 or int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV prefill compressor post-QDQ fusion "
                f"requires start_pos=0 and k==kv_len, got start_pos={int(start_pos)} "
                f"k={int(k)} kv_len={int(kv_len)}"
            )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        write_cache_key: tuple[Any, ...] = ()
        if bool(write_swa_state_cache):
            write_cache_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(kv_score_state, "shape", ())),
                str(getattr(kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape", ())),
                str(getattr(compressed_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(indexer_kv_score_state, "shape", ())),
                str(getattr(indexer_kv_score_state, "dtype", "unknown")),
                tuple(
                    int(dim)
                    for dim in getattr(indexer_compressed_kv_cache, "shape", ())
                ),
                str(getattr(indexer_compressed_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(owner_ids, "shape", ())),
                str(getattr(owner_ids, "dtype", "unknown")),
                int(compressor_ring_size),
                int(compressor_state_tail_len),
                int(indexer_compressor_ring_size),
                int(indexer_compressor_state_tail_len),
                int(max_c_len),
                int(indexer_max_c_len),
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
            tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ())),
            str(getattr(compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_wgate, "shape", ())),
            str(getattr(compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_ape, "shape", ())),
            str(getattr(compressor_ape, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_norm_weight, "shape", ())),
            str(getattr(compressor_norm_weight, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())),
            str(getattr(indexer_compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wgate, "shape", ())),
            str(getattr(indexer_compressor_wgate, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_ape, "shape", ())),
            str(getattr(indexer_compressor_ape, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_norm_weight, "shape", ())
            ),
            str(getattr(indexer_compressor_norm_weight, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_cos_table, "shape", ())),
            str(getattr(compressor_cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_sin_table, "shape", ())),
            str(getattr(compressor_sin_table, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_cos_table, "shape", ())
            ),
            str(getattr(indexer_compressor_cos_table, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_sin_table, "shape", ())
            ),
            str(getattr(indexer_compressor_sin_table, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos),
            int(kv_len),
            int(k),
            int(rows),
            int(k_tile),
            int(compressor_head_dim),
            int(compressor_rope_head_dim),
            int(compressor_block_size),
            float(compressor_fp8_max),
            bool(compressor_rotate),
            bool(compressor_overlap),
            float(compressor_eps),
            int(indexer_compressor_head_dim),
            int(indexer_compressor_rope_head_dim),
            int(indexer_compressor_block_size),
            float(indexer_compressor_fp8_max),
            bool(indexer_compressor_rotate),
            bool(indexer_compressor_overlap),
            float(indexer_compressor_eps),
            bool(write_swa_state_cache),
            write_cache_key,
        )
        name_prefix = "dsv4_product_attention_qkv_indexer_allkv_prefill_post_qdq"
        if bool(write_swa_state_cache):
            name_prefix = (
                "dsv4_product_attention_qkv_indexer_allkv_prefill_post_qdq_"
                "write_swa_dual_state_cache"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"kv{int(kv_len)}_k{int(k)}_rows{int(rows)}_kt{int(k_tile)}_"
            f"kvflat{int(kv_token_bucket)}_cd{int(compressor_head_dim)}_"
            f"icd{int(indexer_compressor_head_dim)}"
        )
        if bool(write_swa_state_cache):
            if (
                swa_kv_cache is None
                or kv_score_state is None
                or compressed_kv_cache is None
                or indexer_kv_score_state is None
                or indexer_compressed_kv_cache is None
                or owner_ids is None
            ):
                raise RuntimeError(
                    "DSV4 product all-KV prefill SWA/dual-state/cache fusion "
                    "requires SWA, both states, both caches, and owner ids"
                )
            fn = graph_attention.attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_from_freq_table_fn
        compile_inputs = [
            _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(wq_a, fallback_dtype=np.float32),
            _sample_array(q_norm, fallback_dtype=np.float32),
            _sample_array(wq_b, fallback_dtype=np.float32),
            _sample_array(wkv, fallback_dtype=np.float32),
            _sample_array(kv_norm, fallback_dtype=np.float32),
            _sample_array(compressor_wkv, fallback_dtype=np.float32),
            _sample_array(compressor_wgate, fallback_dtype=np.float32),
        ]
        if bool(write_swa_state_cache):
            compile_inputs.extend(
                [
                    _sample_array(swa_kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(kv_score_state, fallback_dtype=np.float32),
                    _sample_array(
                        compressed_kv_cache,
                        fallback_dtype=ml_dtypes.bfloat16,
                    ),
                    _sample_array(owner_ids, fallback_dtype=np.int32),
                ]
            )
        compile_inputs.extend(
            [
                _sample_array(compressor_ape, fallback_dtype=np.float32),
                _sample_array(compressor_norm_weight, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wkv, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wgate, fallback_dtype=np.float32),
            ]
        )
        if bool(write_swa_state_cache):
            compile_inputs.extend(
                [
                    _sample_array(indexer_kv_score_state, fallback_dtype=np.float32),
                    _sample_array(
                        indexer_compressed_kv_cache,
                        fallback_dtype=ml_dtypes.bfloat16,
                    ),
                ]
            )
        compile_inputs.extend(
            [
                _sample_array(indexer_compressor_ape, fallback_dtype=np.float32),
                _sample_array(
                    indexer_compressor_norm_weight,
                    fallback_dtype=np.float32,
                ),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(compressor_cos_table, fallback_dtype=np.float32),
                _sample_array(compressor_sin_table, fallback_dtype=np.float32),
                _sample_array(
                    indexer_compressor_cos_table,
                    fallback_dtype=np.float32,
                ),
                _sample_array(
                    indexer_compressor_sin_table,
                    fallback_dtype=np.float32,
                ),
                _sample_array(positions, fallback_dtype=np.int32),
            ]
        )
        compile_kwargs = dict(
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
            name=name,
            additional_compiler_args=getattr(self, "compiler_args", ""),
            build_dir=self.build_dir,
        )
        if bool(write_swa_state_cache):
            compile_kwargs.update(
                compressor_ring_size=int(compressor_ring_size),
                compressor_state_tail_len=int(compressor_state_tail_len),
                max_c_len=int(max_c_len),
                indexer_compressor_ring_size=int(indexer_compressor_ring_size),
                indexer_compressor_state_tail_len=int(
                    indexer_compressor_state_tail_len
                ),
                indexer_max_c_len=int(indexer_max_c_len),
            )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
                if bool(write_swa_state_cache)
                else QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )

    def _attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
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
        max_c_len: int = 0,
        indexer_max_c_len: int = 0,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if int(start_pos) <= 0 or len(x_shape) < 2 or int(x_shape[1]) != 1:
            raise RuntimeError(
                "DSV4 product all-KV decode compressor post-QDQ fusion "
                f"requires decode shape, got x={x_shape} start_pos={int(start_pos)}"
            )
        if int(k) != int(kv_len):
            raise RuntimeError(
                "DSV4 product all-KV decode compressor post-QDQ fusion requires "
                f"k == kv_len, got k={int(k)} kv_len={int(kv_len)}"
            )
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape", ()))
        idx_state_shape = tuple(
            int(dim) for dim in getattr(indexer_kv_score_state, "shape", ())
        )
        owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape", ()))
        end_shape = tuple(int(dim) for dim in getattr(end_positions, "shape", ()))
        write_cache_key: tuple[Any, ...] = ()
        if bool(write_swa_state_cache):
            write_cache_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape", ())),
                str(getattr(compressed_kv_cache, "dtype", "unknown")),
                tuple(
                    int(dim)
                    for dim in getattr(indexer_compressed_kv_cache, "shape", ())
                ),
                str(getattr(indexer_compressed_kv_cache, "dtype", "unknown")),
                int(max_c_len),
                int(indexer_max_c_len),
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
            tuple(int(dim) for dim in getattr(compressor_wkv, "shape", ())),
            str(getattr(compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_wgate, "shape", ())),
            str(getattr(compressor_wgate, "dtype", "unknown")),
            state_shape,
            str(getattr(kv_score_state, "dtype", "unknown")),
            owner_shape,
            str(getattr(owner_ids, "dtype", "unknown")),
            end_shape,
            str(getattr(end_positions, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_ape, "shape", ())),
            str(getattr(compressor_ape, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_norm_weight, "shape", ())),
            str(getattr(compressor_norm_weight, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wkv, "shape", ())),
            str(getattr(indexer_compressor_wkv, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_wgate, "shape", ())),
            str(getattr(indexer_compressor_wgate, "dtype", "unknown")),
            idx_state_shape,
            str(getattr(indexer_kv_score_state, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(indexer_compressor_ape, "shape", ())),
            str(getattr(indexer_compressor_ape, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_norm_weight, "shape", ())
            ),
            str(getattr(indexer_compressor_norm_weight, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_cos_table, "shape", ())),
            str(getattr(compressor_cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_sin_table, "shape", ())),
            str(getattr(compressor_sin_table, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_cos_table, "shape", ())
            ),
            str(getattr(indexer_compressor_cos_table, "dtype", "unknown")),
            tuple(
                int(dim) for dim in getattr(indexer_compressor_sin_table, "shape", ())
            ),
            str(getattr(indexer_compressor_sin_table, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            -1,
            int(kv_len),
            int(k),
            int(rows),
            int(k_tile),
            int(compressor_head_dim),
            int(compressor_state_width),
            int(compressor_ring_size),
            int(compressor_rope_head_dim),
            int(compressor_block_size),
            float(compressor_fp8_max),
            bool(compressor_rotate),
            bool(compressor_overlap),
            float(compressor_eps),
            int(indexer_compressor_head_dim),
            int(indexer_compressor_state_width),
            int(indexer_compressor_ring_size),
            int(indexer_compressor_rope_head_dim),
            int(indexer_compressor_block_size),
            float(indexer_compressor_fp8_max),
            bool(indexer_compressor_rotate),
            bool(indexer_compressor_overlap),
            float(indexer_compressor_eps),
            bool(write_swa_state_cache),
            write_cache_key,
        )
        name_prefix = "dsv4_product_attention_qkv_indexer_allkv_decode_post_qdq"
        if bool(write_swa_state_cache):
            name_prefix = (
                "dsv4_product_attention_qkv_indexer_allkv_decode_post_qdq_"
                "write_swa_dual_state_cache"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"sdyn_kv{int(kv_len)}_k{int(k)}_rows{int(rows)}_kt{int(k_tile)}_"
            f"kvflat{int(kv_token_bucket)}_cd{int(compressor_head_dim)}_"
            f"cw{int(compressor_state_width)}_ring{int(compressor_ring_size)}_"
            f"icd{int(indexer_compressor_head_dim)}_"
            f"icw{int(indexer_compressor_state_width)}_"
            f"iring{int(indexer_compressor_ring_size)}"
        )
        compile_start_pos = 1
        if bool(write_swa_state_cache):
            if (
                swa_kv_cache is None
                or compressed_kv_cache is None
                or indexer_compressed_kv_cache is None
            ):
                raise RuntimeError(
                    "DSV4 product all-KV decode SWA/dual-state/cache fusion "
                    "requires SWA and both compressed-cache tensors"
                )
            fn = graph_attention.attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_from_freq_table_fn
        compile_inputs = [
            _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(wq_a, fallback_dtype=np.float32),
            _sample_array(q_norm, fallback_dtype=np.float32),
            _sample_array(wq_b, fallback_dtype=np.float32),
            _sample_array(wkv, fallback_dtype=np.float32),
            _sample_array(kv_norm, fallback_dtype=np.float32),
            _sample_array(compressor_wkv, fallback_dtype=np.float32),
            _sample_array(compressor_wgate, fallback_dtype=np.float32),
        ]
        if bool(write_swa_state_cache):
            compile_inputs.extend(
                [
                    _sample_array(swa_kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(kv_score_state, fallback_dtype=np.float32),
                    _sample_array(
                        compressed_kv_cache, fallback_dtype=ml_dtypes.bfloat16
                    ),
                ]
            )
        else:
            compile_inputs.append(
                _sample_array(kv_score_state, fallback_dtype=np.float32)
            )
        compile_inputs.extend(
            [
                _sample_array(owner_ids, fallback_dtype=np.int32),
                _sample_array(end_positions, fallback_dtype=np.int32),
                _sample_array(compressor_ape, fallback_dtype=np.float32),
                _sample_array(compressor_norm_weight, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wkv, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_wgate, fallback_dtype=np.float32),
                _sample_array(indexer_kv_score_state, fallback_dtype=np.float32),
            ]
        )
        if bool(write_swa_state_cache):
            compile_inputs.append(
                _sample_array(
                    indexer_compressed_kv_cache, fallback_dtype=ml_dtypes.bfloat16
                )
            )
        compile_inputs.extend(
            [
                _sample_array(indexer_compressor_ape, fallback_dtype=np.float32),
                _sample_array(
                    indexer_compressor_norm_weight, fallback_dtype=np.float32
                ),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(compressor_cos_table, fallback_dtype=np.float32),
                _sample_array(compressor_sin_table, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_cos_table, fallback_dtype=np.float32),
                _sample_array(indexer_compressor_sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
            ]
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
                if bool(write_swa_state_cache)
                else QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
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
                start_pos=int(compile_start_pos),
                kv_len=int(kv_len),
                k=int(k),
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
                **(
                    {
                        "max_c_len": int(max_c_len),
                        "indexer_max_c_len": int(indexer_max_c_len),
                    }
                    if bool(write_swa_state_cache)
                    else {}
                ),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )
