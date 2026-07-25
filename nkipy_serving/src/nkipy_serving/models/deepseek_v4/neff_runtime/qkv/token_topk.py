"""Token-top-k QKV/indexer product kernel cache helpers."""

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


class Dsv4ProductQkvTokenTopkMixin:
    def _attention_qkv_token_topk_prep_kernel_for(
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
        q_softmax_scale: float,
        q_token_bucket: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        max_c_len: int,
        rows: int,
        k_tile: int,
        kv_token_bucket: int = 0,
        return_qr: bool = True,
        dynamic_decode_start_pos: bool = False,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos_key),
            int(max_c_len),
            int(rows),
            int(k_tile),
            int(kv_token_bucket),
            bool(return_qr),
            bool(dynamic_decode_start_pos),
        )
        name = (
            "dsv4_product_attention_qkv_token_topk_prep_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"s{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}_"
            f"c{int(max_c_len)}_"
            f"rows{int(rows)}_k{int(k_tile)}"
            f"{'_kvflat' + str(int(kv_token_bucket)) if int(kv_token_bucket) else ''}"
            f"{'_noqr' if not bool(return_qr) else ''}"
        )
        if bool(dynamic_decode_start_pos):
            if bool(return_qr):
                raise RuntimeError(
                    "DSV4 product dynamic decode QKV/top-k path does not materialize QR"
                )
            fn = graph_attention.attention_qkv_token_topk_prep_decode_no_qr_from_freq_table_fn
        else:
            fn = (
                graph_attention.attention_qkv_token_topk_prep_from_freq_table_fn
                if bool(return_qr)
                else graph_attention.attention_qkv_token_topk_prep_no_qr_from_freq_table_fn
            )
        compile_start_pos = 1 if bool(dynamic_decode_start_pos) else int(start_pos)
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=QkvVariantName.TOKEN_TOPK_PREP,
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
                q_softmax_scale=float(q_softmax_scale),
                q_token_bucket=int(q_token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(compile_start_pos),
                max_c_len=int(max_c_len),
                rows=int(rows),
                k_tile=int(k_tile),
                kv_token_bucket=int(kv_token_bucket),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )

    def _attention_qkv_compressor_token_topk_prep_kernel_for(
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
        kv_token_bucket: int = 0,
        dynamic_decode_start_pos: bool = False,
        write_swa_state: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ape: Any | None = None,
        compressor_ring_size: int = 0,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        start_pos_key = -1 if bool(dynamic_decode_start_pos) else int(start_pos)
        write_swa_state_key: tuple[Any, ...] = ()
        if bool(write_swa_state):
            write_swa_state_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(kv_score_state, "shape", ())),
                str(getattr(kv_score_state, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(owner_ids, "shape", ())),
                str(getattr(owner_ids, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressor_ape, "shape", ())),
                str(getattr(compressor_ape, "dtype", "unknown")),
                int(compressor_ring_size),
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos_key),
            int(max_c_len),
            int(rows),
            int(k_tile),
            int(kv_token_bucket),
            bool(dynamic_decode_start_pos),
            bool(write_swa_state),
            write_swa_state_key,
        )
        name_prefix = "dsv4_product_attention_qkv_compressor_token_topk_prep"
        if bool(write_swa_state):
            if not bool(dynamic_decode_start_pos):
                raise RuntimeError(
                    "DSV4 product QKV/compressor SWA/state write is decode-only"
                )
            name_prefix = (
                "dsv4_product_attention_qkv_compressor_token_topk_write_swa_state"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"s{'dyn' if bool(dynamic_decode_start_pos) else int(start_pos)}_"
            f"c{int(max_c_len)}_"
            f"rows{int(rows)}_k{int(k_tile)}"
            f"{'_kvflat' + str(int(kv_token_bucket)) if int(kv_token_bucket) else ''}"
            "_noqr"
        )
        if bool(write_swa_state):
            fn = graph_attention.attention_qkv_compressor_kv_score_token_topk_prep_write_swa_state_decode_no_qr_from_freq_table_fn
        else:
            fn = (
                graph_attention.attention_qkv_compressor_kv_score_token_topk_prep_decode_no_qr_from_freq_table_fn
                if bool(dynamic_decode_start_pos)
                else graph_attention.attention_qkv_compressor_kv_score_token_topk_prep_no_qr_from_freq_table_fn
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
        ]
        if bool(write_swa_state):
            if (
                swa_kv_cache is None
                or kv_score_state is None
                or owner_ids is None
                or compressor_ape is None
            ):
                raise RuntimeError(
                    "DSV4 product SWA/state write kernel requires cache, state, "
                    "owner IDs, and compressor APE inputs"
                )
            compile_inputs.extend(
                [
                    _sample_array(swa_kv_cache, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(kv_score_state, fallback_dtype=ml_dtypes.bfloat16),
                    _sample_array(owner_ids, fallback_dtype=np.int32),
                    _sample_array(compressor_ape, fallback_dtype=np.float32),
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
            "window_size": int(window_size),
            "ratio": int(ratio),
            "offset": int(offset),
            "start_pos": int(compile_start_pos),
            "max_c_len": int(max_c_len),
            "rows": int(rows),
            "k_tile": int(k_tile),
            "kv_token_bucket": int(kv_token_bucket),
        }
        if bool(write_swa_state):
            compile_kwargs["compressor_ring_size"] = int(compressor_ring_size)
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
                QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE
                if bool(write_swa_state)
                else QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )

    def _attention_qkv_compressor_prefill_post_qdq_token_topk_prep_kernel_for(
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
        kv_token_bucket: int,
        compressor_head_dim: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        write_swa_state_cache: bool = False,
        swa_kv_cache: Any | None = None,
        kv_score_state: Any | None = None,
        compressed_kv_cache: Any | None = None,
        owner_ids: Any | None = None,
        compressor_ring_size: int = 0,
        compressor_state_tail_len: int = 0,
    ) -> Any:
        if int(start_pos) != 0:
            raise RuntimeError(
                "DSV4 product prefill compressor post-QDQ fusion requires "
                f"start_pos=0, got {int(start_pos)}"
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
                tuple(int(dim) for dim in getattr(owner_ids, "shape", ())),
                str(getattr(owner_ids, "dtype", "unknown")),
                int(compressor_ring_size),
                int(compressor_state_tail_len),
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
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_cos_table, "shape", ())),
            str(getattr(compressor_cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_sin_table, "shape", ())),
            str(getattr(compressor_sin_table, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            int(start_pos),
            int(max_c_len),
            int(rows),
            int(k_tile),
            int(kv_token_bucket),
            int(compressor_head_dim),
            int(compressor_rope_head_dim),
            int(compressor_block_size),
            float(compressor_fp8_max),
            bool(compressor_rotate),
            bool(compressor_overlap),
            float(compressor_eps),
            bool(write_swa_state_cache),
            write_cache_key,
        )
        name_prefix = "dsv4_product_attention_qkv_comp_prefill_post_qdq_token_topk"
        if bool(write_swa_state_cache):
            name_prefix = (
                "dsv4_product_attention_qkv_comp_prefill_post_qdq_token_topk_"
                "write_swa_state_cache"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"rows{int(rows)}_k{int(k_tile)}_kvflat{int(kv_token_bucket)}_"
            f"cd{int(compressor_head_dim)}_crd{int(compressor_rope_head_dim)}_"
            f"rot{int(bool(compressor_rotate))}_ov{int(bool(compressor_overlap))}"
        )
        if bool(write_swa_state_cache):
            if (
                swa_kv_cache is None
                or kv_score_state is None
                or compressed_kv_cache is None
                or owner_ids is None
            ):
                raise RuntimeError(
                    "DSV4 product prefill post-QDQ SWA/state/cache fusion "
                    "requires SWA, state, cache, and owner-id tensors"
                )
            fn = graph_attention.attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_no_qr_from_freq_table_fn
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
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(compressor_cos_table, fallback_dtype=np.float32),
                _sample_array(compressor_sin_table, fallback_dtype=np.float32),
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
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=int(kv_token_bucket),
            compressor_head_dim=int(compressor_head_dim),
            compressor_rope_head_dim=int(compressor_rope_head_dim),
            compressor_block_size=int(compressor_block_size),
            compressor_fp8_max=float(compressor_fp8_max),
            compressor_rotate=bool(compressor_rotate),
            compressor_overlap=bool(compressor_overlap),
            compressor_eps=float(compressor_eps),
            name=name,
            additional_compiler_args=getattr(self, "compiler_args", ""),
            build_dir=self.build_dir,
        )
        if bool(write_swa_state_cache):
            compile_kwargs.update(
                compressor_ring_size=int(compressor_ring_size),
                compressor_state_tail_len=int(compressor_state_tail_len),
            )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
                if bool(write_swa_state_cache)
                else QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )

    def _attention_qkv_compressor_decode_post_qdq_token_topk_prep_kernel_for(
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
        kv_token_bucket: int,
        compressor_head_dim: int,
        compressor_state_width: int,
        compressor_ring_size: int,
        compressor_rope_head_dim: int,
        compressor_block_size: int,
        compressor_fp8_max: float,
        compressor_rotate: bool,
        compressor_overlap: bool,
        compressor_eps: float,
        write_swa_state_cache: bool = False,
        compressed_cache_stride: int = 0,
        swa_kv_cache: Any | None = None,
        compressed_kv_cache: Any | None = None,
    ) -> Any:
        if int(start_pos) <= 0:
            raise RuntimeError(
                "DSV4 product decode compressor post-QDQ fusion requires "
                f"decode start_pos > 0, got {int(start_pos)}"
            )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape", ()))
        owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape", ()))
        end_shape = tuple(int(dim) for dim in getattr(end_positions, "shape", ()))
        write_cache_key: tuple[Any, ...] = ()
        if bool(write_swa_state_cache):
            write_cache_key = (
                tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ())),
                str(getattr(swa_kv_cache, "dtype", "unknown")),
                tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape", ())),
                str(getattr(compressed_kv_cache, "dtype", "unknown")),
                int(compressed_cache_stride),
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
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_cos_table, "shape", ())),
            str(getattr(compressor_cos_table, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(compressor_sin_table, "shape", ())),
            str(getattr(compressor_sin_table, "dtype", "unknown")),
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
            int(window_size),
            int(ratio),
            int(offset),
            -1,
            int(max_c_len),
            int(compressed_cache_stride),
            int(rows),
            int(k_tile),
            int(kv_token_bucket),
            int(compressor_head_dim),
            int(compressor_state_width),
            int(compressor_ring_size),
            int(compressor_rope_head_dim),
            int(compressor_block_size),
            float(compressor_fp8_max),
            bool(compressor_rotate),
            bool(compressor_overlap),
            float(compressor_eps),
            bool(write_swa_state_cache),
            write_cache_key,
        )
        name_prefix = "dsv4_product_attention_qkv_comp_decode_post_qdq_token_topk"
        if bool(write_swa_state_cache):
            name_prefix = (
                "dsv4_product_attention_qkv_comp_decode_post_qdq_token_topk_"
                "write_swa_state_cache"
            )
        name = (
            f"{name_prefix}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"h{int(n_heads)}_d{int(head_dim)}_"
            f"w{int(window_size)}_r{int(ratio)}_o{int(offset)}_"
            f"sdyn_c{int(max_c_len)}_rows{int(rows)}_k{int(k_tile)}_"
            f"kvflat{int(kv_token_bucket)}_cd{int(compressor_head_dim)}_"
            f"cw{int(compressor_state_width)}_ring{int(compressor_ring_size)}_"
            f"crd{int(compressor_rope_head_dim)}_"
            f"rot{int(bool(compressor_rotate))}_ov{int(bool(compressor_overlap))}"
        )
        if bool(write_swa_state_cache):
            name = f"{name}_cs{int(compressed_cache_stride)}"
        compile_start_pos = 1
        if bool(write_swa_state_cache):
            if swa_kv_cache is None or compressed_kv_cache is None:
                raise RuntimeError(
                    "DSV4 product decode post-QDQ SWA/state/cache fusion "
                    "requires SWA and compressed-cache tensors"
                )
            fn = graph_attention.attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn
        else:
            fn = graph_attention.attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_no_qr_from_freq_table_fn
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
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(compressor_cos_table, fallback_dtype=np.float32),
                _sample_array(compressor_sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
            ]
        )
        compile_kwargs = {
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "rope_head_dim": int(rope_head_dim),
            "eps": float(eps),
            "block_size": int(block_size),
            "fp8_max": float(fp8_max),
            "q_softmax_scale": float(q_softmax_scale),
            "q_token_bucket": int(q_token_bucket),
            "window_size": int(window_size),
            "ratio": int(ratio),
            "offset": int(offset),
            "start_pos": int(compile_start_pos),
            "max_c_len": int(max_c_len),
            "rows": int(rows),
            "k_tile": int(k_tile),
            "kv_token_bucket": int(kv_token_bucket),
            "compressor_head_dim": int(compressor_head_dim),
            "compressor_state_width": int(compressor_state_width),
            "compressor_ring_size": int(compressor_ring_size),
            "compressor_rope_head_dim": int(compressor_rope_head_dim),
            "compressor_block_size": int(compressor_block_size),
            "compressor_fp8_max": float(compressor_fp8_max),
            "compressor_rotate": bool(compressor_rotate),
            "compressor_overlap": bool(compressor_overlap),
            "compressor_eps": float(compressor_eps),
            "name": name,
            "additional_compiler_args": getattr(self, "compiler_args", ""),
            "build_dir": self.build_dir,
        }
        if bool(write_swa_state_cache):
            compile_kwargs["compressed_cache_stride"] = int(compressed_cache_stride)
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=(
                QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
                if bool(write_swa_state_cache)
                else QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP
            ),
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                fn,
                *compile_inputs,
                **compile_kwargs,
                load=False,
            ),
        )
