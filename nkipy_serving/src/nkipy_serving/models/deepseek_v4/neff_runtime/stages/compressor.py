"""Compressor product kernels for DSV4 product execution."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import _compile_product_kernel
from nkipy_serving.models.deepseek_v4.neff_graphs import common as graph_common
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _as_product_device_input,
    _require_product_device_value,
    _sample_array,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
)


class Dsv4ProductCompressorMixin:
    def _compressor_post_qdq_freq_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        kv_pool: Any,
        norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        bsz: int,
        clen: int,
        source_token_positions: bool = False,
        compress_ratio: int = 1,
        start_pos: int = 0,
        seqlen: int = 0,
        rope_head_dim: int,
        block_size: int,
        fp8_max: float,
        rotate: bool,
        eps: float,
    ) -> Any:
        kv_shape = tuple(int(dim) for dim in getattr(kv_pool, "shape", ()))
        norm_shape = tuple(int(dim) for dim in getattr(norm_weight, "shape", ()))
        cos_shape = tuple(int(dim) for dim in getattr(cos_table, "shape", ()))
        sin_shape = tuple(int(dim) for dim in getattr(sin_table, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        key = (
            kv_shape,
            str(getattr(kv_pool, "dtype", "unknown")),
            norm_shape,
            str(getattr(norm_weight, "dtype", "unknown")),
            cos_shape,
            str(getattr(cos_table, "dtype", "unknown")),
            sin_shape,
            str(getattr(sin_table, "dtype", "unknown")),
            pos_shape,
            str(getattr(positions, "dtype", "unknown")),
            int(bsz),
            int(clen),
            bool(source_token_positions),
            int(compress_ratio),
            int(start_pos),
            int(seqlen),
            int(rope_head_dim),
            int(block_size),
            float(fp8_max),
            bool(rotate),
            float(eps),
        )
        name = (
            "dsv4_product_compressor_post_qdq_freq_table_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in kv_shape)}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"b{int(bsz)}_c{int(clen)}_"
            f"src{int(bool(source_token_positions))}_"
            f"r{int(compress_ratio)}_s{int(start_pos)}_q{int(seqlen)}_"
            f"rd{int(rope_head_dim)}_bs{int(block_size)}_"
            f"rot{int(bool(rotate))}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="compressor_post_qdq_freq_table_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_common.compressor_post_qdq_from_freq_table_fn,
                _sample_array(kv_pool, fallback_dtype=np.float32),
                _sample_array(norm_weight, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                bsz=int(bsz),
                clen=int(clen),
                source_token_positions=bool(source_token_positions),
                compress_ratio=int(compress_ratio),
                start_pos=int(start_pos),
                seqlen=int(seqlen),
                rope_head_dim=int(rope_head_dim),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                rotate=bool(rotate),
                eps=float(eps),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
            ),
        )

    def _compressor_decode_pool_post_qdq_freq_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        kv_score_state: Any,
        owner_ids: Any,
        end_positions: Any,
        norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        bsz: int,
        ratio: int,
        head_dim: int,
        state_width: int,
        ring_size: int,
        overlap: bool,
        source_token_positions: bool = False,
        compress_ratio: int = 1,
        start_pos: int = 0,
        seqlen: int = 0,
        rope_head_dim: int,
        block_size: int,
        fp8_max: float,
        rotate: bool,
        eps: float,
    ) -> Any:
        state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape", ()))
        owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape", ()))
        end_shape = tuple(int(dim) for dim in getattr(end_positions, "shape", ()))
        norm_shape = tuple(int(dim) for dim in getattr(norm_weight, "shape", ()))
        cos_shape = tuple(int(dim) for dim in getattr(cos_table, "shape", ()))
        sin_shape = tuple(int(dim) for dim in getattr(sin_table, "shape", ()))
        pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
        key = (
            state_shape,
            str(getattr(kv_score_state, "dtype", "unknown")),
            owner_shape,
            str(getattr(owner_ids, "dtype", "unknown")),
            end_shape,
            str(getattr(end_positions, "dtype", "unknown")),
            norm_shape,
            str(getattr(norm_weight, "dtype", "unknown")),
            cos_shape,
            str(getattr(cos_table, "dtype", "unknown")),
            sin_shape,
            str(getattr(sin_table, "dtype", "unknown")),
            pos_shape,
            str(getattr(positions, "dtype", "unknown")),
            int(bsz),
            int(ratio),
            int(head_dim),
            int(state_width),
            int(ring_size),
            bool(overlap),
            bool(source_token_positions),
            int(compress_ratio),
            int(start_pos),
            int(seqlen),
            int(rope_head_dim),
            int(block_size),
            float(fp8_max),
            bool(rotate),
            float(eps),
        )
        name = (
            "dsv4_product_compressor_decode_pool_post_qdq_"
            f"t{int(bucket.token_bucket)}_"
            f"state{'x'.join(str(v) for v in state_shape)}_"
            f"b{int(bsz)}_r{int(ratio)}_d{int(head_dim)}_"
            f"w{int(state_width)}_ring{int(ring_size)}_"
            f"ov{int(bool(overlap))}_"
            f"p{'x'.join(str(v) for v in pos_shape)}_"
            f"src{int(bool(source_token_positions))}_"
            f"rd{int(rope_head_dim)}_bs{int(block_size)}_"
            f"rot{int(bool(rotate))}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="compressor_decode_pool_post_qdq_freq_table_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_moe.compressor_decode_pool_post_qdq_from_state_freq_table_fn,
                _sample_array(kv_score_state, fallback_dtype=np.float32),
                _sample_array(owner_ids, fallback_dtype=np.int32),
                _sample_array(end_positions, fallback_dtype=np.int32),
                _sample_array(norm_weight, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                bsz=int(bsz),
                ratio=int(ratio),
                head_dim=int(head_dim),
                state_width=int(state_width),
                ring_size=int(ring_size),
                overlap=bool(overlap),
                source_token_positions=bool(source_token_positions),
                compress_ratio=int(compress_ratio),
                start_pos=int(start_pos),
                seqlen=int(seqlen),
                rope_head_dim=int(rope_head_dim),
                block_size=int(block_size),
                fp8_max=float(fp8_max),
                rotate=bool(rotate),
                eps=float(eps),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
            ),
        )

    def _run_product_compressor_post_qdq_from_freq_table(
        self,
        kv_pool: Any,
        norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        bsz: int,
        clen: int,
        source_token_positions: bool = False,
        compress_ratio: int = 1,
        start_pos: int = 0,
        seqlen: int = 0,
        rope_head_dim: int,
        block_size: int,
        fp8_max: float,
        rotate: bool,
        eps: float,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="compressor_post_qdq_from_freq_table"
        )
        _require_product_device_value(
            kv_pool,
            where="compressor_post_qdq_from_freq_table/kv_pool",
        )
        _require_product_device_value(
            norm_weight,
            where="compressor_post_qdq_from_freq_table/norm_weight",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="compressor",
        )
        positions = self._product_freq_positions_for(bucket, positions)
        kernel = self._compressor_post_qdq_freq_table_kernel_for(
            bucket,
            kv_pool,
            norm_weight,
            cos_table,
            sin_table,
            positions,
            bsz=int(bsz),
            clen=int(clen),
            source_token_positions=bool(source_token_positions),
            compress_ratio=int(compress_ratio),
            start_pos=int(start_pos),
            seqlen=int(seqlen),
            rope_head_dim=int(rope_head_dim),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            rotate=bool(rotate),
            eps=float(eps),
        )
        outputs = dict(_nkipy_output_tensors or {})
        out = outputs.get("output0")
        if out is None:
            out = self._bucket_scratch(
                bucket,
                "compressor_post_qdq_bf16",
                tuple(int(dim) for dim in getattr(kv_pool, "shape", ())),
                ml_dtypes.bfloat16,
            )
        kernel(
            inputs={
                "kv_pool": kv_pool,
                "norm_weight": norm_weight,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs={"output0": out},
        )
        return out

    def _run_product_compressor_decode_pool_post_qdq_from_state_freq_table(
        self,
        kv_score_state: Any,
        owner_ids: Any,
        end_positions: Any,
        norm_weight: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        bsz: int,
        ratio: int,
        head_dim: int,
        state_width: int,
        ring_size: int,
        overlap: bool,
        source_token_positions: bool = False,
        compress_ratio: int = 1,
        start_pos: int = 0,
        seqlen: int = 0,
        rope_head_dim: int,
        block_size: int,
        fp8_max: float,
        rotate: bool,
        eps: float,
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="compressor_decode_pool_post_qdq_from_state_freq_table"
        )
        _require_product_device_value(
            kv_score_state,
            where="compressor_decode_pool_post_qdq/kv_score_state",
        )
        _require_product_device_value(
            norm_weight,
            where="compressor_decode_pool_post_qdq/norm_weight",
        )
        owner_ids = _as_product_device_input(
            owner_ids,
            name="dsv4_product_compressor_decode_owner_ids",
        )
        end_positions = _as_product_device_input(
            end_positions,
            name="dsv4_product_compressor_decode_end_positions",
        )
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="compressor",
        )
        positions = self._product_freq_positions_for(bucket, positions)
        kernel = self._compressor_decode_pool_post_qdq_freq_table_kernel_for(
            bucket,
            kv_score_state,
            owner_ids,
            end_positions,
            norm_weight,
            cos_table,
            sin_table,
            positions,
            bsz=int(bsz),
            ratio=int(ratio),
            head_dim=int(head_dim),
            state_width=int(state_width),
            ring_size=int(ring_size),
            overlap=bool(overlap),
            source_token_positions=bool(source_token_positions),
            compress_ratio=int(compress_ratio),
            start_pos=int(start_pos),
            seqlen=int(seqlen),
            rope_head_dim=int(rope_head_dim),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            rotate=bool(rotate),
            eps=float(eps),
        )
        outputs = dict(_nkipy_output_tensors or {})
        out = outputs.get("output0")
        if out is None:
            out = self._bucket_scratch(
                bucket,
                "compressor_decode_post_qdq_bf16",
                (int(bsz), int(head_dim)),
                ml_dtypes.bfloat16,
            )
        kernel(
            inputs={
                "kv_score_state": kv_score_state,
                "owner_ids": owner_ids,
                "end_positions": end_positions,
                "norm_weight": norm_weight,
                "cos_table": cos_table,
                "sin_table": sin_table,
                "positions": positions,
            },
            outputs={"output0": out},
        )
        return out
