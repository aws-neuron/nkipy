"""DP-attention product kernels for DSV4 execution."""

from __future__ import annotations

import hashlib
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _collective_load_barrier_metadata_for_groups,
    _compile_product_kernel,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import common as graph_common
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _product_executor_coord,
    _product_warmup_trace,
    _require_product_device_value,
    _sample_array,
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _product_canonical_neff_cache_key,
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


class Dsv4ProductDpAttentionMixin:
    def _unload_after_dp_attention_pipeline_kernel_call(self) -> bool:
        keep_loaded = getattr(
            self,
            "_keep_dp_attention_pipeline_collectives_loaded",
            None,
        )
        return not (callable(keep_loaded) and bool(keep_loaded()))

    def _run_product_dp_attention_lane_slice(
        self,
        x: Any,
        *,
        start: int,
        size: int,
    ) -> Any:
        _require_product_device_value(x, where="dp_attention_lane_slice")
        alias = _alias_device_value_first_dim_slice(
            x,
            start=int(start),
            size=int(size),
        )
        if alias is None:
            raise RuntimeError(
                "DSV4 product DP-attention lane slice requires an NRT "
                "first-dimension tensor alias; standalone slice kernels are "
                "not part of the product path"
            )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 3 or int(size) <= 0:
            return alias
        batch_size, seqlen, hidden_size = x_shape
        if int(seqlen) <= 1:
            return alias
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            return alias
        active_rows = int(size) * int(seqlen)
        attention_bucket = self._attention_backend_bucket_for_tokens(
            int(active_rows),
            int(bucket.token_bucket),
            is_decode=False,
        )
        if int(attention_bucket) <= int(active_rows):
            return alias
        compile_bsz, compile_seqlen = self._product_compile_sequence_shape(
            bucket,
            bsz=int(batch_size),
            seqlen=int(seqlen),
        )
        full = self._product_full_value_for(
            x,
            (int(compile_bsz), int(compile_seqlen), int(hidden_size)),
        )
        if full is None:
            return alias
        flat_full = _alias_device_value_shape(
            full,
            (int(compile_bsz) * int(compile_seqlen), int(hidden_size)),
        )
        if flat_full is None:
            return alias
        token_start = int(start) * int(seqlen)
        flat_rows = int(getattr(flat_full, "shape", (0,))[0])
        if token_start < 0 or token_start + int(attention_bucket) > flat_rows:
            return alias
        lane_flat = _alias_device_value_first_dim_slice(
            flat_full,
            start=int(token_start),
            size=int(attention_bucket),
        )
        if lane_flat is None:
            return alias
        lane_full = _alias_device_value_shape(
            lane_flat,
            (1, int(attention_bucket), int(hidden_size)),
        )
        if lane_full is None:
            return alias
        return self._product_active_alias(
            lane_full,
            (int(size), int(seqlen), int(hidden_size)),
        )

    def _sequence_hidden_pad_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        *,
        rows: int,
        hidden_size: int,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        dtype_name = str(getattr(x, "dtype", "unknown")).replace(".", "_")
        rows_i = int(rows)
        hidden_i = int(hidden_size)
        key = (x_shape, dtype_name, rows_i, hidden_i)
        name = (
            "dsv4_product_sequence_hidden_pad_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"rows{rows_i}_h{hidden_i}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="sequence_hidden_pad_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_common.sequence_hidden_pad_fn,
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                rows=rows_i,
                dim=hidden_i,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )

    def _run_product_sequence_hidden_pad(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        *,
        rows: int,
        hidden_size: int,
    ) -> Any:
        _require_product_device_value(x, where="sequence_hidden_pad")
        rows_i = int(rows)
        hidden_i = int(hidden_size)
        if rows_i <= 0 or hidden_i <= 0:
            raise RuntimeError(
                "DSV4 product sequence hidden pad requires positive rows/hidden, "
                f"got rows={rows_i} hidden={hidden_i}"
            )
        kernel = self._sequence_hidden_pad_kernel_for(
            bucket,
            x,
            rows=rows_i,
            hidden_size=hidden_i,
        )
        out = self._bucket_scratch(
            bucket,
            "sequence_hidden_pad",
            (1, rows_i, hidden_i),
            _value_dtype(x, fallback=ml_dtypes.bfloat16),
        )
        _product_warmup_trace(
            _product_executor_coord(self),
            "sequence_hidden_pad kernel_run start "
            f"x_shape={getattr(x, 'shape', None)} rows={rows_i} hidden={hidden_i} "
            f"out_shape={getattr(out, 'shape', None)}",
        )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={"x": x},
            outputs={"output0": out},
            unload_after_call=self._unload_after_dp_attention_pipeline_kernel_call(),
        )
        _product_warmup_trace(
            _product_executor_coord(self),
            "sequence_hidden_pad kernel_run done "
            f"x_shape={getattr(x, 'shape', None)} rows={rows_i} hidden={hidden_i}",
        )
        return out

    def _run_product_dp_attention_flat_zero(
        self,
        x: Any,
        *,
        rows: int,
        hidden_size: int,
    ) -> Any:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(
                "DSV4 product DP-attention flat zero requires an active token bucket"
            )
        _require_product_device_value(x, where="dp_attention_flat_zero")
        return self._bucket_scratch(
            bucket,
            "dp_attention_flat_zero",
            (int(rows), int(hidden_size)),
            np.float32,
        )

    def _run_dp_attention_all_reduce_unpad_post_pre(
        self,
        x: Any,
        *,
        replica_groups: tuple[tuple[int, ...], ...],
        bsz: int,
        seqlen: int,
        hidden_size: int,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        output_bsz: int | None = None,
        output_seqlen: int | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        groups = tuple(tuple(int(rank) for rank in group) for group in replica_groups)
        if not groups or all(len(group) <= 1 for group in groups):
            raise RuntimeError(
                "DSV4 product DP-attention post/pre requires a real DP "
                "all-reduce group; standalone unpad/post-pre kernels are not "
                "part of the product path"
            )
        executor = self.runtime_surface
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(
                "DSV4 product DP-attention all-reduce/post-pre requires an "
                "active token bucket"
            )
        for value, where in (
            (x, "dp_attention_all_reduce_post_pre/x"),
            (residual, "dp_attention_all_reduce_post_pre/residual"),
            (post, "dp_attention_all_reduce_post_pre/post"),
            (comb, "dp_attention_all_reduce_post_pre/comb"),
            (hc_fn, "dp_attention_all_reduce_post_pre/hc_fn"),
            (hc_scale, "dp_attention_all_reduce_post_pre/hc_scale"),
            (hc_base, "dp_attention_all_reduce_post_pre/hc_base"),
            (norm_weight, "dp_attention_all_reduce_post_pre/norm_weight"),
        ):
            _require_product_device_value(value, where=where)
        hc_mult = int(executor.args.hc_mult)
        dispatch_context = getattr(
            self,
            "_active_dp_attention_moe_dispatch_context",
            None,
        )
        logical_bsz = int(output_bsz) if output_bsz is not None else int(bsz)
        logical_seqlen = (
            int(output_seqlen) if output_seqlen is not None else int(seqlen)
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        x_rows = int(x_shape[0]) if x_shape else int(bucket.token_bucket)
        compile_bsz, compile_seqlen = self._dp_attention_post_pre_compile_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            rows=int(x_rows),
            dispatch_context=dispatch_context,
        )
        (
            compile_bsz,
            compile_seqlen,
            residual_kernel,
            post_kernel,
            comb_kernel,
        ) = self._product_promote_mhc_state_shape(
            residual,
            post,
            comb,
            compile_bsz=compile_bsz,
            compile_seqlen=int(compile_seqlen),
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=hc_mult,
        )
        if self._should_split_dp_attention_post_pre(
            x,
            seqlen=int(compile_seqlen),
        ):
            _product_warmup_trace(
                _product_executor_coord(self),
                "prefill DP-attention post/pre concat skipped "
                f"rows={int(getattr(x, 'shape', (0,))[0])} "
                f"max_rows={int(self.product_prefill_dp_attention_post_pre_fusion_max_rows)}",
            )
            return self._run_dp_attention_all_reduce_split_post_pre(
                x,
                replica_groups=groups,
                bsz=int(compile_bsz),
                seqlen=int(compile_seqlen),
                output_bsz=int(logical_bsz),
                output_seqlen=int(logical_seqlen),
                hidden_size=int(hidden_size),
                residual=residual_kernel,
                post=post_kernel,
                comb=comb_kernel,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                norm_weight=norm_weight,
                hc_mult=hc_mult,
                sinkhorn_iters=int(executor.args.hc_sinkhorn_iters),
                norm_eps=float(executor.args.norm_eps),
                hc_eps=float(executor.args.hc_eps),
            )
        if dispatch_context is not None:
            return self._run_dp_attention_all_reduce_post_pre_moe_dispatch(
                x,
                dispatch_context=dispatch_context,
                replica_groups=groups,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                residual=residual_kernel,
                post=post_kernel,
                comb=comb_kernel,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                norm_weight=norm_weight,
            )
        kernel = self._dp_attention_all_reduce_post_pre_kernel_for(
            bucket,
            x,
            residual_kernel,
            post_kernel,
            comb_kernel,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            replica_groups=groups,
            bsz=int(compile_bsz),
            seqlen=int(compile_seqlen),
            hidden_size=int(hidden_size),
            hc_mult=hc_mult,
            sinkhorn_iters=int(executor.args.hc_sinkhorn_iters),
            norm_eps=float(executor.args.norm_eps),
            hc_eps=float(executor.args.hc_eps),
        )
        residual_shape = tuple(
            int(dim) for dim in getattr(residual_kernel, "shape", ())
        )
        if len(residual_shape) < 4:
            raise RuntimeError(
                "DSV4 product DP-attention all-reduce/post-pre expects "
                f"residual [batch, seq, hc, dim], got {residual_shape}"
            )
        h = self._alloc_mhc_post_output(
            bucket,
            residual_shape=residual_shape,
            residual=residual,
            x=x,
        )
        dtype = _value_dtype(residual, fallback=ml_dtypes.bfloat16)
        outputs = {
            "output0": h,
            "output1": self._bucket_scratch(
                bucket,
                "mhc_pre_y",
                (residual_shape[0], residual_shape[1], residual_shape[-1]),
                dtype,
            ),
            "output2": self._bucket_scratch(
                bucket,
                "mhc_pre_post",
                (residual_shape[0], residual_shape[1], hc_mult),
                np.float32,
            ),
            "output3": self._bucket_scratch(
                bucket,
                "mhc_pre_comb",
                (residual_shape[0], residual_shape[1], hc_mult, hc_mult),
                np.float32,
            ),
        }
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "x": x,
                "residual": residual_kernel,
                "post": post_kernel,
                "comb": comb_kernel,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_weight": norm_weight,
            },
            outputs=outputs,
            unload_after_call=self._unload_after_dp_attention_pipeline_kernel_call(),
        )
        return self._alias_mhc_post_pre_outputs(
            outputs,
            bsz=logical_bsz,
            seqlen=logical_seqlen,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )

    def _should_split_dp_attention_post_pre(
        self,
        x: Any,
        *,
        seqlen: int,
    ) -> bool:
        max_rows = int(
            getattr(
                self,
                "product_prefill_dp_attention_post_pre_fusion_max_rows",
                0,
            )
            or 0
        )
        if max_rows <= 0 or int(seqlen) <= 1:
            return False
        shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if not shape:
            return False
        return int(shape[0]) > max_rows

    def _run_dp_attention_all_reduce_split_post_pre(
        self,
        x: Any,
        *,
        replica_groups: tuple[tuple[int, ...], ...],
        bsz: int,
        seqlen: int,
        output_bsz: int | None = None,
        output_seqlen: int | None = None,
        hidden_size: int,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(
            where="DP-attention split post/pre"
        )
        reduced = self._run_product_dp_attention_split_all_reduce(
            x,
            replica_groups=replica_groups,
        )
        kernel = self._dp_attention_unpad_post_pre_kernel_for(
            bucket,
            reduced,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
        )
        residual_shape = tuple(int(dim) for dim in getattr(residual, "shape", ()))
        if len(residual_shape) < 4:
            raise RuntimeError(
                "DSV4 product DP-attention split post/pre expects residual "
                f"[batch, seq, hc, hidden], got {residual_shape}"
            )
        h = self._alloc_mhc_post_output(
            bucket,
            residual_shape=residual_shape,
            residual=residual,
            x=x,
        )
        dtype = _value_dtype(residual, fallback=ml_dtypes.bfloat16)
        outputs = {
            "output0": h,
            "output1": self._bucket_scratch(
                bucket,
                "mhc_pre_y",
                (residual_shape[0], residual_shape[1], residual_shape[-1]),
                dtype,
            ),
            "output2": self._bucket_scratch(
                bucket,
                "mhc_pre_post",
                (residual_shape[0], residual_shape[1], int(hc_mult)),
                np.float32,
            ),
            "output3": self._bucket_scratch(
                bucket,
                "mhc_pre_comb",
                (
                    residual_shape[0],
                    residual_shape[1],
                    int(hc_mult),
                    int(hc_mult),
                ),
                np.float32,
            ),
        }
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "x": reduced,
                "residual": residual,
                "post": post,
                "comb": comb,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_weight": norm_weight,
            },
            outputs=outputs,
            unload_after_call=self._unload_after_dp_attention_pipeline_kernel_call(),
        )
        return self._alias_mhc_post_pre_outputs(
            outputs,
            bsz=int(output_bsz) if output_bsz is not None else int(bsz),
            seqlen=int(output_seqlen) if output_seqlen is not None else int(seqlen),
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )

    def _run_product_dp_attention_split_all_reduce(
        self,
        x: Any,
        *,
        replica_groups: tuple[tuple[int, ...], ...],
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="DP-attention split all-reduce"
        )
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        if len(x_shape) != 2:
            raise RuntimeError(
                "DSV4 product DP-attention split all-reduce expects "
                f"[rows, hidden], got {x_shape}"
            )
        max_rows = int(
            getattr(
                self,
                "product_prefill_dp_attention_post_pre_fusion_max_rows",
                0,
            )
            or 0
        )
        out = self._bucket_scratch(
            bucket,
            "dp_attention_split_all_reduce",
            x_shape,
            _value_dtype(x, fallback=np.float32),
        )
        if max_rows > 0 and int(x_shape[0]) > max_rows:
            _product_warmup_trace(
                _product_executor_coord(self),
                "prefill DP-attention post/pre split uses bucket all-reduce "
                f"rows={int(x_shape[0])} post_pre_max_rows={max_rows}",
            )
        kernel = self._dp_attention_all_reduce_kernel_for(
            bucket,
            x,
            replica_groups=replica_groups,
        )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={"x": x},
            outputs={"output0": out},
            unload_after_call=self._unload_after_dp_attention_pipeline_kernel_call(),
        )
        return out

    def _dp_attention_all_reduce_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        *,
        replica_groups: tuple[tuple[int, ...], ...],
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        groups = tuple(tuple(int(rank) for rank in group) for group in replica_groups)
        key = (
            x_shape,
            str(getattr(x, "dtype", "unknown")).replace(".", "_"),
            groups,
        )
        cached = bucket.kernel_caches["dp_attention_all_reduce_kernels"].get(key)
        if cached is not None:
            return cached
        rank_id, world_size = self._collective_graph_metadata(
            "dp_attention_all_reduce",
            where="DP-attention split all-reduce",
        )
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        barrier_rank_id, barrier_world_size = (
            _collective_load_barrier_metadata_for_groups(
                rank_id=int(rank_id),
                world_size=int(world_size),
                replica_groups=groups,
            )
        )
        name = (
            "dsv4_product_dp_attention_ar_split_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_{group_tag}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="dp_attention_all_reduce_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_common.dp_attention_all_reduce_fn,
                _sample_array(x, fallback_dtype=np.float32),
                replica_groups=groups,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load=False,
                load_barrier_name=(
                    "dsv4_product_dp_attention_ar_split_"
                    f"t{int(bucket.token_bucket)}_"
                    f"x{'x'.join(str(v) for v in x_shape)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=_product_canonical_neff_cache_key(
                    "dsv4_product_dp_attention_ar_split",
                    "v1",
                    key,
                ),
            ),
        )

    def _dp_attention_unpad_post_pre_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        *,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        key = (
            x_shape,
            str(getattr(x, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(residual, "shape", ())),
            str(getattr(residual, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(post, "shape", ())),
            str(getattr(post, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(comb, "shape", ())),
            str(getattr(comb, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(hc_fn, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_scale, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_base, "shape", ())),
            tuple(int(dim) for dim in getattr(norm_weight, "shape", ())),
            int(bsz),
            int(seqlen),
            int(hidden_size),
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
        )
        name = (
            f"dsv4_product_dp_attention_post_pre_split_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="dp_attention_unpad_post_pre_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_common.dp_attention_unpad_reshape_mhc_post_pre_fn,
                _sample_array(x, fallback_dtype=np.float32),
                _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(post, fallback_dtype=np.float32),
                _sample_array(comb, fallback_dtype=np.float32),
                _sample_array(hc_fn, fallback_dtype=np.float32),
                _sample_array(hc_scale, fallback_dtype=np.float32),
                _sample_array(hc_base, fallback_dtype=np.float32),
                _sample_array(norm_weight, fallback_dtype=np.float32),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                hc_mult=int(hc_mult),
                sinkhorn_iters=int(sinkhorn_iters),
                norm_eps=float(norm_eps),
                hc_eps=float(hc_eps),
            ),
        )

    def _dp_attention_all_reduce_post_pre_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        *,
        replica_groups: tuple[tuple[int, ...], ...],
        bsz: int,
        seqlen: int,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> Any:
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        groups = tuple(tuple(int(rank) for rank in group) for group in replica_groups)
        key = (
            x_shape,
            str(getattr(x, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(residual, "shape", ())),
            str(getattr(residual, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(post, "shape", ())),
            str(getattr(post, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(comb, "shape", ())),
            str(getattr(comb, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(hc_fn, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_scale, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_base, "shape", ())),
            tuple(int(dim) for dim in getattr(norm_weight, "shape", ())),
            int(bsz),
            int(seqlen),
            int(hidden_size),
            groups,
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
        )
        cached = bucket.kernel_caches["dp_attention_all_reduce_post_pre_kernels"].get(
            key
        )
        if cached is not None:
            return cached
        rank_id, world_size = self._collective_graph_metadata(
            "dp_attention_all_reduce",
            where="DP-attention all-reduce/post-pre",
        )
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        barrier_rank_id, barrier_world_size = (
            _collective_load_barrier_metadata_for_groups(
                rank_id=int(rank_id),
                world_size=int(world_size),
                replica_groups=groups,
            )
        )
        name = (
            "dsv4_product_dp_attention_ar_post_pre_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_{group_tag}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="dp_attention_all_reduce_post_pre_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_moe.dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn,
                _sample_array(x, fallback_dtype=np.float32),
                _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(post, fallback_dtype=np.float32),
                _sample_array(comb, fallback_dtype=np.float32),
                _sample_array(hc_fn, fallback_dtype=np.float32),
                _sample_array(hc_scale, fallback_dtype=np.float32),
                _sample_array(hc_base, fallback_dtype=np.float32),
                _sample_array(norm_weight, fallback_dtype=np.float32),
                replica_groups=groups,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                hc_mult=int(hc_mult),
                sinkhorn_iters=int(sinkhorn_iters),
                norm_eps=float(norm_eps),
                hc_eps=float(hc_eps),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load=False,
                load_barrier_name=(
                    "dsv4_product_dp_attention_ar_post_pre_"
                    f"t{int(bucket.token_bucket)}_"
                    f"x{'x'.join(str(v) for v in x_shape)}_"
                    f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=(
                    _product_canonical_neff_cache_key(
                        "dsv4_product_dp_attention_ar_post_pre",
                        "v1",
                        key,
                    )
                ),
            ),
        )
