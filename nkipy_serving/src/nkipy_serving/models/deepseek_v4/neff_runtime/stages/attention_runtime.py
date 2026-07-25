"""Attention-layer runtime orchestration for DSV4 product execution."""

from __future__ import annotations

from typing import Any, Mapping

from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _product_executor_coord,
    _product_warmup_trace,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.stages.attention_execution import (
    run_dsv4_attention,
    run_dsv4_attention_with_backend,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)


class Dsv4ProductAttentionRuntimeMixin:
    def _run_attention_replicated(
        self,
        block: Any,
        y: Any,
        metadata: Any | None,
        *,
        layer_id: int,
        start_pos: int,
        device_layer_state: Any,
        token_bucket: int | None = None,
        profile_fields: Mapping[str, Any] | None = None,
    ) -> Any:
        graph = self._attention_graph()
        y_shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
        y_tokens = int(y_shape[0] * y_shape[1]) if len(y_shape) >= 2 else 0
        is_compressed = bool(int(getattr(block.attn, "compress_ratio", 0)) != 0)
        replicated_profile_fields: dict[str, Any] = {
            "layer_id": int(layer_id),
            "start_pos": int(start_pos),
            "token_bucket": None if token_bucket is None else int(token_bucket),
            "y_shape": y_shape,
            "y_tokens": int(y_tokens),
            "compressed": bool(is_compressed),
        }
        if profile_fields is not None:
            replicated_profile_fields.update(dict(profile_fields))

        def _attention_profile(
            stage: str,
            elapsed_s: float,
            fields: dict[str, Any],
        ) -> None:
            event_fields = dict(replicated_profile_fields)
            event_fields.update(fields)
            self._write_product_stage_profile(
                f"attention_replicated_{stage}",
                float(elapsed_s),
                **event_fields,
            )

        attention_bucket = (
            self._compressed_attention_bucket_for(y, token_bucket)
            if is_compressed and y_tokens > 0
            else self._attention_bucket_for(y, token_bucket)
        )
        if metadata is not None and not is_compressed:
            owner_bucket = (
                None
                if getattr(self, "_active_product_bucket", None) is None
                or attention_bucket is None
                else self._ensure_product_bucket(int(attention_bucket))
            )
            backend_bucket = getattr(
                getattr(self, "attention_backend", None),
                "active_bucket",
                None,
            )
            attention_rows = (
                max(int(backend_bucket), int(y_tokens))
                if backend_bucket is not None
                else int(attention_bucket)
            )
            attention_output = self._attention_output_scratch_for(
                owner_bucket,
                rows=attention_rows,
                n_heads=int(getattr(block.attn, "n_heads")),
                head_dim=int(getattr(block.attn, "head_dim")),
            )
            return run_dsv4_attention_with_backend(
                graph,
                block.attn,
                y,
                metadata,
                layer_id=layer_id,
                backend=self.attention_backend,
                attention_output=attention_output,
                attention_scratch=(
                    None
                    if owner_bucket is None
                    else lambda kind, shape, dtype: self._bucket_scratch(
                        owner_bucket,
                        kind,
                        shape,
                        dtype,
                    )
                ),
                attention_profile=_attention_profile,
                attention_hidden_shape=y_shape,
            )
        owner_bucket = (
            None
            if getattr(self, "_active_product_bucket", None) is None
            else getattr(self, "_active_product_bucket", None)
        )
        attention_rows = (
            max(int(attention_bucket), int(y_tokens))
            if attention_bucket is not None
            else int(y_tokens)
        )
        attention_output = self._attention_output_scratch_for(
            owner_bucket,
            rows=attention_rows,
            n_heads=int(getattr(block.attn, "n_heads")),
            head_dim=int(getattr(block.attn, "head_dim")),
        )
        attention_postprocess_output = None
        dp_flat_context = getattr(self, "_attention_out_dp_flat_context", None)
        if (
            is_compressed
            and dp_flat_context is not None
            and owner_bucket is not None
            and attention_rows < int(owner_bucket.token_bucket)
        ):
            canonical_output = self._attention_output_for(
                layer_id,
                int(owner_bucket.token_bucket),
            )
            attention_output_alias = _alias_device_value_first_dim_slice(
                canonical_output,
                start=0,
                size=attention_rows,
            )
            if attention_output_alias is None:
                raise RuntimeError(
                    "DSV4 product compressed DP-attention requires a first-row "
                    "alias from the canonical attention output buffer; "
                    "standalone postprocess bucket variants are not part of "
                    "the product path"
                )
            attention_output = attention_output_alias
            attention_postprocess_output = canonical_output
        owner_ids_host, owner_ids_dev = self._attention_owner_buffers_for(
            owner_bucket,
            rows=attention_rows,
            primary=False,
        )
        needs_distinct_primary_owner = False
        if is_compressed and int(start_pos) == 0:
            win = getattr(block.attn, "window_size", None)
            needs_distinct_primary_owner = (
                win is None or len(y_shape) < 2 or int(y_shape[1]) > int(win)
            )
        primary_owner_ids_host, primary_owner_ids_dev = (None, None)
        if needs_distinct_primary_owner:
            primary_owner_ids_host, primary_owner_ids_dev = (
                self._attention_owner_buffers_for(
                    owner_bucket,
                    rows=attention_rows,
                    primary=True,
                )
            )
        return run_dsv4_attention(
            graph,
            block.attn,
            y,
            int(start_pos),
            backend=self.attention_backend,
            options=self.options,
            build_dir=self.build_dir,
            device_layer_state=device_layer_state,
            metadata=metadata,
            token_bucket=attention_bucket,
            attention_output=attention_output,
            attention_scratch=(
                None
                if owner_bucket is None
                else lambda kind, shape, dtype: self._bucket_scratch(
                    owner_bucket,
                    kind,
                    shape,
                    dtype,
                )
            ),
            owner_ids_host=owner_ids_host,
            owner_ids_dev=owner_ids_dev,
            primary_owner_ids_host=primary_owner_ids_host,
            primary_owner_ids_dev=primary_owner_ids_dev,
            attention_hidden_shape=y_shape,
            attention_postprocess_output=attention_postprocess_output,
            attention_profile=_attention_profile,
            layer_id=layer_id,
        )

    def _run_attention_layer(
        self,
        block: Any,
        y: Any,
        metadata: Any | None,
        *,
        layer_id: int,
        start_pos: int,
        device_layer_state: Any,
        reduce_token_bucket: int | None = None,
        dp_attention_ctx: Any | None = None,
        dp_attention_lane_metadata: Any | None = None,
        is_decode: bool = False,
    ) -> Any:
        ctx = (
            dp_attention_ctx
            if dp_attention_ctx is not None
            else self._dp_attention_lane_context(metadata)
        )
        coord = _product_executor_coord(self)
        if ctx is None:
            _product_warmup_trace(
                coord,
                f"attention replicated start layer={int(layer_id)}",
            )
            out = self._run_attention_replicated(
                block,
                y,
                metadata,
                layer_id=layer_id,
                start_pos=start_pos,
                device_layer_state=device_layer_state,
                token_bucket=reduce_token_bucket,
            )
            _product_warmup_trace(
                coord,
                f"attention replicated done layer={int(layer_id)}",
            )
            return out
        post_pre_context = getattr(self, "_dp_attention_unpad_post_pre_context", None)
        if post_pre_context is None:
            raise RuntimeError(
                "DSV4 product DP-attention requires the fused "
                "dp_attention_ar_post_pre boundary; direct standalone "
                "attention-layer DP reduce is unsupported"
            )
        ref_shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
        if len(ref_shape) != 3:
            raise RuntimeError(
                "DSV4 DP-attention reduce expects [batch, seqlen, hidden], "
                f"got {ref_shape}"
            )
        n_tokens = int(ref_shape[0] * ref_shape[1])
        token_bucket_i = (
            int(reduce_token_bucket) if reduce_token_bucket is not None else n_tokens
        )
        bucket = self._require_active_product_bucket(where="attention_layer")
        # Decode canonicalizes every runtime batch onto one ``max_requests``-wide
        # NEFF (``_product_compile_batch_size`` promotes). The fused post/pre +
        # MoE NEFF reshapes ``x[:compile_batch_size*seqlen]``, so the flat reduce
        # buffer must follow the promoted batch -- the scheduler hands us a raw
        # (often smaller) batch. No-op when ``max_requests==1``.
        compile_batch_size = self._product_compile_batch_size(
            bucket,
            bsz=int(ctx.total_batch_size),
            seqlen=int(ref_shape[1]),
        )
        reduce_rows = self._dp_attention_reduce_rows_for_step(
            token_bucket=int(token_bucket_i),
            total_tokens=int(n_tokens),
            batch_size=int(ctx.total_batch_size),
            seqlen=int(ref_shape[1]),
            is_decode=bool(is_decode),
            compile_batch_size=int(compile_batch_size),
        )
        is_compressed = bool(
            int(getattr(getattr(block, "attn", None), "compress_ratio", 0) or 0)
        )
        collective_specs = self._dp_attention_out_collective_specs_for_step(
            bucket,
            token_bucket=token_bucket_i,
            batch_size=int(ctx.total_batch_size),
            total_tokens=int(n_tokens),
            compressed=is_compressed,
            is_decode=bool(is_decode),
        )
        attention_profile_common = {
            "layer_id": int(layer_id),
            "bucket": int(token_bucket_i),
            "batch_size": int(ctx.total_batch_size),
            "lane": int(ctx.lane),
            "lane_batch_size": int(ctx.batch_size),
            "n_tokens": int(n_tokens),
            "reduce_rows": int(reduce_rows),
            "compressed": bool(is_compressed),
            "is_decode": bool(is_decode),
        }
        _product_warmup_trace(
            coord,
            "dp_attention start "
            f"layer={int(layer_id)} ctx_lane={int(ctx.lane)} "
            f"batch_start={int(ctx.batch_start)} "
            f"batch_size={int(ctx.batch_size)} "
            f"total_batch={int(ctx.total_batch_size)} "
            f"n_tokens={int(n_tokens)} reduce_rows={int(reduce_rows)} "
            f"compressed={bool(is_compressed)} specs={len(collective_specs)}",
        )
        flat_lane = None
        try:
            if int(ctx.batch_size) <= 0:
                # Empty lanes still participate in each load barrier so active
                # lanes can materialize one collective model at a time without
                # keeping every lane variant resident in HBM.
                for spec_index, spec in enumerate(collective_specs):
                    next_spec = (
                        collective_specs[spec_index + 1]
                        if spec_index + 1 < len(collective_specs)
                        else None
                    )
                    if self._is_dp_attention_out_collective_spec_materialized(
                        bucket,
                        spec,
                    ):
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize skip "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=True",
                        )
                    else:
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize start "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            f"start={int(spec.start)} size={int(spec.size)} "
                            f"bsz={int(spec.bsz)} seqlen={int(spec.seqlen)} "
                            f"rows={int(spec.rows)} "
                            f"reduce_rows={int(spec.reduce_rows)} empty_lane=True",
                        )
                        with self._profile_product_stage(
                            "attention_dp_materialize",
                            **attention_profile_common,
                            empty_lane=True,
                            spec_index=int(spec_index),
                            spec_start=int(spec.start),
                            spec_size=int(spec.size),
                        ):
                            self._materialize_dp_attention_out_collective_spec(
                                bucket,
                                spec,
                            )
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize done "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=True",
                        )
                    if (
                        not self._should_defer_transient_dp_attention_unload()
                        and not self._keep_transient_dp_attention_out_collectives_loaded()
                        and not self._dp_attention_out_specs_reuse_loaded_inverse_kernel(
                            spec,
                            next_spec,
                        )
                    ):
                        with self._profile_product_stage(
                            "attention_dp_unload",
                            **attention_profile_common,
                            empty_lane=True,
                            spec_index=int(spec_index),
                        ):
                            self._unload_transient_dp_attention_out_collectives(bucket)
                        _product_warmup_trace(
                            coord,
                            "dp_attention unload done "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=True",
                        )
                    else:
                        _product_warmup_trace(
                            coord,
                            "dp_attention unload defer "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=True",
                        )
            else:
                if int(ctx.batch_start) == 0 and int(ctx.batch_size) == int(
                    ref_shape[0]
                ):
                    lane_y = y
                elif int(ctx.batch_start) == 0:
                    with self._profile_product_stage(
                        "attention_dp_lane_slice",
                        **attention_profile_common,
                    ):
                        lane_y = self._run_product_dp_attention_lane_slice(
                            y,
                            start=int(ctx.batch_start),
                            size=int(ctx.batch_size),
                        )
                else:
                    with self._profile_product_stage(
                        "attention_dp_lane_slice",
                        **attention_profile_common,
                    ):
                        lane_y = self._run_product_dp_attention_lane_slice(
                            y,
                            start=int(ctx.batch_start),
                            size=int(ctx.batch_size),
                        )
                lane_metadata = dp_attention_lane_metadata
                if lane_metadata is None:
                    _product_warmup_trace(
                        coord,
                        f"dp_attention lane_metadata start layer={int(layer_id)}",
                    )
                    with self._profile_product_stage(
                        "attention_dp_lane_metadata",
                        **attention_profile_common,
                    ):
                        lane_metadata = self._prepare_dp_attention_lane_metadata(
                            metadata,
                            ctx,
                        )
                    _product_warmup_trace(
                        coord,
                        f"dp_attention lane_metadata done layer={int(layer_id)}",
                    )
                for spec_index, spec in enumerate(collective_specs):
                    next_spec = (
                        collective_specs[spec_index + 1]
                        if spec_index + 1 < len(collective_specs)
                        else None
                    )
                    if self._is_dp_attention_out_collective_spec_materialized(
                        bucket,
                        spec,
                    ):
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize skip "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=False",
                        )
                    else:
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize start "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            f"start={int(spec.start)} size={int(spec.size)} "
                            f"bsz={int(spec.bsz)} seqlen={int(spec.seqlen)} "
                            f"rows={int(spec.rows)} "
                            f"reduce_rows={int(spec.reduce_rows)} empty_lane=False",
                        )
                        with self._profile_product_stage(
                            "attention_dp_materialize",
                            **attention_profile_common,
                            empty_lane=False,
                            spec_index=int(spec_index),
                            spec_start=int(spec.start),
                            spec_size=int(spec.size),
                        ):
                            self._materialize_dp_attention_out_collective_spec(
                                bucket,
                                spec,
                            )
                        _product_warmup_trace(
                            coord,
                            "dp_attention materialize done "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            "empty_lane=False",
                        )
                    try:
                        if int(ctx.batch_start) != int(spec.start) or int(
                            ctx.batch_size
                        ) != int(spec.size):
                            _product_warmup_trace(
                                coord,
                                "dp_attention spec skip "
                                f"layer={int(layer_id)} spec={int(spec_index)} "
                                f"ctx_start={int(ctx.batch_start)} "
                                f"ctx_size={int(ctx.batch_size)}",
                            )
                            continue
                        previous_dp_flat = getattr(
                            self,
                            "_attention_out_dp_flat_context",
                            None,
                        )
                        compile_seqlen = self._attention_out_dp_flat_compile_seqlen(
                            bucket,
                            bsz=int(spec.bsz),
                            seqlen=int(spec.seqlen),
                            batch_size=int(spec.batch_size),
                            start=int(spec.start),
                            size=int(spec.size),
                            rows=int(spec.reduce_rows),
                            is_decode=bool(spec.is_decode),
                        )
                        self._attention_out_dp_flat_context = {
                            "batch_size": int(spec.batch_size),
                            "start": int(spec.start),
                            "size": int(spec.size),
                            "rows": int(spec.reduce_rows),
                            "compile_seqlen": int(compile_seqlen),
                            "flat_token_range": (
                                not bool(spec.is_decode)
                                and int(compile_seqlen) == int(bucket.token_bucket)
                                and int(spec.reduce_rows) >= int(bucket.token_bucket)
                            ),
                            "token_start": int(ctx.token_start),
                            "token_count": int(ctx.token_end) - int(ctx.token_start),
                        }
                        try:
                            _product_warmup_trace(
                                coord,
                                "dp_attention replicated start "
                                f"layer={int(layer_id)} spec={int(spec_index)}",
                            )
                            with self._profile_product_stage(
                                "attention_replicated",
                                **attention_profile_common,
                                spec_index=int(spec_index),
                            ):
                                lane_out = self._run_attention_replicated(
                                    block,
                                    lane_y,
                                    lane_metadata,
                                    layer_id=layer_id,
                                    start_pos=start_pos,
                                    device_layer_state=device_layer_state,
                                    token_bucket=reduce_token_bucket,
                                    profile_fields={
                                        **attention_profile_common,
                                        "spec_index": int(spec_index),
                                    },
                                )
                            _product_warmup_trace(
                                coord,
                                "dp_attention replicated done "
                                f"layer={int(layer_id)} spec={int(spec_index)}",
                            )
                        finally:
                            self._attention_out_dp_flat_context = previous_dp_flat
                        lane_out_shape = tuple(
                            int(dim) for dim in getattr(lane_out, "shape", ())
                        )
                        _product_warmup_trace(
                            coord,
                            "dp_attention replicated output "
                            f"layer={int(layer_id)} spec={int(spec_index)} "
                            f"shape={lane_out_shape}",
                        )
                        if lane_out_shape == (int(reduce_rows), int(ref_shape[2])):
                            flat_lane = lane_out
                        else:
                            raise RuntimeError(
                                "DSV4 product DP-attention requires attention "
                                "output to produce flat reduce rows directly; "
                                "standalone lane-scatter/flatten/pad kernels "
                                "are not part of the product path. "
                                f"got lane_out_shape={lane_out_shape}, expected="
                                f"{(int(reduce_rows), int(ref_shape[2]))}"
                            )
                    finally:
                        if (
                            not self._should_defer_transient_dp_attention_unload()
                            and not self._keep_transient_dp_attention_out_collectives_loaded()
                            and not self._dp_attention_out_specs_reuse_loaded_inverse_kernel(
                                spec,
                                next_spec,
                            )
                        ):
                            with self._profile_product_stage(
                                "attention_dp_unload",
                                **attention_profile_common,
                                empty_lane=False,
                                spec_index=int(spec_index),
                            ):
                                self._unload_transient_dp_attention_out_collectives(
                                    bucket
                                )
                            _product_warmup_trace(
                                coord,
                                "dp_attention unload done "
                                f"layer={int(layer_id)} spec={int(spec_index)} "
                                "empty_lane=False",
                            )
                        else:
                            _product_warmup_trace(
                                coord,
                                "dp_attention unload defer "
                                f"layer={int(layer_id)} spec={int(spec_index)} "
                                "empty_lane=False",
                            )
                if flat_lane is None:
                    raise RuntimeError(
                        "DSV4 product DP-attention did not execute a lane "
                        f"collective for batch_start={int(ctx.batch_start)}, "
                        f"batch_size={int(ctx.batch_size)}"
                    )
        finally:
            if (
                not self._should_defer_transient_dp_attention_unload()
                and not self._keep_transient_dp_attention_out_collectives_loaded()
            ):
                self._unload_transient_dp_attention_out_collectives(bucket)
        if flat_lane is None:
            with self._profile_product_stage(
                "attention_dp_flat_zero",
                **attention_profile_common,
            ):
                flat_lane = self._run_product_dp_attention_flat_zero(
                    y,
                    rows=reduce_rows,
                    hidden_size=int(ref_shape[2]),
                )
        flat_shape = tuple(int(dim) for dim in getattr(flat_lane, "shape", ()))
        _product_warmup_trace(
            coord,
            "dp_attention all_reduce_post_pre start "
            f"layer={int(layer_id)} flat_shape={flat_shape}",
        )
        with self._profile_product_stage(
            "attention_dp_all_reduce_post_pre",
            **attention_profile_common,
        ):
            out = self._run_dp_attention_all_reduce_unpad_post_pre(
                flat_lane,
                replica_groups=ctx.replica_groups,
                bsz=int(ref_shape[0]),
                seqlen=int(ref_shape[1]),
                hidden_size=int(ref_shape[2]),
                **post_pre_context,
            )
        _product_warmup_trace(
            coord,
            f"dp_attention all_reduce_post_pre done layer={int(layer_id)}",
        )
        return out

    def _run_attention_layer_post_pre(
        self,
        block: Any,
        y: Any,
        metadata: Any | None,
        *,
        layer_id: int,
        start_pos: int,
        device_layer_state: Any,
        reduce_token_bucket: int | None,
        dp_attention_ctx: Any | None,
        dp_attention_lane_metadata: Any | None,
        residual: Any,
        post: Any,
        comb: Any,
        input_ids_for_moe: Any | None = None,
        is_decode: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        ctx = (
            dp_attention_ctx
            if dp_attention_ctx is not None
            else self._dp_attention_lane_context(metadata)
        )
        if ctx is not None:
            # DP-attention must scatter/reduce one hidden tensor per lane
            # before the following mHC boundary. Fuse only after the DP
            # all-reduce when the full rectangular hidden tensor is restored.
            previous_dp = getattr(self, "_dp_attention_unpad_post_pre_context", None)
            previous_moe_dispatch = getattr(
                self,
                "_active_dp_attention_moe_dispatch_context",
                None,
            )
            y_shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
            moe_dispatch_context = None
            if input_ids_for_moe is not None and len(y_shape) == 3:
                moe_dispatch_context = self._make_dp_attention_moe_dispatch_context(
                    block,
                    input_ids=input_ids_for_moe,
                    layer_id=int(layer_id),
                    is_decode=bool(is_decode),
                    token_bucket=reduce_token_bucket,
                    bsz=int(y_shape[0]),
                    seqlen=int(y_shape[1]),
                    hidden_size=int(y_shape[2]),
                )
            self._dp_attention_unpad_post_pre_context = {
                "residual": residual,
                "post": post,
                "comb": comb,
                "hc_fn": block.hc_ffn_fn,
                "hc_scale": block.hc_ffn_scale,
                "hc_base": block.hc_ffn_base,
                "norm_weight": block.ffn_norm,
                "output_bsz": int(y_shape[0]) if len(y_shape) == 3 else None,
                "output_seqlen": int(y_shape[1]) if len(y_shape) == 3 else None,
            }
            self._active_dp_attention_moe_dispatch_context = moe_dispatch_context
            try:
                fused = self._run_attention_layer(
                    block,
                    y,
                    metadata,
                    layer_id=int(layer_id),
                    start_pos=int(start_pos),
                    device_layer_state=device_layer_state,
                    reduce_token_bucket=reduce_token_bucket,
                    dp_attention_ctx=ctx,
                    dp_attention_lane_metadata=dp_attention_lane_metadata,
                    is_decode=bool(is_decode),
                )
            finally:
                self._dp_attention_unpad_post_pre_context = previous_dp
                self._active_dp_attention_moe_dispatch_context = previous_moe_dispatch
            if isinstance(fused, tuple) and len(fused) == 4:
                return fused
            raise RuntimeError(
                "DSV4 product DP-attention must return a fused post/pre tuple; "
                "standalone mHC post/pre is not a valid product boundary"
            )
        raise RuntimeError(
            "DSV4 product attention post/pre requires the fused DP-attention "
            "all-reduce/post-pre boundary; standalone non-DP post/pre kernels "
            "are not part of the product path"
        )
