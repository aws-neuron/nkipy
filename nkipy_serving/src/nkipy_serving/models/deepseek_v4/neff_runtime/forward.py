"""Forward entrypoints for DSV4 product execution."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    product_runtime_layer_profile,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.layer_specs import (
    product_layer_graph_spec,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _product_executor_coord,
    _product_warmup_trace,
)


class Dsv4ProductForwardMixin:
    def forward_hc_stack(
        self,
        input_ids: np.ndarray,
        *,
        start_pos: int = 0,
        metadata: Any | None = None,
        token_bucket: int | None = None,
    ) -> Any:
        del input_ids, start_pos, metadata
        self._require_runtime_token_bucket(token_bucket)
        raise RuntimeError(
            "DSV4 product executor does not expose forward_hc_stack; use "
            "forward_sampled so the final MoE restore, mHC head, and logits "
            "processor stay fused"
        )

    def forward_sampled(
        self,
        input_ids: np.ndarray,
        *,
        start_pos: int = 0,
        metadata: Any | None = None,
        token_bucket: int | None = None,
        sampling_batch: Any | None = None,
    ) -> dict[str, np.ndarray]:
        runtime_token_bucket = self._require_runtime_token_bucket(token_bucket)
        if (
            self.logits_processor is None
            or self.final_norm_dev is None
            or self.lm_head_dev is None
        ):
            raise RuntimeError("DSV4 sampled head was not installed")
        ids = np.asarray(input_ids)
        profile_common: dict[str, Any] = {
            "bucket": int(runtime_token_bucket),
            "ids_shape": tuple(int(dim) for dim in ids.shape),
        }
        forward_profile_t0 = time.perf_counter()
        forward_profile_status = "ok"
        forward_profile_error = ""
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        previous = getattr(self, "_active_product_bucket", None)
        previous_mhc_post_scratch_index = getattr(
            self,
            "_mhc_post_scratch_index",
            0,
        )
        previous_mhc_post_slot1_alias = getattr(
            self,
            "_mhc_post_slot1_alias",
            None,
        )
        previous_product_alias_registry = getattr(
            self,
            "_product_active_alias_full_values",
            None,
        )
        previous_product_alias_ref_registry = getattr(
            self,
            "_product_active_alias_full_values_by_ref",
            None,
        )
        previous_defer_transient_dp_attention_unload = getattr(
            self,
            "_defer_transient_dp_attention_unload",
            None,
        )
        previous_product_layer_graph_profile_fields = getattr(
            self,
            "_active_product_layer_graph_profile_fields",
            None,
        )
        previous_deferred_dp_attention_out_materialized_keys = getattr(
            self,
            "_deferred_dp_attention_out_collective_materialized_keys",
            None,
        )
        # Collective residency is hardwired off (a documented net loss; see
        # _keep_transient_dp_attention_out_collectives_loaded). Prefill keeps a
        # fresh deferred-materialize set and unloads at teardown; decode can
        # unload at existing attention spec boundaries to lower resident NEFF
        # pressure after large prefill buckets.
        keep_dp_attention_out_collectives_loaded = (
            self._keep_transient_dp_attention_out_collectives_loaded()
        )
        materialized_keys: set = set()
        self._active_product_bucket = bucket
        self._mhc_post_scratch_index = 0
        self._mhc_post_slot1_alias = None
        self._product_active_alias_full_values = {}
        self._product_active_alias_full_values_by_ref = {}
        self._defer_transient_dp_attention_unload = True
        self._deferred_dp_attention_out_collective_materialized_keys = materialized_keys
        try:
            with self._profile_product_stage("prepare_inputs", **profile_common):
                device_state = self.device_state
                executor = self.runtime_surface
                bucket, embedding_bsz, embedding_seqlen, _embedding_input_capacity = (
                    self._prepare_embedding_input_ids(ids)
                )
                is_decode = self._is_decode_step(metadata, int(start_pos))
                self._defer_transient_dp_attention_unload = not bool(is_decode)
                _layer_compile_bsz, layer_compile_seqlen = (
                    self._product_compile_sequence_shape(
                        bucket,
                        bsz=int(embedding_bsz),
                        seqlen=int(embedding_seqlen),
                        bucket_single_token=not bool(is_decode),
                    )
                )
                # Used only for warmup-layer signature bucketing before layer 0
                # materializes the real active embedding through the fused path.
                h = self._embedding_full_spec_for_bucket(bucket)
                dp_attention_ctx = self._dp_attention_lane_context(metadata)
                dp_attention_lane_metadata = self._prepare_dp_attention_lane_metadata(
                    metadata,
                    dp_attention_ctx,
                )
                base = self._base_metadata(metadata)
                batch_size = (
                    int(getattr(base, "batch_size"))
                    if base is not None and hasattr(base, "batch_size")
                    else self._batch_size_from_input(ids)
                )
            profile_common.update(
                {
                    "batch_size": int(batch_size),
                    "is_decode": bool(is_decode),
                    "dp_superstep": getattr(metadata, "dp_superstep", None) is not None,
                }
            )
            needs_logprobs = (
                bool(sampling_batch.needs_logprobs) if sampling_batch else False
            )
            use_full_sampler = (
                sampling_batch is not None
                and bool(getattr(sampling_batch, "enabled", False))
            ) or bool(needs_logprobs)
            use_fused_head_top1 = self._can_use_fused_head_top1() and not bool(
                use_full_sampler
            )
            coord = _product_executor_coord(self)
            _product_warmup_trace(
                coord,
                "forward_sampled start "
                f"bucket={int(runtime_token_bucket)} "
                f"ids_shape={tuple(int(dim) for dim in ids.shape)} "
                f"batch={int(batch_size)} "
                f"is_decode={bool(is_decode)} "
                f"dp_superstep={getattr(metadata, 'dp_superstep', None) is not None}",
            )
            last_indices = None
            with self._profile_product_stage("last_indices", **profile_common):
                last_indices = self._last_token_indices_dev_for(
                    ids,
                    metadata=metadata,
                    batch_size=batch_size,
                )
            hidden = None
            fused_top1_output = None
            pending_attn_pre = None
            num_layers = len(executor.blocks)
            for layer_id, block in enumerate(executor.blocks):
                is_last_layer = int(layer_id) == int(num_layers) - 1
                input_boundary = (
                    "embedding_mhc_pre"
                    if pending_attn_pre is None
                    else "previous_shared_restore_post_pre"
                )
                if is_last_layer:
                    output_boundary = (
                        "sampled_head_top1"
                        if use_fused_head_top1
                        else "sampled_head_hidden"
                    )
                else:
                    output_boundary = "shared_restore_post_pre"
                layer_graph = product_layer_graph_spec(
                    block,
                    layer_id=int(layer_id),
                    num_layers=int(num_layers),
                    token_bucket=int(runtime_token_bucket),
                    ids_shape=tuple(int(dim) for dim in ids.shape),
                    batch_size=int(batch_size),
                    seqlen=int(layer_compile_seqlen),
                    is_decode=bool(is_decode),
                    has_dp_attention=dp_attention_ctx is not None,
                    has_dp_superstep=getattr(metadata, "dp_superstep", None)
                    is not None,
                    input_boundary=input_boundary,
                    output_boundary=output_boundary,
                    blockwise_moe_enabled=getattr(self, "blockwise_moe_state", None)
                    is not None,
                )
                layer_profile_common = {
                    **profile_common,
                    **layer_graph.profile_fields(),
                }
                layer_profile_fields = layer_graph.profile_fields()
                self._active_product_layer_graph_profile_fields = layer_profile_fields
                with product_runtime_layer_profile(layer_profile_fields):
                    _product_warmup_trace(
                        coord,
                        f"layer start layer={int(layer_id)} "
                        f"last={bool(is_last_layer)} "
                        f"pending_attn_pre={pending_attn_pre is not None} "
                        f"variant={layer_graph.variant_name}",
                    )
                    if (
                        (not is_last_layer)
                        and pending_attn_pre is not None
                        and self._should_skip_hc_layer_for_warmup(
                            layer_id=int(layer_id),
                            block=block,
                            h=h,
                            metadata=metadata,
                            start_pos=int(start_pos),
                            token_bucket=runtime_token_bucket,
                            is_decode=bool(is_decode),
                        )
                    ):
                        # Warmup dedup skips compiling/running duplicate layers, but
                        # the product path still needs a legal fused handoff for the
                        # next non-skipped layer. Reuse the previous pending tuple;
                        # warmup values are synthetic and only shapes/signatures
                        # matter here.
                        _product_warmup_trace(
                            coord,
                            f"layer skip layer={int(layer_id)}",
                        )
                        continue
                    device_layer_state = device_state.layer(layer_id)
                    if pending_attn_pre is None:
                        if int(layer_id) == 0:
                            _product_warmup_trace(
                                coord,
                                "embedding_mhc_pre start "
                                f"layer={int(layer_id)} bsz={int(embedding_bsz)} "
                                f"seqlen={int(embedding_seqlen)}",
                            )
                            with self._profile_product_stage(
                                "embedding_mhc_pre",
                                **profile_common,
                                layer_id=int(layer_id),
                                bsz=int(embedding_bsz),
                                seqlen=int(embedding_seqlen),
                            ):
                                h, y, post, comb = (
                                    self._run_product_embedding_mhc_pre_from_ids(
                                        bucket,
                                        block.hc_attn_fn,
                                        block.hc_attn_scale,
                                        block.hc_attn_base,
                                        block.attn_norm,
                                        bsz=int(embedding_bsz),
                                        seqlen=int(embedding_seqlen),
                                        is_decode=bool(is_decode),
                                    )
                                )
                            _product_warmup_trace(
                                coord,
                                f"embedding_mhc_pre done layer={int(layer_id)}",
                            )
                            self._mhc_post_slot1_alias = h
                        else:
                            raise RuntimeError(
                                "DSV4 product non-first layer requires the previous "
                                "MoE to return a fused shared-restore/post-pre tuple; "
                                "standalone mHC pre kernels are not part of the "
                                "product path"
                            )
                        attn_residual = h
                    else:
                        h, y, post, comb = pending_attn_pre
                        pending_attn_pre = None
                        attn_residual = h
                    _product_warmup_trace(
                        coord,
                        f"attention start layer={int(layer_id)}",
                    )
                    with self._profile_product_stage(
                        "attention",
                        **layer_profile_common,
                        layer_id=int(layer_id),
                        final=bool(is_last_layer),
                    ):
                        h, y, post, comb = self._run_attention_layer_post_pre(
                            block,
                            y,
                            metadata,
                            layer_id=layer_id,
                            start_pos=int(start_pos),
                            device_layer_state=device_layer_state,
                            reduce_token_bucket=runtime_token_bucket,
                            dp_attention_ctx=dp_attention_ctx,
                            dp_attention_lane_metadata=dp_attention_lane_metadata,
                            residual=attn_residual,
                            post=post,
                            comb=comb,
                            input_ids_for_moe=bucket.input_ids_dev,
                            is_decode=bool(is_decode),
                        )
                    _product_warmup_trace(
                        coord,
                        f"attention done layer={int(layer_id)}",
                    )
                    if is_last_layer:
                        previous_moe_head = getattr(
                            self,
                            "_shared_expert_restore_head_context",
                            None,
                        )
                        head_context = {
                            "residual": h,
                            "post": post,
                            "comb": comb,
                            "head": executor.head,
                            "last_token_indices": last_indices,
                            "token_bucket": runtime_token_bucket,
                            "is_decode": bool(is_decode),
                        }
                        if use_fused_head_top1:
                            head_context.update(
                                {
                                    "final_norm": self.final_norm_dev,
                                    "lm_head": self.lm_head_dev,
                                    "last_token_indices": last_indices,
                                    "top1_values": bucket.head_top1_values,
                                    "top1_indices": bucket.head_top1_indices,
                                }
                            )
                        self._shared_expert_restore_head_context = head_context
                        try:
                            _product_warmup_trace(
                                coord,
                                f"moe start layer={int(layer_id)} final=True",
                            )
                            with self._profile_product_stage(
                                "moe",
                                **layer_profile_common,
                                layer_id=int(layer_id),
                                final=True,
                            ):
                                moe_result = self._run_moe_layer(
                                    block,
                                    y,
                                    ids,
                                    layer_id=int(layer_id),
                                    is_decode=bool(is_decode),
                                    token_bucket=runtime_token_bucket,
                                )
                            _product_warmup_trace(
                                coord,
                                f"moe done layer={int(layer_id)} final=True",
                            )
                        finally:
                            self._shared_expert_restore_head_context = previous_moe_head
                        if use_fused_head_top1:
                            if (
                                isinstance(moe_result, dict)
                                and "top1_values" in moe_result
                                and "top1_indices" in moe_result
                            ):
                                fused_top1_output = moe_result
                            else:
                                raise RuntimeError(
                                    "DSV4 product sampled final MoE must return "
                                    "fused sampled-head top1 output"
                                )
                        elif moe_result is bucket.head_hidden_output:
                            hidden = moe_result
                        else:
                            raise RuntimeError(
                                "DSV4 product sampled final MoE must return "
                                "the fused sampled-head output"
                            )
                    else:
                        next_block = executor.blocks[int(layer_id) + 1]
                        previous_moe_post_pre = getattr(
                            self,
                            "_shared_expert_restore_post_pre_context",
                            None,
                        )
                        self._shared_expert_restore_post_pre_context = {
                            "residual": h,
                            "post": post,
                            "comb": comb,
                            "hc_fn": next_block.hc_attn_fn,
                            "hc_scale": next_block.hc_attn_scale,
                            "hc_base": next_block.hc_attn_base,
                            "norm_weight": next_block.attn_norm,
                            "is_decode": bool(is_decode),
                        }
                        try:
                            _product_warmup_trace(
                                coord,
                                f"moe start layer={int(layer_id)} final=False",
                            )
                            with self._profile_product_stage(
                                "moe",
                                **layer_profile_common,
                                layer_id=int(layer_id),
                                final=False,
                            ):
                                moe_result = self._run_moe_layer(
                                    block,
                                    y,
                                    ids,
                                    layer_id=int(layer_id),
                                    is_decode=bool(is_decode),
                                    token_bucket=runtime_token_bucket,
                                )
                            _product_warmup_trace(
                                coord,
                                f"moe done layer={int(layer_id)} final=False",
                            )
                        finally:
                            self._shared_expert_restore_post_pre_context = (
                                previous_moe_post_pre
                            )
                        if isinstance(moe_result, tuple) and len(moe_result) == 4:
                            h, y, post, comb = moe_result
                        else:
                            raise RuntimeError(
                                "DSV4 product sampled non-final MoE must return "
                                "the fused post/pre tuple"
                            )
                        pending_attn_pre = (h, y, post, comb)
            if previous_product_layer_graph_profile_fields is None:
                if hasattr(self, "_active_product_layer_graph_profile_fields"):
                    delattr(self, "_active_product_layer_graph_profile_fields")
            else:
                self._active_product_layer_graph_profile_fields = (
                    previous_product_layer_graph_profile_fields
                )
            if fused_top1_output is not None:
                _product_warmup_trace(
                    coord,
                    "forward_sampled done fused_top1=True",
                )
                with self._profile_product_stage("output_copy_top1", **profile_common):
                    top1_values = (
                        fused_top1_output["top1_values"]
                        .numpy()[:batch_size]
                        .astype(np.float32, copy=False)
                    )
                    top1_indices = (
                        fused_top1_output["top1_indices"]
                        .numpy()[:batch_size]
                        .astype(np.int32, copy=False)
                    )
                return {
                    "top1_values": top1_values,
                    "top1_indices": top1_indices,
                    "vocab_offset": np.asarray(
                        [int(executor.v4.lm_head_vocab_offset)],
                        dtype=np.int32,
                    ),
                }
            if hidden is None:
                raise RuntimeError(
                    "DSV4 product sampled forward requires at least one "
                    "transformer block for fused final head"
                )
            with self._profile_product_stage("last_indices", **profile_common):
                last_indices = self._compact_last_token_indices_dev_for(
                    batch_size=batch_size,
                )
            with self._profile_product_stage("logits_processor", **profile_common):
                lp_output = self.logits_processor.forward(
                    hidden,
                    self.final_norm_dev,
                    self.lm_head_dev,
                    last_indices,
                    batch_size=batch_size,
                    token_bucket=runtime_token_bucket,
                    sampling_batch=sampling_batch,
                    needs_logprobs=bool(needs_logprobs),
                    logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
                )
            _product_warmup_trace(
                coord,
                "forward_sampled done fused_top1=False",
            )
            with self._profile_product_stage("output_to_shm_dict", **profile_common):
                return lp_output.to_shm_dict(
                    vocab_offset=int(executor.v4.lm_head_vocab_offset),
                )
        except Exception as exc:
            forward_profile_status = "error"
            forward_profile_error = repr(exc)
            raise
        finally:
            if previous_product_layer_graph_profile_fields is None:
                if hasattr(self, "_active_product_layer_graph_profile_fields"):
                    delattr(self, "_active_product_layer_graph_profile_fields")
            else:
                self._active_product_layer_graph_profile_fields = (
                    previous_product_layer_graph_profile_fields
                )
            self._write_product_stage_profile(
                "forward_total",
                time.perf_counter() - forward_profile_t0,
                status=forward_profile_status,
                error=forward_profile_error,
                **profile_common,
            )
            with self._profile_product_stage(
                "attention_dp_unload_deferred",
                **profile_common,
                keep_resident=bool(keep_dp_attention_out_collectives_loaded),
            ):
                self._unload_transient_dp_attention_out_collectives(bucket)
            self._active_product_bucket = previous
            self._mhc_post_scratch_index = previous_mhc_post_scratch_index
            self._mhc_post_slot1_alias = previous_mhc_post_slot1_alias
            if previous_defer_transient_dp_attention_unload is None:
                if hasattr(self, "_defer_transient_dp_attention_unload"):
                    delattr(self, "_defer_transient_dp_attention_unload")
            else:
                self._defer_transient_dp_attention_unload = (
                    previous_defer_transient_dp_attention_unload
                )
            if previous_deferred_dp_attention_out_materialized_keys is None:
                if hasattr(
                    self,
                    "_deferred_dp_attention_out_collective_materialized_keys",
                ):
                    delattr(
                        self,
                        "_deferred_dp_attention_out_collective_materialized_keys",
                    )
            else:
                self._deferred_dp_attention_out_collective_materialized_keys = (
                    previous_deferred_dp_attention_out_materialized_keys
                )
            if previous_product_alias_registry is None:
                if hasattr(self, "_product_active_alias_full_values"):
                    delattr(self, "_product_active_alias_full_values")
            else:
                self._product_active_alias_full_values = previous_product_alias_registry
            if previous_product_alias_ref_registry is None:
                if hasattr(self, "_product_active_alias_full_values_by_ref"):
                    delattr(self, "_product_active_alias_full_values_by_ref")
            else:
                self._product_active_alias_full_values_by_ref = (
                    previous_product_alias_ref_registry
                )
