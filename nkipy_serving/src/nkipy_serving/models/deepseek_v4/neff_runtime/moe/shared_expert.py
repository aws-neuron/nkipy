"""Shared-expert product kernels for DSV4 product execution."""

from __future__ import annotations

import hashlib
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _compile_product_kernel,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _blockwise_moe_ep_tp_groups,
    _require_product_device_value,
    _sample_array,
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _product_canonical_neff_cache_key,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _TensorSpec,
)
from nkipy_serving.runtime.device_tensor import (
    normalize_dtype as _normalize_dtype,
)


class Dsv4ProductSharedExpertMixin:
    def precompile_shared_expert_restore_post_pre_helpers(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile non-final MoE shared restore fused into the next mHC pre."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product shared restore precompile could not infer hidden size"
            )
        args = getattr(self.runtime_surface, "args", None)
        hc_mult = int(getattr(args, "hc_mult", 0) or 0)
        if hc_mult <= 0:
            raise RuntimeError(
                "DSV4 product shared restore precompile requires "
                "runtime_surface.args.hc_mult"
            )
        compile_shape = self._product_compile_sequence_shape(
            bucket,
            bsz=bsz,
            seqlen=seq,
            bucket_single_token=not bool(is_decode),
        )
        compile_shape_candidates = [compile_shape]
        real_shape = (int(bsz), int(seq))
        # Decode lanes and multi-request prefill lanes (bsz>1 cannot promote
        # to the bucket rectangle) serve at the REAL shape.
        if real_shape not in compile_shape_candidates and (
            bool(is_decode) or int(bsz) > 1
        ):
            compile_shape_candidates.append(real_shape)
        bf16_dtype = np.dtype(ml_dtypes.bfloat16)
        f32_dtype = np.dtype(np.float32)
        moe_outputs = (
            tuple(bucket.moe_decode_outputs)
            if bool(is_decode)
            else tuple(bucket.moe_prefill_outputs)
        )
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()))
        if len(blocks) <= 1:
            return
        for compile_bsz, compile_seq in compile_shape_candidates:
            residual = _TensorSpec(
                (compile_bsz, compile_seq, hc_mult, hidden_size), bf16_dtype
            )
            post = _TensorSpec((compile_bsz, compile_seq, hc_mult), f32_dtype)
            comb = _TensorSpec((compile_bsz, compile_seq, hc_mult, hc_mult), f32_dtype)
            for layer_id, block_obj in enumerate(blocks[:-1]):
                ffn = getattr(block_obj, "ffn", None)
                shared = getattr(ffn, "shared", None)
                next_block = blocks[int(layer_id) + 1]
                required = (
                    getattr(shared, "w1", None),
                    getattr(shared, "w3", None),
                    getattr(shared, "w2", None),
                    getattr(next_block, "hc_attn_fn", None),
                    getattr(next_block, "hc_attn_scale", None),
                    getattr(next_block, "hc_attn_base", None),
                    getattr(next_block, "attn_norm", None),
                )
                if shared is None or any(value is None for value in required):
                    continue
                # Serve hands a block-rounded routed buffer covering n_tokens;
                # the per-layer arena entry can be smaller (lane-sized) or
                # missing, so always compile the rounded variant and the arena
                # variant when it differs.
                compile_tokens = int(compile_bsz) * int(compile_seq)
                rounded_rows = ((max(compile_tokens, 1) + 127) // 128) * 128
                acc_candidates = [_TensorSpec((rounded_rows, hidden_size), bf16_dtype)]
                # Decode shared-hidden buffers can be token-bucket-rows wide
                # (serve x_for_shared at the bucket) — cover that variant too.
                bucket_rows = ((int(bucket.token_bucket) + 127) // 128) * 128
                if bucket_rows != rounded_rows:
                    acc_candidates.append(
                        _TensorSpec((bucket_rows, hidden_size), bf16_dtype)
                    )
                if int(layer_id) < len(moe_outputs):
                    arena_acc = moe_outputs[int(layer_id)]
                    arena_shape = tuple(
                        int(dim) for dim in getattr(arena_acc, "shape", ())
                    )
                    # acc rows must cover n_tokens (the fragment slices
                    # routed[:n_tokens]); smaller arena entries are lane
                    # buffers never served at this compile shape.
                    if (
                        len(arena_shape) == 2
                        and int(arena_shape[0]) >= compile_tokens
                        and int(arena_shape[0]) != rounded_rows
                    ):
                        acc_candidates.append(arena_acc)
                acc = acc_candidates[0]
                acc_rows = int(rounded_rows)
                x = _TensorSpec((1, acc_rows, hidden_size), bf16_dtype)
                tp_degree = 1
                tp_replica_groups: tuple[tuple[int, ...], ...] = ()
                if bool(getattr(shared, "tp_sharded", False)):
                    state = getattr(self, "blockwise_moe_state", None)
                    tp_degree = int(getattr(state, "tp_degree", 1) or 1)
                    tp_replica_groups = tuple(
                        tuple(int(rank) for rank in group)
                        for group in tuple(
                            getattr(state, "tp_replica_groups", ()) or ()
                        )
                    )
                moe_replica_groups = _blockwise_moe_ep_tp_groups(
                    getattr(self, "blockwise_moe_state", None)
                )
                for acc in acc_candidates:
                    acc_rows = int(getattr(acc, "shape", (acc_rows,))[0])
                    x = _TensorSpec((1, acc_rows, hidden_size), bf16_dtype)
                    kernel = self._shared_expert_add_restore_post_pre_kernel_for(
                        bucket,
                        acc,
                        x,
                        shared.w1,
                        shared.w3,
                        shared.w2,
                        residual,
                        post,
                        comb,
                        next_block.hc_attn_fn,
                        next_block.hc_attn_scale,
                        next_block.hc_attn_base,
                        next_block.attn_norm,
                        limit=float(getattr(shared, "swiglu_limit", 0.0)),
                        bsz=compile_bsz,
                        seqlen=compile_seq,
                        hidden_size=hidden_size,
                        tp_degree=tp_degree,
                        tp_replica_groups=tp_replica_groups,
                        moe_replica_groups=moe_replica_groups,
                        hc_mult=hc_mult,
                        sinkhorn_iters=int(getattr(args, "hc_sinkhorn_iters")),
                        norm_eps=float(getattr(args, "norm_eps")),
                        hc_eps=float(getattr(args, "hc_eps")),
                    )
                    if self._keep_dp_attention_pipeline_collectives_loaded():
                        self._load_resident_product_kernel(kernel)

    def precompile_lane_head_helpers(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Sampled head is compiled through the fused MoE-restore/head path."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()))
        if not blocks:
            return
        final_ffn = getattr(blocks[-1], "ffn", None)
        shared = getattr(final_ffn, "shared", None)
        head = getattr(self.runtime_surface, "head", None)
        if shared is None or head is None:
            return
        required = (
            getattr(shared, "w1", None),
            getattr(shared, "w3", None),
            getattr(shared, "w2", None),
            getattr(head, "hc_head_fn", None),
            getattr(head, "hc_head_scale", None),
            getattr(head, "hc_head_base", None),
        )
        if any(value is None for value in required):
            return
        hidden_shape = tuple(
            int(dim) for dim in getattr(bucket.head_hidden_output, "shape", ())
        )
        hidden_size = int(hidden_shape[-1]) if hidden_shape else 0
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product lane head precompile could not infer hidden size"
            )
        hc_mult = int(
            getattr(getattr(self.runtime_surface, "args", None), "hc_mult", 0) or 0
        )
        if hc_mult <= 0:
            raise RuntimeError(
                "DSV4 product lane head precompile requires "
                "runtime_surface.args.hc_mult"
            )
        n_tokens = bsz * seq
        if n_tokens > int(runtime_token_bucket):
            raise RuntimeError(
                "DSV4 product lane head precompile token count exceeds bucket: "
                f"tokens={n_tokens}, token_bucket={int(runtime_token_bucket)}"
            )
        compile_bsz, compile_seq = self._product_compile_sequence_shape(
            bucket,
            bsz=bsz,
            seqlen=seq,
            bucket_single_token=not bool(is_decode),
        )
        compile_shapes: list[tuple[int, int, int]] = []
        seen_compile_shapes: set[tuple[int, int]] = set()
        candidate_shape_inputs = [(int(compile_bsz), int(compile_seq))]
        real_shape = (int(bsz), int(seq))
        if real_shape not in candidate_shape_inputs and (
            bool(is_decode) or int(bsz) > 1
        ):
            candidate_shape_inputs.append(real_shape)
        for candidate_bsz, candidate_seq in candidate_shape_inputs:
            candidate_tokens = int(candidate_bsz) * int(candidate_seq)
            candidate = (int(candidate_bsz), int(candidate_seq))
            if (
                candidate_bsz <= 0
                or candidate_seq <= 0
                or candidate_tokens <= 0
                or candidate_tokens > int(runtime_token_bucket)
                or candidate in seen_compile_shapes
            ):
                continue
            compile_shapes.append(
                (int(candidate_bsz), int(candidate_seq), int(candidate_tokens))
            )
            seen_compile_shapes.add(candidate)
        bf16_dtype = np.dtype(ml_dtypes.bfloat16)
        f32_dtype = np.dtype(np.float32)
        tp_degree = 1
        tp_replica_groups: tuple[tuple[int, ...], ...] = ()
        if bool(getattr(shared, "tp_sharded", False)):
            state = getattr(self, "blockwise_moe_state", None)
            tp_degree = int(getattr(state, "tp_degree", 1) or 1)
            tp_replica_groups = tuple(
                tuple(int(rank) for rank in group)
                for group in tuple(getattr(state, "tp_replica_groups", ()) or ())
            )
        moe_replica_groups = _blockwise_moe_ep_tp_groups(
            getattr(self, "blockwise_moe_state", None)
        )
        acc_specs: list[_TensorSpec] = []
        seen_acc_shapes: set[tuple[tuple[int, ...], str]] = set()
        moe_candidates = (
            tuple(bucket.moe_decode_outputs)
            if bool(is_decode)
            else tuple(bucket.moe_prefill_outputs)
        )
        for compile_bsz_i, compile_seq_i, compile_n_tokens_i in compile_shapes:
            residual = _TensorSpec(
                (compile_bsz_i, compile_seq_i, hc_mult, hidden_size), bf16_dtype
            )
            post = _TensorSpec((compile_bsz_i, compile_seq_i, hc_mult), f32_dtype)
            comb = _TensorSpec(
                (compile_bsz_i, compile_seq_i, hc_mult, hc_mult), f32_dtype
            )
            acc_specs.clear()
            seen_acc_shapes.clear()
            for acc in moe_candidates:
                shape = tuple(int(dim) for dim in getattr(acc, "shape", ()))
                if len(shape) != 2 or int(shape[0]) < int(compile_n_tokens_i):
                    continue
                dtype = _normalize_dtype(getattr(acc, "dtype", bf16_dtype), bf16_dtype)
                sig = (shape, str(dtype))
                if sig in seen_acc_shapes:
                    continue
                seen_acc_shapes.add(sig)
                acc_specs.append(_TensorSpec(shape, dtype))
            # Serve hands a block-rounded routed buffer covering the compile
            # tokens; cover it even when no arena candidate matches.
            rounded_rows = ((max(int(compile_n_tokens_i), 1) + 127) // 128) * 128
            bucket_rows = ((int(runtime_token_bucket) + 127) // 128) * 128
            for extra_rows in (rounded_rows, bucket_rows):
                sig = ((int(extra_rows), hidden_size), str(bf16_dtype))
                if sig not in seen_acc_shapes:
                    seen_acc_shapes.add(sig)
                    acc_specs.append(
                        _TensorSpec((int(extra_rows), hidden_size), bf16_dtype)
                    )
            for acc in acc_specs:
                acc_rows = int(acc.shape[0])
                x = _TensorSpec((1, acc_rows, hidden_size), bf16_dtype)
                if self._can_use_fused_head_top1():
                    head_top1_kernel = (
                        self._shared_expert_add_restore_head_top1_kernel_for(
                            bucket,
                            acc,
                            x,
                            shared.w1,
                            shared.w3,
                            shared.w2,
                            residual,
                            post,
                            comb,
                            head,
                            self.final_norm_dev,
                            self.lm_head_dev,
                            bucket.last_token_indices_dev,
                            limit=float(getattr(shared, "swiglu_limit", 0.0)),
                            bsz=compile_bsz_i,
                            seqlen=compile_seq_i,
                            hidden_size=hidden_size,
                            tp_degree=tp_degree,
                            tp_replica_groups=tp_replica_groups,
                            moe_replica_groups=moe_replica_groups,
                            n_tokens=compile_n_tokens_i,
                            rows=int(runtime_token_bucket),
                            lm_norm_eps=float(
                                getattr(self.logits_processor, "_rms_norm_eps", 1e-6)
                            ),
                        )
                    )
                    if self._keep_dp_attention_pipeline_collectives_loaded():
                        self._load_resident_product_kernel(head_top1_kernel)
                head_select_kernel = (
                    self._shared_expert_add_restore_head_select_kernel_for(
                        bucket,
                        acc,
                        x,
                        shared.w1,
                        shared.w3,
                        shared.w2,
                        residual,
                        post,
                        comb,
                        head,
                        bucket.last_token_indices_dev,
                        limit=float(getattr(shared, "swiglu_limit", 0.0)),
                        bsz=compile_bsz_i,
                        seqlen=compile_seq_i,
                        hidden_size=hidden_size,
                        tp_degree=tp_degree,
                        tp_replica_groups=tp_replica_groups,
                        moe_replica_groups=moe_replica_groups,
                        n_tokens=compile_n_tokens_i,
                        rows=int(runtime_token_bucket),
                    )
                )
                if self._keep_dp_attention_pipeline_collectives_loaded():
                    self._load_resident_product_kernel(head_select_kernel)

    def _shared_expert_add_restore_post_pre_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        *,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple,
        moe_replica_groups: tuple,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> Any:
        acc_shape = tuple(int(dim) for dim in getattr(acc, "shape", ()))
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        residual_shape = tuple(int(dim) for dim in getattr(residual, "shape", ()))
        x_dtype = str(getattr(x, "dtype", "unknown")).replace(".", "_")
        acc_dtype = str(getattr(acc, "dtype", "unknown")).replace(".", "_")
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        moe_groups = tuple(
            tuple(int(rank) for rank in group) for group in moe_replica_groups
        )
        key = (
            acc_shape,
            acc_dtype,
            x_shape,
            x_dtype,
            tuple(int(dim) for dim in getattr(w_gate, "shape", ())),
            str(getattr(w_gate, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_up, "shape", ())),
            str(getattr(w_up, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_down, "shape", ())),
            str(getattr(w_down, "dtype", "unknown")).replace(".", "_"),
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
            float(limit),
            (int(bsz), int(seqlen), int(hidden_size)),
            int(tp_degree),
            groups,
            moe_groups,
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
        )
        cached = bucket.kernel_caches["shared_expert_add_restore_post_pre_kernels"].get(
            key
        )
        if cached is not None:
            return cached
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_shared_restore_post_pre_"
            f"t{int(bucket.token_bucket)}_{'x'.join(str(v) for v in x_shape)}_"
            f"acc{'x'.join(str(v) for v in acc_shape)}_"
            f"res{'x'.join(str(v) for v in residual_shape)}_{x_dtype}_"
            f"tp{int(tp_degree)}_{group_tag}"
        )
        if moe_groups:
            moe_group_tag = hashlib.sha1(repr(moe_groups).encode("utf-8")).hexdigest()[
                :8
            ]
            name = f"{name}_moe{moe_group_tag}"
        compile_kwargs: dict[str, Any] = {
            "limit": float(limit),
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "tp_degree": int(tp_degree),
            "tp_replica_groups": groups,
            "moe_replica_groups": moe_groups,
            "hc_mult": int(hc_mult),
            "sinkhorn_iters": int(sinkhorn_iters),
            "norm_eps": float(norm_eps),
            "hc_eps": float(hc_eps),
        }
        needs_collective = int(tp_degree) > 1 or bool(moe_groups)
        if needs_collective:
            state = getattr(self, "blockwise_moe_state", None)
            rank_id = getattr(state, "collective_rank", None)
            world_size = getattr(state, "collective_world_size", None)
            if rank_id is None or world_size is None:
                raise RuntimeError(
                    "DSV4 product shared-restore post/pre requires "
                    "blockwise_moe_state collective_rank/collective_world_size"
                )
            compile_kwargs.update(
                {
                    "cc_enabled": True,
                    "rank_id": int(rank_id),
                    "world_size": int(world_size),
                    "is_spmd": False,
                    "load_barrier_name": name,
                    "canonical_neff_cache_key": (
                        _product_canonical_neff_cache_key(
                            "dsv4_product_shared_restore_post_pre",
                            "v1",
                            key,
                        )
                    ),
                }
            )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="shared_expert_add_restore_post_pre_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_moe.shared_expert_add_restore_mhc_post_pre_fn,
                _sample_array(acc, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_gate, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_up, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_down, fallback_dtype=ml_dtypes.bfloat16),
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
                load=False if needs_collective else True,
                **compile_kwargs,
            ),
        )

    def _shared_expert_add_restore_head_select_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        residual: Any,
        post: Any,
        comb: Any,
        head: Any,
        last_token_indices: Any,
        *,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple,
        moe_replica_groups: tuple,
        n_tokens: int,
        rows: int,
    ) -> Any:
        acc_shape = tuple(int(dim) for dim in getattr(acc, "shape", ()))
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        residual_shape = tuple(int(dim) for dim in getattr(residual, "shape", ()))
        last_idx_shape = tuple(
            int(dim) for dim in getattr(last_token_indices, "shape", ())
        )
        x_dtype = str(getattr(x, "dtype", "unknown")).replace(".", "_")
        acc_dtype = str(getattr(acc, "dtype", "unknown")).replace(".", "_")
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        moe_groups = tuple(
            tuple(int(rank) for rank in group) for group in moe_replica_groups
        )
        key = (
            acc_shape,
            acc_dtype,
            x_shape,
            x_dtype,
            tuple(int(dim) for dim in getattr(w_gate, "shape", ())),
            str(getattr(w_gate, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_up, "shape", ())),
            str(getattr(w_up, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_down, "shape", ())),
            str(getattr(w_down, "dtype", "unknown")).replace(".", "_"),
            residual_shape,
            str(getattr(residual, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(post, "shape", ())),
            str(getattr(post, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(comb, "shape", ())),
            str(getattr(comb, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(head.hc_head_fn, "shape", ())),
            tuple(int(dim) for dim in getattr(head.hc_head_scale, "shape", ())),
            tuple(int(dim) for dim in getattr(head.hc_head_base, "shape", ())),
            last_idx_shape,
            str(getattr(last_token_indices, "dtype", "unknown")).replace(".", "_"),
            float(limit),
            (int(bsz), int(seqlen), int(hidden_size)),
            int(tp_degree),
            groups,
            moe_groups,
            float(head.norm_eps),
            float(head.hc_eps),
            int(n_tokens),
            int(rows),
        )
        cached = bucket.kernel_caches[
            "shared_expert_add_restore_head_select_kernels"
        ].get(key)
        if cached is not None:
            return cached
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_shared_restore_head_select_"
            f"t{int(bucket.token_bucket)}_{'x'.join(str(v) for v in x_shape)}_"
            f"acc{'x'.join(str(v) for v in acc_shape)}_"
            f"res{'x'.join(str(v) for v in residual_shape)}_{x_dtype}_"
            f"tp{int(tp_degree)}_{group_tag}"
        )
        if moe_groups:
            moe_group_tag = hashlib.sha1(repr(moe_groups).encode("utf-8")).hexdigest()[
                :8
            ]
            name = f"{name}_moe{moe_group_tag}"
        compile_kwargs: dict[str, Any] = {
            "limit": float(limit),
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "tp_degree": int(tp_degree),
            "tp_replica_groups": groups,
            "moe_replica_groups": moe_groups,
            "norm_eps": float(head.norm_eps),
            "hc_eps": float(head.hc_eps),
            "n_tokens": int(n_tokens),
            "rows": int(rows),
        }
        needs_collective = int(tp_degree) > 1 or bool(moe_groups)
        if needs_collective:
            state = getattr(self, "blockwise_moe_state", None)
            rank_id = getattr(state, "collective_rank", None)
            world_size = getattr(state, "collective_world_size", None)
            if rank_id is None or world_size is None:
                raise RuntimeError(
                    "DSV4 product shared-restore sampled head-select requires "
                    "blockwise_moe_state collective_rank/collective_world_size"
                )
            compile_kwargs.update(
                {
                    "cc_enabled": True,
                    "rank_id": int(rank_id),
                    "world_size": int(world_size),
                    "is_spmd": False,
                    "load_barrier_name": name,
                    "canonical_neff_cache_key": (
                        _product_canonical_neff_cache_key(
                            "dsv4_product_shared_restore_head_select",
                            "v1",
                            key,
                        )
                    ),
                }
            )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="shared_expert_add_restore_head_select_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_moe.shared_expert_add_restore_mhc_post_head_select_pad_fn,
                _sample_array(acc, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_gate, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_up, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_down, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(post, fallback_dtype=np.float32),
                _sample_array(comb, fallback_dtype=np.float32),
                _sample_array(head.hc_head_fn, fallback_dtype=np.float32),
                _sample_array(head.hc_head_scale, fallback_dtype=np.float32),
                _sample_array(head.hc_head_base, fallback_dtype=np.float32),
                _sample_array(last_token_indices, fallback_dtype=np.int32),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if needs_collective else True,
                **compile_kwargs,
            ),
        )

    def _shared_expert_add_restore_head_top1_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        residual: Any,
        post: Any,
        comb: Any,
        head: Any,
        final_norm: Any,
        lm_head: Any,
        last_token_indices: Any,
        *,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple,
        moe_replica_groups: tuple,
        n_tokens: int,
        rows: int,
        lm_norm_eps: float,
    ) -> Any:
        acc_shape = tuple(int(dim) for dim in getattr(acc, "shape", ()))
        x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
        residual_shape = tuple(int(dim) for dim in getattr(residual, "shape", ()))
        final_norm_shape = tuple(int(dim) for dim in getattr(final_norm, "shape", ()))
        lm_head_shape = tuple(int(dim) for dim in getattr(lm_head, "shape", ()))
        last_idx_shape = tuple(
            int(dim) for dim in getattr(last_token_indices, "shape", ())
        )
        x_dtype = str(getattr(x, "dtype", "unknown")).replace(".", "_")
        acc_dtype = str(getattr(acc, "dtype", "unknown")).replace(".", "_")
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        moe_groups = tuple(
            tuple(int(rank) for rank in group) for group in moe_replica_groups
        )
        key = (
            acc_shape,
            acc_dtype,
            x_shape,
            x_dtype,
            tuple(int(dim) for dim in getattr(w_gate, "shape", ())),
            str(getattr(w_gate, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_up, "shape", ())),
            str(getattr(w_up, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(w_down, "shape", ())),
            str(getattr(w_down, "dtype", "unknown")).replace(".", "_"),
            residual_shape,
            str(getattr(residual, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(post, "shape", ())),
            str(getattr(post, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(comb, "shape", ())),
            str(getattr(comb, "dtype", "unknown")).replace(".", "_"),
            tuple(int(dim) for dim in getattr(head.hc_head_fn, "shape", ())),
            tuple(int(dim) for dim in getattr(head.hc_head_scale, "shape", ())),
            tuple(int(dim) for dim in getattr(head.hc_head_base, "shape", ())),
            final_norm_shape,
            str(getattr(final_norm, "dtype", "unknown")).replace(".", "_"),
            lm_head_shape,
            str(getattr(lm_head, "dtype", "unknown")).replace(".", "_"),
            last_idx_shape,
            str(getattr(last_token_indices, "dtype", "unknown")).replace(".", "_"),
            float(limit),
            (int(bsz), int(seqlen), int(hidden_size)),
            int(tp_degree),
            groups,
            moe_groups,
            float(head.norm_eps),
            float(head.hc_eps),
            int(n_tokens),
            int(rows),
            float(lm_norm_eps),
        )
        cached = bucket.kernel_caches[
            "shared_expert_add_restore_head_top1_kernels"
        ].get(key)
        if cached is not None:
            return cached
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        vocab = int(lm_head_shape[0]) if lm_head_shape else 0
        name = (
            "dsv4_product_shared_restore_head_top1_"
            f"t{int(bucket.token_bucket)}_{'x'.join(str(v) for v in x_shape)}_"
            f"acc{'x'.join(str(v) for v in acc_shape)}_"
            f"res{'x'.join(str(v) for v in residual_shape)}_"
            f"v{int(vocab)}_{x_dtype}_tp{int(tp_degree)}_{group_tag}"
        )
        if moe_groups:
            moe_group_tag = hashlib.sha1(repr(moe_groups).encode("utf-8")).hexdigest()[
                :8
            ]
            name = f"{name}_moe{moe_group_tag}"
        compile_kwargs: dict[str, Any] = {
            "limit": float(limit),
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "tp_degree": int(tp_degree),
            "tp_replica_groups": groups,
            "moe_replica_groups": moe_groups,
            "norm_eps": float(head.norm_eps),
            "hc_eps": float(head.hc_eps),
            "n_tokens": int(n_tokens),
            "rows": int(rows),
            "lm_norm_eps": float(lm_norm_eps),
        }
        needs_collective = int(tp_degree) > 1 or bool(moe_groups)
        if needs_collective:
            state = getattr(self, "blockwise_moe_state", None)
            rank_id = getattr(state, "collective_rank", None)
            world_size = getattr(state, "collective_world_size", None)
            if rank_id is None or world_size is None:
                raise RuntimeError(
                    "DSV4 product shared-restore sampled head top1 requires "
                    "blockwise_moe_state collective_rank/collective_world_size"
                )
            compile_kwargs.update(
                {
                    "cc_enabled": True,
                    "rank_id": int(rank_id),
                    "world_size": int(world_size),
                    "is_spmd": False,
                    "load_barrier_name": name,
                    "canonical_neff_cache_key": (
                        _product_canonical_neff_cache_key(
                            "dsv4_product_shared_restore_head_top1",
                            "v1",
                            key,
                        )
                    ),
                }
            )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="shared_expert_add_restore_head_top1_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_moe.shared_expert_add_restore_mhc_post_head_top1_fn,
                _sample_array(acc, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_gate, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_up, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(w_down, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(post, fallback_dtype=np.float32),
                _sample_array(comb, fallback_dtype=np.float32),
                _sample_array(head.hc_head_fn, fallback_dtype=np.float32),
                _sample_array(head.hc_head_scale, fallback_dtype=np.float32),
                _sample_array(head.hc_head_base, fallback_dtype=np.float32),
                _sample_array(final_norm, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(lm_head, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(last_token_indices, fallback_dtype=np.int32),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if needs_collective else True,
                **compile_kwargs,
            ),
        )

    def _run_product_shared_expert_add_restore(
        self,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        *,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple = (),
        moe_replica_groups: tuple = (),
    ) -> Any:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(
                "DSV4 product shared-expert restore requires an active token bucket"
            )
        _require_product_device_value(acc, where="shared_expert_add_restore/acc")
        _require_product_device_value(x, where="shared_expert_add_restore/x")
        _require_product_device_value(w_gate, where="shared_expert_add_restore/w_gate")
        _require_product_device_value(w_up, where="shared_expert_add_restore/w_up")
        _require_product_device_value(w_down, where="shared_expert_add_restore/w_down")
        post_pre_ctx = getattr(self, "_shared_expert_restore_post_pre_context", None)
        if post_pre_ctx is not None:
            return self._run_product_shared_expert_add_restore_post_pre(
                acc,
                x,
                w_gate,
                w_up,
                w_down,
                residual=post_pre_ctx["residual"],
                post=post_pre_ctx["post"],
                comb=post_pre_ctx["comb"],
                hc_fn=post_pre_ctx["hc_fn"],
                hc_scale=post_pre_ctx["hc_scale"],
                hc_base=post_pre_ctx["hc_base"],
                norm_weight=post_pre_ctx["norm_weight"],
                limit=float(limit),
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=tp_replica_groups,
                moe_replica_groups=moe_replica_groups,
                is_decode=bool(post_pre_ctx.get("is_decode", False)),
            )
        head_ctx = getattr(self, "_shared_expert_restore_head_context", None)
        if head_ctx is not None:
            if "top1_values" in head_ctx:
                return self._run_product_shared_expert_add_restore_head_top1(
                    acc,
                    x,
                    w_gate,
                    w_up,
                    w_down,
                    residual=head_ctx["residual"],
                    post=head_ctx["post"],
                    comb=head_ctx["comb"],
                    head=head_ctx["head"],
                    final_norm=head_ctx["final_norm"],
                    lm_head=head_ctx["lm_head"],
                    last_token_indices=head_ctx["last_token_indices"],
                    top1_values=head_ctx["top1_values"],
                    top1_indices=head_ctx["top1_indices"],
                    token_bucket=head_ctx["token_bucket"],
                    limit=float(limit),
                    bsz=int(bsz),
                    seqlen=int(seqlen),
                    hidden_size=int(hidden_size),
                    tp_degree=int(tp_degree),
                    tp_replica_groups=tp_replica_groups,
                    moe_replica_groups=moe_replica_groups,
                    is_decode=bool(head_ctx.get("is_decode", False)),
                )
            return self._run_product_shared_expert_add_restore_head_hidden(
                acc,
                x,
                w_gate,
                w_up,
                w_down,
                residual=head_ctx["residual"],
                post=head_ctx["post"],
                comb=head_ctx["comb"],
                head=head_ctx["head"],
                last_token_indices=head_ctx["last_token_indices"],
                token_bucket=head_ctx["token_bucket"],
                limit=float(limit),
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=tp_replica_groups,
                moe_replica_groups=moe_replica_groups,
                is_decode=bool(head_ctx.get("is_decode", False)),
            )
        raise RuntimeError(
            "DSV4 product shared-expert restore requires a fused post/pre or "
            "sampled-head context; standalone shared restore kernels are not "
            "part of the product path"
        )

    def _run_product_shared_expert_add_restore_post_pre(
        self,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        *,
        residual: Any,
        post: Any,
        comb: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple = (),
        moe_replica_groups: tuple = (),
        is_decode: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        executor = self.runtime_surface
        bucket = self._require_active_product_bucket(
            where="shared-expert restore mHC post/pre"
        )
        for value, where in (
            (residual, "shared_restore_post_pre/residual"),
            (post, "shared_restore_post_pre/post"),
            (comb, "shared_restore_post_pre/comb"),
            (hc_fn, "shared_restore_post_pre/hc_fn"),
            (hc_scale, "shared_restore_post_pre/hc_scale"),
            (hc_base, "shared_restore_post_pre/hc_base"),
            (norm_weight, "shared_restore_post_pre/norm_weight"),
        ):
            _require_product_device_value(value, where=where)
        hc_mult = int(executor.args.hc_mult)
        compile_bsz, compile_seqlen = self._product_compile_sequence_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            bucket_single_token=not bool(is_decode),
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
            compile_seqlen=compile_seqlen,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=hc_mult,
        )
        kernel = self._shared_expert_add_restore_post_pre_kernel_for(
            bucket,
            acc,
            x,
            w_gate,
            w_up,
            w_down,
            residual_kernel,
            post_kernel,
            comb_kernel,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            limit=float(limit),
            bsz=int(compile_bsz),
            seqlen=int(compile_seqlen),
            hidden_size=int(hidden_size),
            tp_degree=int(tp_degree),
            tp_replica_groups=tp_replica_groups,
            moe_replica_groups=moe_replica_groups,
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
                "DSV4 product shared-restore post/pre expects residual "
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
                (int(compile_bsz), int(compile_seqlen), int(hidden_size)),
                dtype,
            ),
            "output2": self._bucket_scratch(
                bucket,
                "mhc_pre_post",
                (int(compile_bsz), int(compile_seqlen), hc_mult),
                np.float32,
            ),
            "output3": self._bucket_scratch(
                bucket,
                "mhc_pre_comb",
                (int(compile_bsz), int(compile_seqlen), hc_mult, hc_mult),
                np.float32,
            ),
        }
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "acc": acc,
                "x": x,
                "w_gate": w_gate,
                "w_up": w_up,
                "w_down": w_down,
                "residual": residual_kernel,
                "post": post_kernel,
                "comb": comb_kernel,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_weight": norm_weight,
            },
            outputs=outputs,
            unload_after_call=not self._keep_dp_attention_pipeline_collectives_loaded(),
        )
        return self._alias_mhc_post_pre_outputs(
            outputs,
            bsz=bsz,
            seqlen=seqlen,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )

    def _run_product_shared_expert_add_restore_head_hidden(
        self,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        *,
        residual: Any,
        post: Any,
        comb: Any,
        head: Any,
        last_token_indices: Any,
        token_bucket: int,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple = (),
        moe_replica_groups: tuple = (),
        is_decode: bool = False,
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="shared-expert restore sampled head"
        )
        for value, where in (
            (residual, "shared_restore_head/residual"),
            (post, "shared_restore_head/post"),
            (comb, "shared_restore_head/comb"),
            (head.hc_head_fn, "shared_restore_head/hc_head_fn"),
            (head.hc_head_scale, "shared_restore_head/hc_head_scale"),
            (head.hc_head_base, "shared_restore_head/hc_head_base"),
            (last_token_indices, "shared_restore_head/last_token_indices"),
        ):
            _require_product_device_value(value, where=where)
        token_bucket_i = int(token_bucket)
        n_tokens = int(bsz) * int(seqlen)
        if token_bucket_i < n_tokens:
            raise RuntimeError(
                "token_bucket cannot be smaller than real DSV4 hidden rows: "
                f"token_bucket={token_bucket_i}, rows={n_tokens}"
            )
        output_shape = tuple(
            int(dim) for dim in getattr(bucket.head_hidden_output, "shape", ())
        )
        if output_shape != (token_bucket_i, int(hidden_size)):
            raise RuntimeError(
                "DSV4 product shared-restore sampled-head output shape mismatch: "
                f"output={output_shape}, expected={(token_bucket_i, int(hidden_size))}"
            )
        hc_mult = int(
            getattr(getattr(self.runtime_surface, "args", None), "hc_mult", 1)
        )
        compile_bsz, compile_seqlen = self._product_compile_sequence_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            bucket_single_token=not bool(is_decode),
        )
        compile_bsz, compile_seqlen, residual_kernel, post_kernel, comb_kernel = (
            self._product_promote_mhc_state_shape(
                residual,
                post,
                comb,
                compile_bsz=compile_bsz,
                compile_seqlen=compile_seqlen,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                hc_mult=hc_mult,
            )
        )
        compile_n_tokens = int(compile_bsz) * int(compile_seqlen)
        kernel = self._shared_expert_add_restore_head_select_kernel_for(
            bucket,
            acc,
            x,
            w_gate,
            w_up,
            w_down,
            residual_kernel,
            post_kernel,
            comb_kernel,
            head,
            last_token_indices,
            limit=float(limit),
            bsz=int(compile_bsz),
            seqlen=int(compile_seqlen),
            hidden_size=int(hidden_size),
            tp_degree=int(tp_degree),
            tp_replica_groups=tp_replica_groups,
            moe_replica_groups=moe_replica_groups,
            n_tokens=compile_n_tokens,
            rows=token_bucket_i,
        )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "acc": acc,
                "x": x,
                "w_gate": w_gate,
                "w_up": w_up,
                "w_down": w_down,
                "residual": residual_kernel,
                "post": post_kernel,
                "comb": comb_kernel,
                "hc_head_fn_weight": head.hc_head_fn,
                "hc_head_scale": head.hc_head_scale,
                "hc_head_base": head.hc_head_base,
                "last_token_indices": last_token_indices,
            },
            outputs={"output0": bucket.head_hidden_output},
            unload_after_call=not self._keep_dp_attention_pipeline_collectives_loaded(),
        )
        return bucket.head_hidden_output

    def _run_product_shared_expert_add_restore_head_top1(
        self,
        acc: Any,
        x: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        *,
        residual: Any,
        post: Any,
        comb: Any,
        head: Any,
        final_norm: Any,
        lm_head: Any,
        last_token_indices: Any,
        top1_values: Any,
        top1_indices: Any,
        token_bucket: int,
        limit: float,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple = (),
        moe_replica_groups: tuple = (),
        is_decode: bool = False,
    ) -> dict[str, Any]:
        bucket = self._require_active_product_bucket(
            where="shared-expert restore sampled head top1"
        )
        for value, where in (
            (residual, "shared_restore_head_top1/residual"),
            (post, "shared_restore_head_top1/post"),
            (comb, "shared_restore_head_top1/comb"),
            (head.hc_head_fn, "shared_restore_head_top1/hc_head_fn"),
            (head.hc_head_scale, "shared_restore_head_top1/hc_head_scale"),
            (head.hc_head_base, "shared_restore_head_top1/hc_head_base"),
            (final_norm, "shared_restore_head_top1/final_norm"),
            (lm_head, "shared_restore_head_top1/lm_head"),
            (last_token_indices, "shared_restore_head_top1/last_token_indices"),
        ):
            _require_product_device_value(value, where=where)
        token_bucket_i = int(token_bucket)
        n_tokens = int(bsz) * int(seqlen)
        if token_bucket_i < n_tokens:
            raise RuntimeError(
                "token_bucket cannot be smaller than real DSV4 hidden rows: "
                f"token_bucket={token_bucket_i}, rows={n_tokens}"
            )
        hc_mult = int(
            getattr(getattr(self.runtime_surface, "args", None), "hc_mult", 1)
        )
        compile_bsz, compile_seqlen = self._product_compile_sequence_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            bucket_single_token=not bool(is_decode),
        )
        compile_bsz, compile_seqlen, residual_kernel, post_kernel, comb_kernel = (
            self._product_promote_mhc_state_shape(
                residual,
                post,
                comb,
                compile_bsz=compile_bsz,
                compile_seqlen=compile_seqlen,
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                hc_mult=hc_mult,
            )
        )
        compile_n_tokens = int(compile_bsz) * int(compile_seqlen)
        kernel = self._shared_expert_add_restore_head_top1_kernel_for(
            bucket,
            acc,
            x,
            w_gate,
            w_up,
            w_down,
            residual_kernel,
            post_kernel,
            comb_kernel,
            head,
            final_norm,
            lm_head,
            last_token_indices,
            limit=float(limit),
            bsz=int(compile_bsz),
            seqlen=int(compile_seqlen),
            hidden_size=int(hidden_size),
            tp_degree=int(tp_degree),
            tp_replica_groups=tp_replica_groups,
            moe_replica_groups=moe_replica_groups,
            n_tokens=compile_n_tokens,
            rows=token_bucket_i,
            lm_norm_eps=float(
                getattr(getattr(self, "logits_processor", None), "_rms_norm_eps", 1e-6)
            ),
        )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "acc": acc,
                "x": x,
                "w_gate": w_gate,
                "w_up": w_up,
                "w_down": w_down,
                "residual": residual_kernel,
                "post": post_kernel,
                "comb": comb_kernel,
                "hc_head_fn_weight": head.hc_head_fn,
                "hc_head_scale": head.hc_head_scale,
                "hc_head_base": head.hc_head_base,
                "final_norm": final_norm,
                "lm_head": lm_head,
                "last_token_indices": last_token_indices,
            },
            outputs={
                "output0": top1_values,
                "output1": top1_indices,
            },
            unload_after_call=not self._keep_dp_attention_pipeline_collectives_loaded(),
        )
        return {
            "top1_values": top1_values,
            "top1_indices": top1_indices,
        }
