"""Attention-output product kernels for DSV4 product execution."""

from __future__ import annotations

import hashlib
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import _sampled_warmup_trace
from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _collective_load_barrier_metadata_for_groups,
    _compile_product_kernel,
    _ProductPrecompiledKernel,
    _resolve_product_kernel_for_load,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import attention as graph_attention
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _as_product_device_input,
    _require_product_device_value,
    _sample_array,
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _product_canonical_neff_cache_key,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _AttentionOutCollectiveSpec,
    _TensorSpec,
)
from nkipy_serving.runtime.device_tensor import normalize_dtype as _normalize_dtype


class Dsv4ProductAttentionOutMixin:
    @staticmethod
    def _dp_attention_reduce_rows_for_step(
        *,
        token_bucket: int,
        total_tokens: int,
        batch_size: int,
        seqlen: int,
        is_decode: bool,
        compile_batch_size: int | None = None,
    ) -> int:
        """Return the flat DP-attention reduce rows for a runtime step.

        Prefill keeps the product token bucket so prompt lengths share a
        canonical NEFF. Decode has no prompt-length variation, so using the
        active full-batch rows avoids reducing padded scheduler-bucket rows.

        ``compile_batch_size`` is the promoted kernel batch from
        ``_product_compile_batch_size`` (decode canonicalizes every runtime
        batch onto one ``max_requests``-wide NEFF). The fused post/pre + MoE
        NEFF reshapes ``x[:compile_batch_size*seqlen]``, so the reduce buffer
        MUST hold at least that many rows even though the scheduler hands us a
        smaller raw batch. The o-proj scatters the real rows and the promoted
        tail stays zero (``dp_attention_flat_zero``), so the extra rows route
        as empty tokens through MoE and are discarded at the head. No-op when
        ``max_requests==1`` (batch-1 config => promotion is identity).
        """
        full_batch_rows = max(int(total_tokens), int(batch_size) * int(seqlen))
        if compile_batch_size is not None:
            full_batch_rows = max(
                full_batch_rows,
                int(compile_batch_size) * int(seqlen),
            )
        compact_rows = max(2, int(full_batch_rows))
        if bool(is_decode):
            return compact_rows
        return max(compact_rows, int(token_bucket))

    def _dp_attention_out_collective_rows_for_step(
        self,
        *,
        runtime_token_bucket: int,
        total_tokens: int,
        batch_size: int,
        is_decode: bool,
    ) -> int:
        """Return ``spec.rows`` -- the attention-output ``o`` row count for a step.

        The runtime feeds the o-proj an attention output sized to the backend's
        ``_active_bucket``: DECODE pads on batch (decode ladder = request_buckets),
        PREFILL on token count (prefill ladder = token_buckets). Precompile and
        runtime both route through this so the o-proj NEFF shape matches; using
        the raw scheduler ``token_bucket`` for decode builds the wrong o_rows and
        faults as a post-seal late compile (runtime wants o_rows=request_bucket).

        NOTE: single-request decode only. For batch>1 decode the all-reduce/unpad
        path (dp_attention_unpad_reshape_fn) slices ``bsz*seqlen`` rows from the
        same operand, which exceeds a small request bucket -- multi-request decode
        DP-attention geometry is a separate, currently-unsupported path.
        """
        if bool(is_decode):
            return int(
                self._attention_backend_bucket_for_tokens(
                    int(total_tokens),
                    int(batch_size),
                    is_decode=True,
                )
            )
        return int(runtime_token_bucket)

    def _prewarm_attention_out_hidden_collective(
        self,
        bucket: Dsv4ProductBucket,
        *,
        rows: int,
        bsz: int,
        seqlen: int,
        runtime_seqlen: int | None = None,
        total_batch_size: int | None = None,
        start: int = 0,
        size: int | None = None,
        reduce_rows: int | None = None,
        is_decode: bool = False,
        load: bool = False,
        execute: bool = False,
    ) -> None:
        """Compile or materialize product attention-output collectives."""
        rows_i = int(rows)
        bsz_i = int(bsz)
        seq_i = int(seqlen)
        runtime_seq_i = int(runtime_seqlen) if runtime_seqlen is not None else seq_i
        total_batch_i = int(total_batch_size) if total_batch_size is not None else bsz_i
        start_i = int(start)
        size_i = int(size) if size is not None else bsz_i
        reduce_rows_i = (
            int(reduce_rows)
            if reduce_rows is not None
            else max(2, total_batch_i * seq_i, int(bucket.token_bucket))
        )
        if rows_i <= 0 or bsz_i <= 0 or seq_i <= 0:
            return
        warmed_by_bucket = getattr(
            self,
            "_product_attention_out_dp_flat_warmup_rows",
            None,
        )
        if warmed_by_bucket is None:
            warmed_by_bucket = {}
            self._product_attention_out_dp_flat_warmup_rows = warmed_by_bucket
        warmed = warmed_by_bucket.setdefault(int(bucket.token_bucket), set())
        fns = getattr(self, "graph", {})
        base = fns.get("attention_out_flat") if isinstance(fns, dict) else None
        if base is None:
            return
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()))
        if not blocks:
            return
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            n_heads = int(getattr(attn, "n_heads", 0) or 0)
            head_dim = int(getattr(attn, "head_dim", 0) or 0)
            hidden_size = int(getattr(getattr(attn, "wo_b", None), "shape", (0, 0))[0])
            n_groups = int(getattr(attn, "n_groups", 0) or 0)
            tp_degree = int(getattr(attn, "tp_degree", 1))
            if n_heads <= 0 or head_dim <= 0 or hidden_size <= 0 or n_groups <= 0:
                continue
            flat_prefill_bucket = (
                not bool(is_decode)
                and int(seq_i) == int(bucket.token_bucket)
                and int(reduce_rows_i) >= int(bucket.token_bucket)
            )
            if rows_i < bsz_i * seq_i and not flat_prefill_bucket:
                continue
            groups = list(getattr(attn, "tp_replica_groups", ()) or ())
            if not groups:
                groups = [list(range(tp_degree))]
            groups_t = tuple(tuple(int(rank) for rank in group) for group in groups)
            wo_a = getattr(attn, "wo_a", None)
            wo_b = getattr(attn, "wo_b", None)
            wo_a_sig = (
                tuple(int(dim) for dim in getattr(wo_a, "shape", ())),
                str(getattr(wo_a, "dtype", "")),
            )
            wo_b_sig = (
                tuple(int(dim) for dim in getattr(wo_b, "shape", ())),
                str(getattr(wo_b, "dtype", "")),
            )
            # Compressed layers (compress_ratio != 0; the two-source paged path)
            # alias their owner attention output to the canonical full token-bucket
            # buffer at runtime, so they need an EXTRA o-proj variant (o rows =
            # token_bucket) that non-compressed layers do not. Compressed and
            # non-compressed blocks can otherwise share identical head geometry
            # (n_heads/head_dim/wo sig), so the dedup key MUST distinguish them --
            # else a non-compressed block warms the key first and the compressed
            # block is skipped before its extra variant is built (post-seal fault).
            compressed_layer = bool(int(getattr(attn, "compress_ratio", 0) or 0) != 0)
            token_bucket_rows = int(bucket.token_bucket)
            key = (
                "dp_flat",
                rows_i,
                bsz_i,
                seq_i,
                runtime_seq_i,
                total_batch_i,
                start_i,
                size_i,
                reduce_rows_i,
                n_heads,
                head_dim,
                hidden_size,
                n_groups,
                tp_degree,
                groups_t,
                wo_a_sig,
                wo_b_sig,
                compressed_layer,
            )
            if key in warmed and not load:
                continue
            # The o-proj flat output / reduce buffer holds the full-batch token
            # count: the kernel scatters into `kernel_rows` flat rows and the
            # downstream unpad slices `batch_size*seqlen`. `reduce_rows_i` is
            # computed per DP lane (batch_size=1 in the precompile lane loop) but
            # `total_batch_i` is the full DP batch, so promote to cover it.
            # No-op at batch-1 where the two coincide.
            if flat_prefill_bucket:
                kernel_rows = max(int(reduce_rows_i), int(bucket.token_bucket))
            else:
                kernel_rows = max(int(reduce_rows_i), int(total_batch_i) * int(seq_i))
            out = self._bucket_scratch(
                bucket,
                "attention_inverse_rope_out_dp_flat",
                (int(kernel_rows), hidden_size),
                np.float32,
            )
            freqs_cos = getattr(attn, "freqs_cos", None)
            freqs_sin = getattr(attn, "freqs_sin", None)
            rope_head_dim = int(getattr(attn, "rope_head_dim", 0) or 0)

            def _attention_sample_for_rows(sample_rows: int) -> Any:
                sample = None
                if (
                    int(sample_rows) == int(bucket.token_bucket)
                    and bucket.attention_outputs
                ):
                    candidate = bucket.attention_outputs[0]
                    if tuple(int(dim) for dim in getattr(candidate, "shape", ())) == (
                        int(sample_rows),
                        n_heads,
                        head_dim,
                    ):
                        sample = candidate
                if sample is None:
                    sample = self._attention_output_scratch_for(
                        bucket,
                        rows=int(sample_rows),
                        n_heads=n_heads,
                        head_dim=head_dim,
                    )
                return sample

            def _prewarm_project_input_for_rows(
                project_rows: int,
                *,
                execute_kernel: bool,
                force_lane_start: bool = False,
                project_seqlen: int | None = None,
            ) -> None:
                sample = _attention_sample_for_rows(int(project_rows))
                project_input = sample
                project_seqlen_i = (
                    int(project_seqlen) if project_seqlen is not None else int(seq_i)
                )
                flat_token_range = bool(flat_prefill_bucket) and not bool(
                    force_lane_start
                )
                if (
                    freqs_cos is not None
                    and freqs_sin is not None
                    and rope_head_dim > 0
                ):
                    positions = _TensorSpec((int(project_rows),), np.dtype(np.int32))
                    if flat_token_range:
                        token_start = _TensorSpec((1,), np.dtype(np.int32))
                        token_count = _TensorSpec((1,), np.dtype(np.int32))
                        kernel = self._attention_inverse_rope_out_dp_flat_token_range_table_kernel_for(
                            bucket,
                            sample,
                            attn.wo_a,
                            attn.wo_b,
                            freqs_cos,
                            freqs_sin,
                            positions,
                            token_start,
                            token_count,
                            rope_head_dim=rope_head_dim,
                            n_groups=n_groups,
                            rows=int(kernel_rows),
                            hidden_size=hidden_size,
                            tp_degree=tp_degree,
                            tp_replica_groups=groups_t,
                            base=base,
                            load=load,
                        )
                    else:
                        lane_start = _TensorSpec((1,), np.dtype(np.int32))
                        kernel = (
                            self._attention_inverse_rope_out_dp_flat_table_kernel_for(
                                bucket,
                                sample,
                                attn.wo_a,
                                attn.wo_b,
                                freqs_cos,
                                freqs_sin,
                                positions,
                                lane_start,
                                rope_head_dim=rope_head_dim,
                                n_groups=n_groups,
                                bsz=bsz_i,
                                seqlen=project_seqlen_i,
                                batch_size=total_batch_i,
                                start=start_i,
                                size=size_i,
                                rows=int(kernel_rows),
                                hidden_size=hidden_size,
                                tp_degree=tp_degree,
                                tp_replica_groups=groups_t,
                                base=base,
                                load=load,
                            )
                        )
                else:
                    if flat_token_range:
                        token_start = _TensorSpec((1,), np.dtype(np.int32))
                        token_count = _TensorSpec((1,), np.dtype(np.int32))
                        kernel = self._attention_out_dp_flat_token_range_kernel_for(
                            bucket,
                            sample,
                            attn.wo_a,
                            attn.wo_b,
                            token_start,
                            token_count,
                            n_groups=n_groups,
                            rows=int(kernel_rows),
                            hidden_size=hidden_size,
                            tp_degree=tp_degree,
                            tp_replica_groups=groups_t,
                            base=base,
                            load=load,
                        )
                    else:
                        kernel = self._attention_out_dp_flat_kernel_for(
                            bucket,
                            sample,
                            attn.wo_a,
                            attn.wo_b,
                            n_groups=n_groups,
                            bsz=bsz_i,
                            seqlen=project_seqlen_i,
                            batch_size=total_batch_i,
                            start=start_i,
                            size=size_i,
                            rows=int(kernel_rows),
                            hidden_size=hidden_size,
                            tp_degree=tp_degree,
                            tp_replica_groups=groups_t,
                            base=base,
                            load=load,
                        )
                if execute_kernel:
                    if (
                        freqs_cos is not None
                        and freqs_sin is not None
                        and rope_head_dim > 0
                    ):
                        if flat_token_range:
                            token_start_dev, token_count_dev = (
                                self._sync_attention_dp_token_range(
                                    bucket,
                                    token_start=0,
                                    token_count=min(
                                        int(project_rows), int(kernel_rows)
                                    ),
                                )
                            )
                        else:
                            lane_start_dev = self._sync_attention_dp_lane_start(
                                bucket,
                                start_i,
                            )
                        inputs = {
                            "o": project_input,
                            "wo_a": attn.wo_a,
                            "wo_b": attn.wo_b,
                            "cos_table": freqs_cos,
                            "sin_table": freqs_sin,
                            "positions": bucket.freq_positions_dev,
                        }
                        if flat_token_range:
                            inputs["token_start"] = token_start_dev
                            inputs["token_count"] = token_count_dev
                        else:
                            inputs["lane_start"] = lane_start_dev
                    else:
                        inputs = {
                            "o": project_input,
                            "wo_a": attn.wo_a,
                            "wo_b": attn.wo_b,
                        }
                        if flat_token_range:
                            token_start_dev, token_count_dev = (
                                self._sync_attention_dp_token_range(
                                    bucket,
                                    token_start=0,
                                    token_count=min(
                                        int(project_rows), int(kernel_rows)
                                    ),
                                )
                            )
                            inputs["token_start"] = token_start_dev
                            inputs["token_count"] = token_count_dev
                    _run_product_kernel(
                        kernel,
                        build_dir=self.build_dir,
                        inputs=inputs,
                        outputs={"output0": out},
                        unload_after_call=not bool(load),
                    )

            _prewarm_project_input_for_rows(rows_i, execute_kernel=execute)
            # The runtime o-proj `o` input is the owner-lane attention output,
            # sized to the BACKEND's per-lane _active_bucket. Decode pads on batch
            # (decode ladder: a 1-token lane snaps to the 2-row floor), prefill on
            # token count. Mirror the backend's mode here -- omitting is_decode
            # built `o` rows off the prefill ladder (e.g. 256), so the real
            # per-lane decode `o` (2 rows) faulted as a post-seal late compile.
            lane_rows_i = self._attention_backend_bucket_for_tokens(
                bsz_i * runtime_seq_i,
                int(bucket.token_bucket),
                is_decode=bool(is_decode),
            )
            if (
                not execute
                and int(lane_rows_i) != rows_i
                and int(lane_rows_i) >= bsz_i * runtime_seq_i
            ):
                _prewarm_project_input_for_rows(
                    int(lane_rows_i),
                    execute_kernel=False,
                )
            # COMPRESSED decode o-proj: compressed layers (compress_ratio != 0;
            # the two-source paged path `run_dsv4_attention`) alias their owner
            # attention output to the canonical FULL token-bucket buffer
            # (attention_runtime.py: `attention_postprocess_output = canonical`),
            # so the runtime feeds the o-proj `o` rows = token_bucket while the
            # reduce buffer is the promoted decode `reduce_rows`. That
            # (o_rows=token_bucket, reduce_rows=promoted) combo is built by no
            # other path -- spec.rows is the small decode ladder bucket and the
            # 256-o-rows kernel is only ever paired with prefill reduce_rows.
            # Build it here for compressed blocks so it is resident pre-seal.
            if (
                not execute
                and bool(is_decode)
                and compressed_layer
                and token_bucket_rows != rows_i
                and token_bucket_rows != int(lane_rows_i)
                and token_bucket_rows >= bsz_i * seq_i
            ):
                _prewarm_project_input_for_rows(
                    token_bucket_rows,
                    execute_kernel=False,
                )
            warmed.add(key)

    def precompile_dp_attention_reduce_paths(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        total_tokens: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile DP-attention reduce-lane product kernels.

        Empty lanes need flat-zero reduce inputs, and active lanes need the
        attention-output collective shapes registered before lane divergence.
        The heavy collective handles are compiled lazily here and loaded only
        for the active runtime step to stay within full-model HBM.
        """
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bs = int(batch_size)
        total = int(total_tokens)
        if bs <= 0:
            raise RuntimeError(
                f"DSV4 DP-attention reduce precompile batch must be > 0, got {bs}"
            )
        if total <= 0:
            raise RuntimeError(
                "DSV4 DP-attention reduce precompile total_tokens must be > 0, "
                f"got {total}"
            )
        if total % bs != 0:
            raise RuntimeError(
                "DSV4 DP-attention reduce precompile requires rectangular sampled "
                f"shape; total_tokens={total}, batch_size={bs}"
            )
        hidden_size = int(getattr(bucket.head_hidden_output, "shape", (0, 0))[1])
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 DP-attention reduce precompile could not infer hidden size"
            )
        q_len = total // bs
        compile_batch_size = self._product_compile_batch_size(
            bucket,
            bsz=bs,
            seqlen=int(q_len),
        )
        reduce_rows = self._dp_attention_reduce_rows_for_step(
            token_bucket=int(runtime_token_bucket),
            total_tokens=total,
            batch_size=bs,
            seqlen=int(q_len),
            is_decode=bool(is_decode),
            compile_batch_size=int(compile_batch_size),
        )
        lanes = max(
            1,
            int(
                getattr(
                    getattr(self.runtime_surface, "model_config", None),
                    "attention_dp_degree",
                    bs,
                )
                or bs
            ),
        )
        active_lanes = min(bs, lanes)
        base_lane_batch, rem_lane_batch = divmod(bs, active_lanes)
        lane_slices: list[tuple[int, int]] = []
        lane_start = 0
        for lane_idx in range(active_lanes):
            lane_size = base_lane_batch + (1 if lane_idx < rem_lane_batch else 0)
            if lane_size <= 0:
                continue
            lane_slices.append((lane_start, lane_size))
            lane_start += lane_size
        self._bucket_scratch(
            bucket,
            "dp_attention_flat_zero",
            (int(reduce_rows), int(hidden_size)),
            np.float32,
        )
        # Runtime DP-attention can execute attention projection only on owner
        # lanes, while empty lanes can immediately enter the DP all-reduce.
        # Preload the product hidden-output collective shapes on every rank
        # before those lanes diverge, otherwise rank subsets can block on
        # different collective-load barriers. Attention QKV/KV-write remains
        # lane-local, but the post-attention projection consumes product-owned
        # output buffers sized to the runtime bucket.
        for lane_start_i, lane_size_i in lane_slices:
            compile_seqlen = self._attention_out_dp_flat_compile_seqlen(
                bucket,
                bsz=int(lane_size_i),
                seqlen=int(q_len),
                batch_size=bs,
                start=int(lane_start_i),
                size=int(lane_size_i),
                rows=int(reduce_rows),
                is_decode=bool(is_decode),
            )
            spec = _AttentionOutCollectiveSpec(
                rows=self._dp_attention_out_collective_rows_for_step(
                    runtime_token_bucket=int(runtime_token_bucket),
                    total_tokens=total,
                    batch_size=bs,
                    is_decode=bool(is_decode),
                ),
                bsz=int(lane_size_i),
                seqlen=int(q_len),
                batch_size=bs,
                start=int(lane_start_i),
                size=int(lane_size_i),
                reduce_rows=reduce_rows,
                is_decode=bool(is_decode),
            )
            # Collective residency is hardwired off (net loss; see
            # _keep_transient_dp_attention_out_collectives_loaded). Precompile the
            # collective NEFF without loading it; it is loaded+unloaded per step.
            self._prewarm_attention_out_hidden_collective(
                bucket,
                rows=int(spec.rows),
                bsz=int(spec.bsz),
                seqlen=int(compile_seqlen),
                runtime_seqlen=int(spec.seqlen),
                total_batch_size=int(spec.batch_size),
                start=int(spec.start),
                size=int(spec.size),
                reduce_rows=int(spec.reduce_rows),
                is_decode=bool(spec.is_decode),
                load=False,
            )
        groups = tuple(
            tuple(int(rank) for rank in group)
            for group in tuple(self._dp_attention_replica_groups() or ())
        )
        if not groups or all(len(group) <= 1 for group in groups):
            return

        blocks = tuple(getattr(self.runtime_surface, "blocks", ()) or ())
        if not blocks:
            return
        args = getattr(self.runtime_surface, "args", None)
        hc_mult = int(getattr(args, "hc_mult", 0) or 0)
        if hc_mult <= 0:
            raise RuntimeError(
                "DSV4 DP-attention reduce precompile requires runtime_surface.args.hc_mult"
            )
        canonical_post_pre_shape = self._dp_attention_post_pre_compile_shape(
            bucket,
            bsz=bs,
            seqlen=int(q_len),
            rows=int(reduce_rows),
            dispatch_context={"is_decode": bool(is_decode)},
        )
        bf16_dtype = np.dtype(ml_dtypes.bfloat16)
        f32_dtype = np.dtype(np.float32)
        x = _TensorSpec((int(reduce_rows), int(hidden_size)), f32_dtype)
        post_pre_shape_candidates = [canonical_post_pre_shape]

        for compile_bsz, post_pre_seqlen in post_pre_shape_candidates:
            split_post_pre = self._should_split_dp_attention_post_pre(
                x,
                seqlen=int(post_pre_seqlen),
            )
            if split_post_pre:
                kernel = self._dp_attention_all_reduce_kernel_for(
                    bucket,
                    x,
                    replica_groups=groups,
                )
                if self._keep_dp_attention_pipeline_collectives_loaded():
                    self._load_resident_product_kernel(kernel)

            post_pre_bsz = int(compile_bsz)
            post_pre_seqlen_i = int(post_pre_seqlen)
            for compile_bsz in (post_pre_bsz,):
                # The fused unpad slices x[:bsz*seqlen]. When compile_bsz is
                # promoted to max_requests (batch>1 decode), the flat reduce
                # buffer x must hold that many rows -- the per-lane reduce_rows
                # (batch_size=1) is too small. The runtime sizes the o-proj flat
                # output by the full batch, so match it. No-op at batch-1.
                x_for_compile_bsz = _TensorSpec(
                    (
                        max(
                            int(reduce_rows),
                            int(compile_bsz) * int(post_pre_seqlen_i),
                        ),
                        int(hidden_size),
                    ),
                    f32_dtype,
                )
                residual = _TensorSpec(
                    (
                        int(compile_bsz),
                        int(post_pre_seqlen_i),
                        int(hc_mult),
                        int(hidden_size),
                    ),
                    bf16_dtype,
                )
                post = _TensorSpec(
                    (int(compile_bsz), int(post_pre_seqlen_i), int(hc_mult)),
                    f32_dtype,
                )
                comb = _TensorSpec(
                    (
                        int(compile_bsz),
                        int(post_pre_seqlen_i),
                        int(hc_mult),
                        int(hc_mult),
                    ),
                    f32_dtype,
                )
                for block in blocks:
                    required = (
                        getattr(block, "hc_ffn_fn", None),
                        getattr(block, "hc_ffn_scale", None),
                        getattr(block, "hc_ffn_base", None),
                        getattr(block, "ffn_norm", None),
                    )
                    if any(value is None for value in required):
                        continue
                    if split_post_pre:
                        kernel = self._dp_attention_unpad_post_pre_kernel_for(
                            bucket,
                            x,
                            residual,
                            post,
                            comb,
                            block.hc_ffn_fn,
                            block.hc_ffn_scale,
                            block.hc_ffn_base,
                            block.ffn_norm,
                            bsz=int(compile_bsz),
                            seqlen=int(post_pre_seqlen_i),
                            hidden_size=int(hidden_size),
                            hc_mult=int(hc_mult),
                            sinkhorn_iters=int(getattr(args, "hc_sinkhorn_iters")),
                            norm_eps=float(getattr(args, "norm_eps")),
                            hc_eps=float(getattr(args, "hc_eps")),
                        )
                    else:
                        kernel = self._dp_attention_all_reduce_post_pre_kernel_for(
                            bucket,
                            x_for_compile_bsz,
                            residual,
                            post,
                            comb,
                            block.hc_ffn_fn,
                            block.hc_ffn_scale,
                            block.hc_ffn_base,
                            block.ffn_norm,
                            replica_groups=groups,
                            bsz=int(compile_bsz),
                            seqlen=int(post_pre_seqlen_i),
                            hidden_size=int(hidden_size),
                            hc_mult=int(hc_mult),
                            sinkhorn_iters=int(getattr(args, "hc_sinkhorn_iters")),
                            norm_eps=float(getattr(args, "norm_eps")),
                            hc_eps=float(getattr(args, "hc_eps")),
                        )
                    if self._keep_dp_attention_pipeline_collectives_loaded():
                        self._load_resident_product_kernel(kernel)

    def _dp_attention_out_collective_specs_for_step(
        self,
        bucket: Dsv4ProductBucket,
        *,
        token_bucket: int,
        batch_size: int,
        total_tokens: int,
        compressed: bool,
        is_decode: bool = False,
    ) -> list[_AttentionOutCollectiveSpec]:
        bs = int(batch_size)
        total = int(total_tokens)
        if bs <= 0 or total <= 0:
            return []
        if total % bs != 0:
            raise RuntimeError(
                "DSV4 product DP-attention runtime collective load requires "
                f"rectangular sampled shape; total_tokens={total}, batch_size={bs}"
            )
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        q_len = total // bs
        # Live decode batches land BETWEEN ladder points (15 of 16 requests
        # still decoding); warmup only compiles ladder rectangles, so pad the
        # decode batch to its request bucket — padding lanes reduce the zeroed
        # flat scratch. Prefill keeps its real bsz (lane geometry is per-spec).
        if bool(is_decode) and int(q_len) == 1:
            promoted = int(self._product_compile_batch_size(bucket, bsz=bs, seqlen=1))
            if promoted > bs:
                bs = promoted
                total = bs
        compile_batch_size = self._product_compile_batch_size(
            bucket,
            bsz=bs,
            seqlen=int(q_len),
        )
        reduce_rows = self._dp_attention_reduce_rows_for_step(
            token_bucket=int(runtime_token_bucket),
            total_tokens=total,
            batch_size=bs,
            seqlen=int(q_len),
            is_decode=bool(is_decode),
            compile_batch_size=int(compile_batch_size),
        )
        lanes = max(
            1,
            int(
                getattr(
                    getattr(self.runtime_surface, "model_config", None),
                    "attention_dp_degree",
                    bs,
                )
                or bs
            ),
        )
        active_lanes = min(bs, lanes)
        base_lane_batch, rem_lane_batch = divmod(bs, active_lanes)
        lane_slices: list[tuple[int, int]] = []
        lane_start = 0
        for lane_idx in range(active_lanes):
            lane_size = base_lane_batch + (1 if lane_idx < rem_lane_batch else 0)
            if lane_size <= 0:
                continue
            lane_slices.append((lane_start, lane_size))
            lane_start += lane_size

        spec_rows = self._dp_attention_out_collective_rows_for_step(
            runtime_token_bucket=int(runtime_token_bucket),
            total_tokens=total,
            batch_size=bs,
            is_decode=bool(is_decode),
        )
        return [
            _AttentionOutCollectiveSpec(
                rows=int(spec_rows),
                bsz=int(lane_size_i),
                seqlen=int(q_len),
                batch_size=bs,
                start=int(lane_start_i),
                size=int(lane_size_i),
                reduce_rows=int(reduce_rows),
                is_decode=bool(is_decode),
            )
            for lane_start_i, lane_size_i in lane_slices
        ]

    def _materialize_dp_attention_out_collectives_for_step(
        self,
        bucket: Dsv4ProductBucket,
        *,
        token_bucket: int,
        batch_size: int,
        total_tokens: int,
        compressed: bool,
        is_decode: bool = False,
    ) -> None:
        for spec in self._dp_attention_out_collective_specs_for_step(
            bucket,
            token_bucket=token_bucket,
            batch_size=batch_size,
            total_tokens=total_tokens,
            compressed=compressed,
            is_decode=bool(is_decode),
        ):
            self._materialize_dp_attention_out_collective_spec(bucket, spec)

    def _materialize_dp_attention_out_collective_spec(
        self,
        bucket: Dsv4ProductBucket,
        spec: _AttentionOutCollectiveSpec,
    ) -> None:
        materialized_key, compile_seqlen = (
            self._dp_attention_out_collective_materialized_key_for_spec(
                bucket,
                spec,
            )
        )
        materialized_keys = None
        if self._should_defer_transient_dp_attention_unload():
            materialized_keys = getattr(
                self,
                "_deferred_dp_attention_out_collective_materialized_keys",
                None,
            )
            if materialized_keys is None:
                materialized_keys = set()
                self._deferred_dp_attention_out_collective_materialized_keys = (
                    materialized_keys
                )
            if materialized_key in materialized_keys:
                return
        self._prewarm_attention_out_hidden_collective(
            bucket,
            rows=int(spec.rows),
            bsz=int(spec.bsz),
            seqlen=int(compile_seqlen),
            runtime_seqlen=int(spec.seqlen),
            total_batch_size=int(spec.batch_size),
            start=int(spec.start),
            size=int(spec.size),
            reduce_rows=int(spec.reduce_rows),
            is_decode=bool(spec.is_decode),
            load=True,
        )
        if materialized_keys is not None:
            materialized_keys.add(materialized_key)

    def _is_dp_attention_out_collective_spec_materialized(
        self,
        bucket: Dsv4ProductBucket,
        spec: _AttentionOutCollectiveSpec,
    ) -> bool:
        materialized_keys = None
        if self._should_defer_transient_dp_attention_unload():
            materialized_keys = getattr(
                self,
                "_deferred_dp_attention_out_collective_materialized_keys",
                None,
            )
        if materialized_keys is None:
            return False
        materialized_key, _compile_seqlen = (
            self._dp_attention_out_collective_materialized_key_for_spec(
                bucket,
                spec,
            )
        )
        return materialized_key in materialized_keys

    def _dp_attention_out_collective_materialized_key_for_spec(
        self,
        bucket: Dsv4ProductBucket,
        spec: _AttentionOutCollectiveSpec,
    ) -> tuple[tuple[int, int, int, int, int, int, int, int, int, int], int]:
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
        return (
            self._dp_attention_out_collective_materialized_key(
                bucket,
                spec,
                compile_seqlen=int(compile_seqlen),
            ),
            int(compile_seqlen),
        )

    @staticmethod
    def _dp_attention_out_collective_materialized_key(
        bucket: Dsv4ProductBucket,
        spec: _AttentionOutCollectiveSpec,
        *,
        compile_seqlen: int,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            id(bucket),
            int(bucket.token_bucket),
            int(spec.rows),
            int(spec.bsz),
            int(compile_seqlen),
            int(spec.batch_size),
            int(spec.start),
            int(spec.size),
            int(spec.reduce_rows),
            int(bool(spec.is_decode)),
        )

    @staticmethod
    def _unload_transient_dp_attention_out_collectives(
        bucket: Dsv4ProductBucket,
    ) -> None:
        for kernel in (
            tuple(
                bucket.kernel_caches[
                    "attention_inverse_rope_tail_flat_kernels"
                ].values()
            )
            + tuple(bucket.kernel_caches["attention_out_dp_flat_kernels"].values())
            + tuple(
                bucket.kernel_caches[
                    "attention_inverse_rope_out_dp_flat_kernels"
                ].values()
            )
        ):
            if isinstance(kernel, _ProductPrecompiledKernel):
                kernel.unload()

    def _should_defer_transient_dp_attention_unload(self) -> bool:
        return bool(getattr(self, "_defer_transient_dp_attention_unload", False))

    def _keep_transient_dp_attention_out_collectives_loaded(self) -> bool:
        # The cc-enabled DP-attention-out collective NEFF is reloaded per step
        # rather than held resident. Keeping it resident eliminates the
        # ~282ms/step `attention_dp_materialize` reload, but device A/B
        # (2026-05-30) showed residency makes EVERY per-layer kernel ~2x slower
        # (forward_total 1054ms -> 1912ms), a net loss. Layer-NEFF fusion was
        # also tried (2026-06) and is infeasible on this stack (per-replica
        # cc-NEFF exec barrier), so unload-per-step is hardwired.
        return False

    def _keep_dp_attention_pipeline_collectives_loaded(self) -> bool:
        # Pipeline-collective residency is likewise a net loss; unload per call.
        return False

    def _load_resident_product_kernel(self, kernel: Any) -> None:
        if isinstance(kernel, _ProductPrecompiledKernel):
            kernel.load(build_dir=self.build_dir)

    def _dp_attention_out_specs_reuse_loaded_inverse_kernel(
        self,
        current: _AttentionOutCollectiveSpec,
        next_spec: _AttentionOutCollectiveSpec | None,
    ) -> bool:
        if next_spec is None:
            return False
        if (
            int(current.rows) != int(next_spec.rows)
            or int(current.bsz) != int(next_spec.bsz)
            or int(current.seqlen) != int(next_spec.seqlen)
            or int(current.batch_size) != int(next_spec.batch_size)
            or int(current.size) != int(next_spec.size)
            or int(current.reduce_rows) != int(next_spec.reduce_rows)
        ):
            return False
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()))
        if not blocks:
            return False
        has_attention = False
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            has_attention = True
            if (
                getattr(attn, "freqs_cos", None) is None
                or getattr(attn, "freqs_sin", None) is None
                or int(getattr(attn, "rope_head_dim", 0) or 0) <= 0
            ):
                return False
        return has_attention

    def _attention_inverse_rope_tail_flat_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        o: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        rope_head_dim: int,
        load: bool = False,
    ) -> Any:
        shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if len(shape) != 3:
            raise RuntimeError(
                "DSV4 product inverse_rope_tail_flat expects [rows, heads, dim] "
                f"input, got {shape}"
            )
        if int(rope_head_dim) <= 0:
            raise RuntimeError(
                "DSV4 product inverse_rope_tail_flat requires rope_head_dim > 0, "
                f"got {int(rope_head_dim)}"
            )
        key = (
            shape,
            str(np.dtype(_normalize_dtype(_value_dtype(o, fallback=np.float32)))),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(positions, "shape", ())),
            str(getattr(positions, "dtype", "")),
            int(rope_head_dim),
        )
        cached = bucket.kernel_caches["attention_inverse_rope_tail_flat_kernels"].get(
            key
        )
        if cached is not None:
            if load and isinstance(cached, _ProductPrecompiledKernel):
                cached.load(build_dir=self.build_dir)
            return cached
        kernel = self._cached_product_kernel(
            bucket=bucket,
            cache_name="attention_inverse_rope_tail_flat_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.inverse_rope_tail_flat_from_freq_table_fn,
                _sample_array(o, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                rope_head_dim=int(rope_head_dim),
                name=(
                    "dsv4_product_inverse_rope_tail_flat_"
                    f"t{int(bucket.token_bucket)}_"
                    f"{'x'.join(str(v) for v in shape)}_"
                    f"p{'x'.join(str(v) for v in getattr(positions, 'shape', ())) or '0'}_"
                    f"rd{int(rope_head_dim)}"
                ),
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
            ),
        )
        if load and isinstance(kernel, _ProductPrecompiledKernel):
            kernel.load(build_dir=self.build_dir)
        return kernel

    def _attention_out_dp_flat_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        *,
        n_groups: int,
        bsz: int,
        seqlen: int,
        batch_size: int,
        start: int,
        size: int,
        rows: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        base: Any | None = None,
        load: bool = True,
    ) -> Any:
        shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if len(shape) != 3 or int(hidden_size) <= 0:
            raise RuntimeError(
                "DSV4 product attention_out_dp_flat expects "
                f"[rows, heads, dim] input and hidden_size > 0, got {shape}"
            )
        if shape[0] < int(bsz) * int(seqlen):
            raise RuntimeError(
                "DSV4 product attention_out_dp_flat rows are smaller than target "
                f"tokens: rows={shape[0]}, bsz={int(bsz)}, seqlen={int(seqlen)}"
            )
        batch_size_i = int(batch_size)
        start_i = int(start)
        size_i = int(size)
        if start_i < 0 or size_i < 0 or start_i + size_i > batch_size_i:
            raise RuntimeError(
                "DSV4 product attention_out_dp_flat has invalid DP lane range: "
                f"start={start_i}, size={size_i}, batch_size={batch_size_i}"
            )
        rows_i = int(rows)
        if rows_i < batch_size_i * int(seqlen):
            raise RuntimeError(
                "DSV4 product attention_out_dp_flat reduce rows are smaller "
                "than full-batch rows: "
                f"rows={rows_i}, batch_size={batch_size_i}, seqlen={int(seqlen)}"
            )
        compile_batch_size = start_i + size_i
        dtype = _value_dtype(o, fallback=np.float32)
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        key = (
            shape,
            str(np.dtype(_normalize_dtype(dtype))),
            tuple(int(dim) for dim in getattr(wo_a, "shape", ())),
            str(getattr(wo_a, "dtype", "")),
            tuple(int(dim) for dim in getattr(wo_b, "shape", ())),
            str(getattr(wo_b, "dtype", "")),
            int(n_groups),
            int(bsz),
            int(seqlen),
            int(compile_batch_size),
            start_i,
            size_i,
            rows_i,
            int(hidden_size),
            int(tp_degree),
            groups,
        )
        cached = bucket.kernel_caches["attention_out_dp_flat_kernels"].get(key)
        if cached is not None:
            return _resolve_product_kernel_for_load(
                cached,
                build_dir=self.build_dir,
                load=load,
            )
        compile_kwargs: dict[str, Any] = {}
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_attn_out_dp_flat_"
            f"t{int(bucket.token_bucket)}_"
            f"{'x'.join(str(v) for v in shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
            f"b{int(compile_batch_size)}_s{start_i}_n{size_i}_"
            f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
        )
        if int(tp_degree) > 1:
            rank_id, world_size = self._collective_graph_metadata(
                "attention_out_flat",
                where="attention_out_dp_flat",
                base=base,
            )
            barrier_rank_id, barrier_world_size = (
                _collective_load_barrier_metadata_for_groups(
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                    replica_groups=groups,
                )
            )
            compile_kwargs.update(
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load_barrier_name=(
                    "dsv4_product_attn_out_dp_flat_"
                    f"t{int(bucket.token_bucket)}_"
                    f"{'x'.join(str(v) for v in shape)}_"
                    f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
                    f"b{int(compile_batch_size)}_s{start_i}_n{size_i}_"
                    f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=_product_canonical_neff_cache_key(
                    "dsv4_product_attn_out_dp_flat",
                    "v1",
                    key,
                ),
            )
        defer_collective_load = int(tp_degree) > 1
        kernel = self._cached_product_kernel(
            bucket=bucket,
            cache_name="attention_out_dp_flat_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.attention_out_proj_dp_flat_fn,
                _sample_array(o, fallback_dtype=np.float32),
                _sample_array(wo_a, fallback_dtype=np.float32),
                _sample_array(wo_b, fallback_dtype=np.float32),
                n_groups=int(n_groups),
                bsz=int(bsz),
                seqlen=int(seqlen),
                batch_size=int(compile_batch_size),
                start=start_i,
                size=size_i,
                rows=rows_i,
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if defer_collective_load else load,
                **compile_kwargs,
            ),
        )
        return _resolve_product_kernel_for_load(
            kernel,
            build_dir=self.build_dir,
            load=load,
        )

    def _attention_inverse_rope_out_dp_flat_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        lane_start: Any,
        *,
        rope_head_dim: int,
        n_groups: int,
        bsz: int,
        seqlen: int,
        batch_size: int,
        start: int,
        size: int,
        rows: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        base: Any | None = None,
        load: bool = True,
    ) -> Any:
        shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if len(shape) != 3 or int(hidden_size) <= 0:
            raise RuntimeError(
                "DSV4 product inverse_rope attention_out_dp_flat expects "
                f"[rows, heads, dim] input and hidden_size > 0, got {shape}"
            )
        if shape[0] < int(bsz) * int(seqlen):
            raise RuntimeError(
                "DSV4 product inverse_rope attention_out_dp_flat rows are "
                "smaller than target tokens: "
                f"rows={shape[0]}, bsz={int(bsz)}, seqlen={int(seqlen)}"
            )
        batch_size_i = int(batch_size)
        start_i = int(start)
        size_i = int(size)
        if start_i < 0 or size_i < 0 or start_i + size_i > batch_size_i:
            raise RuntimeError(
                "DSV4 product inverse_rope attention_out_dp_flat has invalid "
                f"DP lane range: start={start_i}, size={size_i}, "
                f"batch_size={batch_size_i}"
            )
        rows_i = int(rows)
        compile_batch_size = batch_size_i
        if rows_i < compile_batch_size * int(seqlen):
            raise RuntimeError(
                "DSV4 product inverse_rope attention_out_dp_flat reduce rows "
                "are smaller than full-batch rows: "
                f"rows={rows_i}, batch_size={compile_batch_size}, "
                f"seqlen={int(seqlen)}"
            )
        lane_start_shape = tuple(int(dim) for dim in getattr(lane_start, "shape", ()))
        lane_start_elements = int(np.prod(lane_start_shape)) if lane_start_shape else 1
        if lane_start_elements <= 0:
            raise RuntimeError(
                "DSV4 product inverse_rope attention_out_dp_flat requires a "
                f"non-empty lane_start tensor, got shape={lane_start_shape}"
            )
        dtype = _value_dtype(o, fallback=np.float32)
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        key = (
            shape,
            str(np.dtype(_normalize_dtype(dtype))),
            tuple(int(dim) for dim in getattr(wo_a, "shape", ())),
            str(getattr(wo_a, "dtype", "")),
            tuple(int(dim) for dim in getattr(wo_b, "shape", ())),
            str(getattr(wo_b, "dtype", "")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(positions, "shape", ())),
            str(getattr(positions, "dtype", "")),
            lane_start_shape,
            str(getattr(lane_start, "dtype", "")),
            int(rope_head_dim),
            int(n_groups),
            int(bsz),
            int(seqlen),
            int(compile_batch_size),
            size_i,
            rows_i,
            int(hidden_size),
            int(tp_degree),
            groups,
        )
        cached = bucket.kernel_caches["attention_inverse_rope_out_dp_flat_kernels"].get(
            key
        )
        if cached is not None:
            return _resolve_product_kernel_for_load(
                cached,
                build_dir=self.build_dir,
                load=load,
            )
        compile_kwargs: dict[str, Any] = {}
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_inverse_rope_attn_out_dp_flat_"
            f"t{int(bucket.token_bucket)}_"
            f"{'x'.join(str(v) for v in shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
            f"p{'x'.join(str(v) for v in getattr(positions, 'shape', ())) or '0'}_"
            f"rd{int(rope_head_dim)}_"
            f"b{int(compile_batch_size)}_sdyn_n{size_i}_"
            f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
        )
        if int(tp_degree) > 1:
            rank_id, world_size = self._collective_graph_metadata(
                "attention_out_flat",
                where="inverse_rope_attention_out_dp_flat",
                base=base,
            )
            barrier_rank_id, barrier_world_size = (
                _collective_load_barrier_metadata_for_groups(
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                    replica_groups=groups,
                )
            )
            compile_kwargs.update(
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load_barrier_name=(
                    "dsv4_product_inverse_rope_attn_out_dp_flat_"
                    f"t{int(bucket.token_bucket)}_"
                    f"{'x'.join(str(v) for v in shape)}_"
                    f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
                    f"p{'x'.join(str(v) for v in getattr(positions, 'shape', ())) or '0'}_"
                    f"rd{int(rope_head_dim)}_"
                    f"b{int(compile_batch_size)}_sdyn_n{size_i}_"
                    f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=(
                    _product_canonical_neff_cache_key(
                        "dsv4_product_inverse_rope_attn_out_dp_flat",
                        "v1",
                        key,
                    )
                ),
            )
        defer_collective_load = int(tp_degree) > 1
        kernel = self._cached_product_kernel(
            bucket=bucket,
            cache_name="attention_inverse_rope_out_dp_flat_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.attention_inverse_rope_active_out_proj_dp_flat_dynamic_start_from_freq_table_fn,
                _sample_array(o, fallback_dtype=np.float32),
                _sample_array(wo_a, fallback_dtype=np.float32),
                _sample_array(wo_b, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                _sample_array(lane_start, fallback_dtype=np.int32),
                rope_head_dim=int(rope_head_dim),
                n_groups=int(n_groups),
                bsz=int(bsz),
                seqlen=int(seqlen),
                batch_size=int(compile_batch_size),
                size=size_i,
                rows=rows_i,
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if defer_collective_load else load,
                **compile_kwargs,
            ),
        )
        return _resolve_product_kernel_for_load(
            kernel,
            build_dir=self.build_dir,
            load=load,
        )

    def _attention_out_dp_flat_token_range_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        token_start: Any,
        token_count: Any,
        *,
        n_groups: int,
        rows: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        base: Any | None = None,
        load: bool = True,
    ) -> Any:
        shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if len(shape) != 3 or int(hidden_size) <= 0:
            raise RuntimeError(
                "DSV4 product attention_out token-range expects "
                f"[rows, heads, dim] input and hidden_size > 0, got {shape}"
            )
        rows_i = int(rows)
        dtype = _value_dtype(o, fallback=np.float32)
        token_start_shape = tuple(int(dim) for dim in getattr(token_start, "shape", ()))
        token_count_shape = tuple(int(dim) for dim in getattr(token_count, "shape", ()))
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        key = (
            shape,
            str(np.dtype(_normalize_dtype(dtype))),
            tuple(int(dim) for dim in getattr(wo_a, "shape", ())),
            str(getattr(wo_a, "dtype", "")),
            tuple(int(dim) for dim in getattr(wo_b, "shape", ())),
            str(getattr(wo_b, "dtype", "")),
            token_start_shape,
            str(getattr(token_start, "dtype", "")),
            token_count_shape,
            str(getattr(token_count, "dtype", "")),
            int(n_groups),
            rows_i,
            int(hidden_size),
            int(tp_degree),
            groups,
        )
        cached = bucket.kernel_caches["attention_out_dp_flat_kernels"].get(key)
        if cached is not None:
            return _resolve_product_kernel_for_load(
                cached,
                build_dir=self.build_dir,
                load=load,
            )
        compile_kwargs: dict[str, Any] = {}
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_attn_out_dp_flat_token_range_"
            f"t{int(bucket.token_bucket)}_"
            f"{'x'.join(str(v) for v in shape)}_"
            f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
        )
        if int(tp_degree) > 1:
            rank_id, world_size = self._collective_graph_metadata(
                "attention_out_flat",
                where="attention_out_dp_flat_token_range",
                base=base,
            )
            barrier_rank_id, barrier_world_size = (
                _collective_load_barrier_metadata_for_groups(
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                    replica_groups=groups,
                )
            )
            compile_kwargs.update(
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load_barrier_name=(
                    "dsv4_product_attn_out_dp_flat_token_range_"
                    f"t{int(bucket.token_bucket)}_"
                    f"{'x'.join(str(v) for v in shape)}_"
                    f"r{rows_i}_tp{int(tp_degree)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=_product_canonical_neff_cache_key(
                    "dsv4_product_attn_out_dp_flat_token_range",
                    "v1",
                    key,
                ),
            )
        defer_collective_load = int(tp_degree) > 1
        kernel = self._cached_product_kernel(
            bucket=bucket,
            cache_name="attention_out_dp_flat_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.attention_out_proj_dp_flat_dynamic_token_range_fn,
                _sample_array(o, fallback_dtype=np.float32),
                _sample_array(wo_a, fallback_dtype=np.float32),
                _sample_array(wo_b, fallback_dtype=np.float32),
                _sample_array(token_start, fallback_dtype=np.int32),
                _sample_array(token_count, fallback_dtype=np.int32),
                n_groups=int(n_groups),
                rows=rows_i,
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if defer_collective_load else load,
                **compile_kwargs,
            ),
        )
        return _resolve_product_kernel_for_load(
            kernel,
            build_dir=self.build_dir,
            load=load,
        )

    def _attention_inverse_rope_out_dp_flat_token_range_table_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        token_start: Any,
        token_count: Any,
        *,
        rope_head_dim: int,
        n_groups: int,
        rows: int,
        hidden_size: int,
        tp_degree: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        base: Any | None = None,
        load: bool = True,
    ) -> Any:
        shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if len(shape) != 3 or int(hidden_size) <= 0:
            raise RuntimeError(
                "DSV4 product inverse-RoPE attention_out token-range expects "
                f"[rows, heads, dim] input and hidden_size > 0, got {shape}"
            )
        rows_i = int(rows)
        token_start_shape = tuple(int(dim) for dim in getattr(token_start, "shape", ()))
        token_count_shape = tuple(int(dim) for dim in getattr(token_count, "shape", ()))
        dtype = _value_dtype(o, fallback=np.float32)
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        key = (
            shape,
            str(np.dtype(_normalize_dtype(dtype))),
            tuple(int(dim) for dim in getattr(wo_a, "shape", ())),
            str(getattr(wo_a, "dtype", "")),
            tuple(int(dim) for dim in getattr(wo_b, "shape", ())),
            str(getattr(wo_b, "dtype", "")),
            tuple(int(dim) for dim in getattr(cos_table, "shape", ())),
            str(getattr(cos_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(sin_table, "shape", ())),
            str(getattr(sin_table, "dtype", "")),
            tuple(int(dim) for dim in getattr(positions, "shape", ())),
            str(getattr(positions, "dtype", "")),
            token_start_shape,
            str(getattr(token_start, "dtype", "")),
            token_count_shape,
            str(getattr(token_count, "dtype", "")),
            int(rope_head_dim),
            int(n_groups),
            rows_i,
            int(hidden_size),
            int(tp_degree),
            groups,
        )
        cached = bucket.kernel_caches["attention_inverse_rope_out_dp_flat_kernels"].get(
            key
        )
        if cached is not None:
            return _resolve_product_kernel_for_load(
                cached,
                build_dir=self.build_dir,
                load=load,
            )
        compile_kwargs: dict[str, Any] = {}
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            "dsv4_product_inverse_rope_attn_out_dp_flat_token_range_"
            f"t{int(bucket.token_bucket)}_"
            f"{'x'.join(str(v) for v in shape)}_"
            f"p{'x'.join(str(v) for v in getattr(positions, 'shape', ())) or '0'}_"
            f"rd{int(rope_head_dim)}_r{rows_i}_tp{int(tp_degree)}_{group_tag}"
        )
        if int(tp_degree) > 1:
            rank_id, world_size = self._collective_graph_metadata(
                "attention_out_flat",
                where="inverse_rope_attention_out_dp_flat_token_range",
                base=base,
            )
            barrier_rank_id, barrier_world_size = (
                _collective_load_barrier_metadata_for_groups(
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                    replica_groups=groups,
                )
            )
            compile_kwargs.update(
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load_barrier_name=(
                    "dsv4_product_inverse_rope_attn_out_dp_flat_token_range_"
                    f"t{int(bucket.token_bucket)}_"
                    f"{'x'.join(str(v) for v in shape)}_"
                    f"p{'x'.join(str(v) for v in getattr(positions, 'shape', ())) or '0'}_"
                    f"rd{int(rope_head_dim)}_r{rows_i}_tp{int(tp_degree)}_{group_tag}"
                ),
                load_barrier_rank_id=int(barrier_rank_id),
                load_barrier_world_size=int(barrier_world_size),
                canonical_neff_cache_key=_product_canonical_neff_cache_key(
                    "dsv4_product_inverse_rope_attn_out_dp_flat_token_range",
                    "v1",
                    key,
                ),
            )
        defer_collective_load = int(tp_degree) > 1
        kernel = self._cached_product_kernel(
            bucket=bucket,
            cache_name="attention_inverse_rope_out_dp_flat_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_attention.attention_inverse_rope_out_proj_dp_flat_dynamic_token_range_from_freq_table_fn,
                _sample_array(o, fallback_dtype=np.float32),
                _sample_array(wo_a, fallback_dtype=np.float32),
                _sample_array(wo_b, fallback_dtype=np.float32),
                _sample_array(cos_table, fallback_dtype=np.float32),
                _sample_array(sin_table, fallback_dtype=np.float32),
                _sample_array(positions, fallback_dtype=np.int32),
                _sample_array(token_start, fallback_dtype=np.int32),
                _sample_array(token_count, fallback_dtype=np.int32),
                rope_head_dim=int(rope_head_dim),
                n_groups=int(n_groups),
                rows=rows_i,
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False if defer_collective_load else load,
                **compile_kwargs,
            ),
        )
        return _resolve_product_kernel_for_load(
            kernel,
            build_dir=self.build_dir,
            load=load,
        )

    def _run_product_attention_out_flat_hidden(
        self,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        *,
        n_groups: int,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple[tuple[int, ...], ...] = (),
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        dp_flat_context = getattr(self, "_attention_out_dp_flat_context", None)
        if dp_flat_context is not None:
            context = dict(dp_flat_context)
            compile_seqlen = context.pop("compile_seqlen", None)
            flat_token_range = bool(context.pop("flat_token_range", False))
            token_start = int(context.pop("token_start", 0))
            token_count = int(context.pop("token_count", int(bsz) * int(seqlen)))
            if flat_token_range:
                return self._run_product_attention_out_dp_flat_token_range(
                    o,
                    wo_a,
                    wo_b,
                    n_groups=int(n_groups),
                    rows=int(context["rows"]),
                    token_start=token_start,
                    token_count=token_count,
                    hidden_size=int(hidden_size),
                    tp_degree=int(tp_degree),
                    tp_replica_groups=tuple(
                        tuple(int(rank) for rank in group)
                        for group in tp_replica_groups
                    ),
                )
            return self._run_product_attention_out_dp_flat(
                o,
                wo_a,
                wo_b,
                n_groups=int(n_groups),
                bsz=int(bsz),
                seqlen=(
                    int(compile_seqlen) if compile_seqlen is not None else int(seqlen)
                ),
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=tuple(
                    tuple(int(rank) for rank in group) for group in tp_replica_groups
                ),
                **context,
            )
        del _nkipy_output_tensors
        raise RuntimeError(
            "DSV4 product attention_out_flat_hidden requires a fused "
            "DP-flat context; standalone attention-out kernels are not part "
            "of the product path"
        )

    def _run_product_attention_out_dp_flat_token_range(
        self,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        *,
        n_groups: int,
        rows: int,
        token_start: int,
        token_count: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple[tuple[int, ...], ...] = (),
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        bucket = self._require_active_product_bucket(
            where="attention_out_dp_flat_token_range"
        )
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        _require_product_device_value(o, where="attention_out_dp_flat_token_range/o")
        _require_product_device_value(
            wo_a,
            where="attention_out_dp_flat_token_range/wo_a",
        )
        _require_product_device_value(
            wo_b,
            where="attention_out_dp_flat_token_range/wo_b",
        )
        token_start_dev, token_count_dev = self._sync_attention_dp_token_range(
            bucket,
            token_start=int(token_start),
            token_count=int(token_count),
        )
        base = None
        fns = getattr(self, "graph", {})
        if isinstance(fns, dict):
            base = fns.get("attention_out_flat")
        kernel = self._attention_out_dp_flat_token_range_kernel_for(
            bucket,
            o,
            wo_a,
            wo_b,
            token_start_dev,
            token_count_dev,
            n_groups=int(n_groups),
            rows=int(rows),
            hidden_size=int(hidden_size),
            tp_degree=int(tp_degree),
            tp_replica_groups=groups,
            base=base,
        )
        outputs = dict(_nkipy_output_tensors or {})
        out = outputs.get("output0")
        if out is None:
            out = self._bucket_scratch(
                bucket,
                "attention_out_dp_flat",
                (int(rows), int(hidden_size)),
                np.float32,
            )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "o": o,
                "wo_a": wo_a,
                "wo_b": wo_b,
                "token_start": token_start_dev,
                "token_count": token_count_dev,
            },
            outputs={"output0": out},
            unload_after_call=False,
        )
        return out

    def _run_product_attention_out_dp_flat(
        self,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        *,
        n_groups: int,
        bsz: int,
        seqlen: int,
        batch_size: int,
        start: int,
        size: int,
        rows: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple[tuple[int, ...], ...] = (),
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        bucket = self._require_active_product_bucket(where="attention_out_dp_flat")
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        _require_product_device_value(o, where="attention_out_dp_flat/o")
        _require_product_device_value(wo_a, where="attention_out_dp_flat/wo_a")
        _require_product_device_value(wo_b, where="attention_out_dp_flat/wo_b")
        base = None
        fns = getattr(self, "graph", {})
        if isinstance(fns, dict):
            base = fns.get("attention_out_flat")
        kernel = self._attention_out_dp_flat_kernel_for(
            bucket,
            o,
            wo_a,
            wo_b,
            n_groups=int(n_groups),
            bsz=int(bsz),
            seqlen=int(seqlen),
            batch_size=int(batch_size),
            start=int(start),
            size=int(size),
            rows=int(rows),
            hidden_size=int(hidden_size),
            tp_degree=int(tp_degree),
            tp_replica_groups=groups,
            base=base,
        )
        outputs = dict(_nkipy_output_tensors or {})
        out = outputs.get("output0")
        if out is None:
            out = self._bucket_scratch(
                bucket,
                "attention_out_dp_flat",
                (int(rows), int(hidden_size)),
                np.float32,
            )
        kernel(
            inputs={"o": o, "wo_a": wo_a, "wo_b": wo_b},
            outputs={"output0": out},
        )
        return out

    def _run_product_attention_inverse_rope_out_flat_hidden_from_freq_table(
        self,
        o: Any,
        wo_a: Any,
        wo_b: Any,
        cos_table: Any,
        sin_table: Any,
        positions: Any,
        *,
        rope_head_dim: int,
        n_groups: int,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        tp_degree: int = 1,
        tp_replica_groups: tuple[tuple[int, ...], ...] = (),
        _nkipy_output_tensors: dict[str, Any] | None = None,
    ) -> Any:
        dp_flat_context = getattr(self, "_attention_out_dp_flat_context", None)
        if dp_flat_context is None:
            raise RuntimeError(
                "DSV4 product fused inverse-RoPE attention output requires "
                "the DP-flat context; standalone inverse-RoPE/output projection "
                "is not a product path"
            )
        bucket = self._require_active_product_bucket(
            where="inverse_rope_attention_out_dp_flat"
        )
        o = _as_product_device_input(o, name="dsv4_inverse_rope_attention_out")
        _require_product_device_value(wo_a, where="inverse_rope_attention_out/wo_a")
        _require_product_device_value(wo_b, where="inverse_rope_attention_out/wo_b")
        cos_table, sin_table = self._product_freq_tables_for(
            cos_table,
            sin_table,
            name="attention",
        )
        o_shape = tuple(int(dim) for dim in getattr(o, "shape", ()))
        if not o_shape:
            raise RuntimeError(
                "DSV4 product fused inverse-RoPE attention output expects "
                "non-scalar attention output"
            )
        positions = self._product_freq_positions_for(
            bucket,
            positions,
            rows=int(o_shape[0]),
        )
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        context = dict(dp_flat_context)
        compile_seqlen = context.pop("compile_seqlen", None)
        flat_token_range = bool(context.pop("flat_token_range", False))
        token_start_i = int(context.pop("token_start", 0))
        token_count_i = int(context.pop("token_count", int(bsz) * int(seqlen)))
        batch_size_i = int(context["batch_size"])
        start_i = int(context["start"])
        size_i = int(context["size"])
        rows_i = int(context["rows"])
        aligned_lane_token_range = bool(
            flat_token_range
            and int(size_i) > 0
            and int(token_start_i) == int(start_i) * int(seqlen)
            and int(token_count_i) == int(size_i) * int(seqlen)
            and int(o_shape[0]) == int(token_count_i)
        )
        run_seqlen = int(compile_seqlen) if compile_seqlen is not None else int(seqlen)
        base = None
        fns = getattr(self, "graph", {})
        if isinstance(fns, dict):
            base = fns.get("attention_out_flat")
        _sampled_warmup_trace(
            "attention_out inverse_rope_dp_flat runner start "
            f"o_shape={o_shape} flat_token_range={bool(flat_token_range)} "
            f"aligned_lane={bool(aligned_lane_token_range)} "
            f"bsz={int(bsz)} seqlen={int(seqlen)} run_seqlen={int(run_seqlen)} "
            f"batch_size={int(batch_size_i)} start={int(start_i)} "
            f"size={int(size_i)} rows={int(rows_i)} "
            f"token_start={int(token_start_i)} token_count={int(token_count_i)}",
        )
        inputs = {
            "o": o,
            "wo_a": wo_a,
            "wo_b": wo_b,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "positions": positions,
        }
        if flat_token_range:
            token_start_dev, token_count_dev = self._sync_attention_dp_token_range(
                bucket,
                token_start=token_start_i,
                token_count=token_count_i,
            )
            _sampled_warmup_trace(
                "attention_out inverse_rope_dp_flat kernel_for start "
                f"mode=token_range o_shape={o_shape} rows={int(rows_i)}",
            )
            kernel = (
                self._attention_inverse_rope_out_dp_flat_token_range_table_kernel_for(
                    bucket,
                    o,
                    wo_a,
                    wo_b,
                    cos_table,
                    sin_table,
                    positions,
                    token_start_dev,
                    token_count_dev,
                    rope_head_dim=int(rope_head_dim),
                    n_groups=int(n_groups),
                    rows=rows_i,
                    hidden_size=int(hidden_size),
                    tp_degree=int(tp_degree),
                    tp_replica_groups=groups,
                    base=base,
                    load=False,
                )
            )
            _sampled_warmup_trace(
                "attention_out inverse_rope_dp_flat kernel_for done "
                f"mode=token_range o_shape={o_shape}",
            )
            inputs["token_start"] = token_start_dev
            inputs["token_count"] = token_count_dev
        else:
            lane_start = self._sync_attention_dp_lane_start(bucket, start_i)
            _sampled_warmup_trace(
                "attention_out inverse_rope_dp_flat kernel_for start "
                f"mode=lane_start o_shape={o_shape} rows={int(rows_i)}",
            )
            kernel = self._attention_inverse_rope_out_dp_flat_table_kernel_for(
                bucket,
                o,
                wo_a,
                wo_b,
                cos_table,
                sin_table,
                positions,
                lane_start,
                rope_head_dim=int(rope_head_dim),
                n_groups=int(n_groups),
                bsz=int(bsz),
                seqlen=run_seqlen,
                batch_size=batch_size_i,
                start=start_i,
                size=size_i,
                rows=rows_i,
                hidden_size=int(hidden_size),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                base=base,
                load=False,
            )
            _sampled_warmup_trace(
                "attention_out inverse_rope_dp_flat kernel_for done "
                f"mode=lane_start o_shape={o_shape}",
            )
            inputs["lane_start"] = lane_start
        outputs = dict(_nkipy_output_tensors or {})
        out = outputs.get("output0")
        if out is None:
            out = self._bucket_scratch(
                bucket,
                "attention_inverse_rope_out_dp_flat",
                (rows_i, int(hidden_size)),
                np.float32,
            )
        _sampled_warmup_trace(
            "attention_out inverse_rope_dp_flat kernel_run start "
            f"o_shape={o_shape} rows={int(rows_i)}",
        )
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=inputs,
            outputs={"output0": out},
            unload_after_call=False,
        )
        _sampled_warmup_trace(
            f"attention_out inverse_rope_dp_flat kernel_run done o_shape={o_shape}",
        )
        return out
