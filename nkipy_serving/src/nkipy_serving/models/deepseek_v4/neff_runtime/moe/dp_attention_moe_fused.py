"""Fused DP-attention plus MoE product kernels for DSV4 execution."""

from __future__ import annotations

import hashlib
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _compile_product_kernel,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _blockwise_moe_ep_tp_groups,
    _is_device_value,
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
    _TensorSpec,
)


class Dsv4ProductDpAttentionMoeFusedMixin:
    def _precompile_dp_attention_moe_concat_helpers(
        self,
        bucket: Dsv4ProductBucket,
        *,
        token_bucket: int,
        batch_size: int,
        seqlen: int,
        is_decode: bool,
    ) -> None:
        """Precompile fused DP-attention post/pre + router/dispatch variants."""
        groups = tuple(
            tuple(int(rank) for rank in group)
            for group in tuple(self._dp_attention_replica_groups() or ())
        )
        if not groups or all(len(group) <= 1 for group in groups):
            return
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            return
        # The flat reduce buffer `x` is shared across every promoted compile_bsz
        # candidate below; each fused NEFF reshapes x[:compile_bsz*seq]. Size it
        # for the promoted batch (decode canonicalizes onto one max_requests-wide
        # NEFF) so the largest candidate fits. No-op when max_requests==1.
        compile_batch_size = self._product_compile_batch_size(
            bucket,
            bsz=bsz,
            seqlen=seq,
        )
        reduce_rows = self._dp_attention_reduce_rows_for_step(
            token_bucket=int(token_bucket),
            total_tokens=bsz * seq,
            batch_size=bsz,
            seqlen=seq,
            is_decode=bool(is_decode),
            compile_batch_size=int(compile_batch_size),
        )
        x = _TensorSpec(
            (int(reduce_rows), int(hidden_size)),
            np.dtype(np.float32),
        )
        compile_bsz_canonical, compile_seq = self._dp_attention_post_pre_compile_shape(
            bucket,
            bsz=bsz,
            seqlen=seq,
            rows=int(reduce_rows),
            dispatch_context={"is_decode": bool(is_decode)},
        )
        args = getattr(self.runtime_surface, "args", None)
        hc_mult = int(getattr(args, "hc_mult", 0) or 0)
        if hc_mult <= 0:
            return
        compile_shape_candidates = [(int(compile_bsz_canonical), int(compile_seq))]
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()) or ())
        for compile_bsz, compile_seq_i in compile_shape_candidates:
            if self._should_split_dp_attention_post_pre(x, seqlen=int(compile_seq_i)):
                continue
            for compile_bsz in (int(compile_bsz),):
                residual = _TensorSpec(
                    (
                        int(compile_bsz),
                        int(compile_seq_i),
                        int(hc_mult),
                        int(hidden_size),
                    ),
                    np.dtype(ml_dtypes.bfloat16),
                )
                post = _TensorSpec(
                    (int(compile_bsz), int(compile_seq_i), int(hc_mult)),
                    np.dtype(np.float32),
                )
                comb = _TensorSpec(
                    (
                        int(compile_bsz),
                        int(compile_seq_i),
                        int(hc_mult),
                        int(hc_mult),
                    ),
                    np.dtype(np.float32),
                )
                for layer_id, block in enumerate(blocks):
                    required = (
                        getattr(block, "hc_ffn_fn", None),
                        getattr(block, "hc_ffn_scale", None),
                        getattr(block, "hc_ffn_base", None),
                        getattr(block, "ffn_norm", None),
                    )
                    if any(value is None for value in required):
                        continue
                    dispatch_context = self._make_dp_attention_moe_dispatch_context(
                        block,
                        input_ids=bucket.input_ids_dev,
                        layer_id=int(layer_id),
                        is_decode=bool(is_decode),
                        token_bucket=int(token_bucket),
                        bsz=int(compile_bsz),
                        seqlen=int(compile_seq_i),
                        hidden_size=int(hidden_size),
                    )
                    if dispatch_context is None:
                        continue
                    blockwise_payload = self._dp_attention_moe_blockwise_payload(
                        dispatch_context,
                        rows=int(dispatch_context["rows"]),
                        hidden_size=int(hidden_size),
                    )
                    if blockwise_payload is None:
                        kernel = self._dp_attention_moe_dispatch_kernel_for(
                            bucket,
                            x,
                            residual,
                            post,
                            comb,
                            block.hc_ffn_fn,
                            block.hc_ffn_scale,
                            block.hc_ffn_base,
                            block.ffn_norm,
                            dispatch_context=dispatch_context,
                            replica_groups=groups,
                            bsz=int(compile_bsz),
                            seqlen=int(compile_seq_i),
                            hidden_size=int(hidden_size),
                            hc_mult=int(hc_mult),
                            sinkhorn_iters=int(getattr(args, "hc_sinkhorn_iters")),
                            norm_eps=float(getattr(args, "norm_eps")),
                            hc_eps=float(getattr(args, "hc_eps")),
                        )
                    else:
                        kernel = self._dp_attention_moe_blockwise_kernel_for(
                            bucket,
                            x,
                            residual,
                            post,
                            comb,
                            block.hc_ffn_fn,
                            block.hc_ffn_scale,
                            block.hc_ffn_base,
                            block.ffn_norm,
                            dispatch_context=dispatch_context,
                            blockwise_payload=blockwise_payload,
                            replica_groups=groups,
                            bsz=int(compile_bsz),
                            seqlen=int(compile_seq_i),
                            hidden_size=int(hidden_size),
                            hc_mult=int(hc_mult),
                            sinkhorn_iters=int(getattr(args, "hc_sinkhorn_iters")),
                            norm_eps=float(getattr(args, "norm_eps")),
                            hc_eps=float(getattr(args, "hc_eps")),
                        )
                    if self._keep_dp_attention_pipeline_collectives_loaded():
                        self._load_resident_product_kernel(kernel)

    def _run_dp_attention_all_reduce_post_pre_moe_dispatch(
        self,
        x: Any,
        *,
        dispatch_context: dict[str, Any],
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
    ) -> tuple[Any, Any, Any, Any]:
        executor = self.runtime_surface
        bucket = self._require_active_product_bucket(
            where="DP-attention post/pre MoE dispatch"
        )
        kind = str(dispatch_context.get("kind", ""))
        weight = dispatch_context.get("weight")
        _require_product_device_value(weight, where="dp_attention_moe_dispatch/weight")
        if kind == "hash_no_bias":
            input_ids = dispatch_context.get("input_ids")
            tid2eid = dispatch_context.get("tid2eid")
            _require_product_device_value(
                input_ids,
                where="dp_attention_moe_dispatch/input_ids",
            )
            _require_product_device_value(
                tid2eid,
                where="dp_attention_moe_dispatch/tid2eid",
            )
        elif kind == "learned_with_bias":
            bias = dispatch_context.get("bias")
            _require_product_device_value(
                bias,
                where="dp_attention_moe_dispatch/bias",
            )
        elif kind != "learned_no_bias":
            raise RuntimeError(f"unsupported DP-attention MoE dispatch kind: {kind}")

        hc_mult = int(executor.args.hc_mult)
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
        rows = int(dispatch_context["rows"])
        layer_id = int(dispatch_context["layer_id"])
        is_decode = bool(dispatch_context["is_decode"])
        blockwise_payload = self._dp_attention_moe_blockwise_payload(
            dispatch_context,
            rows=rows,
            hidden_size=int(hidden_size),
        )
        if blockwise_payload is None:
            kernel = self._dp_attention_moe_dispatch_kernel_for(
                bucket,
                x,
                residual_kernel,
                post_kernel,
                comb_kernel,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
                dispatch_context=dispatch_context,
                replica_groups=replica_groups,
                bsz=int(compile_bsz),
                seqlen=int(compile_seqlen),
                hidden_size=int(hidden_size),
                hc_mult=hc_mult,
                sinkhorn_iters=int(executor.args.hc_sinkhorn_iters),
                norm_eps=float(executor.args.norm_eps),
                hc_eps=float(executor.args.hc_eps),
            )
        else:
            kernel = self._dp_attention_moe_blockwise_kernel_for(
                bucket,
                x,
                residual_kernel,
                post_kernel,
                comb_kernel,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
                dispatch_context=dispatch_context,
                blockwise_payload=blockwise_payload,
                replica_groups=replica_groups,
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
                "DSV4 product DP-attention MoE dispatch expects residual "
                f"[batch, seq, hc, hidden], got {residual_shape}"
            )
        h = self._alloc_mhc_post_output(
            bucket,
            residual_shape=residual_shape,
            residual=residual,
            x=x,
        )
        dtype = _value_dtype(residual, fallback=ml_dtypes.bfloat16)
        topk = (
            int(getattr(dispatch_context.get("tid2eid"), "shape", (0, 0))[-1])
            if kind == "hash_no_bias"
            else int(dispatch_context["topk"])
        )
        flat_hidden = self._moe_ep_output_for_layer(
            layer_id=layer_id,
            is_decode=is_decode,
            rows=rows,
            dim=int(hidden_size),
        )
        if flat_hidden is None:
            flat_hidden = self._bucket_scratch(
                bucket,
                "dp_attention_moe_dispatch_hidden",
                (rows, int(hidden_size)),
                ml_dtypes.bfloat16,
            )
        shared_hidden = self._head_hidden_alias_for(
            shape=(1, rows, int(hidden_size)),
            dtype=dtype,
        )
        if shared_hidden is None:
            shared_hidden = self._bucket_scratch(
                bucket,
                "dp_attention_moe_dispatch_shared_hidden",
                (1, rows, int(hidden_size)),
                dtype,
            )
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
        if blockwise_payload is None:
            outputs.update(
                {
                    "output4": flat_hidden,
                    "output5": self._bucket_scratch(
                        bucket,
                        "dp_attention_moe_dispatch_weights",
                        (rows, topk),
                        np.float32,
                    ),
                    "output6": self._bucket_scratch(
                        bucket,
                        "dp_attention_moe_dispatch_indices",
                        (rows, topk),
                        np.int32,
                    ),
                    "output7": shared_hidden,
                }
            )
        elif is_decode:
            outputs.update(
                {
                    "output4": blockwise_payload["moe_output"],
                    "output5": shared_hidden,
                }
            )
        elif bool(blockwise_payload.get("prefill_beta2", False)):
            outputs.update(
                {
                    "output4": blockwise_payload["moe_output"],
                    "output5": shared_hidden,
                }
            )
        else:
            outputs.update(
                {
                    "moe_output": blockwise_payload["moe_output"],
                    "output5": shared_hidden,
                }
            )
        inputs = {
            "x": x,
            "residual": residual_kernel,
            "post": post_kernel,
            "comb": comb_kernel,
            "hc_fn": hc_fn,
            "hc_scale": hc_scale,
            "hc_base": hc_base,
            "norm_weight": norm_weight,
            "weight": weight,
        }
        if kind == "hash_no_bias":
            inputs["input_ids"] = dispatch_context["input_ids"]
            inputs["tid2eid"] = dispatch_context["tid2eid"]
        elif kind == "learned_with_bias":
            inputs["bias"] = dispatch_context["bias"]
        if blockwise_payload is not None:
            if not is_decode:
                if bool(blockwise_payload.get("prefill_beta2", False)):
                    inputs["moe_output"] = blockwise_payload["moe_output"]
                else:
                    inputs["moe_output.must_alias_input"] = blockwise_payload[
                        "moe_output"
                    ]
            inputs.update(
                {
                    "ep_start": blockwise_payload["ep_start"],
                    "gate_up_proj_weight": blockwise_payload["gate_up"],
                    "gate_up_bias_plus1_T_hbm": blockwise_payload["gate_up_bias"],
                    "down_proj_weight": blockwise_payload["down"],
                    "down_bias_broadcasted_hbm": blockwise_payload["down_bias"],
                }
            )
        # Collective loads allocate CCOM bootstrap resources. The fused dispatch
        # handle stays resident across the per-layer loop so full-model runs do
        # not churn load/unload state.
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=inputs,
            outputs=outputs,
            unload_after_call=False,
        )
        pending = getattr(self, "_pending_fused_moe_dispatch", None)
        if pending is None:
            pending = {}
            self._pending_fused_moe_dispatch = pending
        if blockwise_payload is None:
            pending[(layer_id, is_decode)] = (
                outputs["output4"],
                outputs["output5"],
                outputs["output6"],
                outputs["output7"],
            )
        else:
            pending[(layer_id, is_decode)] = {
                "routed": blockwise_payload["moe_output"],
                "shared_hidden": shared_hidden,
            }
        return self._alias_mhc_post_pre_outputs(
            outputs,
            bsz=bsz,
            seqlen=seqlen,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )

    def _dp_attention_moe_blockwise_payload(
        self,
        dispatch_context: dict[str, Any],
        *,
        rows: int,
        hidden_size: int,
    ) -> dict[str, Any] | None:
        state = getattr(self, "blockwise_moe_state", None)
        if state is None:
            return None
        layer_id = int(dispatch_context["layer_id"])
        layers = tuple(getattr(state, "layers", ()))
        if layer_id < 0 or layer_id >= len(layers):
            return None
        layer = layers[layer_id]
        if int(getattr(layer, "n_local_experts", 0) or 0) <= 0:
            return None

        # If a distributed MoE reduce is required, the existing shared-restore
        # graph owns it. Do not silently drop a reduce in the fused product path.
        moe_groups = _blockwise_moe_ep_tp_groups(state)
        if not moe_groups and (
            int(getattr(state, "ep_degree", 1) or 1) > 1
            or int(getattr(state, "tp_degree", 1) or 1) > 1
        ):
            return None

        gate_up = getattr(layer, "gate_up_w", None)
        down = getattr(layer, "down_w", None)
        gate_up_bias = getattr(layer, "gate_up_bias", None)
        if not (
            _is_device_value(gate_up)
            and _is_device_value(down)
            and _is_device_value(gate_up_bias)
        ):
            return None

        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            _down_bias_device_tensor,
        )
        from nkipy_serving.ops.moe.blockwise_index import (
            BLOCK_SIZE as MOE_BLOCK_SIZE,
        )
        from nkipy_serving.ops.moe.blockwise_index import (
            get_n_blocks,
        )
        from nkipy_serving.ops.moe.device_schedule import (
            choose_indexed_flatten_f_len,
            logical_nc_config,
        )

        has_down_bias = getattr(layer, "down_bias_bc", None) is not None
        down_bias = _down_bias_device_tensor(
            getattr(layer, "down_bias_bc", None),
            tensor_cls=_get_device_tensor_cls(),
        )
        moe_output, _moe_ep_output, _moe_tp_output = self._moe_outputs_for(
            layer_id,
            is_decode=bool(dispatch_context["is_decode"]),
        )
        out_shape = tuple(int(dim) for dim in getattr(moe_output, "shape", ()))
        if out_shape != (int(rows), int(hidden_size)):
            return None

        E = int(getattr(layer, "n_local_experts"))
        if not bool(dispatch_context["is_decode"]):
            max_prefill_rows = int(
                getattr(
                    self,
                    "product_prefill_moe_blockwise_fusion_max_rows",
                    0,
                )
                or 0
            )
            if max_prefill_rows > 0 and int(rows) > max_prefill_rows:
                _product_warmup_trace(
                    _product_executor_coord(self),
                    "prefill blockwise MoE concat skipped "
                    f"layer={layer_id} rows={int(rows)} "
                    f"max_rows={max_prefill_rows}",
                )
                return None
        ep_start = _get_device_tensor_cls().from_numpy(
            np.asarray([int(getattr(state, "ep_rank", 0)) * E], dtype=np.int32),
            name="moe_ep_start",
        )
        clamps = (
            state._v4_clamp_kwargs()
            if hasattr(state, "_v4_clamp_kwargs")
            else {
                "gate_clamp_upper": 7.0,
                "gate_clamp_lower": None,
                "up_clamp_upper": 8.0,
                "up_clamp_lower": -6.0,
            }
        )
        payload = {
            "layer": layer,
            "moe_output": moe_output,
            "ep_start": ep_start,
            "gate_up": gate_up,
            "gate_up_bias": gate_up_bias,
            "down": down,
            "down_bias": down_bias,
            "has_down_bias": bool(has_down_bias),
            "prefill_beta2": (
                not bool(dispatch_context["is_decode"]) and not bool(has_down_bias)
            ),
            "gate_clamp_upper": float(clamps.get("gate_clamp_upper", 7.0)),
            "gate_clamp_lower": clamps.get("gate_clamp_lower", None),
            "up_clamp_upper": float(clamps.get("up_clamp_upper", 8.0)),
            "up_clamp_lower": float(clamps.get("up_clamp_lower", -6.0)),
            "local_num_experts": E,
            "experts_per_token": int(getattr(state, "experts_per_token", 0) or 0),
            "moe_replica_groups": tuple(tuple(int(r) for r in g) for g in moe_groups),
        }
        if not bool(dispatch_context["is_decode"]):
            experts_per_token = int(payload["experts_per_token"])
            if experts_per_token <= 0:
                return None
            num_blocks, num_static_blocks = get_n_blocks(
                int(rows),
                experts_per_token,
                E,
            )
            f_len = choose_indexed_flatten_f_len(int(rows))
            payload.update(
                {
                    "num_blocks": int(num_blocks),
                    "num_static_blocks": int(num_static_blocks),
                    "f_len": int(f_len),
                    "output_len": int(num_blocks) * int(MOE_BLOCK_SIZE) + int(rows),
                    "logical_nc_config": int(logical_nc_config()),
                }
            )
        return payload

    def _dp_attention_moe_blockwise_kernel_for(
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
        dispatch_context: dict[str, Any],
        blockwise_payload: dict[str, Any],
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
        kind = str(dispatch_context["kind"])
        phase = "decode" if bool(dispatch_context["is_decode"]) else "prefill"
        common_key = (
            phase,
            kind,
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
            tuple(int(dim) for dim in getattr(dispatch_context["weight"], "shape", ())),
            str(getattr(dispatch_context["weight"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            tuple(
                int(dim) for dim in getattr(blockwise_payload["gate_up"], "shape", ())
            ),
            str(getattr(blockwise_payload["gate_up"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            tuple(
                int(dim)
                for dim in getattr(blockwise_payload["gate_up_bias"], "shape", ())
            ),
            str(getattr(blockwise_payload["gate_up_bias"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            tuple(int(dim) for dim in getattr(blockwise_payload["down"], "shape", ())),
            str(getattr(blockwise_payload["down"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            tuple(
                int(dim) for dim in getattr(blockwise_payload["down_bias"], "shape", ())
            ),
            str(getattr(blockwise_payload["down_bias"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            int(bsz),
            int(seqlen),
            int(hidden_size),
            int(dispatch_context["rows"]),
            str(dispatch_context["score_func"]),
            float(dispatch_context["route_scale"]),
            bool(dispatch_context["normalize"]),
            groups,
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
            int(blockwise_payload["local_num_experts"]),
            int(blockwise_payload["experts_per_token"]),
            bool(blockwise_payload["has_down_bias"]),
            bool(blockwise_payload.get("prefill_beta2", False)),
            float(blockwise_payload["gate_clamp_upper"]),
            blockwise_payload["gate_clamp_lower"],
            float(blockwise_payload["up_clamp_upper"]),
            float(blockwise_payload["up_clamp_lower"]),
        )
        compile_args: list[Any] = [
            _sample_array(x, fallback_dtype=np.float32),
            _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(post, fallback_dtype=np.float32),
            _sample_array(comb, fallback_dtype=np.float32),
            _sample_array(hc_fn, fallback_dtype=np.float32),
            _sample_array(hc_scale, fallback_dtype=np.float32),
            _sample_array(hc_base, fallback_dtype=np.float32),
            _sample_array(norm_weight, fallback_dtype=np.float32),
        ]
        compile_kwargs: dict[str, Any] = {
            "replica_groups": groups,
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "hc_mult": int(hc_mult),
            "sinkhorn_iters": int(sinkhorn_iters),
            "norm_eps": float(norm_eps),
            "hc_eps": float(hc_eps),
            "rows": int(dispatch_context["rows"]),
            "score_func": str(dispatch_context["score_func"]),
            "route_scale": float(dispatch_context["route_scale"]),
            "normalize": bool(dispatch_context["normalize"]),
            "has_down_bias": bool(blockwise_payload["has_down_bias"]),
            "gate_clamp_upper": float(blockwise_payload["gate_clamp_upper"]),
            "gate_clamp_lower": blockwise_payload["gate_clamp_lower"],
            "up_clamp_upper": float(blockwise_payload["up_clamp_upper"]),
            "up_clamp_lower": float(blockwise_payload["up_clamp_lower"]),
        }
        name_kind: str
        if kind == "hash_no_bias":
            key = (
                *common_key,
                tuple(
                    int(dim)
                    for dim in getattr(dispatch_context["input_ids"], "shape", ())
                ),
                str(getattr(dispatch_context["input_ids"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
                tuple(
                    int(dim)
                    for dim in getattr(dispatch_context["tid2eid"], "shape", ())
                ),
                str(getattr(dispatch_context["tid2eid"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
            )
            graph_fn = (
                graph_moe.dp_attention_all_reduce_post_pre_hash_moe_blockwise_decode_no_bias_fn
                if phase == "decode"
                else graph_moe.dp_attention_all_reduce_post_pre_hash_moe_blockwise_prefill_no_bias_fn
            )
            name_kind = "hash_moe_blockwise_no_bias"
            compile_args.extend(
                [
                    _sample_array(
                        dispatch_context["input_ids"], fallback_dtype=np.int32
                    ),
                    _sample_array(
                        dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                    ),
                    _sample_array(dispatch_context["tid2eid"], fallback_dtype=np.int32),
                ]
            )
        elif kind == "learned_no_bias":
            key = (
                *common_key,
                int(dispatch_context["topk"]),
                int(dispatch_context["n_experts"]),
            )
            graph_fn = (
                graph_moe.dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_no_bias_fn
                if phase == "decode"
                else graph_moe.dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_no_bias_fn
            )
            name_kind = "learned_moe_blockwise_no_bias"
            compile_args.append(
                _sample_array(
                    dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                )
            )
            compile_kwargs.update(
                {
                    "topk": int(dispatch_context["topk"]),
                    "n_experts": int(dispatch_context["n_experts"]),
                }
            )
        elif kind == "learned_with_bias":
            key = (
                *common_key,
                tuple(
                    int(dim) for dim in getattr(dispatch_context["bias"], "shape", ())
                ),
                str(getattr(dispatch_context["bias"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
                int(dispatch_context["topk"]),
                int(dispatch_context["n_experts"]),
            )
            graph_fn = (
                graph_moe.dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_with_bias_fn
                if phase == "decode"
                else graph_moe.dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_with_bias_fn
            )
            name_kind = "learned_moe_blockwise_with_bias"
            compile_args.extend(
                [
                    _sample_array(
                        dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                    ),
                    _sample_array(dispatch_context["bias"], fallback_dtype=np.float32),
                ]
            )
            compile_kwargs.update(
                {
                    "topk": int(dispatch_context["topk"]),
                    "n_experts": int(dispatch_context["n_experts"]),
                }
            )
        else:
            raise RuntimeError(f"unsupported DP-attention MoE blockwise kind: {kind}")

        if phase == "prefill":
            key = (
                *key,
                int(blockwise_payload["num_static_blocks"]),
                int(blockwise_payload["num_blocks"]),
                int(blockwise_payload["f_len"]),
                int(blockwise_payload["output_len"]),
                int(blockwise_payload["logical_nc_config"]),
            )
            compile_args.append(
                _sample_array(
                    blockwise_payload["moe_output"],
                    fallback_dtype=ml_dtypes.bfloat16,
                )
            )
            compile_kwargs.update(
                {
                    "num_static_blocks": int(blockwise_payload["num_static_blocks"]),
                    "token_bucket": int(dispatch_context["rows"]),
                    "local_num_experts": int(blockwise_payload["local_num_experts"]),
                    "experts_per_token": int(blockwise_payload["experts_per_token"]),
                    "num_blocks": int(blockwise_payload["num_blocks"]),
                    "f_len": int(blockwise_payload["f_len"]),
                    "output_len": int(blockwise_payload["output_len"]),
                    "logical_nc_config": int(blockwise_payload["logical_nc_config"]),
                }
            )
        compile_args.extend(
            [
                _sample_array(blockwise_payload["ep_start"], fallback_dtype=np.int32),
                _sample_array(
                    blockwise_payload["gate_up"], fallback_dtype=ml_dtypes.bfloat16
                ),
                _sample_array(
                    blockwise_payload["gate_up_bias"],
                    fallback_dtype=ml_dtypes.bfloat16,
                ),
                _sample_array(
                    blockwise_payload["down"], fallback_dtype=ml_dtypes.bfloat16
                ),
                _sample_array(
                    blockwise_payload["down_bias"],
                    fallback_dtype=ml_dtypes.bfloat16,
                ),
            ]
        )
        cached = bucket.kernel_caches["dp_attention_moe_blockwise_kernels"].get(key)
        if cached is not None:
            return cached
        rank_id, world_size = self._collective_graph_metadata(
            "dp_attention_all_reduce",
            where=f"DP-attention {name_kind} {phase}",
        )
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        E = int(blockwise_payload["local_num_experts"])
        phase_name = (
            f"{phase}_beta2"
            if phase == "prefill"
            and bool(blockwise_payload.get("prefill_beta2", False))
            else phase
        )
        name = (
            f"dsv4_product_dp_attention_ar_post_pre_{name_kind}_{phase_name}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
            f"r{int(dispatch_context['rows'])}_e{E}_{group_tag}"
        )
        compiler_args = str(getattr(self, "compiler_args", "") or "")
        if (
            phase == "prefill"
            and bool(blockwise_payload.get("prefill_beta2", False))
            and int(blockwise_payload["logical_nc_config"]) != 1
        ):
            compiler_args = (
                f"{compiler_args} --lnc {int(blockwise_payload['logical_nc_config'])}"
            ).strip()
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="dp_attention_moe_blockwise_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_fn,
                *compile_args,
                name=name,
                additional_compiler_args=compiler_args,
                build_dir=self.build_dir,
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load=False,
                load_barrier_name=name,
                canonical_neff_cache_key=(
                    _product_canonical_neff_cache_key(
                        "dsv4_product_dp_attention_moe_blockwise",
                        "v2",
                        key,
                    )
                ),
                **compile_kwargs,
            ),
        )

    def _dp_attention_moe_dispatch_kernel_for(
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
        dispatch_context: dict[str, Any],
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
        kind = str(dispatch_context["kind"])
        common_key = (
            kind,
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
            tuple(int(dim) for dim in getattr(dispatch_context["weight"], "shape", ())),
            str(getattr(dispatch_context["weight"], "dtype", "unknown")).replace(
                ".",
                "_",
            ),
            int(bsz),
            int(seqlen),
            int(hidden_size),
            int(dispatch_context["rows"]),
            str(dispatch_context["score_func"]),
            float(dispatch_context["route_scale"]),
            bool(dispatch_context["normalize"]),
            groups,
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
        )
        compile_args: list[Any] = [
            _sample_array(x, fallback_dtype=np.float32),
            _sample_array(residual, fallback_dtype=ml_dtypes.bfloat16),
            _sample_array(post, fallback_dtype=np.float32),
            _sample_array(comb, fallback_dtype=np.float32),
            _sample_array(hc_fn, fallback_dtype=np.float32),
            _sample_array(hc_scale, fallback_dtype=np.float32),
            _sample_array(hc_base, fallback_dtype=np.float32),
            _sample_array(norm_weight, fallback_dtype=np.float32),
        ]
        compile_kwargs: dict[str, Any] = {
            "replica_groups": groups,
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "hc_mult": int(hc_mult),
            "sinkhorn_iters": int(sinkhorn_iters),
            "norm_eps": float(norm_eps),
            "hc_eps": float(hc_eps),
            "rows": int(dispatch_context["rows"]),
            "score_func": str(dispatch_context["score_func"]),
            "route_scale": float(dispatch_context["route_scale"]),
            "normalize": bool(dispatch_context["normalize"]),
        }
        if kind == "hash_no_bias":
            cache = bucket.kernel_caches[
                "dp_attention_hash_moe_dispatch_no_bias_kernels"
            ]
            key = (
                *common_key,
                tuple(
                    int(dim)
                    for dim in getattr(dispatch_context["input_ids"], "shape", ())
                ),
                str(getattr(dispatch_context["input_ids"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
                tuple(
                    int(dim)
                    for dim in getattr(dispatch_context["tid2eid"], "shape", ())
                ),
                str(getattr(dispatch_context["tid2eid"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
            )
            graph_fn = (
                graph_moe.dp_attention_all_reduce_post_pre_hash_moe_dispatch_no_bias_fn
            )
            cache_name = "dp_attention_hash_moe_dispatch_no_bias_kernels"
            name_kind = "hash_moe_dispatch_no_bias"
            compile_args.extend(
                [
                    _sample_array(
                        dispatch_context["input_ids"], fallback_dtype=np.int32
                    ),
                    _sample_array(
                        dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                    ),
                    _sample_array(dispatch_context["tid2eid"], fallback_dtype=np.int32),
                ]
            )
        elif kind == "learned_no_bias":
            cache = bucket.kernel_caches[
                "dp_attention_learned_moe_dispatch_no_bias_kernels"
            ]
            key = (
                *common_key,
                int(dispatch_context["topk"]),
                int(dispatch_context["n_experts"]),
            )
            graph_fn = graph_moe.dp_attention_all_reduce_post_pre_learned_moe_dispatch_no_bias_fn
            cache_name = "dp_attention_learned_moe_dispatch_no_bias_kernels"
            name_kind = "learned_moe_dispatch_no_bias"
            compile_args.append(
                _sample_array(
                    dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                )
            )
            compile_kwargs.update(
                {
                    "topk": int(dispatch_context["topk"]),
                    "n_experts": int(dispatch_context["n_experts"]),
                }
            )
        elif kind == "learned_with_bias":
            cache = bucket.kernel_caches[
                "dp_attention_learned_moe_dispatch_with_bias_kernels"
            ]
            key = (
                *common_key,
                tuple(
                    int(dim) for dim in getattr(dispatch_context["bias"], "shape", ())
                ),
                str(getattr(dispatch_context["bias"], "dtype", "unknown")).replace(
                    ".",
                    "_",
                ),
                int(dispatch_context["topk"]),
                int(dispatch_context["n_experts"]),
            )
            graph_fn = graph_moe.dp_attention_all_reduce_post_pre_learned_moe_dispatch_with_bias_fn
            cache_name = "dp_attention_learned_moe_dispatch_with_bias_kernels"
            name_kind = "learned_moe_dispatch_with_bias"
            compile_args.extend(
                [
                    _sample_array(
                        dispatch_context["weight"], fallback_dtype=ml_dtypes.bfloat16
                    ),
                    _sample_array(dispatch_context["bias"], fallback_dtype=np.float32),
                ]
            )
            compile_kwargs.update(
                {
                    "topk": int(dispatch_context["topk"]),
                    "n_experts": int(dispatch_context["n_experts"]),
                }
            )
        else:
            raise RuntimeError(f"unsupported DP-attention MoE dispatch kind: {kind}")

        cached = cache.get(key)
        if cached is not None:
            return cached
        rank_id, world_size = self._collective_graph_metadata(
            "dp_attention_all_reduce",
            where=f"DP-attention {name_kind}",
        )
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        name = (
            f"dsv4_product_dp_attention_ar_post_pre_{name_kind}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(hidden_size)}_"
            f"r{int(dispatch_context['rows'])}_{group_tag}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name=cache_name,
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_fn,
                *compile_args,
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                cc_enabled=True,
                rank_id=int(rank_id),
                world_size=int(world_size),
                is_spmd=False,
                load=False,
                load_barrier_name=name,
                canonical_neff_cache_key=(
                    _product_canonical_neff_cache_key(
                        "dsv4_product_dp_attention_moe_dispatch",
                        "v1",
                        key,
                    )
                ),
                **compile_kwargs,
            ),
        )
