"""MoE dispatch product kernels for DSV4 execution."""

from __future__ import annotations

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
    _product_executor_coord,
    _product_warmup_trace,
    _require_product_device_value,
    _sample_array,
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _TensorSpec,
)


class Dsv4ProductMoeDispatchMixin:
    def _run_moe_layer(
        self,
        block: Any,
        y: Any,
        input_ids: np.ndarray,
        *,
        layer_id: int,
        is_decode: bool,
        token_bucket: int | None = None,
    ) -> Any:
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            MOE_BLOCK_SIZE,
            run_blockwise_moe_decode,
            run_blockwise_moe_prefill,
        )
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.execution import (
            _run_moe,
        )

        bucket = self._require_active_product_bucket(where="MoE layer")
        moe_output, moe_ep_output, moe_tp_output = self._moe_outputs_for(
            layer_id,
            is_decode=bool(is_decode),
        )
        moe_token_bucket = token_bucket
        precomputed_dispatch = None
        precomputed_routed = None
        pending_dispatch = getattr(self, "_pending_fused_moe_dispatch", None)
        if isinstance(pending_dispatch, dict):
            precomputed_dispatch = pending_dispatch.pop(
                (int(layer_id), bool(is_decode)),
                None,
            )
            if precomputed_dispatch is not None:
                if isinstance(precomputed_dispatch, dict):
                    precomputed_routed = precomputed_dispatch
                    precomputed_dispatch = None
                    dispatch_shape = tuple(
                        int(dim)
                        for dim in getattr(precomputed_routed["routed"], "shape", ())
                    )
                else:
                    dispatch_shape = tuple(
                        int(dim)
                        for dim in getattr(precomputed_dispatch[0], "shape", ())
                    )
                if len(dispatch_shape) >= 1 and dispatch_shape[0] > 0:
                    moe_token_bucket = int(dispatch_shape[0])
        previous_layer_id = getattr(self, "_active_moe_layer_id", None)
        previous_is_decode = getattr(self, "_active_moe_is_decode", None)
        self._active_moe_layer_id = int(layer_id)
        self._active_moe_is_decode = bool(is_decode)
        try:
            moe_collective_groups = _blockwise_moe_ep_tp_groups(
                self.blockwise_moe_state
            )
            moe_ops = {
                "force_shared_sequence_pad": True,
                "require_fused_router": True,
                "_profile_stage": (
                    lambda stage, **extra: self._profile_product_stage(
                        stage,
                        layer_id=int(layer_id),
                        final=bool(
                            getattr(
                                self,
                                "_shared_expert_restore_head_context",
                                None,
                            )
                            is not None
                        ),
                        token_bucket=(
                            None if moe_token_bucket is None else int(moe_token_bucket)
                        ),
                        **extra,
                    )
                ),
                "_hash_router_input_ids": bucket.input_ids_dev,
                "hash_moe_dispatch_no_bias": (
                    self._run_product_hash_moe_dispatch_no_bias
                ),
                "learned_moe_dispatch_no_bias": (
                    self._run_product_learned_moe_dispatch_no_bias
                ),
                "learned_moe_dispatch_with_bias": (
                    self._run_product_learned_moe_dispatch_with_bias
                ),
                "shared_expert_add_restore": (
                    self._run_product_shared_expert_add_restore
                ),
                "blockwise_moe_block_size": int(MOE_BLOCK_SIZE),
                "run_blockwise_moe_decode": run_blockwise_moe_decode,
                "run_blockwise_moe_prefill": run_blockwise_moe_prefill,
            }
            if moe_collective_groups:
                moe_ops["fuse_moe_collective_in_shared_restore"] = True
                moe_ops["_moe_collective_replica_groups"] = moe_collective_groups
                moe_ep_output = None
                moe_tp_output = None
            if precomputed_dispatch is not None:
                moe_ops["_precomputed_dispatch"] = precomputed_dispatch
            if precomputed_routed is not None:
                moe_ops["_precomputed_routed"] = precomputed_routed
            return _run_moe(
                self.graph,
                block.ffn,
                y,
                input_ids,
                blockwise_state=self.blockwise_moe_state,
                layer_id=int(layer_id),
                is_decode=bool(is_decode),
                build_dir=self.build_dir,
                token_bucket=moe_token_bucket,
                moe_output=moe_output,
                moe_ep_output=moe_ep_output,
                moe_tp_output=moe_tp_output,
                moe_ops=moe_ops,
            )
        finally:
            self._active_moe_layer_id = previous_layer_id
            self._active_moe_is_decode = previous_is_decode

    def precompile_lane_moe_helpers(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile product MoE router, blockwise, and fused DP-attn+MoE paths."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product lane MoE precompile could not infer hidden size"
            )
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            MOE_BLOCK_SIZE,
            precompile_blockwise_moe_all_reduce,
            precompile_blockwise_moe_decode_router,
            precompile_blockwise_moe_prefill_router,
        )

        n_tokens = bsz * seq
        if bool(is_decode):
            rows = self._decode_moe_rows_for_requests(n_tokens)
        else:
            rows = max(n_tokens, int(runtime_token_bucket))
            block_rows = int(MOE_BLOCK_SIZE)
            rows = ((max(rows, 1) + block_rows - 1) // block_rows) * block_rows
        x = _TensorSpec(
            (int(bsz), int(seq), int(hidden_size)),
            np.dtype(ml_dtypes.bfloat16),
        )
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()) or ())
        if not blocks:
            return
        state = getattr(self, "blockwise_moe_state", None)
        state_layers = tuple(getattr(state, "layers", ()) or ())
        fuses_blockwise_collective = bool(_blockwise_moe_ep_tp_groups(state))
        previous_bucket = getattr(self, "_active_product_bucket", None)
        self._active_product_bucket = bucket
        try:
            for layer_id, block in enumerate(blocks):
                moe = getattr(block, "ffn", None)
                gate = getattr(moe, "gate", None)
                weight = getattr(gate, "weight", None)
                dim = int(getattr(moe, "dim", hidden_size) or hidden_size)
                if gate is None or weight is None or dim <= 0:
                    continue
                score_func = str(getattr(gate, "score_func", "sigmoid"))
                route_scale = float(getattr(gate, "route_scale", 1.0))
                normalize = bool(score_func != "softmax")
                topk = 0
                if bool(getattr(gate, "is_hash", False)):
                    tid2eid = getattr(gate, "tid2eid", None)
                    if tid2eid is None or getattr(gate, "bias", None) is not None:
                        continue
                    topk = int(getattr(tid2eid, "shape", (0, 0))[-1])
                    if topk <= 0:
                        continue
                    self._product_moe_dispatch_kernel_for(
                        bucket,
                        kind="hash_no_bias",
                        x=x,
                        input_ids=bucket.input_ids_dev,
                        weight=weight,
                        tid2eid=tid2eid,
                        bias=None,
                        bsz=bsz,
                        seqlen=seq,
                        dim=dim,
                        rows=rows,
                        score_func=score_func,
                        topk=topk,
                        n_experts=0,
                        route_scale=route_scale,
                        normalize=normalize,
                    )
                else:
                    topk = int(getattr(gate, "topk", 0) or 0)
                    n_experts = int(getattr(weight, "shape", (0,))[0])
                    if topk <= 0 or n_experts <= 0:
                        continue
                    bias = getattr(gate, "bias", None)
                    kind = "learned_no_bias" if bias is None else "learned_with_bias"
                    self._product_moe_dispatch_kernel_for(
                        bucket,
                        kind=kind,
                        x=x,
                        input_ids=None,
                        weight=weight,
                        tid2eid=None,
                        bias=bias,
                        bsz=bsz,
                        seqlen=seq,
                        dim=dim,
                        rows=rows,
                        score_func=score_func,
                        topk=topk,
                        n_experts=n_experts,
                        route_scale=route_scale,
                        normalize=normalize,
                    )
                if (
                    state is not None
                    and int(layer_id) < len(state_layers)
                    and int(
                        getattr(state_layers[int(layer_id)], "n_local_experts", 0) or 0
                    )
                    > 0
                ):
                    weights_dev = self._bucket_scratch(
                        bucket,
                        "moe_dispatch_weights",
                        (int(rows), int(topk)),
                        np.float32,
                    )
                    router_dtype = getattr(weights_dev, "dtype", np.dtype(np.float32))
                    layer = state_layers[int(layer_id)]
                    if bool(is_decode) and int(rows) <= int(MOE_BLOCK_SIZE):
                        precompile_blockwise_moe_decode_router(
                            layer,
                            rows=int(rows),
                            topk=int(topk),
                            router_weights_dtype=router_dtype,
                            state=state,
                            artifacts_dir=self.build_dir,
                        )
                    else:
                        precompile_blockwise_moe_prefill_router(
                            layer,
                            rows=int(rows),
                            topk=int(topk),
                            router_weights_dtype=router_dtype,
                            state=state,
                            artifacts_dir=self.build_dir,
                        )
                    if not fuses_blockwise_collective:
                        precompile_blockwise_moe_all_reduce(
                            rows=int(rows),
                            hidden_size=int(hidden_size),
                            state=state,
                            artifacts_dir=self.build_dir,
                        )

            self._precompile_dp_attention_moe_concat_helpers(
                bucket,
                token_bucket=runtime_token_bucket,
                batch_size=bsz,
                seqlen=seq,
                is_decode=bool(is_decode),
            )
        finally:
            self._active_product_bucket = previous_bucket

    def _make_dp_attention_moe_dispatch_context(
        self,
        block: Any,
        *,
        input_ids: Any,
        layer_id: int,
        is_decode: bool,
        token_bucket: int | None,
        bsz: int,
        seqlen: int,
        hidden_size: int,
    ) -> dict[str, Any] | None:
        moe = getattr(block, "ffn", None)
        gate = getattr(moe, "gate", None)
        if moe is None or gate is None:
            return None
        weight = getattr(gate, "weight", None)
        if weight is None:
            return None
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            MOE_BLOCK_SIZE,
        )

        n_tokens = int(bsz) * int(seqlen)
        if bool(is_decode):
            rows = self._decode_moe_rows_for_requests(n_tokens)
        else:
            rows = max(n_tokens, int(token_bucket) if token_bucket is not None else 0)
            block_rows = int(MOE_BLOCK_SIZE)
            rows = ((max(rows, 1) + block_rows - 1) // block_rows) * block_rows
            max_prefill_rows = int(
                getattr(
                    self,
                    "product_prefill_moe_dispatch_fusion_max_rows",
                    0,
                )
                or 0
            )
            if max_prefill_rows > 0 and int(rows) > max_prefill_rows:
                _product_warmup_trace(
                    _product_executor_coord(self),
                    "prefill dispatch MoE concat skipped "
                    f"layer={layer_id} rows={int(rows)} "
                    f"max_rows={max_prefill_rows}",
                )
                return None
        score_func = str(getattr(gate, "score_func", "sigmoid"))
        common = {
            "layer_id": int(layer_id),
            "is_decode": bool(is_decode),
            "input_ids": input_ids,
            "weight": weight,
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "hidden_size": int(hidden_size),
            "rows": int(rows),
            "score_func": score_func,
            "route_scale": float(getattr(gate, "route_scale", 1.0)),
            "normalize": bool(score_func != "softmax"),
        }
        if bool(getattr(gate, "is_hash", False)):
            if getattr(gate, "bias", None) is not None:
                raise RuntimeError(
                    "DSV4 product DP-attention+MoE fusion does not support "
                    "hash gates with bias"
                )
            tid2eid = getattr(gate, "tid2eid", None)
            if tid2eid is None:
                raise RuntimeError("DSV4 product hash MoE fusion requires gate.tid2eid")
            return {**common, "kind": "hash_no_bias", "tid2eid": tid2eid}

        bias = getattr(gate, "bias", None)
        topk = int(getattr(gate, "topk", 0) or 0)
        n_experts = int(getattr(weight, "shape", (0,))[0])
        if topk <= 0 or n_experts <= 0:
            return None
        if bias is None:
            return {
                **common,
                "kind": "learned_no_bias",
                "topk": topk,
                "n_experts": n_experts,
            }
        return {
            **common,
            "kind": "learned_with_bias",
            "bias": bias,
            "topk": topk,
            "n_experts": n_experts,
        }

    def _run_product_hash_moe_dispatch_no_bias(
        self,
        x: Any,
        input_ids: Any,
        weight: Any,
        tid2eid: Any,
        *,
        bsz: int,
        seqlen: int,
        dim: int,
        rows: int,
        score_func: str,
        route_scale: float,
        normalize: bool,
    ) -> tuple[Any, Any, Any, Any]:
        return self._run_product_moe_dispatch(
            "hash_no_bias",
            x,
            input_ids=input_ids,
            weight=weight,
            tid2eid=tid2eid,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(dim),
            rows=int(rows),
            score_func=str(score_func),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )

    def _run_product_learned_moe_dispatch_no_bias(
        self,
        x: Any,
        weight: Any,
        *,
        bsz: int,
        seqlen: int,
        dim: int,
        rows: int,
        score_func: str,
        topk: int,
        n_experts: int,
        route_scale: float,
        normalize: bool,
    ) -> tuple[Any, Any, Any, Any]:
        return self._run_product_moe_dispatch(
            "learned_no_bias",
            x,
            weight=weight,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(dim),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )

    def _run_product_learned_moe_dispatch_with_bias(
        self,
        x: Any,
        weight: Any,
        bias: Any,
        *,
        bsz: int,
        seqlen: int,
        dim: int,
        rows: int,
        score_func: str,
        topk: int,
        n_experts: int,
        route_scale: float,
        normalize: bool,
    ) -> tuple[Any, Any, Any, Any]:
        return self._run_product_moe_dispatch(
            "learned_with_bias",
            x,
            weight=weight,
            bias=bias,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(dim),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )

    def _run_product_moe_dispatch(
        self,
        kind: str,
        x: Any,
        *,
        weight: Any,
        bsz: int,
        seqlen: int,
        dim: int,
        rows: int,
        score_func: str,
        route_scale: float,
        normalize: bool,
        input_ids: Any | None = None,
        tid2eid: Any | None = None,
        bias: Any | None = None,
        topk: int | None = None,
        n_experts: int | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        bucket = self._require_active_product_bucket(where="MoE dispatch")
        rows_i = int(rows)
        dim_i = int(dim)
        if rows_i <= 0 or dim_i <= 0:
            raise RuntimeError(
                "DSV4 product MoE dispatch requires positive rows/dim, "
                f"got rows={rows_i}, dim={dim_i}"
            )
        _require_product_device_value(x, where="moe_dispatch/x")
        _require_product_device_value(weight, where="moe_dispatch/weight")
        if kind == "hash_no_bias":
            _require_product_device_value(input_ids, where="moe_dispatch/input_ids")
            _require_product_device_value(tid2eid, where="moe_dispatch/tid2eid")
            topk_i = int(getattr(tid2eid, "shape", (0, 0))[-1])
        elif kind == "learned_no_bias":
            topk_i = int(topk or 0)
            n_experts_i = int(n_experts or 0)
            if n_experts_i <= 0:
                raise RuntimeError(
                    "DSV4 product learned MoE dispatch requires n_experts > 0"
                )
        elif kind == "learned_with_bias":
            _require_product_device_value(bias, where="moe_dispatch/bias")
            topk_i = int(topk or 0)
            n_experts_i = int(n_experts or 0)
            if n_experts_i <= 0:
                raise RuntimeError(
                    "DSV4 product learned MoE dispatch requires n_experts > 0"
                )
        else:
            raise RuntimeError(f"unsupported product MoE dispatch kind: {kind}")
        if topk_i <= 0:
            raise RuntimeError(
                f"DSV4 product MoE dispatch requires topk > 0, got {topk_i}"
            )
        if int(seqlen) > 1 and rows_i >= int(bucket.token_bucket):
            compile_bsz, compile_seqlen = self._product_compile_sequence_shape(
                bucket,
                bsz=int(bsz),
                seqlen=int(seqlen),
            )
            x_full = self._product_full_value_for(
                x,
                (int(compile_bsz), int(compile_seqlen), dim_i),
            )
            if x_full is not None:
                x = x_full
                bsz = int(compile_bsz)
                seqlen = int(compile_seqlen)
        kernel = self._product_moe_dispatch_kernel_for(
            bucket,
            kind=kind,
            x=x,
            input_ids=input_ids,
            weight=weight,
            tid2eid=tid2eid,
            bias=bias,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=dim_i,
            rows=rows_i,
            score_func=str(score_func),
            topk=topk_i,
            n_experts=int(n_experts or 0),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
        flat_hidden = self._active_moe_ep_output_for(rows=rows_i, dim=dim_i)
        if flat_hidden is None:
            flat_hidden = self._bucket_scratch(
                bucket,
                "moe_dispatch_hidden",
                (rows_i, dim_i),
                ml_dtypes.bfloat16,
            )
        shared_dtype = _value_dtype(x, fallback=ml_dtypes.bfloat16)
        shared_hidden = self._head_hidden_alias_for(
            shape=(1, rows_i, dim_i),
            dtype=shared_dtype,
        )
        if shared_hidden is None:
            shared_hidden = self._bucket_scratch(
                bucket,
                "moe_dispatch_shared_hidden",
                (1, rows_i, dim_i),
                shared_dtype,
            )
        outputs = {
            "output0": flat_hidden,
            "output1": self._bucket_scratch(
                bucket,
                "moe_dispatch_weights",
                (rows_i, topk_i),
                np.float32,
            ),
            "output2": self._bucket_scratch(
                bucket,
                "moe_dispatch_indices",
                (rows_i, topk_i),
                np.int32,
            ),
            "output3": shared_hidden,
        }
        inputs = {"x": x, "weight": weight}
        if kind == "hash_no_bias":
            inputs["input_ids"] = input_ids
            inputs["tid2eid"] = tid2eid
        elif kind == "learned_with_bias":
            inputs["bias"] = bias
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs=inputs,
            outputs=outputs,
        )
        return (
            outputs["output0"],
            outputs["output1"],
            outputs["output2"],
            outputs["output3"],
        )

    def _product_moe_dispatch_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        *,
        kind: str,
        x: Any,
        input_ids: Any | None,
        weight: Any,
        tid2eid: Any | None,
        bias: Any | None,
        bsz: int,
        seqlen: int,
        dim: int,
        rows: int,
        score_func: str,
        topk: int,
        n_experts: int,
        route_scale: float,
        normalize: bool,
    ) -> Any:
        x_shape = tuple(int(axis) for axis in getattr(x, "shape", ()))
        common_key = (
            kind,
            x_shape,
            str(getattr(x, "dtype", "unknown")).replace(".", "_"),
            tuple(int(axis) for axis in getattr(weight, "shape", ())),
            str(getattr(weight, "dtype", "unknown")).replace(".", "_"),
            int(bsz),
            int(seqlen),
            int(dim),
            int(rows),
            str(score_func),
            float(route_scale),
            bool(normalize),
        )
        compile_kwargs: dict[str, Any] = {
            "bsz": int(bsz),
            "seqlen": int(seqlen),
            "dim": int(dim),
            "rows": int(rows),
            "score_func": str(score_func),
            "route_scale": float(route_scale),
            "normalize": bool(normalize),
        }
        if kind == "hash_no_bias":
            key = (
                *common_key,
                tuple(int(axis) for axis in getattr(input_ids, "shape", ())),
                str(getattr(input_ids, "dtype", "unknown")).replace(".", "_"),
                tuple(int(axis) for axis in getattr(tid2eid, "shape", ())),
                str(getattr(tid2eid, "dtype", "unknown")).replace(".", "_"),
            )
            graph_fn = graph_moe.hash_moe_dispatch_no_bias_fn
            name_kind = "hash_moe_dispatch_no_bias"
            cache_name = "hash_moe_dispatch_no_bias_kernels"
            compile_args = [
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(input_ids, fallback_dtype=np.int32),
                _sample_array(weight, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(tid2eid, fallback_dtype=np.int32),
            ]
        elif kind == "learned_no_bias":
            key = (*common_key, int(topk), int(n_experts))
            graph_fn = graph_moe.learned_moe_dispatch_no_bias_fn
            name_kind = "learned_moe_dispatch_no_bias"
            cache_name = "learned_moe_dispatch_no_bias_kernels"
            compile_args = [
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(weight, fallback_dtype=ml_dtypes.bfloat16),
            ]
            compile_kwargs.update(
                {
                    "topk": int(topk),
                    "n_experts": int(n_experts),
                }
            )
        elif kind == "learned_with_bias":
            key = (
                *common_key,
                tuple(int(axis) for axis in getattr(bias, "shape", ())),
                str(getattr(bias, "dtype", "unknown")).replace(".", "_"),
                int(topk),
                int(n_experts),
            )
            graph_fn = graph_moe.learned_moe_dispatch_with_bias_fn
            name_kind = "learned_moe_dispatch_with_bias"
            cache_name = "learned_moe_dispatch_with_bias_kernels"
            compile_args = [
                _sample_array(x, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(weight, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(bias, fallback_dtype=np.float32),
            ]
            compile_kwargs.update(
                {
                    "topk": int(topk),
                    "n_experts": int(n_experts),
                }
            )
        else:
            raise RuntimeError(f"unsupported product MoE dispatch kind: {kind}")

        name = (
            f"dsv4_product_{name_kind}_"
            f"t{int(bucket.token_bucket)}_"
            f"x{'x'.join(str(v) for v in x_shape)}_"
            f"{int(bsz)}x{int(seqlen)}x{int(dim)}_"
            f"r{int(rows)}"
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
                load=False,
                **compile_kwargs,
            ),
        )
