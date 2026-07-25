"""MoE graph execution helpers for the DSV4 NEFF runtime."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.runtime.device_tensor import is_device_tensor


def _run_moe(
    fns: Dsv4GraphFns,
    moe: Any,
    x: np.ndarray,
    input_ids: np.ndarray,
    *,
    blockwise_state: Any | None,
    layer_id: int,
    is_decode: bool,
    build_dir: str | None,
    token_bucket: int | None = None,
    moe_output: Any | None = None,
    moe_ep_output: Any | None = None,
    moe_tp_output: Any | None = None,
    moe_ops: dict[str, Any] | None = None,
) -> Any:
    """Route + expert compute + shared-expert residual.

    Returns the ``shared_expert_add`` fragment's DeviceTensor output
    (``[bsz, seqlen, dim]``) which chains directly into the downstream
    ``mhc_post`` fragment — no host round-trip at the MoE/mhc boundary.
    """

    # ``x`` is typically a SpikeTensor coming from the ``mhc_pre_apply``
    # device fragment; do the flatten + bf16 cast on device and keep the
    # result resident for the blockwise MoE kernel.
    def _moe_op(name: str) -> Callable[..., Any]:
        if moe_ops is not None and name in moe_ops:
            return moe_ops[name]
        return fns[name]

    profile_stage = (
        moe_ops.get("_profile_stage")
        if moe_ops is not None and callable(moe_ops.get("_profile_stage"))
        else None
    )

    def _profile(name: str, **extra: Any):
        if profile_stage is None:
            return nullcontext()
        return profile_stage(name, **extra)

    blockwise_moe_block_size = int(
        (moe_ops or {}).get("blockwise_moe_block_size", 0) or 0
    )
    precomputed_dispatch = (
        moe_ops.get("_precomputed_dispatch") if moe_ops is not None else None
    )
    precomputed_routed = (
        moe_ops.get("_precomputed_routed") if moe_ops is not None else None
    )
    fuse_moe_collective_in_shared_restore = bool(
        moe_ops is not None
        and moe_ops.get("fuse_moe_collective_in_shared_restore", False)
    )
    moe_replica_groups = (
        tuple(moe_ops.get("_moe_collective_replica_groups", ()))
        if fuse_moe_collective_in_shared_restore and moe_ops is not None
        else ()
    )
    dispatch_x_for_shared = None
    if precomputed_routed is not None:
        routed = precomputed_routed["routed"]
        x_for_shared = precomputed_routed["shared_hidden"]
        restore_shared_shape = True
    elif precomputed_dispatch is not None:
        (
            xf_bf_dev,
            weights,
            indices,
            dispatch_x_for_shared,
        ) = precomputed_dispatch
    else:
        hash_dispatch = (
            moe_ops.get("hash_moe_dispatch_no_bias")
            if moe_ops is not None and bool(getattr(moe.gate, "is_hash", False))
            else None
        )
    if (
        precomputed_routed is None
        and precomputed_dispatch is None
        and callable(hash_dispatch)
        and getattr(moe.gate, "bias", None) is None
    ):
        input_ids_for_hash = (
            moe_ops.get("_hash_router_input_ids", input_ids)
            if moe_ops is not None
            else input_ids
        )
        real_tokens = int(np.prod(tuple(int(dim) for dim in x.shape[:-1])))
        dispatch_rows = max(
            real_tokens,
            int(token_bucket) if token_bucket is not None else 0,
        )
        if not bool(is_decode):
            if blockwise_moe_block_size <= 0:
                raise RuntimeError("DSV4 MoE dispatch requires blockwise block size")
            block = int(blockwise_moe_block_size)
            dispatch_rows = ((max(dispatch_rows, 1) + block - 1) // block) * block
        with _profile(
            "moe_dispatch",
            dispatch_kind="hash",
            rows=int(dispatch_rows),
            real_tokens=int(real_tokens),
        ):
            xf_bf_dev, weights, indices, dispatch_x_for_shared = hash_dispatch(
                x,
                input_ids_for_hash,
                moe.gate.weight,
                moe.gate.tid2eid,
                bsz=int(x.shape[0]),
                seqlen=int(x.shape[1]),
                dim=int(moe.dim),
                rows=int(dispatch_rows),
                score_func=str(moe.gate.score_func),
                route_scale=float(moe.gate.route_scale),
                normalize=bool(moe.gate.score_func != "softmax"),
            )
    elif precomputed_routed is None and precomputed_dispatch is None:
        learned_dispatch = None
        if moe_ops is not None and not bool(getattr(moe.gate, "is_hash", False)):
            learned_dispatch = moe_ops.get(
                (
                    "learned_moe_dispatch_no_bias"
                    if getattr(moe.gate, "bias", None) is None
                    else "learned_moe_dispatch_with_bias"
                )
            )
        if callable(learned_dispatch):
            real_tokens = int(np.prod(tuple(int(dim) for dim in x.shape[:-1])))
            dispatch_rows = max(
                real_tokens,
                int(token_bucket) if token_bucket is not None else 0,
            )
            if not bool(is_decode):
                if blockwise_moe_block_size <= 0:
                    raise RuntimeError(
                        "DSV4 MoE dispatch requires blockwise block size"
                    )
                block = int(blockwise_moe_block_size)
                dispatch_rows = ((max(dispatch_rows, 1) + block - 1) // block) * block
            dispatch_kwargs = {
                "bsz": int(x.shape[0]),
                "seqlen": int(x.shape[1]),
                "dim": int(moe.dim),
                "rows": int(dispatch_rows),
                "score_func": str(moe.gate.score_func),
                "topk": int(moe.gate.topk),
                "n_experts": int(moe.gate.weight.shape[0]),
                "route_scale": float(moe.gate.route_scale),
                "normalize": bool(moe.gate.score_func != "softmax"),
            }
            if getattr(moe.gate, "bias", None) is None:
                with _profile(
                    "moe_dispatch",
                    dispatch_kind="learned_no_bias",
                    rows=int(dispatch_rows),
                    real_tokens=int(real_tokens),
                ):
                    xf_bf_dev, weights, indices, dispatch_x_for_shared = (
                        learned_dispatch(
                            x,
                            moe.gate.weight,
                            **dispatch_kwargs,
                        )
                    )
            else:
                with _profile(
                    "moe_dispatch",
                    dispatch_kind="learned_with_bias",
                    rows=int(dispatch_rows),
                    real_tokens=int(real_tokens),
                ):
                    xf_bf_dev, weights, indices, dispatch_x_for_shared = (
                        learned_dispatch(
                            x,
                            moe.gate.weight,
                            moe.gate.bias,
                            **dispatch_kwargs,
                        )
                    )
        else:
            gate_kind = (
                "hash" if bool(getattr(moe.gate, "is_hash", False)) else "learned"
            )
            has_bias = getattr(moe.gate, "bias", None) is not None
            raise RuntimeError(
                "DSV4 product MoE requires fused router dispatch; "
                f"unsupported gate kind={gate_kind}, bias={has_bias}"
            )

    # Blockwise kernel path: learned-router and hash-router layers both
    # arrive here as device-resident top-k tensors with the same contract.
    if (
        precomputed_routed is None
        and blockwise_state is not None
        and layer_id < len(blockwise_state.layers)
        and blockwise_state.layers[layer_id].n_local_experts > 0
    ):
        run_blockwise_moe_decode = (
            moe_ops.get("run_blockwise_moe_decode") if moe_ops is not None else None
        )
        run_blockwise_moe_prefill = (
            moe_ops.get("run_blockwise_moe_prefill") if moe_ops is not None else None
        )
        if (
            blockwise_moe_block_size <= 0
            or not callable(run_blockwise_moe_decode)
            or not callable(run_blockwise_moe_prefill)
        ):
            raise RuntimeError(
                "DSV4 MoE execution requires caller-provided blockwise MoE ops"
            )

        layer = blockwise_state.layers[layer_id]
        n_tokens = int(xf_bf_dev.shape[0])
        router_on_device = is_device_tensor(weights) and is_device_tensor(indices)
        if not router_on_device:
            raise RuntimeError(
                "DSV4 blockwise MoE requires device-resident router weights and indices"
            )
        force_shared_sequence_pad = bool(
            moe_ops is not None and moe_ops.get("force_shared_sequence_pad", False)
        )
        target_rows = max(
            n_tokens,
            int(token_bucket) if token_bucket is not None else 0,
        )
        if not bool(is_decode):
            block = int(blockwise_moe_block_size)
            target_rows = ((max(target_rows, 1) + block - 1) // block) * block
        use_bucket_pad = target_rows > n_tokens
        restore_shared_shape = bool(use_bucket_pad or force_shared_sequence_pad)
        if use_bucket_pad:
            raise RuntimeError(
                "DSV4 product MoE fused router dispatch must produce "
                f"bucket-padded rows: got {n_tokens}, need {target_rows}"
            )
        else:
            hidden_for_moe = xf_bf_dev
            weights_for_moe = weights
            indices_for_moe = indices
            if dispatch_x_for_shared is not None:
                x_for_shared = dispatch_x_for_shared
            elif force_shared_sequence_pad:
                raise RuntimeError(
                    "DSV4 product MoE fused router dispatch must produce "
                    "shared hidden padding output"
                )
            else:
                x_for_shared = x

        T = int(hidden_for_moe.shape[0])
        if bool(is_decode) and T <= int(blockwise_moe_block_size):
            with _profile(
                "moe_blockwise",
                moe_kernel="decode",
                rows=int(T),
                target_rows=int(target_rows),
                router_on_device=bool(router_on_device),
                skip_all_reduce=bool(fuse_moe_collective_in_shared_restore),
            ):
                routed = run_blockwise_moe_decode(
                    layer,
                    hidden_states=hidden_for_moe,
                    weights=weights_for_moe,
                    indices=indices_for_moe,
                    state=blockwise_state,
                    artifacts_dir=build_dir,
                    return_device=True,
                    output=moe_output,
                    ep_output=None
                    if fuse_moe_collective_in_shared_restore
                    else moe_ep_output,
                    tp_output=None
                    if fuse_moe_collective_in_shared_restore
                    else moe_tp_output,
                    skip_all_reduce=fuse_moe_collective_in_shared_restore,
                )
        else:
            with _profile(
                "moe_blockwise",
                moe_kernel="prefill",
                rows=int(T),
                target_rows=int(target_rows),
                router_on_device=bool(router_on_device),
                skip_all_reduce=bool(fuse_moe_collective_in_shared_restore),
            ):
                routed = run_blockwise_moe_prefill(
                    layer,
                    hidden_states=hidden_for_moe,
                    weights=weights_for_moe,
                    indices=indices_for_moe,
                    state=blockwise_state,
                    artifacts_dir=build_dir,
                    return_device=True,
                    output=moe_output,
                    ep_output=None
                    if fuse_moe_collective_in_shared_restore
                    else moe_ep_output,
                    tp_output=None
                    if fuse_moe_collective_in_shared_restore
                    else moe_tp_output,
                    skip_all_reduce=fuse_moe_collective_in_shared_restore,
                )
    elif precomputed_routed is None:
        raise RuntimeError(
            "DSV4 sampled forward requires blockwise MoE state for device "
            f"expert compute (missing layer_id={layer_id})"
        )

    # ``shared_expert_add_fn`` reshapes the flat routed output to match
    # ``x`` internally, so the DeviceTensor output already has the shape
    # downstream ``mhc_post`` expects. No download here.
    if restore_shared_shape:
        restore_op = (
            moe_ops.get("shared_expert_add_restore") if moe_ops is not None else None
        )
        if callable(restore_op):
            with _profile(
                "moe_shared_restore",
                tp_sharded=bool(getattr(moe.shared, "tp_sharded", False)),
                moe_collective=bool(moe_replica_groups),
            ):
                return restore_op(
                    routed,
                    x_for_shared,
                    moe.shared.w1,
                    moe.shared.w3,
                    moe.shared.w2,
                    limit=float(moe.shared.swiglu_limit),
                    bsz=int(x.shape[0]),
                    seqlen=int(x.shape[1]),
                    hidden_size=int(moe.dim),
                    tp_degree=(
                        int(getattr(blockwise_state, "tp_degree", 1))
                        if bool(getattr(moe.shared, "tp_sharded", False))
                        else 1
                    ),
                    tp_replica_groups=(
                        tuple(getattr(blockwise_state, "tp_replica_groups", ()))
                        if bool(getattr(moe.shared, "tp_sharded", False))
                        else ()
                    ),
                    moe_replica_groups=moe_replica_groups,
                )
    raise RuntimeError(
        "DSV4 product MoE requires fused shared-expert restore; "
        "shared_expert_add/attention_hidden_reshape is not a valid "
        "product boundary"
    )
