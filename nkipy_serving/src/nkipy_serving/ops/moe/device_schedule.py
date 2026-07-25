"""Device-side blockwise MoE scheduling for DSV4.

The prefill blockwise MoE kernel consumes:

* ``expert_affinities_masked_hbm``: flattened local affinities ``[T*E, 1]``.
* ``token_position_to_id``: block slot to token id, ``[N, B]``.
* ``block_to_expert``: block to local expert id, ``[N]``.

The production path builds these tensors from router top-k outputs on device.
This module vendors and uses the same nkilib subkernels NxD uses:
``find_nonzero_indices`` and ``indexed_flatten``. Router top-k outputs stay as
DeviceTensors, and only MoE scheduling metadata tensors are materialized on
device.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.ops.moe.blockwise_index import (
    BLOCK_SIZE as MOE_BLOCK_SIZE,
)
from nkipy_serving.ops.moe.blockwise_index import (
    ControlType,
    get_n_blocks,
)
from nkipy_serving.runtime.device_tensor import is_device_tensor
from nkipy_serving.runtime.kernel_compile import compile_and_load_neff_with_lock

_SKIP_DMA = int(ControlType.SKIP_DMA.value)


def _nki_spec_dtype(dtype: Any) -> Any:
    text = str(dtype)
    if text in ("bfloat16", "bf16"):
        return ml_dtypes.bfloat16
    if text in ("float32", "f32"):
        return np.float32
    if text in ("float16", "f16"):
        return np.float16
    if text in ("int32", "i32"):
        return np.int32
    if text in ("uint32", "ui32"):
        return np.uint32
    if text in ("int8", "i8"):
        return np.int8
    if text in ("uint8", "ui8"):
        return np.uint8
    return np.dtype(text)


def wrap_nki_framework_kernel(
    kernel: Any,
    *,
    lnc: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
):
    """Wrap current beta2 ``nki.framework.kernel.Kernel`` as a HLO custom call.

    NKIPy's generic ``wrap_nki_kernel(..., is_nki_beta_2_version=True)`` targets
    a different package layout than the installed Neuron SDK. This bridge
    compiles framework kernels directly through the Neuron frontend and emits
    the HLO custom-call metadata NKIPy expects.
    """
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.framework.compiled import CompileKernel
    from nkipy.core.backend import get_backend
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    if get_backend() == "cpu":
        raise NotImplementedError("CPU execution is not supported for NKI custom ops")

    kwargs = dict(kwargs or {})
    kernel = kernel[int(lnc)]
    stable_artifacts = os.getenv("NKIPY_SERVING_NKI_BIR_ARTIFACTS_DIR")
    if stable_artifacts:
        kernel_name = "".join(
            ch if ch.isalnum() or ch in "_.-" else "_" for ch in type(kernel).__name__
        )
        artifacts = str(Path(stable_artifacts) / (kernel_name or "kernel"))
        Path(artifacts).mkdir(parents=True, exist_ok=True)
    else:
        artifacts = tempfile.mkdtemp(prefix="dsv4_nki_bir_")
    compile_kernel = kernel._to_subclass(
        CompileKernel,
        _frontend_cls=TracerFrontend,
        artifacts_dir=artifacts,
        target="trn2",
    )
    bound = compile_kernel._bind_args(args, kwargs)
    numpy_bound = {
        name: (
            np.empty(value.shape, dtype=value.dtype)
            if isinstance(value, NKIPyTensorRef)
            else value
        )
        for name, value in bound.items()
    }
    frontend = TracerFrontend(enable_backend_opt=False)
    bir, result = compile_to_bir(
        compile_kernel,
        frontend=frontend,
        inputs=numpy_bound,
        compile_opts=compile_kernel._compile_opts(),
    )
    backend_config = compile_kernel._build_backend_config(bir, result)

    tensor_operands = [
        value for value in bound.values() if isinstance(value, NKIPyTensorRef)
    ]
    hlo_operands = [value.backend_tensor for value in tensor_operands]
    output_shapes = [
        tuple(int(dim) for dim in spec.shape) for spec in result.output_specs
    ]
    output_dtypes = [_nki_spec_dtype(spec.dtype) for spec in result.output_specs]
    attrs = {
        "custom_call_target": "AwsNeuronCustomNativeKernel",
        "backend_config": base64.b64encode(
            json.dumps(backend_config).encode("utf-8")
        ).decode("ascii"),
    }
    if bool(result.has_collectives):
        attrs["has_collectives"] = True

    ctx = get_hlo_context()
    if len(output_shapes) == 1 and not bool(result.is_tuple_return):
        out = ctx.build_op(
            "custom-call",
            hlo_operands,
            output_shapes[0],
            output_dtypes[0],
            attrs,
        )
        return NKIPyTensorRef(out)

    attrs["is_tuple"] = True
    result_tensor = ctx.build_op(
        "custom-call",
        hlo_operands,
        output_shapes,
        output_dtypes,
        attrs,
    )
    return tuple(
        NKIPyTensorRef(
            ctx.build_op(
                "get-tuple-element",
                [result_tensor],
                output_shapes[i],
                output_dtypes[i],
                {"tuple_index": i},
            )
        )
        for i in range(len(output_shapes))
    )


@dataclass(frozen=True)
class DeviceMoESchedule:
    """Device-resident MoE schedule tensors plus static shape metadata."""

    expert_affinities_masked: Any
    token_position_to_id: Any
    block_to_expert: Any
    num_blocks: int
    num_static_blocks: int


def _tensor_sample_dtype(value: Any) -> Any:
    return _nki_spec_dtype(getattr(value, "dtype", ml_dtypes.bfloat16))


def _dtype_cache_name(dtype: Any) -> str:
    text = str(np.dtype(dtype))
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)


def logical_nc_config() -> int:
    raw = os.getenv("NEURON_LOGICAL_NC_CONFIG", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid NEURON_LOGICAL_NC_CONFIG={raw!r}") from exc
    if value not in (1, 2):
        raise RuntimeError(
            "DSV4 device MoE schedule currently supports "
            f"NEURON_LOGICAL_NC_CONFIG 1 or 2, got {value}"
        )
    return value


def choose_indexed_flatten_f_len(token_bucket: int) -> int:
    """Pick an indexed-flatten width compatible with block offsets.

    ``indexed_flatten`` reshapes token lists as ``[E, T // f_len, f_len]`` and
    row offsets are measured in those ``f_len`` chunks.  Therefore ``f_len``
    must divide both the token bucket and the MoE block size; otherwise
    ``start * block_size // f_len`` points at the wrong slot.  The kernel also
    requires ``T // f_len`` to be a multiple of 16.
    """
    T = int(token_bucket)
    if T <= 0:
        raise RuntimeError(f"token_bucket must be > 0, got {T}")
    if T % MOE_BLOCK_SIZE != 0:
        raise NotImplementedError(
            "device MoE schedule requires token_bucket to be a multiple of "
            f"{MOE_BLOCK_SIZE}, got {T}"
        )
    max_f = min(MOE_BLOCK_SIZE, max(1, T // 16))
    for f_len in range(max_f, 0, -1):
        if MOE_BLOCK_SIZE % f_len == 0 and T % f_len == 0 and (T // f_len) % 16 == 0:
            return f_len
    raise NotImplementedError(
        "no indexed_flatten f_len satisfies block-size and tile constraints "
        f"for token_bucket={T}"
    )


def build_local_expert_affinities_oracle(
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
) -> np.ndarray:
    """CPU oracle for the device top-k-to-local-affinity graph."""
    w = np.asarray(weights, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.int32)
    if w.shape != idx.shape:
        raise ValueError(f"weights/indices shape mismatch: {w.shape} vs {idx.shape}")
    T, K = idx.shape
    E = int(local_num_experts)
    e0 = int(ep_rank) * E if int(ep_degree) > 1 else 0
    out = np.zeros((T, E), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            e = int(idx[t, k])
            if e < 0:
                continue
            e_local = e - e0
            if 0 <= e_local < E:
                out[t, e_local] += float(w[t, k])
    return out.astype(ml_dtypes.bfloat16)


def _local_expert_affinities_fn(
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
) -> np.ndarray:
    """Graph-traceable top-k scatter to local ``[T, E_local]`` affinities."""
    import ml_dtypes as _ml

    E = int(local_num_experts)
    idx = indices.astype(np.int32)
    w = weights.astype(np.float32)
    e0 = np.int32(int(ep_rank) * E if int(ep_degree) > 1 else 0)
    experts = np.arange(E, dtype=np.int32)
    local_idx = idx - e0
    matches = local_idx[..., None] == experts[None, None, :]
    valid = idx[..., None] >= np.int32(0)
    if int(ep_degree) > 1:
        valid = valid & (idx[..., None] < np.int32(int(ep_rank + 1) * E))
    aff = np.sum(np.where(matches & valid, w[..., None], np.float32(0.0)), axis=1)
    return aff.astype(_ml.bfloat16)


def local_expert_affinities_dynamic_ep_fn(
    weights: np.ndarray,
    indices: np.ndarray,
    ep_start: np.ndarray,
    *,
    local_num_experts: int,
) -> np.ndarray:
    """Graph-traceable top-k scatter with runtime EP local-expert offset."""
    import ml_dtypes as _ml

    E = int(local_num_experts)
    idx = indices.astype(np.int32)
    w = weights.astype(np.float32)
    e0 = ep_start.astype(np.int32).reshape(-1)[:1]
    experts = np.arange(E, dtype=np.int32)
    local_idx = idx - e0.reshape(1, 1)
    matches = local_idx[..., None] == experts[None, None, :]
    e0_3d = e0.reshape(1, 1, 1)
    valid = (idx[..., None] >= e0_3d) & (
        idx[..., None] < (e0 + np.int32(E)).reshape(1, 1, 1)
    )
    aff = np.sum(np.where(matches & valid, w[..., None], np.float32(0.0)), axis=1)
    return aff.astype(_ml.bfloat16)


def prefill_moe_schedule_oracle(
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    local_num_experts: int,
    experts_per_token: int,
    ep_degree: int,
    ep_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """CPU oracle for the device prefill schedule wrapper."""
    aff = build_local_expert_affinities_oracle(
        weights,
        indices,
        local_num_experts=local_num_experts,
        ep_degree=ep_degree,
        ep_rank=ep_rank,
    )
    local_indices = np.where(aff.astype(np.float32) != 0.0, 0, _SKIP_DMA)
    rows: list[np.ndarray] = []
    for e in range(int(local_num_experts)):
        token_ids = np.nonzero(aff[:, e].astype(np.float32) != 0.0)[0].astype(np.int32)
        row = np.full((aff.shape[0],), _SKIP_DMA, dtype=np.int32)
        row[: token_ids.size] = token_ids
        rows.append(row)
    del local_indices
    token_lists = (
        np.stack(rows, axis=0) if rows else np.zeros((0, aff.shape[0]), dtype=np.int32)
    )

    num_blocks, num_static_blocks = get_n_blocks(
        int(aff.shape[0]), int(experts_per_token), int(local_num_experts)
    )
    counts = (token_lists != _SKIP_DMA).sum(axis=1).astype(np.int32)
    blocks_per_expert = (
        (counts + int(MOE_BLOCK_SIZE) - 1) // int(MOE_BLOCK_SIZE)
    ).astype(np.int32)
    cum = np.cumsum(blocks_per_expert, axis=0).astype(np.int32)
    starts = np.concatenate((np.zeros((1,), dtype=np.int32), cum[:-1]), axis=0)
    token_position_to_id = np.full(
        (int(num_blocks), int(MOE_BLOCK_SIZE)),
        _SKIP_DMA,
        dtype=np.int32,
    )
    for e, start in enumerate(starts):
        flat_start = int(start) * int(MOE_BLOCK_SIZE)
        ids = token_lists[e]
        ids = ids[ids != _SKIP_DMA]
        flat = token_position_to_id.reshape(-1)
        flat[flat_start : flat_start + ids.size] = ids

    block_ids = np.arange(int(num_blocks), dtype=np.int32)
    if int(local_num_experts) > 1:
        raw_b2e = np.sum(
            block_ids[:, None] >= cum[:-1][None, :],
            axis=1,
        ).astype(np.int32)
    else:
        raw_b2e = np.zeros((int(num_blocks),), dtype=np.int32)
    if int(num_blocks) > 1:
        prev = np.concatenate(
            (np.full((1,), -2, dtype=np.int32), raw_b2e[:-1]),
            axis=0,
        )
        block_to_expert = np.where(raw_b2e == prev, _SKIP_DMA, raw_b2e).astype(np.int32)
    else:
        block_to_expert = raw_b2e

    return (
        aff.reshape(int(aff.shape[0]) * int(local_num_experts), 1),
        token_position_to_id,
        block_to_expert,
        int(num_blocks),
        int(num_static_blocks),
    )


def _make_prefill_index_entry(
    *,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
):
    """Create the first prefill schedule graph.

    This graph computes local affinities, token lists, row offsets, and
    block-to-expert metadata. ``indexed_flatten`` intentionally runs in a
    second device launch because consuming a tuple output from one beta2 custom
    call directly inside another custom call has a layout handoff issue under
    NKIPy HLO.
    """
    import nki.language as nl

    from nkipy_serving.ops.moe.find_nonzero_indices import (
        find_nonzero_indices,
    )

    T = int(token_bucket)
    E = int(local_num_experts)
    B = int(MOE_BLOCK_SIZE)
    n_blocks = int(num_blocks)
    f = int(f_len)
    lnc = int(logical_nc_config)

    def _entry(weights, indices, ep_start):
        aff = local_expert_affinities_dynamic_ep_fn(
            weights,
            indices,
            ep_start,
            local_num_experts=E,
        )
        aff_for_index = aff.astype(np.float32)
        token_lists, counts = wrap_nki_framework_kernel(
            find_nonzero_indices,
            lnc=lnc,
            args=(aff_for_index, None, E, T, nl.int32),
        )

        counts_i = counts.astype(np.int32)
        blocks_per_expert = np.floor(
            (counts_i.astype(np.float32) + np.float32(B - 1)) / np.float32(B)
        ).astype(np.int32)
        cum = np.cumsum(blocks_per_expert, axis=0).astype(np.int32)
        starts = np.concatenate((np.zeros((1,), dtype=np.int32), cum[:-1]), axis=0)
        row_offsets = (starts * np.int32(B // f)).astype(np.int32)

        block_ids = np.arange(n_blocks, dtype=np.int32)
        if E > 1:
            raw_b2e = np.sum(
                (block_ids[:, None] >= cum[:-1][None, :]).astype(np.int32),
                axis=1,
            ).astype(np.int32)
        else:
            raw_b2e = np.zeros((n_blocks,), dtype=np.int32)
        if n_blocks > 1:
            prev = np.concatenate(
                (np.full((1,), -2, dtype=np.int32), raw_b2e[:-1]),
                axis=0,
            )
            block_to_expert = np.where(
                raw_b2e == prev,
                np.int32(_SKIP_DMA),
                raw_b2e,
            ).astype(np.int32)
        else:
            block_to_expert = raw_b2e

        return (
            aff.reshape(T * E, 1),
            token_lists,
            row_offsets,
            block_to_expert,
        )

    return _entry


def make_prefill_fused_entry(
    *,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    compress_block_to_expert: bool = True,
):
    """Create a fused prefill schedule graph.

    The graph keeps ``find_nonzero_indices`` from the index stage, then does the
    final blockwise flattening with traceable NumPy scatter logic instead of a
    second ``indexed_flatten`` custom call.  Invalid token slots are redirected
    into the padded tail so they cannot overwrite valid block slots.
    """
    import nki.language as nl

    from nkipy_serving.ops.moe.find_nonzero_indices import (
        find_nonzero_indices,
    )

    T = int(token_bucket)
    E = int(local_num_experts)
    B = int(MOE_BLOCK_SIZE)
    n_blocks = int(num_blocks)
    out_len = int(output_len)
    lnc = int(logical_nc_config)

    def _entry(weights, indices, ep_start):
        from nkipy.core import ops as nkipy_ops

        aff = local_expert_affinities_dynamic_ep_fn(
            weights,
            indices,
            ep_start,
            local_num_experts=E,
        )
        aff_for_index = aff.astype(np.float32)
        token_lists, counts = wrap_nki_framework_kernel(
            find_nonzero_indices,
            lnc=lnc,
            args=(aff_for_index, None, E, T, nl.int32),
        )

        counts_i = counts.astype(np.int32)
        blocks_per_expert = np.floor(
            (counts_i.astype(np.float32) + np.float32(B - 1)) / np.float32(B)
        ).astype(np.int32)
        cum = np.cumsum(blocks_per_expert, axis=0).astype(np.int32)
        starts = np.concatenate((np.zeros((1,), dtype=np.int32), cum[:-1]), axis=0)

        block_ids = np.arange(n_blocks, dtype=np.int32)
        if E > 1:
            raw_b2e = np.sum(
                (block_ids[:, None] >= cum[:-1][None, :]).astype(np.int32),
                axis=1,
            ).astype(np.int32)
        else:
            raw_b2e = np.zeros((n_blocks,), dtype=np.int32)
        if bool(compress_block_to_expert) and n_blocks > 1:
            prev = np.concatenate(
                (np.full((1,), -2, dtype=np.int32), raw_b2e[:-1]),
                axis=0,
            )
            block_to_expert = np.where(
                raw_b2e == prev,
                np.int32(_SKIP_DMA),
                raw_b2e,
            ).astype(np.int32)
        else:
            block_to_expert = raw_b2e

        flat_len = np.int32(n_blocks * B)
        padded = nkipy_ops.full((out_len,), np.int32(_SKIP_DMA), np.dtype(np.int32))
        slots = np.arange(T, dtype=np.int32)
        dst = starts[:, None] * np.int32(B) + slots[None, :]
        valid = (token_lists != np.int32(_SKIP_DMA)) & (dst < flat_len)
        tail_slots = flat_len + (slots[None, :] % np.int32(out_len - n_blocks * B))
        dst_safe = np.where(valid, dst, tail_slots).astype(np.int32).reshape(E * T)
        vals = np.where(valid, token_lists, np.int32(_SKIP_DMA)).astype(np.int32)
        padded = nkipy_ops.put_along_axis(
            padded,
            dst_safe,
            vals.reshape(E * T),
            axis=0,
        )
        token_position_to_id = padded[: n_blocks * B].reshape(n_blocks, B)

        return (
            aff.reshape(T * E, 1),
            token_position_to_id.astype(np.int32),
            block_to_expert,
        )

    return _entry


def _make_prefill_flatten_entry(
    *,
    token_bucket: int,
    local_num_experts: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
):
    """Create the second prefill schedule graph around ``indexed_flatten``."""
    from nkipy_serving.ops.moe.indexed_flatten import indexed_flatten

    B = int(MOE_BLOCK_SIZE)
    n_blocks = int(num_blocks)
    f = int(f_len)
    out_len = int(output_len)
    lnc = int(logical_nc_config)

    def _entry(token_lists, row_offsets):
        token_position_to_id_padded = wrap_nki_framework_kernel(
            indexed_flatten,
            lnc=lnc,
            args=(token_lists, f, out_len, row_offsets, None, _SKIP_DMA),
        )
        return (
            token_position_to_id_padded[0 : n_blocks * B]
            .reshape(n_blocks, B)
            .astype(np.int32)
        )

    return _entry


def _make_decode_schedule_entry(
    *,
    token_bucket: int,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
):
    E = int(local_num_experts)

    def _entry(weights, indices):
        aff = _local_expert_affinities_fn(
            weights,
            indices,
            local_num_experts=E,
            ep_degree=int(ep_degree),
            ep_rank=int(ep_rank),
        )
        return aff.T.astype(aff.dtype)

    return _entry


def run_prefill_moe_schedule_device(
    *,
    weights: Any,
    indices: Any,
    local_num_experts: int,
    experts_per_token: int,
    ep_degree: int,
    ep_rank: int,
    artifacts_dir: str | Path | None = None,
    kernel_cache: dict[tuple[Any, ...], Any] | None = None,
    fuse_schedule: bool = True,
    _device_kernel_cls: Any | None = None,
    _device_tensor_cls: Any | None = None,
) -> DeviceMoESchedule:
    """Build prefill MoE schedule tensors without downloading router outputs."""
    if not (is_device_tensor(weights) and is_device_tensor(indices)):
        raise TypeError("weights and indices must be device tensors")

    T, K = (int(v) for v in weights.shape)
    if tuple(int(v) for v in indices.shape) != (T, K):
        raise ValueError(
            f"weights/indices shape mismatch: {tuple(weights.shape)} vs {tuple(indices.shape)}"
        )
    E = int(local_num_experts)
    num_blocks, num_static_blocks = get_n_blocks(T, int(experts_per_token), E)
    f_len = choose_indexed_flatten_f_len(T)
    output_len = int(num_blocks) * int(MOE_BLOCK_SIZE) + T
    lnc = logical_nc_config()

    if _device_kernel_cls is None or _device_tensor_cls is None:
        from nkipy_serving.models._device_utils import (
            _get_device_kernel_cls,
            _get_device_tensor_cls,
        )

        if _device_kernel_cls is None:
            _device_kernel_cls = _get_device_kernel_cls()
        if _device_tensor_cls is None:
            _device_tensor_cls = _get_device_tensor_cls()

    cache = kernel_cache if kernel_cache is not None else {}
    if bool(fuse_schedule):
        fused_cache_key = (
            "prefill_fused",
            T,
            K,
            E,
            int(experts_per_token),
            int(num_blocks),
            int(f_len),
            int(output_len),
            int(lnc),
            str(getattr(weights, "dtype", "unknown")),
        )
        fused_kernel = cache.get(fused_cache_key)
        if fused_kernel is None:
            entry = make_prefill_fused_entry(
                token_bucket=T,
                local_num_experts=E,
                experts_per_token=int(experts_per_token),
                num_blocks=int(num_blocks),
                f_len=int(f_len),
                output_len=int(output_len),
                logical_nc_config=int(lnc),
            )
            weights_sample_dtype = _tensor_sample_dtype(weights)
            weights_sample = np.zeros((T, K), dtype=weights_sample_dtype)
            indices_sample = np.zeros((T, K), dtype=np.int32)
            dtype_name = _dtype_cache_name(weights_sample_dtype)
            fused_name = (
                f"dsv4_moe_sched_prefill_fused_t{T}_e{E}_k{K}"
                f"_x{int(experts_per_token)}"
                f"_n{int(num_blocks)}_f{int(f_len)}_o{int(output_len)}"
                f"_lnc{int(lnc)}_{dtype_name}"
            )
            fused_kernel = compile_and_load_neff_with_lock(
                _device_kernel_cls,
                entry,
                weights_sample,
                indices_sample,
                np.zeros((1,), dtype=np.int32),
                name=fused_name,
                build_dir=str(artifacts_dir) if artifacts_dir else None,
                namespace="moe_schedule",
            )
            cache[fused_cache_key] = fused_kernel

        aff_dev = _device_tensor_cls.from_numpy(
            np.zeros((T * E, 1), dtype=ml_dtypes.bfloat16),
            name="moe_aff",
        )
        tp_dev = _device_tensor_cls.from_numpy(
            np.full((int(num_blocks), int(MOE_BLOCK_SIZE)), _SKIP_DMA, dtype=np.int32),
            name="moe_tp2id",
        )
        b2e_dev = _device_tensor_cls.from_numpy(
            np.zeros((int(num_blocks),), dtype=np.int32),
            name="moe_b2e",
        )
        ep_start_dev = _device_tensor_cls.from_numpy(
            np.asarray([int(ep_rank) * E], dtype=np.int32),
            name="moe_ep_start",
        )
        fused_kernel(
            inputs={"weights": weights, "indices": indices, "ep_start": ep_start_dev},
            outputs={
                "output0": aff_dev,
                "output1": tp_dev,
                "output2": b2e_dev,
            },
        )
        return DeviceMoESchedule(
            expert_affinities_masked=aff_dev,
            token_position_to_id=tp_dev,
            block_to_expert=b2e_dev,
            num_blocks=int(num_blocks),
            num_static_blocks=int(num_static_blocks),
        )

    index_cache_key = (
        "prefill_index",
        T,
        K,
        E,
        int(experts_per_token),
        int(num_blocks),
        int(f_len),
        int(output_len),
        int(lnc),
        str(getattr(weights, "dtype", "unknown")),
    )
    index_kernel = cache.get(index_cache_key)
    if index_kernel is None:
        entry = _make_prefill_index_entry(
            token_bucket=T,
            local_num_experts=E,
            experts_per_token=int(experts_per_token),
            num_blocks=int(num_blocks),
            f_len=int(f_len),
            output_len=int(output_len),
            logical_nc_config=int(lnc),
        )
        weights_sample_dtype = _tensor_sample_dtype(weights)
        weights_sample = np.zeros((T, K), dtype=weights_sample_dtype)
        indices_sample = np.zeros((T, K), dtype=np.int32)
        dtype_name = _dtype_cache_name(weights_sample_dtype)
        index_name = (
            f"dsv4_moe_sched_prefill_index_t{T}_e{E}_k{K}"
            f"_x{int(experts_per_token)}"
            f"_n{int(num_blocks)}_f{int(f_len)}_o{int(output_len)}"
            f"_lnc{int(lnc)}_{dtype_name}"
        )
        index_kernel = compile_and_load_neff_with_lock(
            _device_kernel_cls,
            entry,
            weights_sample,
            indices_sample,
            np.zeros((1,), dtype=np.int32),
            name=index_name,
            build_dir=str(artifacts_dir) if artifacts_dir else None,
            namespace="moe_schedule",
        )
        cache[index_cache_key] = index_kernel

    flatten_cache_key = (
        "prefill_flatten",
        T,
        E,
        int(num_blocks),
        int(f_len),
        int(output_len),
        int(lnc),
    )
    flatten_kernel = cache.get(flatten_cache_key)
    if flatten_kernel is None:
        entry = _make_prefill_flatten_entry(
            token_bucket=T,
            local_num_experts=E,
            num_blocks=int(num_blocks),
            f_len=int(f_len),
            output_len=int(output_len),
            logical_nc_config=int(lnc),
        )
        token_lists_sample = np.full((E, T), _SKIP_DMA, dtype=np.int32)
        row_offsets_sample = np.zeros((E,), dtype=np.int32)
        flatten_name = (
            f"dsv4_moe_sched_prefill_flatten_t{T}_e{E}"
            f"_n{int(num_blocks)}_f{int(f_len)}_o{int(output_len)}"
            f"_lnc{int(lnc)}"
        )
        flatten_kernel = compile_and_load_neff_with_lock(
            _device_kernel_cls,
            entry,
            token_lists_sample,
            row_offsets_sample,
            name=flatten_name,
            build_dir=str(artifacts_dir) if artifacts_dir else None,
            namespace="moe_schedule",
        )
        cache[flatten_cache_key] = flatten_kernel

    aff_dev = _device_tensor_cls.from_numpy(
        np.zeros((T * E, 1), dtype=ml_dtypes.bfloat16),
        name="moe_aff",
    )
    token_lists_dev = _device_tensor_cls.from_numpy(
        np.full((E, T), _SKIP_DMA, dtype=np.int32),
        name="moe_token_lists",
    )
    row_offsets_dev = _device_tensor_cls.from_numpy(
        np.zeros((E,), dtype=np.int32),
        name="moe_row_offsets",
    )
    tp_dev = _device_tensor_cls.from_numpy(
        np.full((int(num_blocks), int(MOE_BLOCK_SIZE)), _SKIP_DMA, dtype=np.int32),
        name="moe_tp2id",
    )
    b2e_dev = _device_tensor_cls.from_numpy(
        np.zeros((int(num_blocks),), dtype=np.int32),
        name="moe_b2e",
    )
    ep_start_dev = _device_tensor_cls.from_numpy(
        np.asarray([int(ep_rank) * E], dtype=np.int32),
        name="moe_ep_start",
    )
    index_kernel(
        inputs={"weights": weights, "indices": indices, "ep_start": ep_start_dev},
        outputs={
            "output0": aff_dev,
            "output1": token_lists_dev,
            "output2": row_offsets_dev,
            "output3": b2e_dev,
        },
    )
    flatten_kernel(
        inputs={"token_lists": token_lists_dev, "row_offsets": row_offsets_dev},
        outputs={"output0": tp_dev},
    )
    return DeviceMoESchedule(
        expert_affinities_masked=aff_dev,
        token_position_to_id=tp_dev,
        block_to_expert=b2e_dev,
        num_blocks=int(num_blocks),
        num_static_blocks=int(num_static_blocks),
    )


def run_decode_moe_schedule_device(
    *,
    weights: Any,
    indices: Any,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
    artifacts_dir: str | Path | None = None,
    kernel_cache: dict[tuple[Any, ...], Any] | None = None,
    _device_kernel_cls: Any | None = None,
    _device_tensor_cls: Any | None = None,
) -> DeviceMoESchedule:
    """Build decode affinities/static block metadata on device."""
    if not (is_device_tensor(weights) and is_device_tensor(indices)):
        raise TypeError("weights and indices must be device tensors")

    T, K = (int(v) for v in weights.shape)
    if T > int(MOE_BLOCK_SIZE):
        raise ValueError(f"decode schedule requires T <= {MOE_BLOCK_SIZE}, got {T}")
    if tuple(int(v) for v in indices.shape) != (T, K):
        raise ValueError(
            f"weights/indices shape mismatch: {tuple(weights.shape)} vs {tuple(indices.shape)}"
        )
    E = int(local_num_experts)

    if _device_kernel_cls is None or _device_tensor_cls is None:
        from nkipy_serving.models._device_utils import (
            _get_device_kernel_cls,
            _get_device_tensor_cls,
        )

        if _device_kernel_cls is None:
            _device_kernel_cls = _get_device_kernel_cls()
        if _device_tensor_cls is None:
            _device_tensor_cls = _get_device_tensor_cls()

    cache = kernel_cache if kernel_cache is not None else {}
    cache_key = (
        "decode",
        T,
        K,
        E,
        int(ep_degree),
        int(ep_rank),
        str(getattr(weights, "dtype", "unknown")),
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        entry = _make_decode_schedule_entry(
            token_bucket=T,
            local_num_experts=E,
            ep_degree=int(ep_degree),
            ep_rank=int(ep_rank),
        )
        weights_sample_dtype = _tensor_sample_dtype(weights)
        weights_sample = np.zeros((T, K), dtype=weights_sample_dtype)
        indices_sample = np.zeros((T, K), dtype=np.int32)
        dtype_name = _dtype_cache_name(weights_sample_dtype)
        schedule_name = (
            f"dsv4_moe_sched_decode_t{T}_e{E}_k{K}"
            f"_ep{int(ep_degree)}r{int(ep_rank)}_{dtype_name}"
        )
        kernel = compile_and_load_neff_with_lock(
            _device_kernel_cls,
            entry,
            weights_sample,
            indices_sample,
            name=schedule_name,
            build_dir=str(artifacts_dir) if artifacts_dir else None,
            namespace="moe_schedule",
        )
        cache[cache_key] = kernel

    aff_t_dev = _device_tensor_cls.from_numpy(
        np.zeros((E, T), dtype=ml_dtypes.bfloat16),
        name="moe_aff_T",
    )
    token_position_to_id = np.full(
        (E, int(MOE_BLOCK_SIZE)),
        _SKIP_DMA,
        dtype=np.int32,
    )
    token_position_to_id[:, :T] = np.arange(T, dtype=np.int32)[None, :]
    tp_dev = _device_tensor_cls.from_numpy(
        token_position_to_id,
        name="moe_decode_tp2id",
    )
    b2e_dev = _device_tensor_cls.from_numpy(
        np.arange(E, dtype=np.int32),
        name="moe_decode_b2e",
    )
    kernel(
        inputs={"weights": weights, "indices": indices},
        outputs={"output0": aff_t_dev},
    )
    return DeviceMoESchedule(
        expert_affinities_masked=aff_t_dev,
        token_position_to_id=tp_dev,
        block_to_expert=b2e_dev,
        num_blocks=E,
        num_static_blocks=E,
    )
