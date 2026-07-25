"""Compressor execution helpers for DSV4 sampled runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.execution_capabilities import (
    Dsv4ExecutionCapabilities,
)
from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _state_owner_ids_from_batch,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    decode_pool_from_device_state as _decode_pool_from_device_state,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    mirror_compressor_input_to_device_state as _mirror_compressor_input_to_device_state,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    prefill_pool_from_device_slab as _prefill_pool_from_device_slab,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_decode_dual_state_cache_swa_scatter_device as _run_compressor_decode_dual_state_cache_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_decode_dual_state_swa_scatter_device as _run_compressor_decode_dual_state_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_decode_state_cache_scatter_device as _run_compressor_decode_state_cache_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_decode_state_cache_swa_scatter_device as _run_compressor_decode_state_cache_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_decode_state_swa_scatter_device as _run_compressor_decode_state_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_prefill_dual_state_cache_swa_scatter_device as _run_compressor_prefill_dual_state_cache_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_prefill_state_cache_scatter_device as _run_compressor_prefill_state_cache_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_prefill_state_cache_swa_scatter_device as _run_compressor_prefill_state_cache_swa_scatter_device,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_compressor_scatter_rows_device as _run_compressor_scatter_rows_device,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)


@dataclass(slots=True)
class Dsv4DeferredSwaMirror:
    swa_kv_cache: Any
    swa_rows: Any
    window_size: int
    start_pos: int
    bsz: int
    seqlen: int
    owner_ids: np.ndarray
    owner_ids_dev: Any | None = None
    positions_dev: Any | None = None


@dataclass(slots=True)
class Dsv4DeferredIndexerState:
    compressor: Any
    kv: Any
    score: Any
    device_state: Any
    consumed: bool = False
    prefill_scatter_rows: Any | None = None
    decode_scatter_rows: Any | None = None


def _state_attr_has_tensor_ref(state: Any, attr: str) -> bool:
    return hasattr(getattr(state, attr, None), "tensor_ref")


def _prefill_state_tail_len(
    *,
    start_pos: int,
    seqlen: int,
    ratio: int,
    device_state: Any,
    compressor: Any,
) -> int:
    if int(start_pos) != 0 or int(ratio) <= 0:
        return 0
    state_spec = getattr(device_state, "spec", None)
    overlap = bool(
        getattr(state_spec, "overlap", getattr(compressor, "overlap", False))
    )
    if overlap:
        return min(int(seqlen), int(ratio) + int(seqlen) % int(ratio))
    return int(seqlen) % int(ratio)


def _trim_prefill_scatter_rows(rows: Any, pool_shape: tuple[int, int]) -> Any:
    rows_shape = tuple(int(dim) for dim in getattr(rows, "shape", ()))
    if not rows_shape or rows_shape == pool_shape:
        return rows
    # A bucketed prologue (short lane re-aliased up to its compile bucket)
    # returns rows at bucket clen; the real rows are the leading pool rows.
    if (
        len(rows_shape) == 2
        and rows_shape[0] > pool_shape[0]
        and rows_shape[1] == pool_shape[1]
    ):
        return _alias_device_value_first_dim_slice(
            rows,
            start=0,
            size=int(pool_shape[0]),
        )
    raise RuntimeError(
        "DSV4 precomputed compressor prefill rows shape mismatch: "
        f"got {rows_shape}, expected {pool_shape}"
    )


def _run_compressor(
    fns: Dsv4GraphFns,
    compressor: Any,
    x: np.ndarray,
    start_pos: int,
    *,
    build_dir: str | None,
    device_state: Any,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    token_positions: Any | None = None,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None = None,
    precomputed_kv_score: tuple[Any, Any] | None = None,
    precomputed_prefill_scatter_rows: Any | None = None,
    precomputed_decode_scatter_rows: Any | None = None,
    deferred_swa_mirror: Dsv4DeferredSwaMirror | None = None,
    deferred_indexer_state: Dsv4DeferredIndexerState | None = None,
    real_prefill_seqlen: int | None = None,
) -> None:
    """Mutate the persistent compressor state for one compressed layer.

    Device-only: a ``device_state`` is required. The kv/score projections
    feed the device slab scatter and the prefill/decode pool kernels;
    the pooled output flows through the fused post+qdq+scatter tail.
    """
    caps = Dsv4ExecutionCapabilities.from_graph_fns(fns)
    bsz, seqlen, _ = x.shape
    ratio = int(compressor.compress_ratio)
    token_owner_ids = _state_owner_ids_from_batch(
        bsz=bsz,
        seqlen=seqlen,
        owner_ids=owner_ids,
    )
    request_owner_ids = token_owner_ids[::seqlen].astype(np.int32, copy=False)
    request_owner_ids_dev = None
    if owner_ids_dev is not None and int(seqlen) == 1:
        request_owner_ids_dev = _alias_device_value_first_dim_slice(
            owner_ids_dev,
            start=0,
            size=int(bsz),
        )
    token_owner_ids_dev = None
    if owner_ids_dev is not None and int(seqlen) > 1:
        token_owner_ids_dev = _alias_device_value_first_dim_slice(
            owner_ids_dev,
            start=0,
            size=int(bsz) * int(seqlen),
        )
    request_positions_dev = None
    if token_positions is not None and int(seqlen) == 1:
        request_positions_dev = _alias_device_value_first_dim_slice(
            token_positions,
            start=0,
            size=int(bsz),
        )

    def _scratch(
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        if attention_scratch is None:
            return None
        return attention_scratch(
            kind,
            tuple(int(dim) for dim in shape),
            dtype,
        )

    def _output_kwargs(name: str, out: Any | None) -> dict[str, Any]:
        if out is None:
            return {}
        return {"_nkipy_output_tensors": {name: out}}

    width = int(getattr(compressor.wkv, "shape", (0,))[0])
    flat_shape = (int(bsz) * int(seqlen), width)
    if precomputed_kv_score is None:
        if caps.require_precomputed_compressor_kv_score:
            raise RuntimeError(
                "DSV4 product compressor requires KV/score from the fused "
                "attention QKV prologue"
            )
        # Fused two-linear + reshape-to-[B*S, W] + bf16 cast stays on device.
        kv_bf_dev, score_bf_dev = fns["compressor_kv_score_bf16"](
            x,
            compressor.wkv,
            compressor.wgate,
            **(
                {}
                if attention_scratch is None
                else {
                    "_nkipy_output_tensors": {
                        "output0": _scratch(
                            "compressor_kv_bf16",
                            flat_shape,
                            ml_dtypes.bfloat16,
                        ),
                        "output1": _scratch(
                            "compressor_score_bf16",
                            flat_shape,
                            ml_dtypes.bfloat16,
                        ),
                    }
                }
            ),
        )
    else:
        kv_bf_dev, score_bf_dev = precomputed_kv_score
    prefill_compile_seqlen = int(seqlen)
    bucketed_real_prefill_seqlen = real_prefill_seqlen
    if int(start_pos) == 0 and int(bsz) > 0:
        kv_shape = tuple(int(dim) for dim in getattr(kv_bf_dev, "shape", ()))
        if len(kv_shape) == 2 and int(kv_shape[0]) % int(bsz) == 0:
            kv_seqlen = int(kv_shape[0]) // int(bsz)
            if kv_seqlen > int(prefill_compile_seqlen):
                prefill_compile_seqlen = kv_seqlen
                if bucketed_real_prefill_seqlen is None:
                    bucketed_real_prefill_seqlen = int(seqlen)
    state_spec = getattr(device_state, "spec", None)
    compressor_overlap = bool(
        getattr(state_spec, "overlap", getattr(compressor, "overlap", False))
    )
    prefill_state_tail_len = _prefill_state_tail_len(
        start_pos=int(start_pos),
        seqlen=int(seqlen),
        ratio=int(ratio),
        device_state=device_state,
        compressor=compressor,
    )
    decode_compression_boundary = (int(start_pos) + 1) % int(ratio) == 0
    defer_decode_state_write = (
        int(start_pos) != 0
        and int(seqlen) == 1
        and decode_compression_boundary
        and precomputed_decode_scatter_rows is not None
        and _state_attr_has_tensor_ref(device_state, "kv_score_state")
        and _state_attr_has_tensor_ref(device_state, "compressed_kv_cache")
        and not isinstance(kv_bf_dev, np.ndarray)
        and not isinstance(score_bf_dev, np.ndarray)
        and not isinstance(precomputed_decode_scatter_rows, np.ndarray)
    )
    defer_decode_swa_state_write = (
        deferred_swa_mirror is not None
        and int(start_pos) != 0
        and int(seqlen) == 1
        and (
            not decode_compression_boundary
            or (
                precomputed_decode_scatter_rows is not None
                and not isinstance(precomputed_decode_scatter_rows, np.ndarray)
                and _state_attr_has_tensor_ref(device_state, "compressed_kv_cache")
            )
        )
        and _state_attr_has_tensor_ref(device_state, "kv_score_state")
        and not isinstance(kv_bf_dev, np.ndarray)
        and not isinstance(score_bf_dev, np.ndarray)
    )
    defer_prefill_state_write = (
        int(start_pos) == 0
        and int(seqlen) >= int(ratio)
        and int(prefill_state_tail_len) > 0
        and precomputed_prefill_scatter_rows is not None
        and _state_attr_has_tensor_ref(device_state, "kv_score_state")
        and _state_attr_has_tensor_ref(device_state, "compressed_kv_cache")
        and not isinstance(kv_bf_dev, np.ndarray)
        and not isinstance(score_bf_dev, np.ndarray)
        and not isinstance(precomputed_prefill_scatter_rows, np.ndarray)
    )
    defer_prefill_swa_state_write = (
        defer_prefill_state_write and deferred_swa_mirror is not None
    )
    # For non-overlap layers (ratio-128) with bsz=1 prefill: the bucketed serve
    # path produces keep = token_bucket % ratio = 0, so mirror_ returns early
    # without writing state. ALL prefill state writes for these layers are dead
    # at serve (never hit). Skip to avoid per-length NEFF explosion.
    _skip_dead_state_write = bool(
        int(start_pos) == 0
        and int(bsz) == 1
        and int(ratio) > 1
        and not compressor_overlap
    )
    if (
        not defer_decode_state_write
        and not defer_decode_swa_state_write
        and not defer_prefill_state_write
        and not _skip_dead_state_write
    ):
        _mirror_compressor_input_to_device_state(
            compressor,
            kv_bf_dev,
            score_bf_dev,
            start_pos,
            bsz=bsz,
            seqlen=seqlen,
            device_state=device_state,
            build_dir=build_dir,
            owner_ids=token_owner_ids,
            owner_ids_dev=request_owner_ids_dev,
            positions_dev=request_positions_dev,
        )

    cutoff = 0
    if start_pos == 0:
        effective_seqlen = (
            int(prefill_compile_seqlen)
            if bucketed_real_prefill_seqlen is not None
            else int(seqlen)
        )
        should_compress = effective_seqlen >= ratio
        remainder = effective_seqlen % ratio
        cutoff = effective_seqlen - remainder
        if not should_compress:
            return
        pool_shape = (
            int(bsz) * (int(cutoff) // int(ratio)),
            int(compressor.head_dim),
        )
        clen = cutoff // ratio
        if precomputed_prefill_scatter_rows is not None:
            precomputed_prefill_scatter_rows = _trim_prefill_scatter_rows(
                precomputed_prefill_scatter_rows,
                pool_shape,
            )
            if defer_prefill_swa_state_write:
                if deferred_swa_mirror is None:
                    raise RuntimeError(
                        "DSV4 deferred prefill SWA write requires deferred_swa_mirror"
                    )
                indexer_prefill_rows = (
                    None
                    if deferred_indexer_state is None
                    else deferred_indexer_state.prefill_scatter_rows
                )
                indexer_prefill_rows_shape = tuple(
                    int(dim) for dim in getattr(indexer_prefill_rows, "shape", ())
                )
                main_prefill_rows_shape = tuple(
                    int(dim)
                    for dim in getattr(precomputed_prefill_scatter_rows, "shape", ())
                )
                indexer_state_spec = (
                    None
                    if deferred_indexer_state is None
                    else getattr(deferred_indexer_state.device_state, "spec", None)
                )
                indexer_compressor = (
                    None
                    if deferred_indexer_state is None
                    else deferred_indexer_state.compressor
                )
                indexer_ratio = (
                    0
                    if indexer_compressor is None
                    else int(getattr(indexer_compressor, "compress_ratio", 0))
                )
                indexer_overlap = bool(
                    getattr(
                        indexer_state_spec,
                        "overlap",
                        getattr(indexer_compressor, "overlap", False),
                    )
                )
                if indexer_overlap and indexer_ratio > 0:
                    indexer_tail_len = min(
                        int(seqlen),
                        int(indexer_ratio) + int(seqlen) % int(indexer_ratio),
                    )
                elif indexer_ratio > 0:
                    indexer_tail_len = int(seqlen) % int(indexer_ratio)
                else:
                    indexer_tail_len = 0
                if (
                    deferred_indexer_state is not None
                    and not bool(deferred_indexer_state.consumed)
                    and not isinstance(deferred_indexer_state.kv, np.ndarray)
                    and not isinstance(deferred_indexer_state.score, np.ndarray)
                    and indexer_prefill_rows is not None
                    and not isinstance(indexer_prefill_rows, np.ndarray)
                    and indexer_prefill_rows_shape
                    and main_prefill_rows_shape
                    and int(indexer_prefill_rows_shape[0])
                    == int(main_prefill_rows_shape[0])
                    and int(indexer_tail_len) == int(prefill_state_tail_len)
                    and _state_attr_has_tensor_ref(
                        deferred_indexer_state.device_state,
                        "kv_score_state",
                    )
                    and _state_attr_has_tensor_ref(
                        deferred_indexer_state.device_state,
                        "compressed_kv_cache",
                    )
                ):
                    _run_compressor_prefill_dual_state_cache_swa_scatter_device(
                        compressor,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        swa_start_pos=int(deferred_swa_mirror.start_pos),
                        swa_bsz=int(deferred_swa_mirror.bsz),
                        swa_seqlen=int(deferred_swa_mirror.seqlen),
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        scatter_rows=precomputed_prefill_scatter_rows,
                        indexer_compressor=deferred_indexer_state.compressor,
                        indexer_kv=deferred_indexer_state.kv,
                        indexer_score=deferred_indexer_state.score,
                        indexer_scatter_rows=indexer_prefill_rows,
                        bsz=bsz,
                        seqlen=effective_seqlen,
                        clen=clen,
                        device_state=device_state,
                        indexer_device_state=deferred_indexer_state.device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=token_owner_ids,
                        owner_ids_dev=token_owner_ids_dev,
                        positions_dev=token_positions,
                        swa_owner_ids=deferred_swa_mirror.owner_ids,
                        swa_owner_ids_dev=deferred_swa_mirror.owner_ids_dev,
                        swa_positions_dev=deferred_swa_mirror.positions_dev,
                        owner_id_stride=effective_seqlen,
                        real_seqlen=bucketed_real_prefill_seqlen,
                    )
                    deferred_indexer_state.consumed = True
                else:
                    _run_compressor_prefill_state_cache_swa_scatter_device(
                        compressor,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        swa_start_pos=int(deferred_swa_mirror.start_pos),
                        swa_bsz=int(deferred_swa_mirror.bsz),
                        swa_seqlen=int(deferred_swa_mirror.seqlen),
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        scatter_rows=precomputed_prefill_scatter_rows,
                        bsz=bsz,
                        seqlen=effective_seqlen,
                        clen=clen,
                        device_state=device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=token_owner_ids,
                        owner_ids_dev=token_owner_ids_dev,
                        positions_dev=token_positions,
                        swa_owner_ids=deferred_swa_mirror.owner_ids,
                        swa_owner_ids_dev=deferred_swa_mirror.owner_ids_dev,
                        swa_positions_dev=deferred_swa_mirror.positions_dev,
                        owner_id_stride=effective_seqlen,
                        real_seqlen=bucketed_real_prefill_seqlen,
                    )
            elif defer_prefill_state_write:
                _run_compressor_prefill_state_cache_scatter_device(
                    compressor,
                    kv=kv_bf_dev,
                    score=score_bf_dev,
                    scatter_rows=precomputed_prefill_scatter_rows,
                    bsz=bsz,
                    seqlen=seqlen,
                    clen=clen,
                    device_state=device_state,
                    build_dir=build_dir,
                    owner_ids=token_owner_ids,
                    owner_ids_dev=token_owner_ids_dev,
                    positions_dev=token_positions,
                    owner_id_stride=seqlen,
                )
            else:
                _run_compressor_scatter_rows_device(
                    compressor,
                    start_pos,
                    scatter_rows=precomputed_prefill_scatter_rows,
                    bsz=bsz,
                    clen=clen,
                    device_state=device_state,
                    build_dir=build_dir,
                    owner_ids=request_owner_ids,
                    token_owner_ids_dev=token_owner_ids_dev,
                    owner_id_stride=seqlen,
                )
            return
        if remainder == 0:
            kv_for_pool: Any = kv_bf_dev
            score_for_pool: Any = score_bf_dev
        else:
            prefix_aliasable = caps.prefix_two_token_flats_aliases_prefix and (
                int(bsz) == 1 or int(cutoff) == int(seqlen)
            )
            kv_for_pool, score_for_pool = fns["prefix_two_token_flats"](
                kv_bf_dev,
                score_bf_dev,
                bsz=bsz,
                seqlen=seqlen,
                cutoff=cutoff,
                **(
                    {}
                    if attention_scratch is None or prefix_aliasable
                    else {
                        "_nkipy_output_tensors": {
                            "output0": _scratch(
                                "compressor_prefix_kv",
                                (int(bsz) * int(cutoff), width),
                                ml_dtypes.bfloat16,
                            ),
                            "output1": _scratch(
                                "compressor_prefix_score",
                                (int(bsz) * int(cutoff), width),
                                ml_dtypes.bfloat16,
                            ),
                        }
                    }
                ),
            )
        kv_pool = _prefill_pool_from_device_slab(
            compressor,
            kv_for_pool,
            score_for_pool,
            bsz=bsz,
            seqlen=cutoff,
            build_dir=build_dir,
            device_state=device_state,
            return_device=True,
            output=_scratch("compressor_prefill_pool", pool_shape, np.float32),
        )
    else:
        if precomputed_prefill_scatter_rows is not None:
            raise RuntimeError(
                "DSV4 precomputed compressor prefill rows cannot be used on decode"
            )
        should_compress = (start_pos + 1) % ratio == 0
        if not should_compress:
            if precomputed_decode_scatter_rows is not None:
                raise RuntimeError(
                    "DSV4 precomputed compressor decode rows were provided on a "
                    f"non-compression boundary: start_pos={int(start_pos)} "
                    f"ratio={int(ratio)}"
                )
            if defer_decode_swa_state_write:
                if deferred_swa_mirror is None:
                    raise RuntimeError(
                        "DSV4 deferred decode SWA write requires deferred_swa_mirror"
                    )
                if (
                    deferred_indexer_state is not None
                    and not bool(deferred_indexer_state.consumed)
                    and not isinstance(deferred_indexer_state.kv, np.ndarray)
                    and not isinstance(deferred_indexer_state.score, np.ndarray)
                    and _state_attr_has_tensor_ref(
                        deferred_indexer_state.device_state,
                        "kv_score_state",
                    )
                ):
                    _run_compressor_decode_dual_state_swa_scatter_device(
                        compressor,
                        start_pos,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        indexer_compressor=deferred_indexer_state.compressor,
                        indexer_kv=deferred_indexer_state.kv,
                        indexer_score=deferred_indexer_state.score,
                        bsz=bsz,
                        device_state=device_state,
                        indexer_device_state=deferred_indexer_state.device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=request_owner_ids,
                        owner_ids_dev=request_owner_ids_dev,
                        positions_dev=request_positions_dev,
                    )
                    deferred_indexer_state.consumed = True
                else:
                    _run_compressor_decode_state_swa_scatter_device(
                        compressor,
                        start_pos,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        bsz=bsz,
                        device_state=device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=request_owner_ids,
                        owner_ids_dev=request_owner_ids_dev,
                        positions_dev=request_positions_dev,
                    )
            return
        pool_shape = (int(bsz), int(compressor.head_dim))
        if precomputed_decode_scatter_rows is not None:
            rows_shape = tuple(
                int(dim)
                for dim in getattr(precomputed_decode_scatter_rows, "shape", ())
            )
            if rows_shape and rows_shape != pool_shape:
                raise RuntimeError(
                    "DSV4 precomputed compressor decode rows shape mismatch: "
                    f"got {rows_shape}, expected {pool_shape}"
                )
            if defer_decode_swa_state_write:
                if deferred_swa_mirror is None:
                    raise RuntimeError(
                        "DSV4 deferred decode SWA write requires deferred_swa_mirror"
                    )
                if (
                    deferred_indexer_state is not None
                    and not bool(deferred_indexer_state.consumed)
                    and not isinstance(deferred_indexer_state.kv, np.ndarray)
                    and not isinstance(deferred_indexer_state.score, np.ndarray)
                    and not isinstance(
                        deferred_indexer_state.decode_scatter_rows,
                        np.ndarray,
                    )
                    and deferred_indexer_state.decode_scatter_rows is not None
                    and _state_attr_has_tensor_ref(
                        deferred_indexer_state.device_state,
                        "kv_score_state",
                    )
                    and _state_attr_has_tensor_ref(
                        deferred_indexer_state.device_state,
                        "compressed_kv_cache",
                    )
                ):
                    _run_compressor_decode_dual_state_cache_swa_scatter_device(
                        compressor,
                        start_pos,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        scatter_rows=precomputed_decode_scatter_rows,
                        indexer_compressor=deferred_indexer_state.compressor,
                        indexer_kv=deferred_indexer_state.kv,
                        indexer_score=deferred_indexer_state.score,
                        indexer_scatter_rows=deferred_indexer_state.decode_scatter_rows,
                        bsz=bsz,
                        device_state=device_state,
                        indexer_device_state=deferred_indexer_state.device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=request_owner_ids,
                        owner_ids_dev=request_owner_ids_dev,
                        positions_dev=request_positions_dev,
                    )
                    deferred_indexer_state.consumed = True
                else:
                    _run_compressor_decode_state_cache_swa_scatter_device(
                        compressor,
                        start_pos,
                        swa_kv_cache=deferred_swa_mirror.swa_kv_cache,
                        swa_rows=deferred_swa_mirror.swa_rows,
                        kv=kv_bf_dev,
                        score=score_bf_dev,
                        scatter_rows=precomputed_decode_scatter_rows,
                        bsz=bsz,
                        device_state=device_state,
                        window_size=int(deferred_swa_mirror.window_size),
                        build_dir=build_dir,
                        owner_ids=request_owner_ids,
                        owner_ids_dev=request_owner_ids_dev,
                        positions_dev=request_positions_dev,
                    )
            elif defer_decode_state_write:
                _run_compressor_decode_state_cache_scatter_device(
                    compressor,
                    start_pos,
                    kv=kv_bf_dev,
                    score=score_bf_dev,
                    scatter_rows=precomputed_decode_scatter_rows,
                    bsz=bsz,
                    device_state=device_state,
                    build_dir=build_dir,
                    owner_ids=request_owner_ids,
                    owner_ids_dev=request_owner_ids_dev,
                    positions_dev=request_positions_dev,
                )
            else:
                _run_compressor_scatter_rows_device(
                    compressor,
                    start_pos,
                    scatter_rows=precomputed_decode_scatter_rows,
                    bsz=bsz,
                    clen=1,
                    device_state=device_state,
                    build_dir=build_dir,
                    owner_ids=request_owner_ids,
                    owner_ids_dev=request_owner_ids_dev,
                    positions_dev=request_positions_dev,
                    token_owner_ids_dev=token_owner_ids_dev,
                    owner_id_stride=seqlen,
                )
            return
        kv_pool = None
        clen = 1

    if start_pos == 0:
        freq_positions = np.arange(0, int(cutoff), int(ratio), dtype=np.int32)
    else:
        freq_positions = np.array(
            [int(start_pos) + 1 - int(ratio)],
            dtype=np.int32,
        )
    freq_positions_input = token_positions
    source_token_positions = freq_positions_input is not None
    if freq_positions_input is None:
        freq_positions_input = freq_positions
    freq_start_pos = 0 if int(start_pos) == 0 else 1
    freq_seqlen = int(prefill_compile_seqlen) if freq_start_pos == 0 else 1
    post_qdq_from_table = fns.get("compressor_post_qdq_from_freq_table")
    decode_pool_post_qdq_from_table = fns.get(
        "compressor_decode_pool_post_qdq_from_state_freq_table"
    )
    freqs_cos = getattr(compressor, "freqs_cos", None)
    freqs_sin = getattr(compressor, "freqs_sin", None)
    can_fuse_post_qdq_from_table = (
        callable(post_qdq_from_table)
        and freqs_cos is not None
        and freqs_sin is not None
    )
    can_fuse_decode_pool_post_qdq_from_table = (
        int(start_pos) != 0
        and callable(decode_pool_post_qdq_from_table)
        and freqs_cos is not None
        and freqs_sin is not None
        and hasattr(device_state, "kv_score_state")
        and hasattr(device_state, "spec")
    )
    if caps.require_fused_compressor_post_qdq and (
        not can_fuse_post_qdq_from_table
        and not can_fuse_decode_pool_post_qdq_from_table
    ):
        raise RuntimeError(
            "DSV4 product compressor requires fused post-QDQ frequency-table "
            "path; standalone compressor post-pool/norm/qDQ fragments are not "
            "part of the product path"
        )
    # Post-pool + qdq + compressed-KV scatter, fused on device. Non-rotating
    # main compressors keep head_dim on the free axis and support V4's d=512.
    # Rotating indexer compressors still need the Hadamard kernel envelope.
    if compressor.eps != 1e-6 or (
        bool(compressor.rotate)
        and (
            compressor.head_dim > 128 or compressor.head_dim & (compressor.head_dim - 1)
        )
    ):
        raise RuntimeError(
            f"compressor head_dim={compressor.head_dim}, "
            f"eps={compressor.eps} outside NKI kernel envelope "
            "(non-rotating d=512 or rotating d<=128 power-of-two, eps == 1e-6)"
        )
    post_shape = (int(bsz) * int(clen), int(compressor.head_dim))
    rotate = bool(compressor.rotate)
    qdq_block_size = 32 if rotate else 64

    if can_fuse_decode_pool_post_qdq_from_table:
        spec = device_state.spec
        end_positions = (
            request_positions_dev
            if request_positions_dev is not None
            else np.full((int(bsz),), int(start_pos), dtype=np.int32)
        )
        scatter_rows = decode_pool_post_qdq_from_table(
            device_state.kv_score_state,
            (
                request_owner_ids_dev
                if request_owner_ids_dev is not None
                else request_owner_ids
            ),
            end_positions,
            compressor.norm_weight,
            freqs_cos,
            freqs_sin,
            freq_positions_input,
            bsz=bsz,
            ratio=ratio,
            head_dim=int(compressor.head_dim),
            state_width=int(spec.state_width),
            ring_size=int(spec.ring_size),
            overlap=bool(spec.overlap),
            source_token_positions=source_token_positions,
            compress_ratio=ratio,
            start_pos=freq_start_pos,
            seqlen=freq_seqlen,
            rope_head_dim=int(compressor.rope_head_dim),
            block_size=int(qdq_block_size),
            fp8_max=240.0,
            rotate=rotate,
            eps=float(compressor.eps),
            **_output_kwargs(
                "output0",
                _scratch(
                    "compressor_decode_post_qdq_bf16",
                    post_shape,
                    ml_dtypes.bfloat16,
                ),
            ),
        )
        _run_compressor_scatter_rows_device(
            compressor,
            start_pos,
            scatter_rows=scatter_rows,
            bsz=bsz,
            clen=clen,
            device_state=device_state,
            build_dir=build_dir,
            owner_ids=request_owner_ids,
            owner_ids_dev=request_owner_ids_dev,
            positions_dev=request_positions_dev,
            token_owner_ids_dev=token_owner_ids_dev,
            owner_id_stride=seqlen,
        )
        return

    if int(start_pos) != 0 and kv_pool is None:
        kv_pool = _decode_pool_from_device_state(
            device_state,
            bsz=bsz,
            end_pos=int(start_pos),
            build_dir=build_dir,
            return_device=True,
            owner_ids=request_owner_ids,
            owner_ids_dev=request_owner_ids_dev,
            end_positions_dev=request_positions_dev,
            output=_scratch("compressor_decode_pool", pool_shape, np.float32),
        )

    if can_fuse_post_qdq_from_table:
        scatter_rows = post_qdq_from_table(
            kv_pool,
            compressor.norm_weight,
            freqs_cos,
            freqs_sin,
            freq_positions_input,
            bsz=bsz,
            clen=clen,
            source_token_positions=source_token_positions,
            compress_ratio=ratio,
            start_pos=freq_start_pos,
            seqlen=freq_seqlen,
            rope_head_dim=int(compressor.rope_head_dim),
            block_size=int(qdq_block_size),
            fp8_max=240.0,
            rotate=rotate,
            eps=float(compressor.eps),
            **_output_kwargs(
                "output0",
                _scratch("compressor_post_qdq_bf16", post_shape, ml_dtypes.bfloat16),
            ),
        )
        _run_compressor_scatter_rows_device(
            compressor,
            start_pos,
            scatter_rows=scatter_rows,
            bsz=bsz,
            clen=clen,
            device_state=device_state,
            build_dir=build_dir,
            owner_ids=request_owner_ids,
            owner_ids_dev=request_owner_ids_dev,
            positions_dev=request_positions_dev,
        )
        return

    # The product graph always provides one of the two fused post-QDQ paths
    # above (enforced by the ``_product_require_fused_compressor_post_qdq``
    # raise), so the standalone post-pool/norm/qDQ scatter fallback is
    # unreachable in any live runtime.
    raise RuntimeError(
        "DSV4 product compressor requires fused post-QDQ frequency-table path"
    )
