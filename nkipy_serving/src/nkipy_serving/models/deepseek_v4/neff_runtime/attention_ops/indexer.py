"""Indexer execution helpers for DSV4 sampled runtime."""

from __future__ import annotations

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
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.compressor import (
    _run_compressor,
)
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    mirror_compressor_input_to_device_state as _mirror_compressor_input_to_device_state,
)
from nkipy_serving.ops.deepseek_v4.indexer_state import (
    indexer_score_from_device_cache_adapter as _indexer_score_from_device_cache_adapter,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)


def _sparse_attention_topk_shape(
    *,
    start_pos: int,
    seqlen: int,
    k: int,
    sparse_attention_rows: int | None,
    sparse_attention_k_tile: int | None,
    sparse_attention_window_size: int | None,
) -> tuple[int, int] | None:
    if (
        sparse_attention_rows is None
        or sparse_attention_k_tile is None
        or sparse_attention_window_size is None
    ):
        return None
    win_width = (
        int(sparse_attention_window_size)
        if int(start_pos) > 0
        else min(int(seqlen), int(sparse_attention_window_size))
    )
    k_raw = int(win_width) + int(k)
    k_padded = (
        (k_raw + int(sparse_attention_k_tile) - 1) // int(sparse_attention_k_tile)
    ) * int(sparse_attention_k_tile)
    return int(k_padded), int(sparse_attention_rows)


def _topk_output_tensors(
    scratch: Callable[[str, tuple[int, ...], Any], Any | None],
    *,
    shape: tuple[int, int] | None,
    topk_name: str = "output0",
    mask_name: str = "output1",
) -> dict[str, Any]:
    if shape is None:
        return {}
    k_padded, rows = shape
    return {
        topk_name: scratch(
            "attention_topk_t",
            (int(k_padded), int(rows)),
            np.int32,
        ),
        mask_name: scratch(
            "attention_topk_mask",
            (int(rows), int(k_padded)),
            ml_dtypes.bfloat16,
        ),
    }


def _prefill_bucketed_indexer_x_alias(
    x: Any,
    *,
    bsz: int,
    live_seqlen: int,
    bucket_rows: int,
) -> tuple[Any, int]:
    if int(bucket_rows) <= int(bsz) * int(live_seqlen):
        return x, int(live_seqlen)
    if int(bsz) <= 0 or int(bucket_rows) % int(bsz) != 0:
        raise RuntimeError(
            "DSV4 bucketed indexer rows must divide batch size, got "
            f"rows={int(bucket_rows)} bsz={int(bsz)}"
        )
    bucket_seqlen = int(bucket_rows) // int(bsz)
    x_shape = tuple(int(dim) for dim in getattr(x, "shape", ()))
    if len(x_shape) != 3:
        raise RuntimeError(f"DSV4 bucketed indexer x must be [b,s,h], got {x_shape}")
    x_bucket = _alias_device_value_shape(
        x,
        (int(bsz), int(bucket_seqlen), int(x_shape[2])),
        default_name="dsv4_indexer_bucketed_x",
    )
    if x_bucket is None:
        raise RuntimeError(
            "DSV4 bucketed indexer fallback requires x to be backed by a "
            f"bucket-sized device buffer; got x_shape={x_shape} bucket_rows={int(bucket_rows)}"
        )
    return x_bucket, int(bucket_seqlen)


def _run_indexer(
    fns: Dsv4GraphFns,
    indexer: Any,
    x: np.ndarray,
    qr: Any | None,
    start_pos: int,
    offset: int,
    *,
    build_dir: str | None,
    device_state: Any,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    token_positions: Any | None = None,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None = None,
    sparse_attention_rows: int | None = None,
    sparse_attention_k_tile: int | None = None,
    sparse_attention_window_size: int | None = None,
    precomputed_compressor_kv_score: tuple[Any, Any] | None = None,
    precomputed_compressor_decode_scatter_rows: Any | None = None,
    precomputed_compressor_state_written: bool = False,
    precomputed_qw: tuple[Any, Any] | None = None,
    precomputed_empty_topk: tuple[Any, Any] | None = None,
) -> Any:
    """Score indexer queries and return the top-k as a DeviceTensor.

    ``qr`` may be numpy or DeviceTensor — it flows into
    ``indexer_project`` which auto-uploads numpy. The DSV4 contract pins
    ``indexer.head_dim == 128`` so the NKI score kernel always applies.
    """
    caps = Dsv4ExecutionCapabilities.from_graph_fns(fns)
    if indexer.head_dim != 128:
        raise RuntimeError(f"indexer.head_dim={indexer.head_dim} != 128 — unsupported")
    bsz, seqlen, _ = x.shape
    if precomputed_qw is not None and int(start_pos) == 0:
        q_T_pre, _w_pre = precomputed_qw
        q_T_shape = tuple(int(dim) for dim in getattr(q_T_pre, "shape", ()))
        if q_T_shape:
            x, seqlen = _prefill_bucketed_indexer_x_alias(
                x,
                bsz=int(bsz),
                live_seqlen=int(seqlen),
                bucket_rows=int(q_T_shape[0]),
            )
    token_owner_ids = _state_owner_ids_from_batch(
        bsz=bsz,
        seqlen=seqlen,
        owner_ids=owner_ids,
    )
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
    ratio = int(indexer.compress_ratio)
    rd = int(indexer.rope_head_dim)
    end_pos = start_pos + seqlen

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

    def _empty_topk_output_shapes() -> tuple[int, int] | None:
        return _sparse_attention_topk_shape(
            start_pos=int(start_pos),
            seqlen=int(seqlen),
            k=1,
            sparse_attention_rows=sparse_attention_rows,
            sparse_attention_k_tile=sparse_attention_k_tile,
            sparse_attention_window_size=sparse_attention_window_size,
        )

    def _empty_topk_output_tensors(
        *,
        topk_name: str = "output0",
        mask_name: str = "output1",
    ) -> dict[str, Any]:
        shape = _empty_topk_output_shapes()
        if attention_scratch is None or shape is None:
            return {}
        return _topk_output_tensors(
            _scratch,
            shape=shape,
            topk_name=topk_name,
            mask_name=mask_name,
        )

    def _empty_compressed_kv_topk() -> Any:
        token_topk_prep = fns.get("topk_tokens_concat_pad_sparse_attention_prep")
        if callable(token_topk_prep) and _empty_topk_output_shapes() is not None:
            output_tensors = _empty_topk_output_tensors()
            output_kwargs = (
                {"_nkipy_output_tensors": output_tensors} if output_tensors else {}
            )
            return token_topk_prep(
                x,
                window_size=int(sparse_attention_window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=0,
                rows=int(sparse_attention_rows),
                k_tile=int(sparse_attention_k_tile),
                **output_kwargs,
            )
        return fns["invalid_topk_from_tokens"](x, k=1)

    kv_len = end_pos // ratio
    if kv_len <= 0:
        if precomputed_empty_topk is not None:
            if precomputed_compressor_kv_score is None:
                raise RuntimeError(
                    "DSV4 precomputed empty indexer top-k requires "
                    "precomputed compressor KV/score"
                )
            kv_bf_dev, score_bf_dev = precomputed_compressor_kv_score
            _mirror_compressor_input_to_device_state(
                indexer.compressor,
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
            return precomputed_empty_topk
        if caps.require_precomputed_empty_indexer_topk:
            raise RuntimeError(
                "DSV4 product empty-indexer path requires top-k from the "
                "fused attention QKV/indexer prologue"
            )
        fused_empty = fns.get("compressor_kv_score_token_topk_prep")
        if callable(fused_empty) and _empty_topk_output_shapes() is not None:
            width = int(getattr(indexer.compressor.wkv, "shape", (0,))[0])
            output_tensors: dict[str, Any] = {}
            if attention_scratch is not None:
                flat_shape = (int(bsz) * int(seqlen), width)
                output_tensors.update(
                    {
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
                )
                output_tensors.update(
                    _empty_topk_output_tensors(
                        topk_name="output2",
                        mask_name="output3",
                    )
                )
            output_kwargs = (
                {"_nkipy_output_tensors": output_tensors} if output_tensors else {}
            )
            fused_args = (
                (x, indexer.compressor.wkv, indexer.compressor.wgate)
                if token_positions is None
                else (
                    x,
                    indexer.compressor.wkv,
                    indexer.compressor.wgate,
                    token_positions,
                )
            )
            kv_bf_dev, score_bf_dev, topk_t, mask = fused_empty(
                *fused_args,
                window_size=int(sparse_attention_window_size),
                ratio=int(ratio),
                offset=int(offset),
                start_pos=int(start_pos),
                max_c_len=0,
                rows=int(sparse_attention_rows),
                k_tile=int(sparse_attention_k_tile),
                **output_kwargs,
            )
            _mirror_compressor_input_to_device_state(
                indexer.compressor,
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
            return topk_t, mask
        _run_compressor(
            fns,
            indexer.compressor,
            x,
            start_pos,
            build_dir=build_dir,
            device_state=device_state,
            owner_ids=token_owner_ids,
            owner_ids_dev=owner_ids_dev,
            token_positions=token_positions,
            attention_scratch=attention_scratch,
        )
        return _empty_compressed_kv_topk()

    q_shape = (
        int(bsz),
        int(seqlen),
        int(indexer.n_heads),
        int(indexer.head_dim),
    )
    q_T_shape = (
        int(bsz) * int(seqlen),
        int(indexer.head_dim),
        int(indexer.n_heads),
    )
    w_flat_shape = (int(bsz) * int(seqlen), int(indexer.n_heads))
    q_w_outputs = (
        {}
        if attention_scratch is None
        else {
            "_nkipy_output_tensors": {
                "output0": _scratch(
                    "indexer_score_q_t",
                    q_T_shape,
                    ml_dtypes.bfloat16,
                ),
                "output1": _scratch(
                    "indexer_score_weights",
                    w_flat_shape,
                    np.float32,
                ),
            }
        }
    )
    fused_indexer_compressor_qw_prep = fns.get(
        "indexer_compressor_kv_score_project_qw_prep_from_freq_table"
    )
    freq_table_project_qw_prep = fns.get("indexer_project_qw_prep_from_freq_table")
    freqs_cos = getattr(indexer.compressor, "freqs_cos", None)
    freqs_sin = getattr(indexer.compressor, "freqs_sin", None)
    if precomputed_qw is not None:
        q_T_dev, w_dev = precomputed_qw
    elif caps.require_precomputed_indexer_qw:
        raise RuntimeError(
            "DSV4 product indexer path requires Q/W prep from the fused "
            "attention QKV/indexer prologue"
        )
    elif qr is None:
        raise RuntimeError(
            "DSV4 indexer path requires QR unless Q/W prep was precomputed"
        )
    elif (
        callable(fused_indexer_compressor_qw_prep)
        and getattr(indexer.compressor, "wkv", None) is not None
        and getattr(indexer.compressor, "wgate", None) is not None
        and freqs_cos is not None
        and freqs_sin is not None
    ):
        positions = token_positions
        if positions is None:
            positions = np.arange(
                int(start_pos),
                int(start_pos) + int(seqlen),
                dtype=np.int32,
            )
        width = int(getattr(indexer.compressor.wkv, "shape", (0,))[0])
        fused_outputs: dict[str, Any] = {}
        if attention_scratch is not None:
            flat_shape = (int(bsz) * int(seqlen), width)
            fused_outputs = {
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
                "output2": _scratch(
                    "indexer_score_q_t",
                    q_T_shape,
                    ml_dtypes.bfloat16,
                ),
                "output3": _scratch(
                    "indexer_score_weights",
                    w_flat_shape,
                    np.float32,
                ),
            }
        kv_bf_dev, score_bf_dev, q_T_dev, w_dev = fused_indexer_compressor_qw_prep(
            x,
            indexer.compressor.wkv,
            indexer.compressor.wgate,
            qr,
            indexer.wq_b,
            indexer.weights_proj,
            freqs_cos,
            freqs_sin,
            positions,
            score_scale=float(indexer.softmax_scale * indexer.n_heads**-0.5),
            n_heads=int(indexer.n_heads),
            head_dim=int(indexer.head_dim),
            rope_head_dim=rd,
            block_size=32,
            fp8_max=240.0,
            **({"_nkipy_output_tensors": fused_outputs} if fused_outputs else {}),
        )
        precomputed_compressor_kv_score = (kv_bf_dev, score_bf_dev)
    elif (
        callable(freq_table_project_qw_prep)
        and freqs_cos is not None
        and freqs_sin is not None
    ):
        positions = token_positions
        if positions is None:
            positions = np.arange(
                int(start_pos),
                int(start_pos) + int(seqlen),
                dtype=np.int32,
            )
        q_T_dev, w_dev = freq_table_project_qw_prep(
            qr,
            indexer.wq_b,
            x,
            indexer.weights_proj,
            freqs_cos,
            freqs_sin,
            positions,
            score_scale=float(indexer.softmax_scale * indexer.n_heads**-0.5),
            n_heads=int(indexer.n_heads),
            head_dim=int(indexer.head_dim),
            rope_head_dim=rd,
            block_size=32,
            fp8_max=240.0,
            **q_w_outputs,
        )
    else:
        fc = indexer.compressor.freqs_cis[start_pos : start_pos + seqlen]
        cos = fc.real.astype(np.float32)
        sin = fc.imag.astype(np.float32)
        fused_project_qw_prep = fns.get("indexer_project_qw_prep")
        if callable(fused_project_qw_prep):
            q_T_dev, w_dev = fused_project_qw_prep(
                qr,
                indexer.wq_b,
                x,
                indexer.weights_proj,
                cos,
                sin,
                score_scale=float(indexer.softmax_scale * indexer.n_heads**-0.5),
                n_heads=int(indexer.n_heads),
                head_dim=int(indexer.head_dim),
                rope_head_dim=rd,
                block_size=32,
                fp8_max=240.0,
                **q_w_outputs,
            )
        else:
            q_flat_shape = (
                int(bsz),
                int(seqlen),
                int(indexer.n_heads * indexer.head_dim),
            )
            weights_shape = (int(bsz), int(seqlen), int(indexer.n_heads))
            project_outputs = None
            if attention_scratch is not None:
                project_outputs = {
                    "output0": _scratch("indexer_project_q", q_flat_shape, np.float32),
                    "output1": _scratch(
                        "indexer_project_weights",
                        weights_shape,
                        np.float32,
                    ),
                }

            q_flat, weights_dev = fns["indexer_project"](
                qr,
                indexer.wq_b,
                x,
                indexer.weights_proj,
                score_scale=float(indexer.softmax_scale * indexer.n_heads**-0.5),
                **(
                    {}
                    if project_outputs is None
                    else {"_nkipy_output_tensors": project_outputs}
                ),
            )
            q = None
            q_flat_shape_actual = tuple(
                int(dim) for dim in getattr(q_flat, "shape", ())
            )
            if q_flat_shape_actual and int(np.prod(q_flat_shape_actual)) == int(
                np.prod(q_shape),
            ):
                q = _alias_device_value_shape(q_flat, q_shape)
            if q is None:
                q = fns["indexer_q_reshape"](
                    q_flat,
                    bsz=int(bsz),
                    seqlen=int(seqlen),
                    n_heads=int(indexer.n_heads),
                    head_dim=int(indexer.head_dim),
                    **_output_kwargs(
                        "output0",
                        _scratch("indexer_q_reshape", q_shape, np.float32),
                    ),
                )
            q_dev = fns["indexer_q_transform"](
                q,
                cos,
                sin,
                rope_head_dim=rd,
                block_size=32,
                fp8_max=240.0,
                **_output_kwargs(
                    "output0",
                    _scratch("indexer_q_transform", q_shape, np.float32),
                ),
            )

            q_T_dev, w_dev = fns["indexer_score_qw_prep"](
                q_dev,
                weights_dev,
                **q_w_outputs,
            )

    if not bool(precomputed_compressor_state_written):
        _run_compressor(
            fns,
            indexer.compressor,
            x,
            start_pos,
            build_dir=build_dir,
            device_state=device_state,
            owner_ids=token_owner_ids,
            owner_ids_dev=owner_ids_dev,
            token_positions=token_positions,
            attention_scratch=attention_scratch,
            precomputed_kv_score=precomputed_compressor_kv_score,
            precomputed_decode_scatter_rows=precomputed_compressor_decode_scatter_rows,
        )

    if kv_len > 0:
        # q_T [B,d,h] bf16 and w [B,h] fp32 stay on device end-to-end:
        # the trace function hands DeviceTensors straight to the NKI
        # score kernel — no host round-trip.
        idx_score_flat = _indexer_score_from_device_cache_adapter(
            q_T_dev,
            w_dev,
            device_state=device_state,
            bsz=bsz,
            seqlen=seqlen,
            kv_len=kv_len,
            build_dir=build_dir,
            return_device=True,
            owner_ids=token_owner_ids,
            owner_ids_dev=(
                request_owner_ids_dev
                if request_owner_ids_dev is not None
                else token_owner_ids_dev
            ),
            output=_scratch(
                "indexer_score_flat",
                (int(bsz) * int(seqlen), int(kv_len)),
                np.float32,
            ),
        )
        k = min(int(indexer.index_topk), kv_len)
        indexer_sparse_attention_prep = fns.get("indexer_sparse_attention_prep_static")
        if (
            callable(indexer_sparse_attention_prep)
            and sparse_attention_rows is not None
            and sparse_attention_k_tile is not None
            and sparse_attention_window_size is not None
        ):
            sparse_topk_shape = _sparse_attention_topk_shape(
                start_pos=int(start_pos),
                seqlen=int(seqlen),
                k=int(k),
                sparse_attention_rows=sparse_attention_rows,
                sparse_attention_k_tile=sparse_attention_k_tile,
                sparse_attention_window_size=sparse_attention_window_size,
            )
            output_kwargs = (
                {}
                if attention_scratch is None
                else {
                    "_nkipy_output_tensors": _topk_output_tensors(
                        _scratch,
                        shape=sparse_topk_shape,
                    )
                }
            )
            sparse_prep_args = (
                (idx_score_flat, x)
                if (
                    token_positions is None
                    or not caps.indexer_sparse_prep_accepts_positions
                )
                else (idx_score_flat, x, token_positions)
            )
            return indexer_sparse_attention_prep(
                *sparse_prep_args,
                bsz=int(bsz),
                seqlen=int(seqlen),
                kv_len=int(kv_len),
                k=int(k),
                ratio=int(ratio),
                offset=int(offset),
                prefill=bool(start_pos == 0),
                window_size=int(sparse_attention_window_size),
                start_pos=int(start_pos),
                rows=int(sparse_attention_rows),
                k_tile=int(sparse_attention_k_tile),
                **output_kwargs,
            )
        indexer_topk_static = fns.get("indexer_topk_static")
        if callable(indexer_topk_static):
            return indexer_topk_static(
                idx_score_flat,
                bsz=int(bsz),
                seqlen=int(seqlen),
                kv_len=int(kv_len),
                k=int(k),
                ratio=int(ratio),
                offset=int(offset),
                prefill=bool(start_pos == 0),
                **_output_kwargs(
                    "output0",
                    _scratch(
                        "indexer_topk_rebase",
                        (int(bsz), int(seqlen), int(k)),
                        np.int32,
                    ),
                ),
            )
        idx_score = fns["indexer_score_reshape"](
            idx_score_flat,
            bsz=int(bsz),
            seqlen=int(seqlen),
            kv_len=int(kv_len),
            **_output_kwargs(
                "output0",
                _scratch(
                    "indexer_score_reshape",
                    (int(bsz), int(seqlen), int(kv_len)),
                    np.float32,
                ),
            ),
        )

    if start_pos == 0:
        idx_score = fns["causal_mask_add"](
            idx_score,
            seqlen=int(seqlen),
            ratio=int(ratio),
            kv_len=int(kv_len),
            **_output_kwargs(
                "output0",
                _scratch(
                    "indexer_causal_scores",
                    (int(bsz), int(seqlen), int(kv_len)),
                    np.float32,
                ),
            ),
        )

    k = min(int(indexer.index_topk), kv_len)
    topk = fns["topk_idx"](idx_score, k=int(k), t=int(kv_len))
    topk_rebase_static = fns.get("topk_rebase_static")
    if topk_rebase_static is not None:
        return topk_rebase_static(
            topk,
            seqlen=int(seqlen),
            ratio=int(ratio),
            offset=int(offset),
            prefill=bool(start_pos == 0),
            **_output_kwargs(
                "output0",
                _scratch(
                    "indexer_topk_rebase",
                    (int(bsz), int(seqlen), int(k)),
                    np.int32,
                ),
            ),
        )
    seqlen_arr = np.arange(1, seqlen + 1, dtype=np.int32)
    return fns["topk_rebase"](
        topk,
        seqlen_arr,
        ratio=int(ratio),
        offset=int(offset),
        prefill=bool(start_pos == 0),
        **_output_kwargs(
            "output0",
            _scratch(
                "indexer_topk_rebase",
                (int(bsz), int(seqlen), int(k)),
                np.int32,
            ),
        ),
    )
