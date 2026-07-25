"""QKV indexer product precompile helpers."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _TensorSpec,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    bucketed_prefill_token_topk_compile_shape as _bucketed_prefill_token_topk_compile_shape,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    bucketed_prefill_token_topk_shape as _bucketed_prefill_token_topk_shape,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    prefill_token_topk_compile_bucket_lengths as _prefill_token_topk_compile_bucket_lengths,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    prefill_token_topk_offset as _prefill_token_topk_offset,
)
from nkipy_serving.ops.deepseek_v4.indexer_state import (
    precompile_indexer_score_from_cache_device,
)


def _device_layer_state_for(device_state: Any, layer_id: int) -> Any | None:
    try:
        return device_state.layer(layer_id)
    except (AttributeError, IndexError, TypeError):
        return None


def _qkv_weights(attn: Any) -> tuple[Any, Any, Any, Any, Any]:
    return (
        getattr(attn, "wq_a", None),
        getattr(attn, "q_norm", None),
        getattr(attn, "wq_b", None),
        getattr(attn, "wkv", None),
        getattr(attn, "kv_norm", None),
    )


def _decode_start_positions(seq: int, ratio: int) -> tuple[int, ...]:
    seq_i = int(seq)
    return tuple(
        sorted({1, max(1, seq_i - 1), seq_i, seq_i + 1, max(1, int(ratio) - 1)})
    )


def _continuation_query_lengths(seqlen: int) -> tuple[int, ...]:
    seq_i = int(seqlen)
    if seq_i <= 1:
        return (1,)
    return _unique_positive_lengths(1, (seq_i + 1) // 2, seq_i)


def _freq_table_spec(
    freqs_cos: Any,
    freqs_sin: Any,
    rope_head_dim: int,
    dtype: Any,
    *,
    fallback_len: int | None = None,
) -> _TensorSpec | None:
    if freqs_cos is None or freqs_sin is None or int(rope_head_dim) <= 0:
        return None
    table_len = int(getattr(freqs_cos, "shape", (0,))[0])
    if table_len <= 0:
        if fallback_len is None:
            return None
        table_len = int(fallback_len)
    if table_len <= 0:
        return None
    return _TensorSpec((table_len, max(1, int(rope_head_dim) // 2)), dtype)


def _freq_table_spec_from_len(
    table_len: int,
    rope_head_dim: int,
    dtype: Any,
) -> _TensorSpec:
    return _TensorSpec(
        (int(table_len), max(1, int(rope_head_dim) // 2)),
        dtype,
    )


def _unique_positive_lengths(*lengths: int) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for length in lengths:
        length_i = int(length)
        if length_i <= 0 or length_i in seen:
            continue
        seen.add(length_i)
        out.append(length_i)
    return tuple(out)


def _qkv_row_bucket_variants(
    owner: Any,
    bucket: Dsv4ProductBucket,
    *,
    token_count: int,
    is_decode: bool,
    include_backend_bucket: bool,
    include_step_bucket: bool,
) -> tuple[int, ...]:
    rows = [
        int(
            owner._compressed_attention_bucket_for_tokens(
                int(token_count),
                int(bucket.token_bucket),
            )
        )
    ]
    if include_backend_bucket:
        rows.append(
            int(
                owner._attention_backend_bucket_for_tokens(
                    int(token_count),
                    int(bucket.token_bucket),
                    is_decode=bool(is_decode),
                )
            )
        )
    if include_step_bucket:
        rows.append(int(bucket.token_bucket))
    return _unique_positive_lengths(*rows)


def _qkv_batch_variants(batch_size: int) -> tuple[int, ...]:
    # Decode lanes can serve at lane batch 1 even when the warmup decode bucket
    # pads batch to 2. Prefill keeps per-request seqlen on DP lanes.
    return _unique_positive_lengths(int(batch_size), 1 if int(batch_size) > 1 else 0)


def _qkv_token_topk_warmup_variants(
    *,
    owner: Any,
    is_decode: bool,
    batch_size: int,
    seqlen: int,
    ratio: int,
    token_bucket: int,
    window_size: int,
    layer_decode_max_c_len: int,
    decode_start_positions: tuple[int, ...],
    k_tile: int = K_TILE,
) -> tuple[tuple[int, int, int, int], ...]:
    variants: list[tuple[int, int, int, int]] = []
    if not bool(is_decode):
        # First-chunk prefill serves at the bucket; continuation chunks
        # (start_pos != 0) keep real length, so keep both.
        bucket_prefill_len = (
            int(token_bucket)
            if int(batch_size) == 1
            and int(token_bucket) >= (2 * max(1, int(ratio)) - 1)
            else 0
        )
        sub_bucket_prefill_lens = (
            tuple(
                int(bucket)
                for bucket in (
                    tuple(owner._configured_product_token_buckets())
                    + tuple(owner._configured_product_decode_buckets())
                )
                if int(bucket) > 0 and int(bucket) < int(token_bucket)
            )
            if int(batch_size) == 1 and int(ratio) > 0
            else ()
        )
        token_topk_bucket_lens = (
            _prefill_token_topk_compile_bucket_lengths(
                token_bucket=int(token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                k_tile=int(k_tile),
            )
            if int(batch_size) == 1 and int(ratio) > 0
            else ()
        )
        prefill_lengths = (
            _unique_positive_lengths(
                *sub_bucket_prefill_lens,
                *token_topk_bucket_lens,
                bucket_prefill_len,
                int(seqlen) if int(seqlen) > int(token_bucket) else 0,
            )
            if int(batch_size) == 1
            else _unique_positive_lengths(int(seqlen))
        )
        for prefill_len_i in prefill_lengths:
            variants.append(
                (
                    prefill_len_i,
                    0,
                    int(layer_decode_max_c_len),
                    _prefill_token_topk_offset(
                        seqlen=int(prefill_len_i),
                        window_size=int(window_size),
                    ),
                )
            )
    for start_pos_i in decode_start_positions:
        for query_len_i in _continuation_query_lengths(int(seqlen)):
            variants.append(
                (
                    int(query_len_i),
                    int(start_pos_i),
                    int(layer_decode_max_c_len),
                    int(window_size),
                )
            )
    return tuple(variants)


def _empty_indexer_warmup_variants(
    *,
    is_decode: bool,
    seqlen: int,
    window_size: int,
    decode_window_width: int,
    decode_start_positions: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    prefill_offset = (
        int(window_size) if int(seqlen) <= int(window_size) else int(seqlen)
    )
    variants = []
    if not bool(is_decode):
        variants.append((int(seqlen), 0, prefill_offset))
        if int(seqlen) != 1:
            variants.append((1, 0, prefill_offset))
    variants.extend(
        (
            1,
            int(start_pos_i),
            int(decode_window_width),
        )
        for start_pos_i in decode_start_positions
    )
    return tuple(variants)


def _all_kv_warmup_variants(
    *,
    owner: Any,
    is_decode: bool,
    batch_size: int,
    seqlen: int,
    ratio: int,
    token_bucket: int,
    window_size: int,
    decode_window_width: int,
    decode_start_positions: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    variants: list[tuple[int, int, int]] = []
    if not bool(is_decode):
        # Bucketed prefill: the all-KV indexer prologue is the only family the
        # runner re-aliases to the token bucket. Keep real length too for
        # chunked continuation steps.
        bucket_prefill_len = (
            int(token_bucket) if int(batch_size) == 1 and int(ratio) > 0 else 0
        )
        sub_bucket_prefill_lens = (
            tuple(
                int(bucket)
                for bucket in (
                    tuple(owner._configured_product_token_buckets())
                    + tuple(owner._configured_product_decode_buckets())
                )
                if int(bucket) > 0 and int(bucket) < int(token_bucket)
            )
            if int(batch_size) == 1 and int(ratio) > 0
            else ()
        )
        prefill_lengths = (
            _unique_positive_lengths(
                *sub_bucket_prefill_lens,
                bucket_prefill_len,
                int(seqlen) if int(seqlen) > int(token_bucket) else 0,
            )
            if int(batch_size) == 1
            else _unique_positive_lengths(int(seqlen))
        )
        for prefill_len_i in prefill_lengths:
            variants.append(
                (
                    prefill_len_i,
                    0,
                    int(window_size)
                    if prefill_len_i <= int(window_size)
                    else prefill_len_i,
                )
            )
    variants.extend(
        (
            1,
            int(start_pos_i),
            int(decode_window_width),
        )
        for start_pos_i in decode_start_positions
    )
    return tuple(variants)


class Dsv4ProductQkvIndexerPrecompileMixin:
    def _precompile_compressor_post_qdq_freq_table(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seq: int,
        ratio: int,
        kv_len: int,
        attn: Any,
        indexer: Any,
        compressor: Any,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        skip_if_qkv_fused: bool,
    ) -> None:
        freqs_cos = getattr(compressor, "freqs_cos", None)
        freqs_sin = getattr(compressor, "freqs_sin", None)
        norm_weight = getattr(compressor, "norm_weight", None)
        rope_head_dim = int(getattr(compressor, "rope_head_dim", 0) or 0)
        head_dim = int(getattr(compressor, "head_dim", 0) or 0)
        freq_table = _freq_table_spec(
            freqs_cos,
            freqs_sin,
            rope_head_dim,
            f32_dtype,
            fallback_len=int(getattr(self.runtime_surface, "max_seq_len", 1)),
        )
        if freq_table is None or head_dim <= 0:
            return

        # Product serving derives compressor post-pool RoPE rows from the
        # backend token-position vector. Mirror the runtime prefill gate so
        # short decode/one-token buckets do not compile impossible
        # clen*ratio > seqlen shapes.
        cutoff_i = int(seq) - (int(seq) % int(ratio))
        clen_i = int(cutoff_i) // int(ratio)
        if cutoff_i <= 0 or clen_i <= 0:
            return
        if int(bsz) * int(seq) > int(bucket.token_bucket):
            return

        comp_wkv = getattr(compressor, "wkv", None)
        comp_wgate = getattr(compressor, "wgate", None)
        comp_ape = getattr(compressor, "ape", None)
        comp_norm = getattr(compressor, "norm_weight", None)
        attn_freqs_cos = getattr(attn, "freqs_cos", None)
        attn_freqs_sin = getattr(attn, "freqs_sin", None)
        attn_table_len = int(getattr(attn_freqs_cos, "shape", (0,))[0])
        qkv_weights = _qkv_weights(attn)
        index_topk = int(getattr(indexer, "index_topk", 0) or 0)
        k_i = min(index_topk, kv_len) if kv_len > 0 else 0
        idx_compressor = getattr(indexer, "compressor", None)
        idx_can_fuse = (
            indexer is not None
            and kv_len > 0
            and int(k_i) == int(kv_len)
            and getattr(idx_compressor, "wkv", None) is not None
            and getattr(idx_compressor, "wgate", None) is not None
            and getattr(idx_compressor, "ape", None) is not None
            and getattr(idx_compressor, "norm_weight", None) is not None
            and getattr(idx_compressor, "freqs_cos", None) is not None
            and getattr(idx_compressor, "freqs_sin", None) is not None
        )
        prefill_post_qdq_fused = bool(
            comp_wkv is not None
            and comp_wgate is not None
            and comp_ape is not None
            and comp_norm is not None
            and attn_freqs_cos is not None
            and attn_freqs_sin is not None
            and attn_table_len > 0
            and not any(value is None for value in qkv_weights)
            and (indexer is None or idx_can_fuse)
        )
        if bool(skip_if_qkv_fused) and prefill_post_qdq_fused:
            return

        self._compressor_post_qdq_freq_table_kernel_for(
            bucket,
            _TensorSpec((int(bsz) * int(clen_i), head_dim), f32_dtype),
            _TensorSpec(
                (head_dim,),
                _value_dtype(norm_weight, fallback=ml_dtypes.bfloat16),
            ),
            freq_table,
            freq_table,
            _TensorSpec((int(bucket.token_bucket),), i32_dtype),
            bsz=bsz,
            clen=int(clen_i),
            source_token_positions=True,
            compress_ratio=int(ratio),
            start_pos=0,
            seqlen=seq,
            rope_head_dim=rope_head_dim,
            block_size=32 if bool(getattr(compressor, "rotate", False)) else 64,
            fp8_max=240.0,
            rotate=bool(getattr(compressor, "rotate", False)),
            eps=float(getattr(compressor, "eps", 1e-6)),
        )

    def _precompile_empty_indexer_compressor_token_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seq: int,
        ratio: int,
        is_decode: bool,
        window_size: int,
        decode_window_width: int,
        decode_start_positions: tuple[int, ...],
        hidden_size: int,
        attn: Any,
        indexer: Any,
        compressor: Any,
        bf16_dtype: np.dtype,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        idx_compressor = getattr(indexer, "compressor", None)
        idx_wkv = getattr(idx_compressor, "wkv", None)
        idx_wgate = getattr(idx_compressor, "wgate", None)
        if (
            int(hidden_size) <= 0
            or idx_wkv is None
            or idx_wgate is None
            or int(window_size) <= 0
        ):
            return

        qkv_weights = _qkv_weights(attn)
        comp_wkv = getattr(compressor, "wkv", None)
        comp_wgate = getattr(compressor, "wgate", None)
        attn_freqs_cos = getattr(attn, "freqs_cos", None)
        attn_freqs_sin = getattr(attn, "freqs_sin", None)
        attn_n_heads = int(getattr(attn, "n_heads", 0) or 0)
        attn_head_dim = int(getattr(attn, "head_dim", 0) or 0)
        attn_rope_head_dim = int(getattr(attn, "rope_head_dim", 0) or 0)
        attn_freq_table = _freq_table_spec(
            attn_freqs_cos,
            attn_freqs_sin,
            attn_rope_head_dim,
            f32_dtype,
        )
        variants = _empty_indexer_warmup_variants(
            is_decode=bool(is_decode),
            seqlen=seq,
            window_size=window_size,
            decode_window_width=decode_window_width,
            decode_start_positions=decode_start_positions,
        )
        for query_len_i, start_pos_i, offset_i in variants:
            hidden = _TensorSpec((bsz, query_len_i, hidden_size), bf16_dtype)
            can_fuse_qkv = (
                comp_wkv is not None
                and comp_wgate is not None
                and attn_freq_table is not None
                and attn_n_heads > 0
                and attn_head_dim > 0
                and attn_rope_head_dim > 0
                and not any(value is None for value in qkv_weights)
            )
            if not can_fuse_qkv:
                raise RuntimeError(
                    "DSV4 product empty-indexer warmup requires "
                    "fused attention QKV/indexer-compressor top-k"
                )
            rows_variants = _qkv_row_bucket_variants(
                self,
                bucket,
                token_count=bsz * query_len_i,
                is_decode=int(start_pos_i) > 0,
                include_backend_bucket=True,
                include_step_bucket=int(start_pos_i) == 0,
            )
            for active_rows in sorted(rows_variants):
                self._attention_qkv_empty_indexer_compressor_token_topk_prep_kernel_for(
                    bucket,
                    hidden,
                    qkv_weights[0],
                    qkv_weights[1],
                    qkv_weights[2],
                    qkv_weights[3],
                    qkv_weights[4],
                    comp_wkv,
                    comp_wgate,
                    idx_wkv,
                    idx_wgate,
                    attn_freq_table,
                    attn_freq_table,
                    _TensorSpec((bsz * query_len_i,), i32_dtype),
                    n_heads=attn_n_heads,
                    head_dim=attn_head_dim,
                    rope_head_dim=attn_rope_head_dim,
                    eps=float(getattr(attn, "eps", 1e-6)),
                    block_size=64,
                    fp8_max=240.0,
                    q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
                    q_token_bucket=int(active_rows),
                    kv_token_bucket=int(active_rows),
                    window_size=int(window_size),
                    ratio=int(ratio),
                    offset=int(offset_i),
                    start_pos=int(start_pos_i),
                    max_c_len=0,
                    rows=int(active_rows),
                    k_tile=int(k_tile),
                    dynamic_decode_start_pos=int(start_pos_i) > 0,
                )

    def _precompile_indexer_score_from_cache_fallback(
        self,
        *,
        rows: int,
        n_heads: int,
        kv_len: int,
        device_indexer_state: Any | None,
        bf16_dtype: np.dtype,
        f32_dtype: np.dtype,
    ) -> None:
        if int(rows) <= 0 or int(n_heads) <= 0 or int(kv_len) <= 0:
            return
        if device_indexer_state is None:
            return
        compressed_kv_cache = getattr(
            device_indexer_state,
            "compressed_kv_cache",
            None,
        )
        spec = getattr(device_indexer_state, "spec", None)
        max_compressed_len = int(getattr(spec, "max_compressed_len", 0) or 0)
        kv_cache_shape = tuple(
            int(dim) for dim in getattr(compressed_kv_cache, "shape", ())
        )
        if (
            max_compressed_len <= 0
            or len(kv_cache_shape) != 2
            or int(kv_cache_shape[1]) != 128
        ):
            return
        precompile_indexer_score_from_cache_device(
            q_T_shape=(int(rows), 128, int(n_heads)),
            q_T_dtype=bf16_dtype,
            kv_cache_shape=kv_cache_shape,
            kv_cache_dtype=getattr(compressed_kv_cache, "dtype", bf16_dtype),
            owner_ids_shape=(int(rows),),
            w_shape=(int(rows), int(n_heads)),
            w_dtype=f32_dtype,
            kv_len=int(kv_len),
            max_compressed_len=int(max_compressed_len),
            artifacts_dir=getattr(self, "build_dir", None),
        )

    def _precompile_indexer_sparse_attention_fallback(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seq: int,
        kv_len: int,
        k: int,
        ratio: int,
        is_decode: bool,
        window_size: int,
        decode_window_width: int,
        decode_start_positions: tuple[int, ...],
        hidden_size: int,
        indexer_n_heads: int,
        device_indexer_state: Any | None,
        bf16_dtype: np.dtype,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        prefill_score = _TensorSpec((bsz * seq, kv_len), f32_dtype)
        prefill_x = _TensorSpec((bsz, seq, hidden_size), bf16_dtype)
        prefill_offset_tensor = _TensorSpec((1, 1), i32_dtype)
        prefill_rows = self._compressed_attention_bucket_for_tokens(
            bsz * int(seq),
            int(bucket.token_bucket),
        )
        prefill_offset = int(window_size) if seq <= window_size else int(seq)
        if not bool(is_decode):
            self._precompile_indexer_score_from_cache_fallback(
                rows=int(bsz) * int(seq),
                n_heads=int(indexer_n_heads),
                kv_len=int(kv_len),
                device_indexer_state=device_indexer_state,
                bf16_dtype=bf16_dtype,
                f32_dtype=f32_dtype,
            )
            self._indexer_sparse_attention_prep_static_kernel_for(
                bucket,
                prefill_score,
                prefill_x,
                None,
                offset_tensor=prefill_offset_tensor,
                bsz=int(bsz),
                seqlen=int(seq),
                kv_len=int(kv_len),
                k=int(k),
                ratio=int(ratio),
                offset=int(prefill_offset),
                prefill=True,
                window_size=int(window_size),
                start_pos=0,
                rows=int(prefill_rows),
                k_tile=int(k_tile),
                dynamic_prefill_offset=True,
            )

        decode_score = _TensorSpec((bsz, kv_len), f32_dtype)
        decode_x = _TensorSpec((bsz, 1, hidden_size), bf16_dtype)
        decode_rows = self._compressed_attention_bucket_for_tokens(
            bsz,
            int(bucket.token_bucket),
        )
        for start_pos_i in decode_start_positions:
            self._precompile_indexer_score_from_cache_fallback(
                rows=int(bsz),
                n_heads=int(indexer_n_heads),
                kv_len=int(kv_len),
                device_indexer_state=device_indexer_state,
                bf16_dtype=bf16_dtype,
                f32_dtype=f32_dtype,
            )
            self._indexer_sparse_attention_prep_static_kernel_for(
                bucket,
                decode_score,
                decode_x,
                _TensorSpec((int(bucket.token_bucket), 1), i32_dtype),
                bsz=int(bsz),
                seqlen=1,
                kv_len=int(kv_len),
                k=int(k),
                ratio=int(ratio),
                offset=int(decode_window_width),
                prefill=False,
                window_size=int(window_size),
                start_pos=int(start_pos_i),
                rows=int(decode_rows),
                k_tile=int(k_tile),
                dynamic_decode_start_pos=True,
            )

    def _precompile_plain_token_topk_bucket_pads(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_size: int,
        token_bucket: int,
        window_size: int,
        ratio: int,
        bf16_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        if int(hidden_size) <= 0 or int(token_bucket) <= 1 or int(ratio) <= 1:
            return
        kernel_for = getattr(self, "_sequence_hidden_pad_kernel_for", None)
        if not callable(kernel_for):
            return
        # The plain compressor-token-topk fallback is selected for first-chunk
        # prompts shorter than the compressor ratio. Bucket those live lengths to
        # the representative top-k width class and warm the copy/pad support
        # kernels that materialize the bucketed x tensor.
        for live_len in range(1, min(int(token_bucket), int(ratio))):
            compile_shape = _bucketed_prefill_token_topk_compile_shape(
                (1, int(live_len), int(hidden_size)),
                canonical_rows=int(token_bucket),
                q_token_bucket=int(token_bucket),
                kv_token_bucket=int(token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                offset=_prefill_token_topk_offset(
                    seqlen=int(live_len),
                    window_size=int(window_size),
                ),
                start_pos=0,
                max_c_len=0,
                k_tile=int(k_tile),
            )
            if compile_shape is None:
                continue
            _, compile_seqlen, _, _ = compile_shape
            if int(compile_seqlen) <= int(live_len):
                continue
            kernel_for(
                bucket,
                _TensorSpec((1, int(live_len), int(hidden_size)), bf16_dtype),
                rows=int(compile_seqlen),
                hidden_size=int(hidden_size),
            )

    def _precompile_no_indexer_qkv_token_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seq: int,
        ratio: int,
        is_decode: bool,
        window_size: int,
        layer_decode_max_c_len: int,
        decode_start_positions: tuple[int, ...],
        attn: Any,
        compressor: Any,
        device_layer_state: Any,
        device_comp_state: Any,
        bf16_dtype: np.dtype,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            return
        freqs_cos = getattr(attn, "freqs_cos", None)
        freqs_sin = getattr(attn, "freqs_sin", None)
        q_low_dim = int(getattr(attn.q_norm, "shape", (0,))[0])
        n_heads = int(getattr(attn, "n_heads", 0) or 0)
        head_dim = int(getattr(attn, "head_dim", 0) or 0)
        rope_head_dim = int(getattr(attn, "rope_head_dim", 0) or 0)
        freq_table = _freq_table_spec(
            freqs_cos,
            freqs_sin,
            rope_head_dim,
            f32_dtype,
        )
        if (
            freq_table is None
            or q_low_dim <= 0
            or n_heads <= 0
            or head_dim <= 0
            or rope_head_dim <= 0
        ):
            return
        qkv_weights = _qkv_weights(attn)
        if any(value is None for value in qkv_weights):
            return
        comp_wkv = getattr(compressor, "wkv", None)
        comp_wgate = getattr(compressor, "wgate", None)
        if (
            not bool(is_decode)
            and int(bsz) == 1
            and comp_wkv is not None
            and comp_wgate is not None
        ):
            self._precompile_plain_token_topk_bucket_pads(
                bucket,
                hidden_size=int(hidden_size),
                token_bucket=int(bucket.token_bucket),
                window_size=int(window_size),
                ratio=int(ratio),
                bf16_dtype=bf16_dtype,
                k_tile=int(k_tile),
            )
        token_variants = _qkv_token_topk_warmup_variants(
            owner=self,
            is_decode=bool(is_decode),
            batch_size=bsz,
            seqlen=seq,
            ratio=ratio,
            token_bucket=int(bucket.token_bucket),
            window_size=window_size,
            layer_decode_max_c_len=layer_decode_max_c_len,
            decode_start_positions=decode_start_positions,
            k_tile=int(k_tile),
        )
        for bsz_i in _qkv_batch_variants(bsz):
            for query_len, start_pos_i, max_c_len_i, offset_i in token_variants:
                query_len_i = int(query_len)
                if query_len_i <= 0:
                    continue
                rows_variants = _qkv_row_bucket_variants(
                    self,
                    bucket,
                    token_count=bsz_i * query_len_i,
                    is_decode=int(start_pos_i) > 0,
                    include_backend_bucket=int(start_pos_i) == 0,
                    include_step_bucket=int(start_pos_i) == 0,
                )
                x_spec = _TensorSpec(
                    (bsz_i, query_len_i, hidden_size),
                    bf16_dtype,
                )
                for active_rows in sorted(rows_variants):
                    if (
                        not bool(is_decode)
                        and int(start_pos_i) == 0
                        and int(bsz_i) == 1
                        and int(active_rows) > int(query_len_i)
                    ):
                        pad_kernel_for = getattr(
                            self, "_sequence_hidden_pad_kernel_for", None
                        )
                        if callable(pad_kernel_for):
                            pad_kernel_for(
                                bucket,
                                x_spec,
                                rows=int(active_rows),
                                hidden_size=int(hidden_size),
                            )
                    if comp_wkv is not None and comp_wgate is not None:
                        comp_ape = getattr(compressor, "ape", None)
                        comp_norm = getattr(compressor, "norm_weight", None)
                        comp_freqs_cos = getattr(compressor, "freqs_cos", None)
                        comp_freqs_sin = getattr(compressor, "freqs_sin", None)
                        comp_table_len = int(getattr(comp_freqs_cos, "shape", (0,))[0])
                        comp_rope_head_dim = int(
                            getattr(compressor, "rope_head_dim", 0) or 0
                        )
                        can_fuse_prefill_post_qdq = bool(
                            int(start_pos_i) == 0
                            and query_len_i >= int(ratio)
                            and comp_ape is not None
                            and comp_norm is not None
                            and comp_freqs_cos is not None
                            and comp_freqs_sin is not None
                            and comp_table_len > 0
                            and comp_rope_head_dim > 0
                        )
                        comp_state_spec = getattr(device_comp_state, "spec", None)
                        comp_state_overlap = bool(
                            getattr(
                                comp_state_spec,
                                "overlap",
                                getattr(compressor, "overlap", False),
                            )
                        )
                        compressor_prefill_state_tail_len = 0
                        if int(start_pos_i) == 0 and int(ratio) > 0:
                            compressor_prefill_state_tail_len = (
                                min(
                                    int(query_len_i),
                                    int(ratio) + int(query_len_i) % int(ratio),
                                )
                                if comp_state_overlap
                                else int(query_len_i) % int(ratio)
                            )
                        can_fuse_prefill_write_cache = bool(
                            can_fuse_prefill_post_qdq
                            and int(compressor_prefill_state_tail_len) > 0
                            and device_layer_state is not None
                            and device_comp_state is not None
                            and hasattr(device_layer_state, "swa_kv_cache")
                            and hasattr(device_comp_state, "kv_score_state")
                            and hasattr(device_comp_state, "compressed_kv_cache")
                        )
                        can_fuse_decode_post_qdq = bool(
                            int(start_pos_i) > 0
                            and query_len_i == 1
                            and (int(start_pos_i) + 1) % int(ratio) == 0
                            and comp_ape is not None
                            and comp_norm is not None
                            and comp_freqs_cos is not None
                            and comp_freqs_sin is not None
                            and comp_table_len > 0
                            and comp_rope_head_dim > 0
                        )
                        can_fuse_decode_swa_state_write = bool(
                            int(start_pos_i) > 0
                            and query_len_i == 1
                            and (int(start_pos_i) + 1) % int(ratio) != 0
                            and comp_ape is not None
                            and device_layer_state is not None
                            and device_comp_state is not None
                            and hasattr(device_layer_state, "swa_kv_cache")
                            and hasattr(device_comp_state, "kv_score_state")
                        )
                        can_fuse_decode_post_qdq_write_cache = bool(
                            can_fuse_decode_post_qdq
                            and device_layer_state is not None
                            and device_comp_state is not None
                            and hasattr(device_layer_state, "swa_kv_cache")
                            and hasattr(device_comp_state, "kv_score_state")
                            and hasattr(device_comp_state, "compressed_kv_cache")
                            and hasattr(device_comp_state, "spec")
                        )
                        if can_fuse_prefill_post_qdq:
                            comp_freq_table = _freq_table_spec_from_len(
                                comp_table_len,
                                comp_rope_head_dim,
                                f32_dtype,
                            )
                            self._attention_qkv_compressor_prefill_post_qdq_token_topk_prep_kernel_for(
                                bucket,
                                x_spec,
                                qkv_weights[0],
                                qkv_weights[1],
                                qkv_weights[2],
                                qkv_weights[3],
                                qkv_weights[4],
                                comp_wkv,
                                comp_wgate,
                                comp_ape,
                                comp_norm,
                                freq_table,
                                freq_table,
                                comp_freq_table,
                                comp_freq_table,
                                _TensorSpec((bsz_i * query_len_i,), i32_dtype),
                                n_heads=int(n_heads),
                                head_dim=int(head_dim),
                                rope_head_dim=int(rope_head_dim),
                                eps=float(getattr(attn, "eps", 1e-6)),
                                block_size=64,
                                fp8_max=240.0,
                                q_softmax_scale=float(
                                    getattr(attn, "softmax_scale", 1.0)
                                ),
                                q_token_bucket=int(active_rows),
                                window_size=int(window_size),
                                ratio=int(ratio),
                                offset=int(offset_i),
                                start_pos=int(start_pos_i),
                                max_c_len=int(max_c_len_i),
                                rows=int(active_rows),
                                k_tile=int(k_tile),
                                kv_token_bucket=int(active_rows),
                                compressor_head_dim=int(
                                    getattr(compressor, "head_dim", 0) or 0
                                ),
                                compressor_rope_head_dim=comp_rope_head_dim,
                                compressor_block_size=(
                                    32
                                    if bool(getattr(compressor, "rotate", False))
                                    else 64
                                ),
                                compressor_fp8_max=240.0,
                                compressor_rotate=bool(
                                    getattr(compressor, "rotate", False)
                                ),
                                compressor_overlap=bool(
                                    getattr(compressor, "overlap", False)
                                ),
                                compressor_eps=float(getattr(compressor, "eps", 1e-6)),
                            )
                            if can_fuse_prefill_write_cache:
                                comp_ring_size = int(
                                    getattr(comp_state_spec, "ring_size", 0)
                                    or ((2 if comp_state_overlap else 1) * int(ratio))
                                )
                                self._attention_qkv_compressor_prefill_post_qdq_token_topk_prep_kernel_for(
                                    bucket,
                                    x_spec,
                                    qkv_weights[0],
                                    qkv_weights[1],
                                    qkv_weights[2],
                                    qkv_weights[3],
                                    qkv_weights[4],
                                    comp_wkv,
                                    comp_wgate,
                                    comp_ape,
                                    comp_norm,
                                    freq_table,
                                    freq_table,
                                    comp_freq_table,
                                    comp_freq_table,
                                    _TensorSpec((bsz_i * query_len_i,), i32_dtype),
                                    n_heads=int(n_heads),
                                    head_dim=int(head_dim),
                                    rope_head_dim=int(rope_head_dim),
                                    eps=float(getattr(attn, "eps", 1e-6)),
                                    block_size=64,
                                    fp8_max=240.0,
                                    q_softmax_scale=float(
                                        getattr(attn, "softmax_scale", 1.0)
                                    ),
                                    q_token_bucket=int(active_rows),
                                    window_size=int(window_size),
                                    ratio=int(ratio),
                                    offset=int(offset_i),
                                    start_pos=int(start_pos_i),
                                    max_c_len=int(max_c_len_i),
                                    rows=int(active_rows),
                                    k_tile=int(k_tile),
                                    kv_token_bucket=int(active_rows),
                                    compressor_head_dim=int(
                                        getattr(compressor, "head_dim", 0) or 0
                                    ),
                                    compressor_rope_head_dim=comp_rope_head_dim,
                                    compressor_block_size=(
                                        32
                                        if bool(getattr(compressor, "rotate", False))
                                        else 64
                                    ),
                                    compressor_fp8_max=240.0,
                                    compressor_rotate=bool(
                                        getattr(compressor, "rotate", False)
                                    ),
                                    compressor_overlap=bool(comp_state_overlap),
                                    compressor_eps=float(
                                        getattr(compressor, "eps", 1e-6)
                                    ),
                                    write_swa_state_cache=True,
                                    swa_kv_cache=device_layer_state.swa_kv_cache,
                                    kv_score_state=device_comp_state.kv_score_state,
                                    compressed_kv_cache=(
                                        device_comp_state.compressed_kv_cache
                                    ),
                                    owner_ids=_TensorSpec(
                                        (int(active_rows),),
                                        i32_dtype,
                                    ),
                                    compressor_ring_size=int(comp_ring_size),
                                    compressor_state_tail_len=int(
                                        compressor_prefill_state_tail_len
                                    ),
                                )
                        elif can_fuse_decode_post_qdq:
                            comp_freq_table = _freq_table_spec_from_len(
                                comp_table_len,
                                comp_rope_head_dim,
                                f32_dtype,
                            )
                            comp_head_dim = int(getattr(compressor, "head_dim", 0) or 0)
                            comp_overlap = bool(getattr(compressor, "overlap", False))
                            comp_coff = 2 if comp_overlap else 1
                            comp_state_width = int(comp_coff) * int(comp_head_dim)
                            comp_ring_size = int(comp_coff) * int(ratio)
                            comp_state_owners = max(
                                1,
                                int(
                                    getattr(
                                        self.runtime_surface,
                                        "max_batch_size",
                                        bsz_i,
                                    )
                                    or bsz_i
                                ),
                            )
                            kv_score_state_input = (
                                device_comp_state.kv_score_state
                                if can_fuse_decode_post_qdq_write_cache
                                else _TensorSpec(
                                    (
                                        int(comp_state_owners) * int(comp_ring_size),
                                        2 * int(comp_state_width),
                                    ),
                                    f32_dtype,
                                )
                            )
                            self._attention_qkv_compressor_decode_post_qdq_token_topk_prep_kernel_for(
                                bucket,
                                x_spec,
                                qkv_weights[0],
                                qkv_weights[1],
                                qkv_weights[2],
                                qkv_weights[3],
                                qkv_weights[4],
                                comp_wkv,
                                comp_wgate,
                                kv_score_state_input,
                                _TensorSpec((bsz_i,), i32_dtype),
                                _TensorSpec((bsz_i,), i32_dtype),
                                comp_ape,
                                comp_norm,
                                freq_table,
                                freq_table,
                                comp_freq_table,
                                comp_freq_table,
                                _TensorSpec((bsz_i * query_len_i,), i32_dtype),
                                n_heads=int(n_heads),
                                head_dim=int(head_dim),
                                rope_head_dim=int(rope_head_dim),
                                eps=float(getattr(attn, "eps", 1e-6)),
                                block_size=64,
                                fp8_max=240.0,
                                q_softmax_scale=float(
                                    getattr(attn, "softmax_scale", 1.0)
                                ),
                                q_token_bucket=int(active_rows),
                                window_size=int(window_size),
                                ratio=int(ratio),
                                offset=int(offset_i),
                                start_pos=int(start_pos_i),
                                max_c_len=int(max_c_len_i),
                                rows=int(active_rows),
                                k_tile=int(k_tile),
                                kv_token_bucket=int(active_rows),
                                compressor_head_dim=int(comp_head_dim),
                                compressor_state_width=int(comp_state_width),
                                compressor_ring_size=int(comp_ring_size),
                                compressor_rope_head_dim=comp_rope_head_dim,
                                compressor_block_size=(
                                    32
                                    if bool(getattr(compressor, "rotate", False))
                                    else 64
                                ),
                                compressor_fp8_max=240.0,
                                compressor_rotate=bool(
                                    getattr(compressor, "rotate", False)
                                ),
                                compressor_overlap=bool(comp_overlap),
                                compressor_eps=float(getattr(compressor, "eps", 1e-6)),
                                write_swa_state_cache=bool(
                                    can_fuse_decode_post_qdq_write_cache
                                ),
                                compressed_cache_stride=(
                                    int(device_comp_state.spec.max_compressed_len)
                                    if can_fuse_decode_post_qdq_write_cache
                                    else 0
                                ),
                                swa_kv_cache=(
                                    device_layer_state.swa_kv_cache
                                    if can_fuse_decode_post_qdq_write_cache
                                    else None
                                ),
                                compressed_kv_cache=(
                                    device_comp_state.compressed_kv_cache
                                    if can_fuse_decode_post_qdq_write_cache
                                    else None
                                ),
                            )
                        elif can_fuse_decode_swa_state_write:
                            comp_overlap = bool(getattr(compressor, "overlap", False))
                            comp_coff = 2 if comp_overlap else 1
                            comp_ring_size = int(comp_coff) * int(ratio)
                            self._attention_qkv_compressor_token_topk_prep_kernel_for(
                                bucket,
                                x_spec,
                                qkv_weights[0],
                                qkv_weights[1],
                                qkv_weights[2],
                                qkv_weights[3],
                                qkv_weights[4],
                                comp_wkv,
                                comp_wgate,
                                freq_table,
                                freq_table,
                                _TensorSpec((bsz_i * query_len_i,), i32_dtype),
                                n_heads=int(n_heads),
                                head_dim=int(head_dim),
                                rope_head_dim=int(rope_head_dim),
                                eps=float(getattr(attn, "eps", 1e-6)),
                                block_size=64,
                                fp8_max=240.0,
                                q_softmax_scale=float(
                                    getattr(attn, "softmax_scale", 1.0)
                                ),
                                q_token_bucket=int(active_rows),
                                window_size=int(window_size),
                                ratio=int(ratio),
                                offset=int(offset_i),
                                start_pos=int(start_pos_i),
                                max_c_len=int(max_c_len_i),
                                rows=int(active_rows),
                                k_tile=int(k_tile),
                                kv_token_bucket=int(active_rows),
                                dynamic_decode_start_pos=True,
                                write_swa_state=True,
                                swa_kv_cache=device_layer_state.swa_kv_cache,
                                kv_score_state=device_comp_state.kv_score_state,
                                owner_ids=_TensorSpec((bsz_i,), i32_dtype),
                                compressor_ape=comp_ape,
                                compressor_ring_size=int(comp_ring_size),
                            )
                        else:
                            simple_x_spec = x_spec
                            simple_positions_spec = _TensorSpec(
                                (bsz_i * query_len_i,),
                                i32_dtype,
                            )
                            bucketed_shape = _bucketed_prefill_token_topk_shape(
                                tuple(int(dim) for dim in x_spec.shape),
                                canonical_rows=int(active_rows),
                                q_token_bucket=int(active_rows),
                                kv_token_bucket=int(active_rows),
                                window_size=int(window_size),
                                ratio=int(ratio),
                                offset=int(offset_i),
                                start_pos=int(start_pos_i),
                                max_c_len=int(max_c_len_i),
                                k_tile=int(k_tile),
                            )
                            simple_specs = [(simple_x_spec, simple_positions_spec)]
                            if bucketed_shape is not None:
                                # Serve buckets only when its own helper check
                                # passes; keep the raw variant compiled too.
                                full_bsz, full_seqlen, full_rows = bucketed_shape
                                simple_specs.insert(
                                    0,
                                    (
                                        _TensorSpec(
                                            (
                                                int(full_bsz),
                                                int(full_seqlen),
                                                hidden_size,
                                            ),
                                            bf16_dtype,
                                        ),
                                        _TensorSpec((int(full_rows),), i32_dtype),
                                    ),
                                )
                            for (
                                simple_x_spec,
                                simple_positions_spec,
                            ) in simple_specs:
                                self._attention_qkv_compressor_token_topk_prep_kernel_for(
                                    bucket,
                                    simple_x_spec,
                                    qkv_weights[0],
                                    qkv_weights[1],
                                    qkv_weights[2],
                                    qkv_weights[3],
                                    qkv_weights[4],
                                    comp_wkv,
                                    comp_wgate,
                                    freq_table,
                                    freq_table,
                                    simple_positions_spec,
                                    n_heads=int(n_heads),
                                    head_dim=int(head_dim),
                                    rope_head_dim=int(rope_head_dim),
                                    eps=float(getattr(attn, "eps", 1e-6)),
                                    block_size=64,
                                    fp8_max=240.0,
                                    q_softmax_scale=float(
                                        getattr(attn, "softmax_scale", 1.0)
                                    ),
                                    q_token_bucket=int(active_rows),
                                    window_size=int(window_size),
                                    ratio=int(ratio),
                                    offset=int(offset_i),
                                    start_pos=int(start_pos_i),
                                    max_c_len=int(max_c_len_i),
                                    rows=int(active_rows),
                                    k_tile=int(k_tile),
                                    kv_token_bucket=int(active_rows),
                                    dynamic_decode_start_pos=(
                                        int(start_pos_i) > 0 and query_len_i == 1
                                    ),
                                )
                    else:
                        self._attention_qkv_token_topk_prep_kernel_for(
                            bucket,
                            x_spec,
                            qkv_weights[0],
                            qkv_weights[1],
                            qkv_weights[2],
                            qkv_weights[3],
                            qkv_weights[4],
                            freq_table,
                            freq_table,
                            _TensorSpec((bsz_i * query_len_i,), i32_dtype),
                            n_heads=int(n_heads),
                            head_dim=int(head_dim),
                            rope_head_dim=int(rope_head_dim),
                            eps=float(getattr(attn, "eps", 1e-6)),
                            block_size=64,
                            fp8_max=240.0,
                            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
                            q_token_bucket=int(active_rows),
                            window_size=int(window_size),
                            ratio=int(ratio),
                            offset=int(offset_i),
                            start_pos=int(start_pos_i),
                            max_c_len=int(max_c_len_i),
                            rows=int(active_rows),
                            k_tile=int(k_tile),
                            kv_token_bucket=int(active_rows),
                            return_qr=False,
                            dynamic_decode_start_pos=(
                                int(start_pos_i) > 0 and query_len_i == 1
                            ),
                        )
        return

    def _precompile_default_all_kv_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_spec: _TensorSpec,
        qkv_weights: tuple[Any, Any, Any, Any, Any],
        comp_wkv: Any,
        comp_wgate: Any,
        idx_wkv: Any,
        idx_wgate: Any,
        attn_freq_table: _TensorSpec,
        positions_spec: _TensorSpec,
        attn: Any,
        attn_n_heads: int,
        attn_head_dim: int,
        attn_rope_head_dim: int,
        active_rows: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        kv_len: int,
        k: int,
        k_tile: int,
    ) -> None:
        self._attention_qkv_indexer_compressor_all_kv_topk_prep_kernel_for(
            bucket,
            hidden_spec,
            qkv_weights[0],
            qkv_weights[1],
            qkv_weights[2],
            qkv_weights[3],
            qkv_weights[4],
            comp_wkv,
            comp_wgate,
            idx_wkv,
            idx_wgate,
            attn_freq_table,
            attn_freq_table,
            positions_spec,
            n_heads=attn_n_heads,
            head_dim=attn_head_dim,
            rope_head_dim=attn_rope_head_dim,
            eps=float(getattr(attn, "eps", 1e-6)),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
            q_token_bucket=int(active_rows),
            kv_token_bucket=int(active_rows),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            kv_len=int(kv_len),
            k=int(k),
            rows=int(active_rows),
            k_tile=int(k_tile),
            dynamic_decode_start_pos=int(start_pos) > 0,
        )

    def _precompile_decode_state_write_all_kv_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_spec: _TensorSpec,
        qkv_weights: tuple[Any, Any, Any, Any, Any],
        comp_wkv: Any,
        comp_wgate: Any,
        idx_wkv: Any,
        idx_wgate: Any,
        attn_freq_table: _TensorSpec,
        positions_spec: _TensorSpec,
        attn: Any,
        attn_n_heads: int,
        attn_head_dim: int,
        attn_rope_head_dim: int,
        active_rows: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        kv_len: int,
        k: int,
        bsz: int,
        i32_dtype: np.dtype,
        compressor: Any,
        idx_compressor: Any,
        device_layer_state: Any,
        device_comp_state: Any,
        device_indexer_state: Any,
        comp_ape: Any,
        idx_comp_ape: Any,
        k_tile: int,
    ) -> None:
        comp_overlap = bool(getattr(compressor, "overlap", False))
        comp_ring_size = int(2 if comp_overlap else 1) * int(ratio)
        idx_comp_overlap = bool(getattr(idx_compressor, "overlap", False))
        idx_comp_ring_size = int(2 if idx_comp_overlap else 1) * int(ratio)
        self._attention_qkv_indexer_compressor_all_kv_topk_prep_kernel_for(
            bucket,
            hidden_spec,
            qkv_weights[0],
            qkv_weights[1],
            qkv_weights[2],
            qkv_weights[3],
            qkv_weights[4],
            comp_wkv,
            comp_wgate,
            idx_wkv,
            idx_wgate,
            attn_freq_table,
            attn_freq_table,
            positions_spec,
            n_heads=attn_n_heads,
            head_dim=attn_head_dim,
            rope_head_dim=attn_rope_head_dim,
            eps=float(getattr(attn, "eps", 1e-6)),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
            q_token_bucket=int(active_rows),
            kv_token_bucket=int(active_rows),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            kv_len=int(kv_len),
            k=int(k),
            rows=int(active_rows),
            k_tile=int(k_tile),
            dynamic_decode_start_pos=True,
            write_swa_dual_state=True,
            swa_kv_cache=device_layer_state.swa_kv_cache,
            kv_score_state=device_comp_state.kv_score_state,
            indexer_kv_score_state=device_indexer_state.kv_score_state,
            owner_ids=_TensorSpec((bsz,), i32_dtype),
            compressor_ape=comp_ape,
            indexer_compressor_ape=idx_comp_ape,
            compressor_ring_size=int(comp_ring_size),
            indexer_compressor_ring_size=int(idx_comp_ring_size),
        )

    def _precompile_prefill_post_qdq_all_kv_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_spec: _TensorSpec,
        qkv_weights: tuple[Any, Any, Any, Any, Any],
        comp_wkv: Any,
        comp_wgate: Any,
        idx_wkv: Any,
        idx_wgate: Any,
        attn_freq_table: _TensorSpec,
        attn: Any,
        attn_n_heads: int,
        attn_head_dim: int,
        attn_rope_head_dim: int,
        active_rows: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        kv_len: int,
        k: int,
        bsz: int,
        query_len: int,
        compressor: Any,
        idx_compressor: Any,
        comp_ape: Any,
        comp_norm: Any,
        idx_comp_ape: Any,
        idx_comp_norm: Any,
        comp_table_len: int,
        idx_comp_table_len: int,
        comp_rope_head_dim: int,
        idx_comp_rope_head_dim: int,
        comp_state_spec: Any,
        idx_state_spec: Any,
        device_layer_state: Any,
        device_comp_state: Any,
        device_indexer_state: Any,
        comp_state_overlap: bool,
        idx_state_overlap: bool,
        idx_ratio: int,
        compressor_state_tail_len: int,
        indexer_state_tail_len: int,
        write_state_cache: bool,
        layer_decode_max_c_len: int,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        comp_freq_table = _freq_table_spec_from_len(
            comp_table_len,
            comp_rope_head_dim,
            f32_dtype,
        )
        idx_comp_freq_table = _freq_table_spec_from_len(
            idx_comp_table_len,
            idx_comp_rope_head_dim,
            f32_dtype,
        )
        prefill_post_qdq_args = (
            bucket,
            hidden_spec,
            qkv_weights[0],
            qkv_weights[1],
            qkv_weights[2],
            qkv_weights[3],
            qkv_weights[4],
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_wkv,
            idx_wgate,
            idx_comp_ape,
            idx_comp_norm,
            attn_freq_table,
            attn_freq_table,
            comp_freq_table,
            comp_freq_table,
            idx_comp_freq_table,
            idx_comp_freq_table,
            _TensorSpec((bsz * int(query_len),), i32_dtype),
        )
        prefill_post_qdq_kwargs = dict(
            n_heads=attn_n_heads,
            head_dim=attn_head_dim,
            rope_head_dim=attn_rope_head_dim,
            eps=float(getattr(attn, "eps", 1e-6)),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
            q_token_bucket=int(active_rows),
            kv_token_bucket=int(active_rows),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            kv_len=int(kv_len),
            k=int(k),
            rows=int(active_rows),
            k_tile=int(k_tile),
            compressor_head_dim=int(getattr(compressor, "head_dim", 0) or 0),
            compressor_rope_head_dim=comp_rope_head_dim,
            compressor_block_size=(
                32 if bool(getattr(compressor, "rotate", False)) else 64
            ),
            compressor_fp8_max=240.0,
            compressor_rotate=bool(getattr(compressor, "rotate", False)),
            compressor_overlap=bool(getattr(compressor, "overlap", False)),
            compressor_eps=float(getattr(compressor, "eps", 1e-6)),
            indexer_compressor_head_dim=int(
                getattr(idx_compressor, "head_dim", 0) or 0
            ),
            indexer_compressor_rope_head_dim=idx_comp_rope_head_dim,
            indexer_compressor_block_size=(
                32 if bool(getattr(idx_compressor, "rotate", False)) else 64
            ),
            indexer_compressor_fp8_max=240.0,
            indexer_compressor_rotate=bool(getattr(idx_compressor, "rotate", False)),
            indexer_compressor_overlap=bool(getattr(idx_compressor, "overlap", False)),
            indexer_compressor_eps=float(getattr(idx_compressor, "eps", 1e-6)),
        )
        self._attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_kernel_for(
            *prefill_post_qdq_args,
            **prefill_post_qdq_kwargs,
        )
        if not write_state_cache:
            return

        comp_ring_size = int(
            getattr(comp_state_spec, "ring_size", 0)
            or ((2 if comp_state_overlap else 1) * int(ratio))
        )
        idx_ring_size = int(
            getattr(idx_state_spec, "ring_size", 0)
            or ((2 if idx_state_overlap else 1) * int(idx_ratio))
        )
        write_prefill_kwargs = {
            **prefill_post_qdq_kwargs,
            "write_swa_state_cache": True,
            "swa_kv_cache": device_layer_state.swa_kv_cache,
            "kv_score_state": device_comp_state.kv_score_state,
            "compressed_kv_cache": device_comp_state.compressed_kv_cache,
            "indexer_kv_score_state": device_indexer_state.kv_score_state,
            "indexer_compressed_kv_cache": (device_indexer_state.compressed_kv_cache),
            "owner_ids": _TensorSpec((int(active_rows),), i32_dtype),
            "compressor_ring_size": int(comp_ring_size),
            "compressor_state_tail_len": int(compressor_state_tail_len),
            "indexer_compressor_ring_size": int(idx_ring_size),
            "indexer_compressor_state_tail_len": int(indexer_state_tail_len),
            "max_c_len": int(
                getattr(comp_state_spec, "max_compressed_len", 0)
                or layer_decode_max_c_len
            ),
            "indexer_max_c_len": int(
                getattr(idx_state_spec, "max_compressed_len", 0)
                or layer_decode_max_c_len
            ),
            "compressor_overlap": bool(comp_state_overlap),
            "indexer_compressor_overlap": bool(idx_state_overlap),
        }
        self._attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_kernel_for(
            *prefill_post_qdq_args,
            **write_prefill_kwargs,
        )

    def _precompile_decode_post_qdq_all_kv_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        hidden_spec: _TensorSpec,
        qkv_weights: tuple[Any, Any, Any, Any, Any],
        comp_wkv: Any,
        comp_wgate: Any,
        idx_wkv: Any,
        idx_wgate: Any,
        attn_freq_table: _TensorSpec,
        attn: Any,
        attn_n_heads: int,
        attn_head_dim: int,
        attn_rope_head_dim: int,
        active_rows: int,
        window_size: int,
        ratio: int,
        offset: int,
        start_pos: int,
        kv_len: int,
        k: int,
        bsz: int,
        query_len: int,
        compressor: Any,
        idx_compressor: Any,
        comp_ape: Any,
        comp_norm: Any,
        idx_comp_ape: Any,
        idx_comp_norm: Any,
        comp_table_len: int,
        idx_comp_table_len: int,
        comp_rope_head_dim: int,
        idx_comp_rope_head_dim: int,
        comp_state_spec: Any,
        idx_state_spec: Any,
        device_layer_state: Any,
        device_comp_state: Any,
        device_indexer_state: Any,
        comp_state_overlap: bool,
        idx_state_overlap: bool,
        write_state_cache: bool,
        layer_decode_max_c_len: int,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        comp_freq_table = _freq_table_spec_from_len(
            comp_table_len,
            comp_rope_head_dim,
            f32_dtype,
        )
        idx_comp_freq_table = _freq_table_spec_from_len(
            idx_comp_table_len,
            idx_comp_rope_head_dim,
            f32_dtype,
        )
        comp_head_dim = int(getattr(compressor, "head_dim", 0) or 0)
        comp_overlap = bool(getattr(compressor, "overlap", False))
        comp_state_width = int(2 if comp_overlap else 1) * int(comp_head_dim)
        comp_ring_size = int(2 if comp_overlap else 1) * int(ratio)
        idx_comp_head_dim = int(getattr(idx_compressor, "head_dim", 0) or 0)
        idx_comp_overlap = bool(getattr(idx_compressor, "overlap", False))
        idx_comp_state_width = int(2 if idx_comp_overlap else 1) * int(
            idx_comp_head_dim
        )
        idx_comp_ring_size = int(2 if idx_comp_overlap else 1) * int(ratio)
        state_owners = max(
            1,
            int(getattr(self.runtime_surface, "max_batch_size", bsz) or bsz),
        )
        positions_spec = _TensorSpec((bsz * int(query_len),), i32_dtype)
        self._attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_kernel_for(
            bucket,
            hidden_spec,
            qkv_weights[0],
            qkv_weights[1],
            qkv_weights[2],
            qkv_weights[3],
            qkv_weights[4],
            comp_wkv,
            comp_wgate,
            _TensorSpec(
                (
                    int(state_owners) * int(comp_ring_size),
                    2 * int(comp_state_width),
                ),
                f32_dtype,
            ),
            _TensorSpec((bsz,), i32_dtype),
            _TensorSpec((bsz,), i32_dtype),
            comp_ape,
            comp_norm,
            idx_wkv,
            idx_wgate,
            _TensorSpec(
                (
                    int(state_owners) * int(idx_comp_ring_size),
                    2 * int(idx_comp_state_width),
                ),
                f32_dtype,
            ),
            idx_comp_ape,
            idx_comp_norm,
            attn_freq_table,
            attn_freq_table,
            comp_freq_table,
            comp_freq_table,
            idx_comp_freq_table,
            idx_comp_freq_table,
            positions_spec,
            n_heads=attn_n_heads,
            head_dim=attn_head_dim,
            rope_head_dim=attn_rope_head_dim,
            eps=float(getattr(attn, "eps", 1e-6)),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
            q_token_bucket=int(active_rows),
            kv_token_bucket=int(active_rows),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            kv_len=int(kv_len),
            k=int(k),
            rows=int(active_rows),
            k_tile=int(k_tile),
            compressor_head_dim=int(comp_head_dim),
            compressor_state_width=int(comp_state_width),
            compressor_ring_size=int(comp_ring_size),
            compressor_rope_head_dim=comp_rope_head_dim,
            compressor_block_size=(
                32 if bool(getattr(compressor, "rotate", False)) else 64
            ),
            compressor_fp8_max=240.0,
            compressor_rotate=bool(getattr(compressor, "rotate", False)),
            compressor_overlap=comp_overlap,
            compressor_eps=float(getattr(compressor, "eps", 1e-6)),
            indexer_compressor_head_dim=int(idx_comp_head_dim),
            indexer_compressor_state_width=int(idx_comp_state_width),
            indexer_compressor_ring_size=int(idx_comp_ring_size),
            indexer_compressor_rope_head_dim=idx_comp_rope_head_dim,
            indexer_compressor_block_size=(
                32 if bool(getattr(idx_compressor, "rotate", False)) else 64
            ),
            indexer_compressor_fp8_max=240.0,
            indexer_compressor_rotate=bool(getattr(idx_compressor, "rotate", False)),
            indexer_compressor_overlap=idx_comp_overlap,
            indexer_compressor_eps=float(getattr(idx_compressor, "eps", 1e-6)),
        )
        if not write_state_cache:
            return

        write_comp_state_width = int(
            getattr(comp_state_spec, "state_width", 0) or comp_state_width
        )
        write_comp_ring_size = int(
            getattr(comp_state_spec, "ring_size", 0) or comp_ring_size
        )
        write_idx_state_width = int(
            getattr(idx_state_spec, "state_width", 0) or idx_comp_state_width
        )
        write_idx_ring_size = int(
            getattr(idx_state_spec, "ring_size", 0) or idx_comp_ring_size
        )
        self._attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_kernel_for(
            bucket,
            hidden_spec,
            qkv_weights[0],
            qkv_weights[1],
            qkv_weights[2],
            qkv_weights[3],
            qkv_weights[4],
            comp_wkv,
            comp_wgate,
            device_comp_state.kv_score_state,
            _TensorSpec((bsz,), i32_dtype),
            _TensorSpec((bsz,), i32_dtype),
            comp_ape,
            comp_norm,
            idx_wkv,
            idx_wgate,
            device_indexer_state.kv_score_state,
            idx_comp_ape,
            idx_comp_norm,
            attn_freq_table,
            attn_freq_table,
            comp_freq_table,
            comp_freq_table,
            idx_comp_freq_table,
            idx_comp_freq_table,
            positions_spec,
            n_heads=attn_n_heads,
            head_dim=attn_head_dim,
            rope_head_dim=attn_rope_head_dim,
            eps=float(getattr(attn, "eps", 1e-6)),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(getattr(attn, "softmax_scale", 1.0)),
            q_token_bucket=int(active_rows),
            kv_token_bucket=int(active_rows),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            kv_len=int(kv_len),
            k=int(k),
            rows=int(active_rows),
            k_tile=int(k_tile),
            compressor_head_dim=int(comp_head_dim),
            compressor_state_width=int(write_comp_state_width),
            compressor_ring_size=int(write_comp_ring_size),
            compressor_rope_head_dim=comp_rope_head_dim,
            compressor_block_size=(
                32 if bool(getattr(compressor, "rotate", False)) else 64
            ),
            compressor_fp8_max=240.0,
            compressor_rotate=bool(getattr(compressor, "rotate", False)),
            compressor_overlap=bool(comp_state_overlap),
            compressor_eps=float(getattr(compressor, "eps", 1e-6)),
            indexer_compressor_head_dim=int(idx_comp_head_dim),
            indexer_compressor_state_width=int(write_idx_state_width),
            indexer_compressor_ring_size=int(write_idx_ring_size),
            indexer_compressor_rope_head_dim=idx_comp_rope_head_dim,
            indexer_compressor_block_size=(
                32 if bool(getattr(idx_compressor, "rotate", False)) else 64
            ),
            indexer_compressor_fp8_max=240.0,
            indexer_compressor_rotate=bool(getattr(idx_compressor, "rotate", False)),
            indexer_compressor_overlap=bool(idx_state_overlap),
            indexer_compressor_eps=float(getattr(idx_compressor, "eps", 1e-6)),
            write_swa_state_cache=True,
            swa_kv_cache=device_layer_state.swa_kv_cache,
            compressed_kv_cache=device_comp_state.compressed_kv_cache,
            indexer_compressed_kv_cache=device_indexer_state.compressed_kv_cache,
            max_c_len=int(
                getattr(comp_state_spec, "max_compressed_len", 0)
                or layer_decode_max_c_len
            ),
            indexer_max_c_len=int(
                getattr(idx_state_spec, "max_compressed_len", 0)
                or layer_decode_max_c_len
            ),
        )

    def _precompile_all_kv_indexer_compressor_topk(
        self,
        bucket: Dsv4ProductBucket,
        *,
        layer_id: int,
        bsz: int,
        seq: int,
        ratio: int,
        kv_len: int,
        k: int,
        index_topk: int,
        is_decode: bool,
        window_size: int,
        decode_window_width: int,
        decode_start_positions: tuple[int, ...],
        layer_decode_max_c_len: int,
        hidden_size: int,
        attn: Any,
        compressor: Any,
        idx_compressor: Any,
        idx_wkv: Any,
        idx_wgate: Any,
        comp_wkv: Any,
        comp_wgate: Any,
        attn_freq_table: _TensorSpec,
        attn_n_heads: int,
        attn_head_dim: int,
        attn_rope_head_dim: int,
        qkv_weights: tuple[Any, Any, Any, Any, Any],
        bf16_dtype: np.dtype,
        f32_dtype: np.dtype,
        i32_dtype: np.dtype,
        k_tile: int,
    ) -> None:
        variants = _all_kv_warmup_variants(
            owner=self,
            is_decode=bool(is_decode),
            batch_size=bsz,
            seqlen=seq,
            ratio=ratio,
            token_bucket=int(bucket.token_bucket),
            window_size=window_size,
            decode_window_width=decode_window_width,
            decode_start_positions=decode_start_positions,
        )
        for query_len_i, start_pos_i, offset_i in variants:
            rows_variants = _qkv_row_bucket_variants(
                self,
                bucket,
                token_count=bsz * int(query_len_i),
                is_decode=int(start_pos_i) > 0,
                include_backend_bucket=True,
                include_step_bucket=int(start_pos_i) == 0,
            )
            for active_rows in rows_variants:
                # offset == the runner-published compiled offset:
                # win for short lanes (query_len<=win), else the
                # query length itself (== compile_seqlen, the axis
                # the runner bakes — NOT active_rows, the row-pad
                # bucket). Set at variant build (win|query_len).
                if int(start_pos_i) == 0:
                    compile_kv_len_i = (
                        int(query_len_i) // int(ratio) if int(ratio) > 0 else 0
                    )
                    compile_k_i = min(
                        int(index_topk),
                        int(compile_kv_len_i),
                    )
                else:
                    compile_kv_len_i = int(kv_len)
                    compile_k_i = int(k)
                if int(start_pos_i) > 0 and int(query_len_i) == 1:
                    compile_kv_len_i = self._product_all_kv_decode_compile_kv_len(
                        kv_len=int(kv_len),
                        seqlen=int(query_len_i),
                        window_size=int(window_size),
                        k_tile=int(k_tile),
                    )
                    compile_k_i = int(compile_kv_len_i)
                if int(compile_k_i) != int(compile_kv_len_i):
                    # This helper precompiles only the all-KV family. Bucketed
                    # warmup can enumerate larger prefill lanes whose
                    # compressed length exceeds index_topk; those shapes are
                    # partial top-k and are covered by the table-family warmup.
                    continue
                hidden_spec = _TensorSpec(
                    (bsz, int(query_len_i), hidden_size),
                    bf16_dtype,
                )
                comp_ape = getattr(compressor, "ape", None)
                comp_norm = getattr(compressor, "norm_weight", None)
                comp_freqs_cos = getattr(compressor, "freqs_cos", None)
                comp_freqs_sin = getattr(compressor, "freqs_sin", None)
                comp_table_len = int(getattr(comp_freqs_cos, "shape", (0,))[0])
                idx_comp_ape = getattr(idx_compressor, "ape", None)
                idx_comp_norm = getattr(
                    idx_compressor,
                    "norm_weight",
                    None,
                )
                idx_comp_freqs_cos = getattr(
                    idx_compressor,
                    "freqs_cos",
                    None,
                )
                idx_comp_freqs_sin = getattr(
                    idx_compressor,
                    "freqs_sin",
                    None,
                )
                idx_comp_table_len = int(getattr(idx_comp_freqs_cos, "shape", (0,))[0])
                comp_rope_head_dim = int(getattr(compressor, "rope_head_dim", 0) or 0)
                idx_comp_rope_head_dim = int(
                    getattr(idx_compressor, "rope_head_dim", 0) or 0
                )
                can_fuse_prefill_post_qdq = bool(
                    int(start_pos_i) == 0
                    and int(query_len_i) >= int(ratio)
                    and comp_ape is not None
                    and comp_norm is not None
                    and comp_freqs_cos is not None
                    and comp_freqs_sin is not None
                    and comp_table_len > 0
                    and idx_comp_ape is not None
                    and idx_comp_norm is not None
                    and idx_comp_freqs_cos is not None
                    and idx_comp_freqs_sin is not None
                    and idx_comp_table_len > 0
                )
                can_fuse_decode_post_qdq = bool(
                    int(start_pos_i) > 0
                    and int(query_len_i) == 1
                    and (int(start_pos_i) + 1) % int(ratio) == 0
                    and comp_ape is not None
                    and comp_norm is not None
                    and comp_freqs_cos is not None
                    and comp_freqs_sin is not None
                    and comp_table_len > 0
                    and comp_rope_head_dim > 0
                    and idx_comp_ape is not None
                    and idx_comp_norm is not None
                    and idx_comp_freqs_cos is not None
                    and idx_comp_freqs_sin is not None
                    and idx_comp_table_len > 0
                    and idx_comp_rope_head_dim > 0
                )
                device_layer_state = _device_layer_state_for(
                    getattr(self, "device_state", None),
                    layer_id,
                )
                device_comp_state = getattr(
                    device_layer_state,
                    "compressor",
                    None,
                )
                device_indexer_state = getattr(
                    device_layer_state,
                    "indexer",
                    None,
                )
                comp_state_spec = getattr(
                    device_comp_state,
                    "spec",
                    None,
                )
                idx_state_spec = getattr(
                    device_indexer_state,
                    "spec",
                    None,
                )
                comp_state_overlap = bool(
                    getattr(
                        comp_state_spec,
                        "overlap",
                        getattr(compressor, "overlap", False),
                    )
                )
                idx_state_overlap = bool(
                    getattr(
                        idx_state_spec,
                        "overlap",
                        getattr(idx_compressor, "overlap", False),
                    )
                )
                compressor_prefill_state_tail_len = 0
                if int(start_pos_i) == 0 and int(ratio) > 0:
                    compressor_prefill_state_tail_len = (
                        min(
                            int(query_len_i),
                            int(ratio) + int(query_len_i) % int(ratio),
                        )
                        if comp_state_overlap
                        else int(query_len_i) % int(ratio)
                    )
                idx_ratio = int(
                    getattr(idx_compressor, "compress_ratio", int(ratio)) or int(ratio)
                )
                indexer_prefill_state_tail_len = 0
                if int(start_pos_i) == 0 and int(idx_ratio) > 0:
                    indexer_prefill_state_tail_len = (
                        min(
                            int(query_len_i),
                            int(idx_ratio) + int(query_len_i) % int(idx_ratio),
                        )
                        if idx_state_overlap
                        else int(query_len_i) % int(idx_ratio)
                    )
                can_fuse_prefill_swa_dual_state_cache = bool(
                    can_fuse_prefill_post_qdq
                    and int(compressor_prefill_state_tail_len) > 0
                    and int(indexer_prefill_state_tail_len) > 0
                    and device_layer_state is not None
                    and device_comp_state is not None
                    and device_indexer_state is not None
                    and hasattr(device_layer_state, "swa_kv_cache")
                    and hasattr(device_comp_state, "kv_score_state")
                    and hasattr(device_comp_state, "compressed_kv_cache")
                    and hasattr(device_indexer_state, "kv_score_state")
                    and hasattr(
                        device_indexer_state,
                        "compressed_kv_cache",
                    )
                )
                can_fuse_decode_swa_dual_state_write = bool(
                    int(start_pos_i) > 0
                    and int(query_len_i) == 1
                    and (int(start_pos_i) + 1) % int(ratio) != 0
                    and comp_ape is not None
                    and idx_comp_ape is not None
                    and device_layer_state is not None
                    and device_comp_state is not None
                    and device_indexer_state is not None
                    and hasattr(device_layer_state, "swa_kv_cache")
                    and hasattr(device_comp_state, "kv_score_state")
                    and hasattr(device_indexer_state, "kv_score_state")
                )
                can_fuse_decode_post_qdq_write_cache = bool(
                    can_fuse_decode_post_qdq
                    and device_layer_state is not None
                    and device_comp_state is not None
                    and device_indexer_state is not None
                    and comp_state_spec is not None
                    and idx_state_spec is not None
                    and hasattr(device_layer_state, "swa_kv_cache")
                    and hasattr(device_comp_state, "kv_score_state")
                    and hasattr(device_comp_state, "compressed_kv_cache")
                    and hasattr(device_indexer_state, "kv_score_state")
                    and hasattr(
                        device_indexer_state,
                        "compressed_kv_cache",
                    )
                )
                if can_fuse_prefill_post_qdq:
                    self._precompile_prefill_post_qdq_all_kv_topk(
                        bucket,
                        hidden_spec=hidden_spec,
                        qkv_weights=qkv_weights,
                        comp_wkv=comp_wkv,
                        comp_wgate=comp_wgate,
                        idx_wkv=idx_wkv,
                        idx_wgate=idx_wgate,
                        attn_freq_table=attn_freq_table,
                        attn=attn,
                        attn_n_heads=attn_n_heads,
                        attn_head_dim=attn_head_dim,
                        attn_rope_head_dim=attn_rope_head_dim,
                        active_rows=int(active_rows),
                        window_size=window_size,
                        ratio=ratio,
                        offset=int(offset_i),
                        start_pos=int(start_pos_i),
                        kv_len=int(compile_kv_len_i),
                        k=int(compile_k_i),
                        bsz=bsz,
                        query_len=int(query_len_i),
                        compressor=compressor,
                        idx_compressor=idx_compressor,
                        comp_ape=comp_ape,
                        comp_norm=comp_norm,
                        idx_comp_ape=idx_comp_ape,
                        idx_comp_norm=idx_comp_norm,
                        comp_table_len=comp_table_len,
                        idx_comp_table_len=idx_comp_table_len,
                        comp_rope_head_dim=comp_rope_head_dim,
                        idx_comp_rope_head_dim=idx_comp_rope_head_dim,
                        comp_state_spec=comp_state_spec,
                        idx_state_spec=idx_state_spec,
                        device_layer_state=device_layer_state,
                        device_comp_state=device_comp_state,
                        device_indexer_state=device_indexer_state,
                        comp_state_overlap=comp_state_overlap,
                        idx_state_overlap=idx_state_overlap,
                        idx_ratio=idx_ratio,
                        compressor_state_tail_len=(compressor_prefill_state_tail_len),
                        indexer_state_tail_len=(indexer_prefill_state_tail_len),
                        write_state_cache=bool(can_fuse_prefill_swa_dual_state_cache),
                        layer_decode_max_c_len=layer_decode_max_c_len,
                        f32_dtype=f32_dtype,
                        i32_dtype=i32_dtype,
                        k_tile=int(k_tile),
                    )
                elif can_fuse_decode_post_qdq:
                    self._precompile_decode_post_qdq_all_kv_topk(
                        bucket,
                        hidden_spec=hidden_spec,
                        qkv_weights=qkv_weights,
                        comp_wkv=comp_wkv,
                        comp_wgate=comp_wgate,
                        idx_wkv=idx_wkv,
                        idx_wgate=idx_wgate,
                        attn_freq_table=attn_freq_table,
                        attn=attn,
                        attn_n_heads=attn_n_heads,
                        attn_head_dim=attn_head_dim,
                        attn_rope_head_dim=attn_rope_head_dim,
                        active_rows=int(active_rows),
                        window_size=window_size,
                        ratio=ratio,
                        offset=int(offset_i),
                        start_pos=int(start_pos_i),
                        kv_len=int(compile_kv_len_i),
                        k=int(compile_k_i),
                        bsz=bsz,
                        query_len=int(query_len_i),
                        compressor=compressor,
                        idx_compressor=idx_compressor,
                        comp_ape=comp_ape,
                        comp_norm=comp_norm,
                        idx_comp_ape=idx_comp_ape,
                        idx_comp_norm=idx_comp_norm,
                        comp_table_len=comp_table_len,
                        idx_comp_table_len=idx_comp_table_len,
                        comp_rope_head_dim=comp_rope_head_dim,
                        idx_comp_rope_head_dim=idx_comp_rope_head_dim,
                        comp_state_spec=comp_state_spec,
                        idx_state_spec=idx_state_spec,
                        device_layer_state=device_layer_state,
                        device_comp_state=device_comp_state,
                        device_indexer_state=device_indexer_state,
                        comp_state_overlap=comp_state_overlap,
                        idx_state_overlap=idx_state_overlap,
                        write_state_cache=bool(can_fuse_decode_post_qdq_write_cache),
                        layer_decode_max_c_len=layer_decode_max_c_len,
                        f32_dtype=f32_dtype,
                        i32_dtype=i32_dtype,
                        k_tile=int(k_tile),
                    )
                elif can_fuse_decode_swa_dual_state_write:
                    self._precompile_decode_state_write_all_kv_topk(
                        bucket,
                        hidden_spec=hidden_spec,
                        qkv_weights=qkv_weights,
                        comp_wkv=comp_wkv,
                        comp_wgate=comp_wgate,
                        idx_wkv=idx_wkv,
                        idx_wgate=idx_wgate,
                        attn_freq_table=attn_freq_table,
                        positions_spec=_TensorSpec(
                            (bsz * int(query_len_i),),
                            i32_dtype,
                        ),
                        attn=attn,
                        attn_n_heads=attn_n_heads,
                        attn_head_dim=attn_head_dim,
                        attn_rope_head_dim=attn_rope_head_dim,
                        active_rows=int(active_rows),
                        window_size=window_size,
                        ratio=ratio,
                        offset=int(offset_i),
                        start_pos=int(start_pos_i),
                        kv_len=int(compile_kv_len_i),
                        k=int(compile_k_i),
                        bsz=bsz,
                        i32_dtype=i32_dtype,
                        compressor=compressor,
                        idx_compressor=idx_compressor,
                        device_layer_state=device_layer_state,
                        device_comp_state=device_comp_state,
                        device_indexer_state=device_indexer_state,
                        comp_ape=comp_ape,
                        idx_comp_ape=idx_comp_ape,
                        k_tile=int(k_tile),
                    )
                else:
                    self._precompile_default_all_kv_topk(
                        bucket,
                        hidden_spec=hidden_spec,
                        qkv_weights=qkv_weights,
                        comp_wkv=comp_wkv,
                        comp_wgate=comp_wgate,
                        idx_wkv=idx_wkv,
                        idx_wgate=idx_wgate,
                        attn_freq_table=attn_freq_table,
                        positions_spec=_TensorSpec(
                            (bsz * int(query_len_i),),
                            i32_dtype,
                        ),
                        attn=attn,
                        attn_n_heads=attn_n_heads,
                        attn_head_dim=attn_head_dim,
                        attn_rope_head_dim=attn_rope_head_dim,
                        active_rows=int(active_rows),
                        window_size=window_size,
                        ratio=ratio,
                        offset=int(offset_i),
                        start_pos=int(start_pos_i),
                        kv_len=int(compile_kv_len_i),
                        k=int(compile_k_i),
                        k_tile=int(k_tile),
                    )
        return

    def _precompile_lane_attention_indexer_helpers(
        self,
        bucket: Dsv4ProductBucket,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile compressed-attention indexer top-k lane shapes."""
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        bf16_dtype = np.dtype(ml_dtypes.bfloat16)
        f32_dtype = np.dtype(np.float32)
        i32_dtype = np.dtype(np.int32)
        from nkipy_serving.models.deepseek_v4.constants import K_TILE

        decode_max_c_len = int(
            getattr(
                getattr(self, "options", None),
                "index_construction_max_c_len",
                0,
            )
            or 0
        )
        if decode_max_c_len <= 0:
            ratios = tuple(
                int(r)
                for r in getattr(
                    getattr(self.runtime_surface, "args", None), "compress_ratios", ()
                )
                if int(r) > 0
            )
            if ratios:
                decode_max_c_len = max(
                    int(getattr(self.runtime_surface, "max_seq_len", 0) or 0)
                    // min(ratios),
                    1,
                )
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            indexer = getattr(attn, "indexer", None)
            if attn is None:
                continue
            ratio = int(
                getattr(indexer, "compress_ratio", 0)
                or getattr(attn, "compress_ratio", 0)
                or 0
            )
            if ratio <= 0:
                continue
            device_layer_state = _device_layer_state_for(
                getattr(self, "device_state", None),
                layer_id,
            )
            device_comp_state = getattr(
                device_layer_state,
                "compressor",
                None,
            )
            layer_decode_max_c_len = int(
                getattr(
                    getattr(device_comp_state, "spec", None),
                    "max_compressed_len",
                    0,
                )
                or int(decode_max_c_len)
            )
            kv_len = int(seq) // ratio
            decode_start_positions = _decode_start_positions(seq, ratio)
            compressor = getattr(attn, "compressor", None)
            if compressor is not None:
                self._precompile_compressor_post_qdq_freq_table(
                    bucket,
                    bsz=bsz,
                    seq=seq,
                    ratio=ratio,
                    kv_len=kv_len,
                    attn=attn,
                    indexer=indexer,
                    compressor=compressor,
                    f32_dtype=f32_dtype,
                    i32_dtype=i32_dtype,
                    skip_if_qkv_fused=True,
                )
            indexer_compressor = getattr(indexer, "compressor", None)
            if indexer_compressor is not None and indexer_compressor is not compressor:
                indexer_ratio = int(
                    getattr(indexer_compressor, "compress_ratio", 0)
                    or getattr(indexer, "compress_ratio", 0)
                    or ratio
                )
                if indexer_ratio > 0:
                    self._precompile_compressor_post_qdq_freq_table(
                        bucket,
                        bsz=bsz,
                        seq=seq,
                        ratio=indexer_ratio,
                        kv_len=int(seq) // int(indexer_ratio),
                        attn=attn,
                        indexer=indexer,
                        compressor=indexer_compressor,
                        f32_dtype=f32_dtype,
                        i32_dtype=i32_dtype,
                        skip_if_qkv_fused=False,
                    )
            window_size = int(getattr(attn, "window_size", 0) or 0)
            decode_window_width = window_size if window_size > 0 else 1
            if indexer is None:
                self._precompile_no_indexer_qkv_token_topk(
                    bucket,
                    bsz=bsz,
                    seq=seq,
                    ratio=ratio,
                    is_decode=bool(is_decode),
                    window_size=window_size,
                    layer_decode_max_c_len=layer_decode_max_c_len,
                    decode_start_positions=decode_start_positions,
                    attn=attn,
                    compressor=compressor,
                    device_layer_state=device_layer_state,
                    device_comp_state=device_comp_state,
                    bf16_dtype=bf16_dtype,
                    f32_dtype=f32_dtype,
                    i32_dtype=i32_dtype,
                    k_tile=int(K_TILE),
                )
                continue
            else:
                index_topk = int(getattr(indexer, "index_topk", 0) or 0)
                k = min(index_topk, kv_len)
                hidden_size = self._product_hidden_size_for_bucket(bucket)
                if not bool(is_decode) and int(kv_len) > 0:
                    self._precompile_empty_indexer_compressor_token_topk(
                        bucket,
                        bsz=bsz,
                        seq=1,
                        ratio=ratio,
                        is_decode=False,
                        window_size=window_size,
                        decode_window_width=decode_window_width,
                        decode_start_positions=(),
                        hidden_size=hidden_size,
                        attn=attn,
                        indexer=indexer,
                        compressor=compressor,
                        bf16_dtype=bf16_dtype,
                        f32_dtype=f32_dtype,
                        i32_dtype=i32_dtype,
                        k_tile=int(K_TILE),
                    )
                if kv_len <= 0:
                    self._precompile_empty_indexer_compressor_token_topk(
                        bucket,
                        bsz=bsz,
                        seq=seq,
                        ratio=ratio,
                        is_decode=bool(is_decode),
                        window_size=window_size,
                        decode_window_width=decode_window_width,
                        decode_start_positions=decode_start_positions,
                        hidden_size=hidden_size,
                        attn=attn,
                        indexer=indexer,
                        compressor=compressor,
                        bf16_dtype=bf16_dtype,
                        f32_dtype=f32_dtype,
                        i32_dtype=i32_dtype,
                        k_tile=int(K_TILE),
                    )
                    continue
                if index_topk <= 0:
                    continue
                q_low_dim = int(getattr(indexer.wq_b, "shape", (0, 0))[1])
                n_heads = int(getattr(indexer, "n_heads", 0) or 0)
                head_dim = int(getattr(indexer, "head_dim", 0) or 0)
                rope_head_dim = int(getattr(indexer, "rope_head_dim", 0) or 0)
                if hidden_size > 0 and q_low_dim > 0 and n_heads > 0 and head_dim > 0:
                    idx_compressor = getattr(indexer, "compressor", None)
                    idx_wkv = getattr(idx_compressor, "wkv", None)
                    idx_wgate = getattr(idx_compressor, "wgate", None)
                    attn_freqs_cos = getattr(attn, "freqs_cos", None)
                    attn_freqs_sin = getattr(attn, "freqs_sin", None)
                    attn_table_len = int(getattr(attn_freqs_cos, "shape", (0,))[0])
                    attn_n_heads = int(getattr(attn, "n_heads", 0) or 0)
                    attn_head_dim = int(getattr(attn, "head_dim", 0) or 0)
                    attn_rope_head_dim = int(getattr(attn, "rope_head_dim", 0) or 0)
                    qkv_weights = _qkv_weights(attn)
                    comp_wkv = getattr(compressor, "wkv", None)
                    comp_wgate = getattr(compressor, "wgate", None)
                    attn_freq_table = _freq_table_spec(
                        attn_freqs_cos,
                        attn_freqs_sin,
                        attn_rope_head_dim,
                        f32_dtype,
                    )
                    can_fuse_all_kv_topk = bool(
                        int(k) == int(kv_len)
                        and idx_wkv is not None
                        and idx_wgate is not None
                        and comp_wkv is not None
                        and comp_wgate is not None
                        and attn_freq_table is not None
                        and attn_n_heads > 0
                        and attn_head_dim > 0
                        and attn_rope_head_dim > 0
                        and not any(value is None for value in qkv_weights)
                    )
                    if can_fuse_all_kv_topk:
                        self._precompile_all_kv_indexer_compressor_topk(
                            bucket,
                            layer_id=layer_id,
                            bsz=bsz,
                            seq=seq,
                            ratio=ratio,
                            kv_len=kv_len,
                            k=k,
                            index_topk=index_topk,
                            is_decode=bool(is_decode),
                            window_size=window_size,
                            decode_window_width=decode_window_width,
                            decode_start_positions=decode_start_positions,
                            layer_decode_max_c_len=layer_decode_max_c_len,
                            hidden_size=hidden_size,
                            attn=attn,
                            compressor=compressor,
                            idx_compressor=idx_compressor,
                            idx_wkv=idx_wkv,
                            idx_wgate=idx_wgate,
                            comp_wkv=comp_wkv,
                            comp_wgate=comp_wgate,
                            attn_freq_table=attn_freq_table,
                            attn_n_heads=attn_n_heads,
                            attn_head_dim=attn_head_dim,
                            attn_rope_head_dim=attn_rope_head_dim,
                            qkv_weights=qkv_weights,
                            bf16_dtype=bf16_dtype,
                            f32_dtype=f32_dtype,
                            i32_dtype=i32_dtype,
                            k_tile=int(K_TILE),
                        )
                        continue
                    for query_len_i in sorted({1, int(seq)}):
                        if query_len_i <= 0:
                            continue
                        hidden = _TensorSpec(
                            (bsz, query_len_i, hidden_size),
                            bf16_dtype,
                        )
                        freqs_cos = getattr(
                            getattr(indexer, "compressor", None),
                            "freqs_cos",
                            None,
                        )
                        freqs_sin = getattr(
                            getattr(indexer, "compressor", None),
                            "freqs_sin",
                            None,
                        )
                        score_scale = float(
                            getattr(indexer, "softmax_scale", 1.0) * n_heads**-0.5
                        )
                        freq_table = _freq_table_spec(
                            freqs_cos,
                            freqs_sin,
                            rope_head_dim,
                            f32_dtype,
                            fallback_len=int(
                                getattr(self.runtime_surface, "max_seq_len", 1)
                            ),
                        )
                        if freq_table is not None:
                            positions = _TensorSpec(
                                (bsz * query_len_i,),
                                i32_dtype,
                            )
                            if (
                                idx_wkv is not None
                                and idx_wgate is not None
                                and comp_wkv is not None
                                and comp_wgate is not None
                                and attn_freqs_cos is not None
                                and attn_freqs_sin is not None
                                and attn_table_len > 0
                                and attn_n_heads > 0
                                and attn_head_dim > 0
                                and attn_rope_head_dim > 0
                                and not any(value is None for value in qkv_weights)
                            ):
                                active_rows = (
                                    self._compressed_attention_bucket_for_tokens(
                                        bsz * query_len_i,
                                        int(bucket.token_bucket),
                                    )
                                )
                                attn_freq_table = _freq_table_spec(
                                    attn_freqs_cos,
                                    attn_freqs_sin,
                                    attn_rope_head_dim,
                                    f32_dtype,
                                )
                                self._attention_qkv_indexer_compressor_table_kernel_for(
                                    bucket,
                                    hidden,
                                    qkv_weights[0],
                                    qkv_weights[1],
                                    qkv_weights[2],
                                    qkv_weights[3],
                                    qkv_weights[4],
                                    comp_wkv,
                                    comp_wgate,
                                    idx_wkv,
                                    idx_wgate,
                                    indexer.wq_b,
                                    indexer.weights_proj,
                                    attn_freq_table,
                                    attn_freq_table,
                                    freq_table,
                                    freq_table,
                                    positions,
                                    n_heads=attn_n_heads,
                                    head_dim=attn_head_dim,
                                    rope_head_dim=attn_rope_head_dim,
                                    eps=float(getattr(attn, "eps", 1e-6)),
                                    block_size=64,
                                    fp8_max=240.0,
                                    q_softmax_scale=float(
                                        getattr(attn, "softmax_scale", 1.0)
                                    ),
                                    q_token_bucket=int(active_rows),
                                    kv_token_bucket=int(active_rows),
                                    indexer_score_scale=score_scale,
                                    indexer_n_heads=n_heads,
                                    indexer_head_dim=head_dim,
                                    indexer_rope_head_dim=rope_head_dim,
                                    indexer_block_size=32,
                                    indexer_fp8_max=240.0,
                                )
                                comp_ape = getattr(compressor, "ape", None)
                                idx_comp_ape = getattr(idx_compressor, "ape", None)
                                device_layer_state = _device_layer_state_for(
                                    getattr(self, "device_state", None),
                                    layer_id,
                                )
                                device_comp_state = getattr(
                                    device_layer_state,
                                    "compressor",
                                    None,
                                )
                                device_indexer_state = getattr(
                                    device_layer_state,
                                    "indexer",
                                    None,
                                )
                                can_fuse_decode_table_swa_dual_state_write = bool(
                                    int(query_len_i) == 1
                                    and int(ratio) > 0
                                    and int(window_size) > 0
                                    and comp_ape is not None
                                    and idx_comp_ape is not None
                                    and device_layer_state is not None
                                    and device_comp_state is not None
                                    and device_indexer_state is not None
                                    and hasattr(device_layer_state, "swa_kv_cache")
                                    and hasattr(device_comp_state, "kv_score_state")
                                    and hasattr(
                                        device_indexer_state,
                                        "kv_score_state",
                                    )
                                )
                                if can_fuse_decode_table_swa_dual_state_write:
                                    comp_overlap = bool(
                                        getattr(compressor, "overlap", False)
                                    )
                                    comp_coff = 2 if comp_overlap else 1
                                    comp_ring_size = int(comp_coff) * int(ratio)
                                    idx_comp_overlap = bool(
                                        getattr(idx_compressor, "overlap", False)
                                    )
                                    idx_comp_coff = 2 if idx_comp_overlap else 1
                                    idx_comp_ring_size = int(idx_comp_coff) * int(ratio)
                                    self._attention_qkv_indexer_compressor_table_kernel_for(
                                        bucket,
                                        hidden,
                                        qkv_weights[0],
                                        qkv_weights[1],
                                        qkv_weights[2],
                                        qkv_weights[3],
                                        qkv_weights[4],
                                        comp_wkv,
                                        comp_wgate,
                                        idx_wkv,
                                        idx_wgate,
                                        indexer.wq_b,
                                        indexer.weights_proj,
                                        attn_freq_table,
                                        attn_freq_table,
                                        freq_table,
                                        freq_table,
                                        positions,
                                        n_heads=attn_n_heads,
                                        head_dim=attn_head_dim,
                                        rope_head_dim=attn_rope_head_dim,
                                        eps=float(getattr(attn, "eps", 1e-6)),
                                        block_size=64,
                                        fp8_max=240.0,
                                        q_softmax_scale=float(
                                            getattr(attn, "softmax_scale", 1.0)
                                        ),
                                        q_token_bucket=int(active_rows),
                                        kv_token_bucket=int(active_rows),
                                        indexer_score_scale=score_scale,
                                        indexer_n_heads=n_heads,
                                        indexer_head_dim=head_dim,
                                        indexer_rope_head_dim=rope_head_dim,
                                        indexer_block_size=32,
                                        indexer_fp8_max=240.0,
                                        dynamic_decode_start_pos=True,
                                        write_swa_dual_state=True,
                                        swa_kv_cache=(device_layer_state.swa_kv_cache),
                                        kv_score_state=(
                                            device_comp_state.kv_score_state
                                        ),
                                        indexer_kv_score_state=(
                                            device_indexer_state.kv_score_state
                                        ),
                                        owner_ids=_TensorSpec((bsz,), i32_dtype),
                                        compressor_ape=comp_ape,
                                        indexer_compressor_ape=idx_comp_ape,
                                        window_size=int(window_size),
                                        ratio=int(ratio),
                                        start_pos=1,
                                        compressor_ring_size=int(comp_ring_size),
                                        indexer_compressor_ring_size=int(
                                            idx_comp_ring_size
                                        ),
                                    )
                            else:
                                raise RuntimeError(
                                    "DSV4 product indexer warmup requires fused "
                                    "attention QKV/compressor/indexer QW prep"
                                )
                        else:
                            raise RuntimeError(
                                "DSV4 product indexer warmup requires device "
                                "frequency tables and fused QKV/indexer prep"
                            )
                self._precompile_indexer_sparse_attention_fallback(
                    bucket,
                    bsz=bsz,
                    seq=seq,
                    kv_len=kv_len,
                    k=k,
                    ratio=ratio,
                    is_decode=bool(is_decode),
                    window_size=window_size,
                    decode_window_width=decode_window_width,
                    decode_start_positions=decode_start_positions,
                    hidden_size=hidden_size,
                    indexer_n_heads=n_heads,
                    device_indexer_state=device_indexer_state,
                    bf16_dtype=bf16_dtype,
                    f32_dtype=f32_dtype,
                    i32_dtype=i32_dtype,
                    k_tile=int(K_TILE),
                )
                continue
