"""QKV variant output allocation for DSV4 attention."""

from __future__ import annotations

from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.shapes import (
    bucketed_prefill_token_topk_compile_shape as _bucketed_prefill_token_topk_compile_shape,
)


def _build_compressed_attention_qkv_outputs(
    *,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None,
    attn: Any,
    x_hidden_size: int,
    bsz: int,
    seqlen: int,
    active_bucket: int,
    head_dim: int,
    n_heads: int,
    q_low_dim: int,
    qkv_fuses_q_scale: bool,
    qkv_outputs_flat_kv: bool,
    token_topk_k_padded: int,
    token_topk_offset: int,
    token_topk_max_c_len: int,
    win: int,
    ratio: int,
    start_pos: int,
    product_shape_aliases: bool,
    compressor_wkv: Any,
    indexer_compressor_wkv: Any,
    indexer_compressor: Any,
    indexer_obj: Any,
    indexer_k: int,
    use_qkv_token_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state: bool,
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache: bool,
    use_qkv_indexer_compressor_table: bool,
    use_qkv_indexer_compressor_table_write_swa_state: bool,
    use_qkv_empty_indexer_compressor_topk: bool,
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep: bool,
    use_qkv_compressor_token_topk_prep: bool,
    use_qkv_compressor_token_topk_prep_write_swa_state: bool,
    use_qkv_compressor_prefill_post_qdq_token_topk_prep: bool,
    use_qkv_compressor_decode_post_qdq_token_topk_prep: bool,
    use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache: bool,
    use_qkv_compressor_table: bool,
) -> dict[str, Any | None] | None:
    if attention_scratch is None:
        return None

    def _scratch(
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        return attention_scratch(
            kind,
            tuple(int(dim) for dim in shape),
            dtype,
        )

    q_output_kind = "attention_q_scaled_t" if qkv_fuses_q_scale else "attention_qkv_q"
    q_output_shape = (
        (active_bucket, head_dim, n_heads)
        if qkv_fuses_q_scale
        else (int(bsz), int(seqlen), n_heads, head_dim)
    )
    q_output_dtype = ml_dtypes.bfloat16 if qkv_fuses_q_scale else np.float32
    qkv_outputs = {
        "output0": _scratch(q_output_kind, q_output_shape, q_output_dtype),
        "output1": _scratch(
            "attention_qkv_kv_flat" if qkv_outputs_flat_kv else "attention_qkv_kv",
            (
                (int(active_bucket), head_dim)
                if qkv_outputs_flat_kv
                else (int(bsz), int(seqlen), head_dim)
            ),
            np.float32,
        ),
    }
    if use_qkv_token_topk_prep:
        qkv_outputs["output2"] = _scratch(
            "attention_topk_t",
            (int(token_topk_k_padded), int(active_bucket)),
            np.int32,
        )
        qkv_outputs["output3"] = _scratch(
            "attention_topk_mask",
            (int(active_bucket), int(token_topk_k_padded)),
            ml_dtypes.bfloat16,
        )
    elif (
        not use_qkv_indexer_compressor_all_kv_topk_prep
        and not use_qkv_indexer_compressor_table
        and not use_qkv_empty_indexer_compressor_topk
    ):
        qkv_outputs["output2"] = _scratch(
            "attention_qkv_qr",
            (int(bsz), int(seqlen), q_low_dim),
            np.float32,
        )
    if (
        use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
        or use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache
    ):
        sparse_win_width = int(win)
        sparse_comp_width = int(indexer_k)
        sparse_k_raw = int(sparse_win_width) + int(sparse_comp_width)
        sparse_k_padded = ((sparse_k_raw + int(K_TILE) - 1) // int(K_TILE)) * int(
            K_TILE
        )
        qkv_outputs["output2"] = _scratch(
            "attention_topk_t",
            (int(sparse_k_padded), int(active_bucket)),
            np.int32,
        )
        qkv_outputs["output3"] = _scratch(
            "attention_topk_mask",
            (int(active_bucket), int(sparse_k_padded)),
            ml_dtypes.bfloat16,
        )
    elif use_qkv_indexer_compressor_table_write_swa_state:
        qkv_outputs["output2"] = _scratch(
            "indexer_score_q_t",
            (
                int(bsz) * int(seqlen),
                int(indexer_obj.head_dim),
                int(indexer_obj.n_heads),
            ),
            ml_dtypes.bfloat16,
        )
        qkv_outputs["output3"] = _scratch(
            "indexer_score_weights",
            (int(bsz) * int(seqlen), int(indexer_obj.n_heads)),
            np.float32,
        )
    elif (
        use_qkv_indexer_compressor_all_kv_topk_prep
        or use_qkv_indexer_compressor_table
        or use_qkv_empty_indexer_compressor_topk
    ):
        comp_width = int(getattr(compressor_wkv, "shape", (0,))[0])
        idx_comp_width = int(getattr(indexer_compressor_wkv, "shape", (0,))[0])
        comp_shape = (int(bsz) * int(seqlen), comp_width)
        idx_comp_shape = (int(bsz) * int(seqlen), idx_comp_width)
        qkv_outputs["output2"] = _scratch(
            "compressor_kv_bf16",
            comp_shape,
            ml_dtypes.bfloat16,
        )
        qkv_outputs["output3"] = _scratch(
            "compressor_score_bf16",
            comp_shape,
            ml_dtypes.bfloat16,
        )
        qkv_outputs["output4"] = _scratch(
            "compressor_kv_bf16",
            idx_comp_shape,
            ml_dtypes.bfloat16,
        )
        qkv_outputs["output5"] = _scratch(
            "compressor_score_bf16",
            idx_comp_shape,
            ml_dtypes.bfloat16,
        )
        if use_qkv_indexer_compressor_table:
            qkv_outputs["output6"] = _scratch(
                "indexer_score_q_t",
                (
                    int(bsz) * int(seqlen),
                    int(indexer_obj.head_dim),
                    int(indexer_obj.n_heads),
                ),
                ml_dtypes.bfloat16,
            )
            qkv_outputs["output7"] = _scratch(
                "indexer_score_weights",
                (int(bsz) * int(seqlen), int(indexer_obj.n_heads)),
                np.float32,
            )
        else:
            sparse_win_width = (
                int(win) if int(start_pos) > 0 else min(int(seqlen), int(win))
            )
            sparse_comp_width = (
                int(indexer_k) if use_qkv_indexer_compressor_all_kv_topk_prep else 1
            )
            sparse_k_raw = int(sparse_win_width) + int(sparse_comp_width)
            sparse_k_padded = ((sparse_k_raw + int(K_TILE) - 1) // int(K_TILE)) * int(
                K_TILE
            )
            qkv_outputs["output6"] = _scratch(
                "attention_topk_t",
                (int(sparse_k_padded), int(active_bucket)),
                np.int32,
            )
            qkv_outputs["output7"] = _scratch(
                "attention_topk_mask",
                (int(active_bucket), int(sparse_k_padded)),
                ml_dtypes.bfloat16,
            )
            if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep:
                clen = int(seqlen) // int(ratio)
                qkv_outputs["output8"] = _scratch(
                    "compressor_post_qdq_bf16",
                    (int(bsz) * int(clen), int(attn.compressor.head_dim)),
                    ml_dtypes.bfloat16,
                )
                qkv_outputs["output9"] = _scratch(
                    "compressor_post_qdq_bf16",
                    (int(bsz) * int(clen), int(indexer_compressor.head_dim)),
                    ml_dtypes.bfloat16,
                )
            elif use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep:
                qkv_outputs["output8"] = _scratch(
                    "compressor_decode_post_qdq_bf16",
                    (int(bsz), int(attn.compressor.head_dim)),
                    ml_dtypes.bfloat16,
                )
                qkv_outputs["output9"] = _scratch(
                    "compressor_decode_post_qdq_bf16",
                    (int(bsz), int(indexer_compressor.head_dim)),
                    ml_dtypes.bfloat16,
                )
    elif (
        use_qkv_compressor_token_topk_prep
        and not use_qkv_compressor_token_topk_prep_write_swa_state
        and not use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache
    ) or use_qkv_compressor_table:
        comp_width = int(getattr(compressor_wkv, "shape", (0,))[0])
        comp_tokens = int(bsz) * int(seqlen)
        simple_bucketed_token_topk_rows = 0
        if (
            product_shape_aliases
            and use_qkv_compressor_token_topk_prep
            and not use_qkv_compressor_prefill_post_qdq_token_topk_prep
            and not use_qkv_compressor_decode_post_qdq_token_topk_prep
            and not use_qkv_compressor_token_topk_prep_write_swa_state
            and not use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache
            and qkv_outputs_flat_kv
        ):
            simple_bucketed_token_topk = _bucketed_prefill_token_topk_compile_shape(
                (int(bsz), int(seqlen), int(x_hidden_size)),
                canonical_rows=int(active_bucket),
                q_token_bucket=int(active_bucket),
                kv_token_bucket=int(active_bucket),
                window_size=int(win),
                ratio=int(ratio),
                offset=int(token_topk_offset),
                start_pos=int(start_pos),
                max_c_len=int(token_topk_max_c_len),
                k_tile=int(K_TILE),
            )
            if simple_bucketed_token_topk is not None:
                simple_bucketed_token_topk_rows = int(simple_bucketed_token_topk[2])
        if simple_bucketed_token_topk_rows > 0:
            comp_tokens = int(simple_bucketed_token_topk_rows)
        comp_shape = (comp_tokens, comp_width)
        output_offset = 4 if use_qkv_compressor_token_topk_prep else 3
        qkv_outputs[f"output{output_offset}"] = _scratch(
            "compressor_kv_bf16",
            comp_shape,
            ml_dtypes.bfloat16,
        )
        qkv_outputs[f"output{output_offset + 1}"] = _scratch(
            "compressor_score_bf16",
            comp_shape,
            ml_dtypes.bfloat16,
        )
        if use_qkv_compressor_prefill_post_qdq_token_topk_prep:
            clen = int(seqlen) // int(ratio)
            qkv_outputs["output6"] = _scratch(
                "compressor_post_qdq_bf16",
                (int(bsz) * int(clen), int(attn.compressor.head_dim)),
                ml_dtypes.bfloat16,
            )
        elif use_qkv_compressor_decode_post_qdq_token_topk_prep:
            qkv_outputs["output6"] = _scratch(
                "compressor_decode_post_qdq_bf16",
                (int(bsz), int(attn.compressor.head_dim)),
                ml_dtypes.bfloat16,
            )
    return qkv_outputs
