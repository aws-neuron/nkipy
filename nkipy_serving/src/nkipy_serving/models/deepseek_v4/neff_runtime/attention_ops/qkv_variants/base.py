"""Base/default QKV variant runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _attention_qkv_quant,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvResult,
    Dsv4CompressedAttentionQkvSetup,
)
from nkipy_serving.models.deepseek_v4.variants import QkvVariantName


def _run_compressed_attention_base_qkv(
    *,
    qkv_setup: Dsv4CompressedAttentionQkvSetup,
    fns: Dsv4GraphFns,
    x: np.ndarray,
    attn: Any,
    freqs: np.ndarray | None,
    freqs_cos: Any,
    freqs_sin: Any,
    qkv_positions_input: Any,
    qkv_fuses_q_scale: bool,
    active_bucket: int,
) -> Dsv4CompressedAttentionQkvResult:
    variant = qkv_setup.variant
    qkv_outputs = qkv_setup.outputs
    qkv_outputs_flat_kv = bool(qkv_setup.qkv_outputs_flat_kv)
    compressor_wkv = qkv_setup.compressor_wkv
    compressor_wgate = qkv_setup.compressor_wgate
    token_topk_offset = int(qkv_setup.token_topk_offset)
    qkv_compressor_table = qkv_setup.qkv_compressor_table

    if variant.name == QkvVariantName.COMPRESSOR_TABLE:
        q_dev, kv_dev, qr_dev, comp_kv_dev, comp_score_dev = qkv_compressor_table(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            freqs_cos,
            freqs_sin,
            qkv_positions_input,
            n_heads=int(attn.n_heads),
            head_dim=int(attn.head_dim),
            rope_head_dim=int(attn.rope_head_dim),
            eps=float(attn.eps),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(attn.softmax_scale),
            q_token_bucket=int(active_bucket),
            kv_token_bucket=int(active_bucket),
            **(
                {"_nkipy_output_tensors": qkv_outputs}
                if qkv_outputs is not None
                else {}
            ),
        )
        return Dsv4CompressedAttentionQkvResult(
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=qr_dev,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            token_topk_offset=int(token_topk_offset),
        )

    q_dev, kv_dev, qr_dev = _attention_qkv_quant(
        fns,
        attn,
        x,
        freqs,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        positions=qkv_positions_input,
        q_softmax_scale=(float(attn.softmax_scale) if qkv_fuses_q_scale else None),
        q_token_bucket=active_bucket if qkv_fuses_q_scale else None,
        kv_token_bucket=active_bucket if qkv_outputs_flat_kv else None,
        output_tensors=qkv_outputs,
    )
    return Dsv4CompressedAttentionQkvResult(
        q_dev=q_dev,
        kv_dev=kv_dev,
        qr_dev=qr_dev,
        token_topk_offset=int(token_topk_offset),
    )
