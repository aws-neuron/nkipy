"""QKV variant setup objects and setup construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _decode_positions_1d_alias,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.compressor import (
    Dsv4DeferredIndexerState,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.outputs import (
    _build_compressed_attention_qkv_outputs,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.selection import (
    _select_qkv_variant_name,
)
from nkipy_serving.models.deepseek_v4.variants import (
    GraphVariantName,
    VariantOutputs,
    VariantSpec,
    variant_spec,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)


def _compressor_kernel_kwargs(compressor: Any, *, prefix: str = "compressor") -> dict:
    """Static compressor kernel params, splatted at each QKV dispatch call.

    Groups the per-branch caravan in one place. ``overlap`` is intentionally
    NOT included — call sites pass it from the device-state spec or the
    compressor object depending on path.
    """
    return {
        f"{prefix}_head_dim": int(compressor.head_dim),
        f"{prefix}_rope_head_dim": int(compressor.rope_head_dim),
        f"{prefix}_block_size": 32 if bool(compressor.rotate) else 64,
        f"{prefix}_fp8_max": 240.0,
        f"{prefix}_rotate": bool(compressor.rotate),
        f"{prefix}_eps": float(compressor.eps),
    }


def _decode_owner_pos_aliases(
    *,
    bsz: int,
    owner_ids_dev: Any | None,
    device_token_positions: Any | None,
) -> tuple[Any | None, Any | None]:
    """Slice the first ``bsz`` rows of the decode owner-ids and token-positions
    device buffers and canonicalize positions to shape ``(bsz,)``.

    The ~7 decode QKV-prep dispatch branches each need the same alias preamble.
    Returns ``(decode_owner_ids_dev, decode_positions_1d)``; either is ``None``
    when its source buffer is ``None`` (callers fall back to host arrays).
    """
    decode_owner_ids_dev = (
        None
        if owner_ids_dev is None
        else _alias_device_value_first_dim_slice(
            owner_ids_dev,
            start=0,
            size=int(bsz),
        )
    )
    decode_positions_dev = (
        None
        if device_token_positions is None
        else _alias_device_value_first_dim_slice(
            device_token_positions,
            start=0,
            size=int(bsz),
        )
    )
    decode_positions_1d = _decode_positions_1d_alias(
        decode_positions_dev,
        bsz=int(bsz),
    )
    return decode_owner_ids_dev, decode_positions_1d


def _has_tensor_ref(value: Any) -> bool:
    return hasattr(value, "tensor_ref")


def _state_attr_has_tensor_ref(state: Any, attr: str) -> bool:
    return _has_tensor_ref(getattr(state, attr, None))


def _owner_ids_supported_for_bucketed_prefill(
    owner_ids: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
) -> bool:
    try:
        flat = np.asarray(owner_ids).reshape(-1)
    except (TypeError, ValueError):
        return False
    bsz_i = int(bsz)
    if int(flat.size) == bsz_i:
        return True
    # Serving prefill carries one owner id per token for bsz==1. The bucketed
    # scatter reduces this to one request owner, so only accept it when every
    # live token names the same owner.
    seqlen_i = int(seqlen)
    if bsz_i == 1 and seqlen_i > 0 and int(flat.size) == seqlen_i:
        return bool(np.all(flat == flat[0]))
    return False


def _should_bucket_indexer_all_kv_prefill(
    *,
    can_write_state_cache: bool,
    ratio: int,
    active_bucket: int,
    bsz: int,
    seqlen: int,
    owner_ids: np.ndarray,
) -> bool:
    """Whether all-KV prefill should use the bucketed QKV + guarded scatter path.

    The real prompt can be small enough to use the all-KV indexer path even when
    the bucket's compressed length exceeds the configured indexer top-k. That is
    still bucket-safe: the bucketed prologue publishes the compiled shape and the
    dedicated state/cache scatter masks padded compressed columns with
    ``cache_real_clen``. Keep the predicate independent of index_topk so a live
    prompt length never falls through to a live-shaped cache-writing graph.
    """

    ratio_i = int(ratio)
    return bool(
        can_write_state_cache
        and ratio_i > 0
        and int(active_bucket) >= 2 * ratio_i - 1
        and int(bsz) == 1
        and _owner_ids_supported_for_bucketed_prefill(
            owner_ids,
            bsz=int(bsz),
            seqlen=int(seqlen),
        )
    )


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
        getattr(
            state_spec,
            "overlap",
            getattr(compressor, "overlap", False),
        )
    )
    if overlap:
        return min(int(seqlen), int(ratio) + int(seqlen) % int(ratio))
    return int(seqlen) % int(ratio)


def _decode_token_topk_max_c_len(
    *,
    options: Any,
    device_layer_state: Any,
) -> int:
    comp_spec = getattr(
        getattr(device_layer_state, "compressor", None),
        "spec",
        None,
    )
    max_c_len = int(getattr(comp_spec, "max_compressed_len", 0) or 0)
    if max_c_len <= 0:
        max_c_len = int(options.index_construction_max_c_len)
    if max_c_len <= 0:
        raise RuntimeError("decode compressed-topk requires max compressed length > 0")
    return max_c_len


@dataclass(slots=True)
class Dsv4CompressedAttentionQkvResult:
    handled: bool = True
    q_dev: Any | None = None
    kv_dev: Any | None = None
    qr_dev: Any | None = None
    topk_t_dev: Any | None = None
    mask_dev: Any | None = None
    precomputed_compressor_kv_score: tuple[Any, Any] | None = None
    precomputed_compressor_prefill_scatter_rows: Any | None = None
    precomputed_compressor_decode_scatter_rows: Any | None = None
    indexer_precomputed_compressor_kv_score: tuple[Any, Any] | None = None
    indexer_precomputed_compressor_decode_scatter_rows: Any | None = None
    indexer_precomputed_compressor_state_written: bool = False
    indexer_precomputed_qw: tuple[Any, Any] | None = None
    indexer_precomputed_empty_topk: tuple[Any, Any] | None = None
    deferred_indexer_state: Dsv4DeferredIndexerState | None = None
    compressor_state_swa_write_fused: bool = False
    bucketed_prefill_done: bool = False
    bucketed_kv_primary: Any | None = None
    token_topk_offset: int = 0
    all_kv_offset: int = 0
    attention_rows: int | None = None


@dataclass(slots=True)
class Dsv4CompressedAttentionQkvSetup:
    variant: VariantSpec
    outputs: dict[str, Any | None] | None
    qkv_outputs_flat_kv: bool
    compressor_wkv: Any | None = None
    compressor_wgate: Any | None = None
    compressor_ape: Any | None = None
    compressor_norm_weight: Any | None = None
    compressor_freqs_cos: Any | None = None
    compressor_freqs_sin: Any | None = None
    indexer_obj: Any | None = None
    indexer_compressor: Any | None = None
    indexer_compressor_wkv: Any | None = None
    indexer_compressor_wgate: Any | None = None
    indexer_freqs_cos: Any | None = None
    indexer_freqs_sin: Any | None = None
    compressor_prefill_state_tail_len: int = 0
    indexer_prefill_state_tail_len: int = 0
    compressed_kv_len: int = 0
    indexer_k: int = 0
    token_topk_offset: int = 0
    token_topk_max_c_len: int = 0
    token_topk_k_padded: int = 0
    use_qkv_compressor_prefill_post_qdq_token_topk_bucketed: bool = False
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed: bool = False
    qkv_token_topk_prep: Any | None = None
    qkv_compressor_token_topk_prep: Any | None = None
    qkv_compressor_token_topk_prep_write_swa_state: Any | None = None
    qkv_compressor_prefill_post_qdq_token_topk_prep: Any | None = None
    qkv_compressor_decode_post_qdq_token_topk_prep: Any | None = None
    qkv_compressor_table: Any | None = None
    qkv_indexer_compressor_table: Any | None = None
    qkv_indexer_compressor_table_write_swa_state: Any | None = None
    qkv_indexer_compressor_all_kv_topk_prep: Any | None = None
    qkv_indexer_compressor_all_kv_topk_prep_write_swa_state: Any | None = None
    qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep: Any | None = None
    qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep: Any | None = None
    qkv_empty_indexer_compressor_topk: Any | None = None

    @property
    def path(self) -> GraphVariantName:
        return self.variant.name

    @property
    def variant_outputs(self) -> VariantOutputs:
        return VariantOutputs(
            tensors=self.outputs,
            flat_kv=bool(self.qkv_outputs_flat_kv),
        )


def build_compressed_attention_qkv_setup(
    *,
    fns: Dsv4GraphFns,
    x: np.ndarray,
    attn: Any,
    options: Any,
    device_layer_state: Any,
    owner_ids: np.ndarray,
    qkv_positions: np.ndarray,
    qkv_positions_input: Any,
    freqs_cos: Any,
    freqs_sin: Any,
    qkv_fuses_q_scale: bool,
    product_shape_aliases: bool,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None,
    active_bucket: int,
    q_low_dim: int,
    n_heads: int,
    head_dim: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    win: int,
    ratio: int,
    prefill_device_primary: bool,
) -> Dsv4CompressedAttentionQkvSetup:
    qkv_token_topk_prep = fns.get("attention_qkv_token_topk_prep_from_freq_table")
    qkv_compressor_token_topk_prep = fns.get(
        "attention_qkv_compressor_kv_score_token_topk_prep_from_freq_table"
    )
    qkv_compressor_token_topk_prep_write_swa_state = fns.get(
        "attention_qkv_compressor_kv_score_token_topk_prep_write_swa_state_from_freq_table"
    )
    qkv_compressor_prefill_post_qdq_token_topk_prep = fns.get(
        "attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_from_freq_table"
    )
    qkv_compressor_decode_post_qdq_token_topk_prep = fns.get(
        "attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_from_freq_table"
    )
    qkv_compressor_table = fns.get("attention_qkv_compressor_kv_score_from_freq_table")
    qkv_indexer_compressor_table = fns.get(
        "attention_qkv_indexer_compressor_qw_prep_from_freq_table"
    )
    qkv_indexer_compressor_table_write_swa_state = fns.get(
        "attention_qkv_indexer_compressor_qw_prep_write_swa_state_from_freq_table"
    )
    qkv_indexer_compressor_all_kv_topk_prep = fns.get(
        "attention_qkv_indexer_compressor_all_kv_topk_from_freq_table"
    )
    qkv_indexer_compressor_all_kv_topk_prep_write_swa_state = fns.get(
        "attention_qkv_indexer_compressor_all_kv_topk_write_swa_state_from_freq_table"
    )
    qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep = fns.get(
        "attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_from_freq_table"
    )
    qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep = fns.get(
        "attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_from_freq_table"
    )
    qkv_empty_indexer_compressor_topk = fns.get(
        "attention_qkv_empty_indexer_compressor_topk_from_freq_table"
    )

    compressor_obj = getattr(attn, "compressor", None)
    compressor_wkv = getattr(compressor_obj, "wkv", None)
    compressor_wgate = getattr(compressor_obj, "wgate", None)
    compressor_ape = getattr(compressor_obj, "ape", None)
    compressor_norm_weight = getattr(compressor_obj, "norm_weight", None)
    compressor_freqs_cos = getattr(compressor_obj, "freqs_cos", None)
    compressor_freqs_sin = getattr(compressor_obj, "freqs_sin", None)
    qkv_outputs_flat_kv = qkv_fuses_q_scale and bool(
        fns.get("_attention_qkv_table_outputs_flat_kv", False)
    )
    use_qkv_token_topk_prep = bool(
        ratio
        and attn.indexer is None
        and qkv_fuses_q_scale
        and callable(qkv_token_topk_prep)
    )

    indexer_obj = getattr(attn, "indexer", None)
    indexer_compressor = getattr(indexer_obj, "compressor", None)
    indexer_compressor_wkv = getattr(indexer_compressor, "wkv", None)
    indexer_compressor_wgate = getattr(indexer_compressor, "wgate", None)
    indexer_freqs_cos = getattr(indexer_compressor, "freqs_cos", None)
    indexer_freqs_sin = getattr(indexer_compressor, "freqs_sin", None)

    compressor_prefill_state_tail_len = _prefill_state_tail_len(
        start_pos=int(start_pos),
        seqlen=int(seqlen),
        ratio=int(ratio),
        device_state=getattr(device_layer_state, "compressor", None),
        compressor=getattr(attn, "compressor", None),
    )

    indexer_ratio = (
        0
        if indexer_compressor is None
        else int(getattr(indexer_compressor, "compress_ratio", int(ratio)))
    )
    indexer_prefill_state_tail_len = _prefill_state_tail_len(
        start_pos=int(start_pos),
        seqlen=int(seqlen),
        ratio=int(indexer_ratio),
        device_state=getattr(device_layer_state, "indexer", None),
        compressor=indexer_compressor,
    )

    compressed_kv_len = int(start_pos + seqlen) // int(ratio) if ratio else 0
    indexer_topk = int(getattr(indexer_obj, "index_topk", 0) or 0)
    indexer_k = min(indexer_topk, compressed_kv_len) if compressed_kv_len > 0 else 0

    use_qkv_indexer_compressor_all_kv_topk_prep = bool(
        ratio
        and compressed_kv_len > 0
        and indexer_k == compressed_kv_len
        and indexer_obj is not None
        and qkv_fuses_q_scale
        and qkv_outputs_flat_kv
        and callable(qkv_indexer_compressor_all_kv_topk_prep)
        and compressor_wkv is not None
        and compressor_wgate is not None
        and indexer_compressor_wkv is not None
        and indexer_compressor_wgate is not None
        and indexer_freqs_cos is not None
        and indexer_freqs_sin is not None
    )
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep = bool(
        int(start_pos) == 0
        and use_qkv_indexer_compressor_all_kv_topk_prep
        and callable(qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep)
        and compressor_ape is not None
        and compressor_norm_weight is not None
        and compressor_freqs_cos is not None
        and compressor_freqs_sin is not None
        and getattr(indexer_compressor, "ape", None) is not None
        and getattr(indexer_compressor, "norm_weight", None) is not None
        and indexer_freqs_cos is not None
        and indexer_freqs_sin is not None
    )
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache = (
        bool(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
            and int(compressor_prefill_state_tail_len) > 0
            and int(indexer_prefill_state_tail_len) > 0
            and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
            and _state_attr_has_tensor_ref(
                device_layer_state.compressor, "kv_score_state"
            )
            and _state_attr_has_tensor_ref(
                device_layer_state.compressor,
                "compressed_kv_cache",
            )
            and _state_attr_has_tensor_ref(device_layer_state.indexer, "kv_score_state")
            and _state_attr_has_tensor_ref(
                device_layer_state.indexer,
                "compressed_kv_cache",
            )
        )
    )

    bkt_ratio = int(ratio) if int(ratio) > 0 else 0
    # DP-split bs>1 prefill presents one request per attention lane (local
    # bsz==1) while owner_ids can describe the full request batch. The bucketed
    # tail scatter has no multi-request gather path, so keep it on request-owner
    # or single-request per-token owner layouts and let other shapes use the
    # safe non-bucketed writer.
    bucketed_prefill_owner_ids_local = _owner_ids_supported_for_bucketed_prefill(
        owner_ids,
        bsz=int(bsz),
        seqlen=int(seqlen),
    )
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed = _should_bucket_indexer_all_kv_prefill(
        can_write_state_cache=bool(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache
        ),
        ratio=int(ratio),
        active_bucket=int(active_bucket),
        bsz=int(bsz),
        seqlen=int(seqlen),
        owner_ids=owner_ids,
    )
    if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed:
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache = False

    use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state = bool(
        int(start_pos) != 0
        and int(seqlen) == 1
        and int(ratio) > 0
        and (int(start_pos) + 1) % int(ratio) != 0
        and use_qkv_indexer_compressor_all_kv_topk_prep
        and callable(qkv_indexer_compressor_all_kv_topk_prep_write_swa_state)
        and compressor_ape is not None
        and getattr(indexer_compressor, "ape", None) is not None
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
        and hasattr(device_layer_state.indexer, "kv_score_state")
        and hasattr(device_layer_state.indexer, "spec")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
        and _state_attr_has_tensor_ref(device_layer_state.indexer, "kv_score_state")
    )
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep = bool(
        int(start_pos) != 0
        and int(seqlen) == 1
        and int(ratio) > 0
        and int(start_pos + 1) % int(ratio) == 0
        and use_qkv_indexer_compressor_all_kv_topk_prep
        and callable(qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep)
        and compressor_ape is not None
        and compressor_norm_weight is not None
        and compressor_freqs_cos is not None
        and compressor_freqs_sin is not None
        and getattr(indexer_compressor, "ape", None) is not None
        and getattr(indexer_compressor, "norm_weight", None) is not None
        and indexer_freqs_cos is not None
        and indexer_freqs_sin is not None
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
        and hasattr(device_layer_state.indexer, "kv_score_state")
        and hasattr(device_layer_state.indexer, "spec")
    )
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache = bool(
        use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
        and _state_attr_has_tensor_ref(
            device_layer_state.compressor,
            "compressed_kv_cache",
        )
        and _state_attr_has_tensor_ref(device_layer_state.indexer, "kv_score_state")
        and _state_attr_has_tensor_ref(
            device_layer_state.indexer,
            "compressed_kv_cache",
        )
    )
    use_qkv_indexer_compressor_table = bool(
        ratio
        and compressed_kv_len > 0
        and not use_qkv_indexer_compressor_all_kv_topk_prep
        and indexer_obj is not None
        and qkv_fuses_q_scale
        and qkv_outputs_flat_kv
        and callable(qkv_indexer_compressor_table)
        and compressor_wkv is not None
        and compressor_wgate is not None
        and indexer_compressor_wkv is not None
        and indexer_compressor_wgate is not None
        and getattr(indexer_obj, "wq_b", None) is not None
        and getattr(indexer_obj, "weights_proj", None) is not None
        and indexer_freqs_cos is not None
        and indexer_freqs_sin is not None
    )
    use_qkv_indexer_compressor_table_write_swa_state = bool(
        int(start_pos) != 0
        and int(seqlen) == 1
        and int(ratio) > 0
        and (int(start_pos) + 1) % int(ratio) != 0
        and use_qkv_indexer_compressor_table
        and callable(qkv_indexer_compressor_table_write_swa_state)
        and compressor_ape is not None
        and getattr(indexer_compressor, "ape", None) is not None
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
        and hasattr(device_layer_state.indexer, "kv_score_state")
        and hasattr(device_layer_state.indexer, "spec")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
        and _state_attr_has_tensor_ref(device_layer_state.indexer, "kv_score_state")
    )
    use_qkv_empty_indexer_compressor_topk = bool(
        ratio
        and compressed_kv_len <= 0
        and indexer_obj is not None
        and qkv_fuses_q_scale
        and qkv_outputs_flat_kv
        and callable(qkv_empty_indexer_compressor_topk)
        and compressor_wkv is not None
        and compressor_wgate is not None
        and indexer_compressor_wkv is not None
        and indexer_compressor_wgate is not None
    )
    use_qkv_compressor_token_topk_prep = bool(
        use_qkv_token_topk_prep
        and callable(qkv_compressor_token_topk_prep)
        and compressor_wkv is not None
        and compressor_wgate is not None
    )
    use_qkv_compressor_token_topk_prep_write_swa_state = bool(
        int(start_pos) != 0
        and int(seqlen) == 1
        and int(ratio) > 0
        and (int(start_pos) + 1) % int(ratio) != 0
        and use_qkv_compressor_token_topk_prep
        and callable(qkv_compressor_token_topk_prep_write_swa_state)
        and compressor_ape is not None
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
    )
    use_qkv_compressor_prefill_post_qdq_token_topk_prep = bool(
        int(start_pos) == 0
        and int(seqlen) >= int(ratio)
        and use_qkv_compressor_token_topk_prep
        and callable(qkv_compressor_prefill_post_qdq_token_topk_prep)
        and compressor_ape is not None
        and compressor_norm_weight is not None
        and compressor_freqs_cos is not None
        and compressor_freqs_sin is not None
    )
    use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache = bool(
        use_qkv_compressor_prefill_post_qdq_token_topk_prep
        and int(compressor_prefill_state_tail_len) > 0
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
        and _state_attr_has_tensor_ref(
            device_layer_state.compressor,
            "compressed_kv_cache",
        )
    )
    use_qkv_compressor_prefill_post_qdq_token_topk_bucketed = bool(
        use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache
        and bkt_ratio > 0
        and int(seqlen) >= int(bkt_ratio)
        and int(active_bucket) >= 2 * int(bkt_ratio) - 1
        and int(bsz) == 1
        and bucketed_prefill_owner_ids_local
    )
    if use_qkv_compressor_prefill_post_qdq_token_topk_bucketed:
        use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache = False
    use_qkv_compressor_decode_post_qdq_token_topk_prep = bool(
        int(start_pos) != 0
        and int(seqlen) == 1
        and int(start_pos + 1) % int(ratio) == 0
        and use_qkv_compressor_token_topk_prep
        and callable(qkv_compressor_decode_post_qdq_token_topk_prep)
        and compressor_ape is not None
        and compressor_norm_weight is not None
        and compressor_freqs_cos is not None
        and compressor_freqs_sin is not None
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
    )
    use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache = bool(
        use_qkv_compressor_decode_post_qdq_token_topk_prep
        and callable(qkv_compressor_decode_post_qdq_token_topk_prep)
        and _state_attr_has_tensor_ref(device_layer_state, "swa_kv_cache")
        and hasattr(device_layer_state.compressor, "kv_score_state")
        and hasattr(device_layer_state.compressor, "spec")
        and _state_attr_has_tensor_ref(device_layer_state.compressor, "kv_score_state")
        and _state_attr_has_tensor_ref(
            device_layer_state.compressor,
            "compressed_kv_cache",
        )
    )
    use_qkv_compressor_table = bool(
        ratio
        and not use_qkv_token_topk_prep
        and not use_qkv_indexer_compressor_table
        and not use_qkv_empty_indexer_compressor_topk
        and qkv_fuses_q_scale
        and qkv_outputs_flat_kv
        and callable(qkv_compressor_table)
        and compressor_wkv is not None
        and compressor_wgate is not None
    )

    token_topk_offset = 0
    token_topk_max_c_len = (
        _decode_token_topk_max_c_len(
            options=options,
            device_layer_state=device_layer_state,
        )
        if int(start_pos) > 0 and int(ratio) > 0 and attn.indexer is None
        else 0
    )
    token_topk_k_padded = 0

    if use_qkv_token_topk_prep:
        if start_pos == 0 and prefill_device_primary:
            token_topk_offset = int(win)
        elif start_pos == 0 and use_qkv_compressor_prefill_post_qdq_token_topk_bucketed:
            token_topk_offset = int(active_bucket)
        elif start_pos == 0:
            token_topk_offset = int(seqlen)
        else:
            token_topk_offset = int(win)
        token_topk_max_c_len = _decode_token_topk_max_c_len(
            options=options,
            device_layer_state=device_layer_state,
        )
        tt_eff_seqlen = (
            int(active_bucket)
            if (
                int(start_pos) == 0
                and use_qkv_compressor_prefill_post_qdq_token_topk_bucketed
            )
            else int(seqlen)
        )
        win_width = (
            int(win) if int(start_pos) > 0 else min(int(tt_eff_seqlen), int(win))
        )
        comp_width = (
            1
            if int(start_pos) == 0 and int(tt_eff_seqlen) // int(ratio) == 0
            else (
                int(token_topk_max_c_len)
                if int(start_pos) > 0
                else int(tt_eff_seqlen) // int(ratio)
            )
        )
        token_topk_k_raw = int(win_width) + int(comp_width)
        token_topk_k_padded = (
            (token_topk_k_raw + int(K_TILE) - 1) // int(K_TILE)
        ) * int(K_TILE)

    qkv_outputs = _build_compressed_attention_qkv_outputs(
        attention_scratch=attention_scratch,
        attn=attn,
        x_hidden_size=int(x.shape[2]),
        bsz=int(bsz),
        seqlen=int(seqlen),
        active_bucket=int(active_bucket),
        head_dim=int(head_dim),
        n_heads=int(n_heads),
        q_low_dim=int(q_low_dim),
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        token_topk_k_padded=int(token_topk_k_padded),
        token_topk_offset=int(token_topk_offset),
        token_topk_max_c_len=int(token_topk_max_c_len),
        win=int(win),
        ratio=int(ratio),
        start_pos=int(start_pos),
        product_shape_aliases=bool(product_shape_aliases),
        compressor_wkv=compressor_wkv,
        indexer_compressor_wkv=indexer_compressor_wkv,
        indexer_compressor=indexer_compressor,
        indexer_obj=indexer_obj,
        indexer_k=int(indexer_k),
        use_qkv_token_topk_prep=bool(use_qkv_token_topk_prep),
        use_qkv_indexer_compressor_all_kv_topk_prep=bool(
            use_qkv_indexer_compressor_all_kv_topk_prep
        ),
        use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state=bool(
            use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
        ),
        use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache=bool(
            use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache
        ),
        use_qkv_indexer_compressor_table=bool(use_qkv_indexer_compressor_table),
        use_qkv_indexer_compressor_table_write_swa_state=bool(
            use_qkv_indexer_compressor_table_write_swa_state
        ),
        use_qkv_empty_indexer_compressor_topk=bool(
            use_qkv_empty_indexer_compressor_topk
        ),
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep=bool(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
        ),
        use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep=bool(
            use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
        ),
        use_qkv_compressor_token_topk_prep=bool(use_qkv_compressor_token_topk_prep),
        use_qkv_compressor_token_topk_prep_write_swa_state=bool(
            use_qkv_compressor_token_topk_prep_write_swa_state
        ),
        use_qkv_compressor_prefill_post_qdq_token_topk_prep=bool(
            use_qkv_compressor_prefill_post_qdq_token_topk_prep
        ),
        use_qkv_compressor_decode_post_qdq_token_topk_prep=bool(
            use_qkv_compressor_decode_post_qdq_token_topk_prep
        ),
        use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache=bool(
            use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache
        ),
        use_qkv_compressor_table=bool(use_qkv_compressor_table),
    )

    variant_name = _select_qkv_variant_name(
        use_qkv_compressor_token_topk_prep_write_swa_state=(
            use_qkv_compressor_token_topk_prep_write_swa_state
        ),
        use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache=(
            use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache
        ),
        use_qkv_compressor_decode_post_qdq_token_topk_prep=(
            use_qkv_compressor_decode_post_qdq_token_topk_prep
        ),
        use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache=(
            use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache
        ),
        use_qkv_compressor_prefill_post_qdq_token_topk_prep=(
            use_qkv_compressor_prefill_post_qdq_token_topk_prep
        ),
        use_qkv_compressor_token_topk_prep=use_qkv_compressor_token_topk_prep,
        use_qkv_token_topk_prep=use_qkv_token_topk_prep,
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache=(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache
        ),
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep=(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
        ),
        use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state=(
            use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
        ),
        use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache=(
            use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache
        ),
        use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep=(
            use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
        ),
        use_qkv_indexer_compressor_all_kv_topk_prep=(
            use_qkv_indexer_compressor_all_kv_topk_prep
        ),
        use_qkv_indexer_compressor_table_write_swa_state=(
            use_qkv_indexer_compressor_table_write_swa_state
        ),
        use_qkv_indexer_compressor_table=use_qkv_indexer_compressor_table,
        use_qkv_empty_indexer_compressor_topk=use_qkv_empty_indexer_compressor_topk,
        use_qkv_compressor_table=use_qkv_compressor_table,
    )
    return Dsv4CompressedAttentionQkvSetup(
        variant=variant_spec(variant_name),
        outputs=qkv_outputs,
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_ape=compressor_ape,
        compressor_norm_weight=compressor_norm_weight,
        compressor_freqs_cos=compressor_freqs_cos,
        compressor_freqs_sin=compressor_freqs_sin,
        indexer_obj=indexer_obj,
        indexer_compressor=indexer_compressor,
        indexer_compressor_wkv=indexer_compressor_wkv,
        indexer_compressor_wgate=indexer_compressor_wgate,
        indexer_freqs_cos=indexer_freqs_cos,
        indexer_freqs_sin=indexer_freqs_sin,
        compressor_prefill_state_tail_len=int(compressor_prefill_state_tail_len),
        indexer_prefill_state_tail_len=int(indexer_prefill_state_tail_len),
        compressed_kv_len=int(compressed_kv_len),
        indexer_k=int(indexer_k),
        token_topk_offset=int(token_topk_offset),
        token_topk_max_c_len=int(token_topk_max_c_len),
        token_topk_k_padded=int(token_topk_k_padded),
        use_qkv_compressor_prefill_post_qdq_token_topk_bucketed=bool(
            use_qkv_compressor_prefill_post_qdq_token_topk_bucketed
        ),
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed=bool(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed
        ),
        qkv_token_topk_prep=qkv_token_topk_prep,
        qkv_compressor_token_topk_prep=qkv_compressor_token_topk_prep,
        qkv_compressor_token_topk_prep_write_swa_state=(
            qkv_compressor_token_topk_prep_write_swa_state
        ),
        qkv_compressor_prefill_post_qdq_token_topk_prep=(
            qkv_compressor_prefill_post_qdq_token_topk_prep
        ),
        qkv_compressor_decode_post_qdq_token_topk_prep=(
            qkv_compressor_decode_post_qdq_token_topk_prep
        ),
        qkv_compressor_table=qkv_compressor_table,
        qkv_indexer_compressor_table=qkv_indexer_compressor_table,
        qkv_indexer_compressor_table_write_swa_state=(
            qkv_indexer_compressor_table_write_swa_state
        ),
        qkv_indexer_compressor_all_kv_topk_prep=(
            qkv_indexer_compressor_all_kv_topk_prep
        ),
        qkv_indexer_compressor_all_kv_topk_prep_write_swa_state=(
            qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
        ),
        qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep=(
            qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
        ),
        qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep=(
            qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
        ),
        qkv_empty_indexer_compressor_topk=qkv_empty_indexer_compressor_topk,
    )
