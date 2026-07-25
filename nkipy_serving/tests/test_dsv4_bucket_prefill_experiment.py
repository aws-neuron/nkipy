from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np

import nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.indexer as indexer_module
import nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_precompile as indexer_precompile_module
from nkipy_serving.models.deepseek_v4.bucket_prefill_experiment import (
    BucketPrefillCase,
    bucket_graph_matches_live_reference,
    bucket_signature_key,
    build_prefill_moe_schedule_from_valid_mask,
    guarded_swa_scatter_matches_tail_reference,
    live_signature_key,
    make_request_major_valid_mask,
    scheduled_token_ids,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.indexer_table import (
    _run_compressed_attention_indexer_table_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvSetup,
    _should_bucket_indexer_all_kv_prefill,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.base import (
    _qkv_write_shape_candidates,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.qkv.indexer_precompile import (
    Dsv4ProductQkvIndexerPrecompileMixin,
    _decode_start_positions,
    _empty_indexer_warmup_variants,
    _qkv_token_topk_warmup_variants,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.manager import (
    Dsv4ProductBucketManagerMixin,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.support_warmup import (
    _bucketed_state_write_warmup_rows,
    _bucketed_two_source_primary_prefix_variants,
    _bucketed_two_source_warmup_specs,
)
from nkipy_serving.models.deepseek_v4.shapes import (
    bucketed_prefill_token_topk_compile_shape,
    prefill_token_topk_compile_bucket_lengths,
)
from nkipy_serving.models.deepseek_v4.variants import (
    QkvVariantName,
    variant_spec,
)
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule


class _FakeDeviceTensor:
    def __init__(self, name: str, shape: tuple[int, ...], dtype=np.float32):
        self.name = name
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.tensor_ref = f"ref:{name}"


def test_bucket_graph_contract_matches_live_rows_for_multiple_lengths() -> None:
    for real_seqlen in (10, 37, 129):
        assert bucket_graph_matches_live_reference(
            BucketPrefillCase(
                batch_size=1,
                real_seqlen=real_seqlen,
                bucket_seqlen=256,
                seed=real_seqlen,
            )
        )


def test_signature_key_uses_bucket_shape_not_live_length() -> None:
    cases = [
        BucketPrefillCase(
            batch_size=1,
            real_seqlen=real_seqlen,
            bucket_seqlen=256,
            seed=real_seqlen,
        )
        for real_seqlen in (10, 37, 129)
    ]

    assert len({bucket_signature_key(case) for case in cases}) == 1
    assert len({live_signature_key(case) for case in cases}) == len(cases)


def test_guarded_swa_scatter_redirects_padding_and_old_rows() -> None:
    cases = [
        BucketPrefillCase(
            batch_size=1,
            real_seqlen=9,
            bucket_seqlen=16,
            window_size=8,
            seed=1,
        ),
        BucketPrefillCase(
            batch_size=2,
            real_seqlen=6,
            bucket_seqlen=16,
            window_size=8,
            seed=2,
        ),
        BucketPrefillCase(
            batch_size=1,
            real_seqlen=25,
            bucket_seqlen=32,
            window_size=8,
            seed=3,
        ),
    ]
    assert all(guarded_swa_scatter_matches_tail_reference(case) for case in cases)


def test_all_kv_prefill_bucket_selection_ignores_bucket_topk_width() -> None:
    assert _should_bucket_indexer_all_kv_prefill(
        can_write_state_cache=True,
        ratio=2,
        active_bucket=256,
        bsz=1,
        seqlen=10,
        owner_ids=np.zeros((10,), dtype=np.int32),
    )


def test_all_kv_prefill_bucket_selection_rejects_mixed_token_owners() -> None:
    assert not _should_bucket_indexer_all_kv_prefill(
        can_write_state_cache=True,
        ratio=2,
        active_bucket=256,
        bsz=1,
        seqlen=10,
        owner_ids=np.arange(10, dtype=np.int32),
    )


def test_token_topk_warmup_covers_retargeted_short_continuation() -> None:
    class _Owner:
        def _configured_product_token_buckets(self) -> tuple[int, ...]:
            return (256, 1024)

        def _configured_product_decode_buckets(self) -> tuple[int, ...]:
            return (1, 16)

    starts = _decode_start_positions(9, 128)
    assert 8 in starts

    variants = set(
        _qkv_token_topk_warmup_variants(
            owner=_Owner(),
            is_decode=True,
            batch_size=1,
            seqlen=9,
            ratio=128,
            token_bucket=16,
            window_size=128,
            layer_decode_max_c_len=16,
            decode_start_positions=starts,
        )
    )

    assert (1, 8, 16, 128) in variants
    assert (5, 8, 16, 128) in variants


def test_token_topk_warmup_uses_request_bucket_for_short_prefill_lane() -> None:
    class _Owner:
        def _configured_product_token_buckets(self) -> tuple[int, ...]:
            return (256, 1024)

        def _configured_product_decode_buckets(self) -> tuple[int, ...]:
            return (1, 16)

    variants = set(
        _qkv_token_topk_warmup_variants(
            owner=_Owner(),
            is_decode=False,
            batch_size=1,
            seqlen=5,
            ratio=128,
            token_bucket=256,
            window_size=128,
            layer_decode_max_c_len=16,
            decode_start_positions=_decode_start_positions(5, 128),
        )
    )

    assert (16, 0, 16, 128) in variants
    assert (5, 0, 16, 128) not in variants


def test_plain_compressor_token_topk_short_prefill_uses_width_bucket() -> None:
    assert prefill_token_topk_compile_bucket_lengths(
        token_bucket=2048,
        window_size=128,
        ratio=128,
        k_tile=128,
    ) == (127, 2048)

    assert bucketed_prefill_token_topk_compile_shape(
        (1, 5, 4096),
        canonical_rows=2048,
        q_token_bucket=2048,
        kv_token_bucket=2048,
        window_size=128,
        ratio=128,
        offset=128,
        start_pos=0,
        max_c_len=32,
        k_tile=128,
    ) == (1, 127, 127, 128)


def test_token_topk_warmup_covers_plain_short_prefill_bucket() -> None:
    class _Owner:
        def _configured_product_token_buckets(self) -> tuple[int, ...]:
            return (2048,)

        def _configured_product_decode_buckets(self) -> tuple[int, ...]:
            return (1,)

    variants = set(
        _qkv_token_topk_warmup_variants(
            owner=_Owner(),
            is_decode=False,
            batch_size=1,
            seqlen=2048,
            ratio=128,
            token_bucket=2048,
            window_size=128,
            layer_decode_max_c_len=32,
            decode_start_positions=_decode_start_positions(2048, 128),
        )
    )

    assert (127, 0, 32, 128) in variants
    assert (2048, 0, 32, 2048) in variants


def test_qkv_write_prefill_warmup_covers_single_query_token_bucket() -> None:
    assert _qkv_write_shape_candidates(
        batch_size=1,
        seqlen=5,
        canonical_shape=(1, 2048),
        candidate_buckets=(2048,),
        is_decode=False,
    ) == ((1, 5), (1, 2048), (1, 1))

    assert _qkv_write_shape_candidates(
        batch_size=1,
        seqlen=5,
        canonical_shape=(1, 2048),
        candidate_buckets=(16,),
        is_decode=True,
    ) == ((1, 5), (1, 2048), (1, 16))


def test_empty_indexer_prefill_warmup_covers_live_single_query() -> None:
    assert _empty_indexer_warmup_variants(
        is_decode=False,
        seqlen=5,
        window_size=128,
        decode_window_width=128,
        decode_start_positions=(1, 5),
    ) == ((5, 0, 128), (1, 0, 128), (1, 1, 128), (1, 5, 128))


def test_indexer_bucketed_precomputed_qw_uses_bucketed_x(monkeypatch) -> None:
    shape_aliases: list[tuple[str, tuple[int, ...]]] = []
    captured: dict[str, object] = {}

    def fake_alias_shape(value, shape, *, default_name=None):
        del default_name
        shape_aliases.append(
            (
                value.name,
                tuple(int(dim) for dim in shape),
            )
        )
        return _FakeDeviceTensor(
            f"{value.name}_bucket",
            tuple(int(dim) for dim in shape),
            value.dtype,
        )

    def fake_first_dim_alias(*args, **kwargs):
        raise AssertionError("precomputed Q/W should not be sliced to live rows")

    def fake_indexer_score(q_T, w, **kwargs):
        captured["q_T_shape"] = tuple(int(dim) for dim in q_T.shape)
        captured["w_shape"] = tuple(int(dim) for dim in w.shape)
        captured["owner_ids_shape"] = tuple(
            int(dim) for dim in np.asarray(kwargs["owner_ids"]).shape
        )
        captured["output_shape"] = tuple(int(dim) for dim in kwargs["output"].shape)
        return _FakeDeviceTensor(
            "indexer_score",
            (int(q_T.shape[0]), int(kwargs["kv_len"])),
            np.float32,
        )

    def fake_sparse_prep(score, x, **kwargs):
        captured["score_shape"] = tuple(int(dim) for dim in score.shape)
        captured["x_shape"] = tuple(int(dim) for dim in x.shape)
        captured["sparse_seqlen"] = int(kwargs["seqlen"])
        return "topk", "mask"

    def fake_scratch(kind, shape, dtype):
        return _FakeDeviceTensor(str(kind), tuple(int(dim) for dim in shape), dtype)

    monkeypatch.setattr(
        indexer_module,
        "_alias_device_value_first_dim_slice",
        fake_first_dim_alias,
    )
    monkeypatch.setattr(
        indexer_module,
        "_alias_device_value_shape",
        fake_alias_shape,
    )
    monkeypatch.setattr(
        indexer_module,
        "_indexer_score_from_device_cache_adapter",
        fake_indexer_score,
    )

    indexer = SimpleNamespace(
        head_dim=128,
        n_heads=4,
        compress_ratio=4,
        rope_head_dim=64,
        index_topk=1,
        compressor=SimpleNamespace(),
    )
    q_t = _FakeDeviceTensor("q_t_bucket", (4096, 128, 4), ml_dtypes.bfloat16)
    w = _FakeDeviceTensor("w_bucket", (4096, 4), np.float32)
    x = _FakeDeviceTensor("x_live", (1, 3500, 4096), ml_dtypes.bfloat16)

    topk, mask = indexer_module._run_indexer(
        {"indexer_sparse_attention_prep_static": fake_sparse_prep},
        indexer,
        x,
        None,
        0,
        128,
        build_dir=None,
        device_state=SimpleNamespace(),
        owner_ids=np.zeros((3500,), dtype=np.int32),
        attention_scratch=fake_scratch,
        sparse_attention_rows=4096,
        sparse_attention_k_tile=128,
        sparse_attention_window_size=128,
        precomputed_compressor_state_written=True,
        precomputed_qw=(q_t, w),
    )

    assert (topk, mask) == ("topk", "mask")
    assert shape_aliases == [("x_live", (1, 4096, 4096))]
    assert captured["q_T_shape"] == (4096, 128, 4)
    assert captured["w_shape"] == (4096, 4)
    assert captured["owner_ids_shape"] == (4096,)
    assert captured["output_shape"] == (4096, 1024)
    assert captured["score_shape"] == (4096, 1024)
    assert captured["x_shape"] == (1, 4096, 4096)
    assert captured["sparse_seqlen"] == 4096


def test_indexer_fallback_warmup_precompiles_bucketed_score(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_precompile_score(**kwargs):
        calls.append(("score", kwargs))

    monkeypatch.setattr(
        indexer_precompile_module,
        "precompile_indexer_score_from_cache_device",
        fake_precompile_score,
    )

    class _Owner(Dsv4ProductQkvIndexerPrecompileMixin):
        build_dir = "/tmp/build_dsv4_test"

        def _compressed_attention_bucket_for_tokens(
            self,
            token_count: int,
            token_bucket: int,
        ) -> int:
            del token_count
            return int(token_bucket)

        def _indexer_sparse_attention_prep_static_kernel_for(self, *args, **kwargs):
            del args
            calls.append(("sparse", kwargs))

    device_indexer_state = SimpleNamespace(
        compressed_kv_cache=_FakeDeviceTensor(
            "indexer_cache",
            (4096, 128),
            ml_dtypes.bfloat16,
        ),
        spec=SimpleNamespace(max_compressed_len=1024),
    )
    owner = _Owner()

    owner._precompile_indexer_sparse_attention_fallback(
        SimpleNamespace(token_bucket=4096),
        bsz=1,
        seq=4096,
        kv_len=1024,
        k=1024,
        ratio=4,
        is_decode=False,
        window_size=128,
        decode_window_width=128,
        decode_start_positions=(),
        hidden_size=4096,
        indexer_n_heads=8,
        device_indexer_state=device_indexer_state,
        bf16_dtype=np.dtype(ml_dtypes.bfloat16),
        f32_dtype=np.dtype(np.float32),
        i32_dtype=np.dtype(np.int32),
        k_tile=128,
    )

    score_call = calls[0][1]
    assert calls[0][0] == "score"
    assert score_call["q_T_shape"] == (4096, 128, 8)
    assert score_call["kv_cache_shape"] == (4096, 128)
    assert score_call["owner_ids_shape"] == (4096,)
    assert score_call["w_shape"] == (4096, 8)
    assert score_call["kv_len"] == 1024
    assert score_call["max_compressed_len"] == 1024
    assert score_call["artifacts_dir"] == "/tmp/build_dsv4_test"
    assert calls[1][0] == "sparse"
    sparse_call = calls[1][1]
    assert sparse_call["dynamic_prefill_offset"] is True
    assert sparse_call["offset_tensor"].shape == (1, 1)


def test_dynamic_offset_sparse_prep_matches_static_offset() -> None:
    score = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.9, 0.2, 0.0],
            [0.5, 0.1, 0.7, 0.2],
        ],
        dtype=np.float32,
    )
    x = np.zeros((1, 4, 8), dtype=np.float32)
    kwargs = dict(
        bsz=1,
        seqlen=4,
        kv_len=4,
        k=2,
        ratio=2,
        prefill=True,
        window_size=2,
        start_pos=0,
        rows=4,
        k_tile=4,
    )

    static = graph_moe.indexer_sparse_attention_prep_static_fn(
        score,
        x,
        offset=7,
        **kwargs,
    )
    dynamic = graph_moe.indexer_sparse_attention_prep_static_dynamic_offset_fn(
        score,
        x,
        np.asarray([[7]], dtype=np.int32),
        **kwargs,
    )

    np.testing.assert_array_equal(dynamic[0], static[0])
    np.testing.assert_array_equal(dynamic[1], static[1])


def test_bucketed_two_source_warmup_covers_short_and_window_prefixes() -> None:
    assert _bucketed_two_source_primary_prefix_variants(
        window_size=128,
        k_padded=256,
    ) == (0, 128)

    specs = _bucketed_two_source_warmup_specs(
        (2048,),
        exact_query_rows=(2,),
        window_size=128,
        ratio=128,
        k_tile=128,
    )
    assert (2, 128, 128) in specs
    assert (2048, 2048, 256) in specs

    indexed_specs = _bucketed_two_source_warmup_specs(
        (4096,),
        exact_query_rows=(2,),
        window_size=128,
        ratio=4,
        k_tile=128,
        index_topk=512,
    )
    assert (2, 128, 128) in indexed_specs
    assert (4096, 4096, 640) in indexed_specs
    assert (4096, 4096, 1152) not in indexed_specs


def test_indexer_table_prefill_can_recover_bucket_backing_shape() -> None:
    class _FakeValue:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = np.dtype("float32")

    class _FakeBucketManager(Dsv4ProductBucketManagerMixin):
        def __init__(self, full_value: _FakeValue) -> None:
            self.full_value = full_value

        def _product_compile_attention_qkv_shape(
            self,
            bucket,
            *,
            bsz: int,
            seqlen: int,
        ) -> tuple[int, int]:
            return int(bsz), int(bucket.token_bucket)

        def _product_full_value_for(self, value, full_shape):
            del value
            if tuple(int(dim) for dim in self.full_value.shape) == tuple(
                int(dim) for dim in full_shape
            ):
                return self.full_value
            return None

    full = _FakeValue((1, 4096, 4096))
    active = _FakeValue((1, 3500, 4096))
    manager = _FakeBucketManager(full)

    bucketed_x, compiled_seqlen, realiased, offset = (
        manager._product_bucketed_prefill_offset(
            SimpleNamespace(token_bucket=4096),
            active,
            bsz=1,
            seqlen=3500,
            hidden_size=4096,
            window_size=0,
            offset=3500,
            bucketed=True,
            max_compile_tokens=4096,
        )
    )

    assert bucketed_x is full
    assert compiled_seqlen == 4096
    assert realiased is True
    assert offset == 4096


def test_indexer_table_qkv_publishes_bucketed_primary_contract() -> None:
    class _FakeTableQkv:
        def __init__(self) -> None:
            self.__self__ = self
            self._product_last_qkv_compiled_offset = (4096, True, 4096)
            self.kwargs: dict[str, object] = {}

        def __call__(self, *args, **kwargs):
            del args
            self.kwargs = kwargs
            return (
                _FakeDeviceTensor("q", (4096, 512, 8), ml_dtypes.bfloat16),
                _FakeDeviceTensor("kv", (4096, 512), np.float32),
                _FakeDeviceTensor("comp_kv", (4096, 512), ml_dtypes.bfloat16),
                _FakeDeviceTensor("comp_score", (4096, 512), ml_dtypes.bfloat16),
                _FakeDeviceTensor("idx_comp_kv", (4096, 128), ml_dtypes.bfloat16),
                _FakeDeviceTensor("idx_comp_score", (4096, 128), ml_dtypes.bfloat16),
                _FakeDeviceTensor("idx_q_t", (4096, 128, 4), ml_dtypes.bfloat16),
                _FakeDeviceTensor("idx_w", (4096, 4), np.float32),
            )

    table_qkv = _FakeTableQkv()
    attn = SimpleNamespace(
        wq_a=object(),
        q_norm=object(),
        wq_b=object(),
        wkv=object(),
        kv_norm=object(),
        n_heads=8,
        head_dim=512,
        rope_head_dim=64,
        eps=1e-6,
        softmax_scale=0.125,
    )
    indexer = SimpleNamespace(
        wq_b=object(),
        weights_proj=object(),
        softmax_scale=1.0,
        n_heads=4,
        head_dim=128,
        rope_head_dim=64,
    )
    qkv_setup = Dsv4CompressedAttentionQkvSetup(
        variant=variant_spec(QkvVariantName.INDEXER_COMPRESSOR_TABLE),
        outputs=None,
        qkv_outputs_flat_kv=True,
        compressor_wkv=object(),
        compressor_wgate=object(),
        indexer_obj=indexer,
        indexer_compressor_wkv=object(),
        indexer_compressor_wgate=object(),
        indexer_freqs_cos=object(),
        indexer_freqs_sin=object(),
        qkv_indexer_compressor_table=table_qkv,
    )

    result = _run_compressed_attention_indexer_table_qkv(
        qkv_setup=qkv_setup,
        x=np.zeros((1, 3500, 4096), dtype=np.float32),
        attn=attn,
        device_layer_state=SimpleNamespace(),
        owner_ids=np.zeros((3500,), dtype=np.int32),
        owner_ids_dev=None,
        device_token_positions=None,
        qkv_positions=np.arange(3500, dtype=np.int32),
        qkv_positions_input=np.arange(3500, dtype=np.int32),
        freqs_cos=object(),
        freqs_sin=object(),
        active_bucket=4096,
        win=128,
        ratio=4,
        start_pos=0,
        bsz=1,
        seqlen=3500,
        prefill_device_primary=False,
    )

    assert result.bucketed_prefill_done is True
    assert result.bucketed_kv_primary is result.kv_dev
    assert result.token_topk_offset == 4096
    assert result.attention_rows == 4096
    assert table_qkv.kwargs["window_size"] == 128


def test_indexer_compressor_post_qdq_warmup_ignores_qkv_fused_skip() -> None:
    class _Owner(Dsv4ProductQkvIndexerPrecompileMixin):
        def __init__(self) -> None:
            self.runtime_surface = SimpleNamespace(max_seq_len=4096)
            self.calls = []

        def _compressor_post_qdq_freq_table_kernel_for(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    freqs = SimpleNamespace(shape=(4096, 32), dtype=np.float32)
    compressor = SimpleNamespace(
        freqs_cos=freqs,
        freqs_sin=freqs,
        norm_weight=SimpleNamespace(shape=(128,), dtype=ml_dtypes.bfloat16),
        head_dim=128,
        rope_head_dim=64,
        rotate=True,
        eps=1e-6,
        compress_ratio=4,
        wkv=object(),
        wgate=object(),
        ape=object(),
    )
    attn = SimpleNamespace(
        freqs_cos=freqs,
        freqs_sin=freqs,
        wq_a=object(),
        q_norm=object(),
        wq_b=object(),
        wkv=object(),
        kv_norm=object(),
    )
    indexer = SimpleNamespace(index_topk=1024, compressor=compressor)
    bucket = SimpleNamespace(token_bucket=4096)
    owner = _Owner()

    owner._precompile_compressor_post_qdq_freq_table(
        bucket,
        bsz=1,
        seq=4096,
        ratio=4,
        kv_len=1024,
        attn=attn,
        indexer=indexer,
        compressor=compressor,
        f32_dtype=np.dtype(np.float32),
        i32_dtype=np.dtype(np.int32),
        skip_if_qkv_fused=True,
    )
    assert owner.calls == []

    owner._precompile_compressor_post_qdq_freq_table(
        bucket,
        bsz=1,
        seq=4096,
        ratio=4,
        kv_len=1024,
        attn=attn,
        indexer=indexer,
        compressor=compressor,
        f32_dtype=np.dtype(np.float32),
        i32_dtype=np.dtype(np.int32),
        skip_if_qkv_fused=False,
    )

    assert len(owner.calls) == 1
    args, kwargs = owner.calls[0]
    assert args[1].shape == (1024, 128)
    assert args[5].shape == (4096,)
    assert kwargs["clen"] == 1024
    assert kwargs["seqlen"] == 4096
    assert kwargs["compress_ratio"] == 4
    assert kwargs["source_token_positions"] is True


def test_compressor_post_qdq_warmup_skips_positions_shorter_than_prefill() -> None:
    class _Owner(Dsv4ProductQkvIndexerPrecompileMixin):
        def __init__(self) -> None:
            self.runtime_surface = SimpleNamespace(max_seq_len=4096)
            self.calls = []

        def _compressor_post_qdq_freq_table_kernel_for(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    freqs = SimpleNamespace(shape=(4096, 32), dtype=np.float32)
    compressor = SimpleNamespace(
        freqs_cos=freqs,
        freqs_sin=freqs,
        norm_weight=SimpleNamespace(shape=(128,), dtype=ml_dtypes.bfloat16),
        head_dim=128,
        rope_head_dim=64,
        rotate=True,
        eps=1e-6,
        compress_ratio=4,
    )
    owner = _Owner()

    owner._precompile_compressor_post_qdq_freq_table(
        SimpleNamespace(token_bucket=256),
        bsz=1,
        seq=512,
        ratio=4,
        kv_len=128,
        attn=SimpleNamespace(),
        indexer=SimpleNamespace(),
        compressor=compressor,
        f32_dtype=np.dtype(np.float32),
        i32_dtype=np.dtype(np.int32),
        skip_if_qkv_fused=False,
    )

    assert owner.calls == []


def test_swa_owner_window_warmup_covers_exact_short_active_rows() -> None:
    assert _bucketed_state_write_warmup_rows(
        (16, 32, 64, 128),
        max_rows=128,
    ) == tuple(range(1, 17)) + (32, 64, 128)

    assert _bucketed_state_write_warmup_rows(
        (16, 32, 64, 128),
        max_rows=8,
    ) == tuple(range(1, 9))


def test_mask_moe_schedule_matches_current_for_single_request_prefix() -> None:
    token_bucket = 16
    real = 5
    topk = np.zeros((token_bucket, 2), dtype=np.int32)
    topk[:, 0] = np.arange(token_bucket, dtype=np.int32) % 4
    topk[:, 1] = (topk[:, 0] + 1) % 4
    valid_mask = np.zeros((token_bucket,), dtype=bool)
    valid_mask[:real] = True

    current_tp, current_b2e, current_nb, current_ns = build_prefill_moe_schedule(
        topk,
        token_bucket=token_bucket,
        real_total_tokens=real,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )
    mask_tp, mask_b2e, mask_nb, mask_ns = build_prefill_moe_schedule_from_valid_mask(
        topk,
        valid_mask=valid_mask,
        token_bucket=token_bucket,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )

    assert current_nb == mask_nb
    assert current_ns == mask_ns
    assert np.array_equal(current_b2e, mask_b2e)
    assert np.array_equal(current_tp, mask_tp)


def test_current_moe_schedule_is_not_request_major_bucket_safe() -> None:
    batch_size = 2
    real = 3
    bucket = 16
    token_bucket = batch_size * bucket
    topk = np.zeros((token_bucket, 2), dtype=np.int32)
    topk[:, 0] = np.arange(token_bucket, dtype=np.int32) % 4
    topk[:, 1] = (topk[:, 0] + 1) % 4
    valid_mask = make_request_major_valid_mask(
        batch_size=batch_size,
        real_seqlen=real,
        bucket_seqlen=bucket,
    )

    token_position_to_id, _b2e, _nb, _ns = build_prefill_moe_schedule(
        topk,
        token_bucket=token_bucket,
        real_total_tokens=batch_size * real,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )

    scheduled = scheduled_token_ids(token_position_to_id)
    valid_rows = {int(i) for i in np.flatnonzero(valid_mask.reshape(-1))}

    assert scheduled != valid_rows
    assert scheduled - valid_rows, "current helper schedules request padding rows"
    assert valid_rows - scheduled, "current helper misses later request live rows"


def test_mask_moe_schedule_handles_request_major_bucket_padding() -> None:
    batch_size = 2
    real = 3
    bucket = 16
    token_bucket = batch_size * bucket
    topk = np.zeros((token_bucket, 2), dtype=np.int32)
    topk[:, 0] = np.arange(token_bucket, dtype=np.int32) % 4
    topk[:, 1] = (topk[:, 0] + 1) % 4
    valid_mask = make_request_major_valid_mask(
        batch_size=batch_size,
        real_seqlen=real,
        bucket_seqlen=bucket,
    )

    token_position_to_id, _b2e, _nb, _ns = build_prefill_moe_schedule_from_valid_mask(
        topk,
        valid_mask=valid_mask,
        token_bucket=token_bucket,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )

    scheduled = scheduled_token_ids(token_position_to_id)
    valid_rows = {int(i) for i in np.flatnonzero(valid_mask.reshape(-1))}
    assert scheduled == valid_rows
