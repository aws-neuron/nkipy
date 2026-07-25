from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import nkipy_serving.models.deepseek_v4.neff_runtime.support_warmup as support_warmup_module
from nkipy_serving.models.deepseek_v4.assembly.warmup_plan import (
    build_dsv4_warmup_plan,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.support_warmup import (
    Dsv4ProductSupportKernelWarmupMixin,
)
from nkipy_serving.models.deepseek_v4.warmup_support import (
    Dsv4DpWarmupPrecompiler,
    _append_dsv4_warmup_compile_entry,
    _collect_dsv4_support_compile_manifest,
    _run_dsv4_warmup_compile_manifest,
)
from nkipy_serving.runtime.precompile_paddings import PrecompilePaddings


def test_dsv4_support_compile_manifest_covers_bucketed_support_paths() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class _FakeOwner:
        def precompile_bucketed_prefill_swa_attention(self, buckets):
            calls.append(("swa_attention", tuple(int(v) for v in buckets)))

        def precompile_bucketed_prefill_two_source_attention(
            self,
            buckets,
            *,
            exact_query_rows=(),
        ):
            calls.append(
                (
                    "two_source",
                    tuple(int(v) for v in buckets),
                    tuple(int(v) for v in exact_query_rows),
                )
            )

        def precompile_swa_owner_window_write_buckets(self, buckets):
            calls.append(("swa_owner_window", tuple(int(v) for v in buckets)))

        def precompile_compressor_state_write_buckets(self, buckets):
            calls.append(("compressor_state", tuple(int(v) for v in buckets)))

        def precompile_compressor_prefill_pool_buckets(self, buckets):
            calls.append(("compressor_prefill_pool", tuple(int(v) for v in buckets)))

        def precompile_compressor_slot_write_buckets(self, buckets):
            calls.append(("compressor_slot_write", tuple(int(v) for v in buckets)))

        def precompile_dual_state_swa_write_buckets(self, buckets):
            calls.append(("dual_state_swa", tuple(int(v) for v in buckets)))

        def precompile_bucketed_single_state_swa_cache_write_buckets(self, buckets):
            calls.append(("single_state_swa_cache", tuple(int(v) for v in buckets)))

    plan = build_dsv4_warmup_plan(
        PrecompilePaddings(
            token_paddings=(256, 1024, 2048),
            bs_paddings=(1, 16),
            max_padded_num_tokens=2048,
            max_padded_batch_size=16,
        ),
        product_warmup_enabled=True,
        has_compressed_layers=True,
        compressed_boundary_pos=127,
    )

    entries = _collect_dsv4_support_compile_manifest(_FakeOwner(), warmup_plan=plan)

    assert [entry[1] for entry in entries] == [
        "precompile bucketed swa attention",
        "precompile bucketed two-source attention",
        "precompile swa owner-window writes",
        "precompile compressor state writes",
        "precompile compressor prefill pool",
        "precompile compressor slot writes",
        "precompile dual-state swa writes",
        "precompile bucketed single-state swa/cache writes",
    ]
    assert entries[0][3]["token_buckets"] == "(256, 1024, 2048)"
    assert entries[1][3]["exact_query_rows"] == "(2,)"
    assert entries[2][3]["buckets"] == "(16, 32, 64, 128)"
    assert entries[4][3]["buckets"] == "(256, 1024, 2048)"
    assert entries[5][3]["buckets"] == "(256, 1024, 2048)"
    assert entries[6][3]["buckets"] == "(1, 16)"

    records: list[tuple[str, dict[str, object]]] = []
    _run_dsv4_warmup_compile_manifest(
        entries,
        manifest_name="support-test",
        rank_msg="rank=0",
        record_warmup=lambda stage, **fields: records.append((stage, fields)),
    )

    assert calls == [
        ("swa_attention", (256, 1024, 2048)),
        ("two_source", (256, 1024, 2048), (2,)),
        ("swa_owner_window", (16, 32, 64, 128)),
        ("compressor_state", (16, 32, 64, 128)),
        ("compressor_prefill_pool", (256, 1024, 2048)),
        ("compressor_slot_write", (256, 1024, 2048)),
        ("dual_state_swa", (1, 16)),
        ("single_state_swa_cache", (256, 1024, 2048)),
    ]
    assert records[-1][0] == "compile manifest support-test"
    assert records[-1][1]["compiled"] == 8
    assert records[-1][1]["skipped_duplicates"] == 0


def test_prefill_pool_warmup_covers_main_and_indexer_compressors(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeDeviceTensor:
        def __init__(self, arr, *, name: str):
            arr = np.asarray(arr)
            self.shape = tuple(int(dim) for dim in arr.shape)
            self.dtype = arr.dtype
            self.name = name
            self.tensor_ref = f"ref:{name}"

        @classmethod
        def from_numpy(cls, arr, *, name: str):
            return cls(arr, name=name)

    def fake_run_prefill_pool_from_slab_device(**kwargs):
        calls.append(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(
        support_warmup_module,
        "_get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )
    monkeypatch.setattr(
        support_warmup_module,
        "run_prefill_pool_from_slab_device",
        fake_run_prefill_pool_from_slab_device,
    )

    main_compressor = SimpleNamespace(
        compress_ratio=4,
        head_dim=512,
        overlap=True,
        ape=np.zeros((4, 1024), dtype=np.float32),
    )
    indexer_compressor = SimpleNamespace(
        compress_ratio=4,
        head_dim=128,
        overlap=True,
        ape=np.zeros((4, 256), dtype=np.float32),
    )

    class _DeviceState:
        def layer(self, layer_id: int):
            assert layer_id == 0
            return SimpleNamespace(
                compressor=SimpleNamespace(
                    spec=SimpleNamespace(
                        compress_ratio=4,
                        head_dim=512,
                        overlap=True,
                        state_width=1024,
                    )
                ),
                indexer=SimpleNamespace(
                    spec=SimpleNamespace(
                        compress_ratio=4,
                        head_dim=128,
                        overlap=True,
                        state_width=256,
                    )
                ),
            )

    class _Owner(Dsv4ProductSupportKernelWarmupMixin):
        build_dir = "/tmp/build_dsv4_test"
        device_state = _DeviceState()
        runtime_surface = SimpleNamespace(
            blocks=(
                SimpleNamespace(
                    attn=SimpleNamespace(
                        compressor=main_compressor,
                        indexer=SimpleNamespace(compressor=indexer_compressor),
                    ),
                ),
            ),
        )

    _Owner().precompile_compressor_prefill_pool_buckets((4096,))

    assert [
        (
            call["kv_new"].shape,
            call["ape"].shape,
            call["ape"].dtype.type,
            call["head_dim"],
            call["overlap"],
        )
        for call in calls
    ] == [
        ((4096, 1024), (4, 1024), np.float32, 512, True),
        ((4096, 256), (4, 256), np.float32, 128, True),
    ]


def test_slot_write_warmup_covers_main_and_indexer_compressors(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeDeviceTensor:
        def __init__(self, arr, *, name: str):
            arr = np.asarray(arr)
            self.shape = tuple(int(dim) for dim in arr.shape)
            self.dtype = arr.dtype
            self.name = name
            self.tensor_ref = f"ref:{name}"

        @classmethod
        def from_numpy(cls, arr, *, name: str):
            return cls(arr, name=name)

    def fake_run_write_kv_slots_device(**kwargs):
        calls.append(kwargs)
        return kwargs["kv_cache"]

    monkeypatch.setattr(
        support_warmup_module,
        "_get_device_tensor_cls",
        lambda: _FakeDeviceTensor,
    )
    monkeypatch.setattr(
        support_warmup_module,
        "run_write_kv_slots_device",
        fake_run_write_kv_slots_device,
    )

    main_compressor = SimpleNamespace(compress_ratio=4)
    indexer_compressor = SimpleNamespace(compress_ratio=4)

    class _DeviceState:
        def layer(self, layer_id: int):
            assert layer_id == 0
            return SimpleNamespace(
                compressor=SimpleNamespace(
                    spec=SimpleNamespace(compress_ratio=4),
                    compressed_kv_cache=_FakeDeviceTensor(
                        np.zeros((2048, 512), dtype=np.float32),
                        name="main_compressed_kv_cache",
                    ),
                ),
                indexer=SimpleNamespace(
                    spec=SimpleNamespace(compress_ratio=4),
                    compressed_kv_cache=_FakeDeviceTensor(
                        np.zeros((2048, 128), dtype=np.float32),
                        name="indexer_compressed_kv_cache",
                    ),
                ),
            )

    class _Owner(Dsv4ProductSupportKernelWarmupMixin):
        build_dir = "/tmp/build_dsv4_test"
        device_state = _DeviceState()
        runtime_surface = SimpleNamespace(
            blocks=(
                SimpleNamespace(
                    attn=SimpleNamespace(
                        compressor=main_compressor,
                        indexer=SimpleNamespace(compressor=indexer_compressor),
                    ),
                ),
            ),
        )

    _Owner().precompile_compressor_slot_write_buckets((4096,))

    assert [
        (
            call["kv_cache"].shape,
            call["kv_new"].shape,
            call["slot_mapping"].shape,
        )
        for call in calls
    ] == [
        ((2048, 512), (1024, 512), (1024,)),
        ((2048, 128), (1024, 128), (1024,)),
    ]


def test_dsv4_warmup_compile_manifest_dedupes_by_key() -> None:
    calls: list[str] = []
    entries = []
    _append_dsv4_warmup_compile_entry(
        entries,
        family="support_attention",
        name="bucketed_swa",
        stage="precompile bucketed swa attention",
        metadata_key=((256, 1024),),
        compile_fn=lambda: calls.append("first"),
    )
    _append_dsv4_warmup_compile_entry(
        entries,
        family="support_attention",
        name="bucketed_swa",
        stage="precompile bucketed swa attention",
        metadata_key=((256, 1024),),
        compile_fn=lambda: calls.append("duplicate"),
    )

    records: list[tuple[str, dict[str, object]]] = []
    _run_dsv4_warmup_compile_manifest(
        entries,
        manifest_name="dedupe-test",
        rank_msg="rank=0",
        record_warmup=lambda stage, **fields: records.append((stage, fields)),
    )

    assert calls == ["first"]
    assert records[-1][0] == "compile manifest dedupe-test"
    assert records[-1][1]["compiled"] == 1
    assert records[-1][1]["skipped_duplicates"] == 1


def test_lane_moe_prefill_warmup_includes_live_single_token_bucket() -> None:
    calls: list[tuple[int, int, int, bool]] = []

    class _FakeOwner:
        def precompile_lane_moe_helpers(
            self,
            token_bucket: int,
            *,
            batch_size: int,
            seqlen: int,
            is_decode: bool = False,
        ) -> None:
            calls.append(
                (
                    int(token_bucket),
                    int(batch_size),
                    int(seqlen),
                    bool(is_decode),
                )
            )

    precompiler = Dsv4DpWarmupPrecompiler(
        _FakeOwner(),
        token_buckets=(2048,),
        decode_continuation_bucket=1,
        forward_mode_extend=1,
        rank_msg="rank=0",
    )
    precompiler._precompile_sampled_lane_helpers(
        step=SimpleNamespace(input_token_bucket=2048),
        sampled_keys=[(2048, 1, 5)],
        is_decode=False,
        step_name="prefill",
    )

    assert calls == [
        (2048, 1, 5, False),
        (2048, 1, 1, False),
        (2048, 1, 2048, False),
    ]
