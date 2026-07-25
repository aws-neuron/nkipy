from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import nkipy_serving.attention.nki_step_inputs as nki_step_inputs_module
from nkipy_serving.attention.nki_blocksparse_flash_attention import (
    build_decode_tile_plan_inplace,
    build_dummy_decode_tile_plan,
    build_dummy_prefill_tile_plan,
    build_prefill_tile_plan_inplace,
)
from nkipy_serving.attention.nki_step_inputs import (
    PreparedNkiStepInputs,
    initialize_prepared_nki_step_inputs,
    prepare_prepared_nki_step_inputs,
)


def _fake_tensor(name: str, shape: tuple[int, ...], dtype: np.dtype) -> SimpleNamespace:
    return SimpleNamespace(name=name, shape=shape, dtype=np.dtype(dtype))


def _prefill_plan(fill: int, *, num_tiles: int = 1) -> dict[str, np.ndarray]:
    return {
        "tile_q_indices": np.full((num_tiles, 128), fill, dtype=np.int32),
        "tile_block_tables": np.full((num_tiles, 32), fill, dtype=np.int32),
        "tile_masks": np.full((128, num_tiles, 1, 1024), fill, dtype=np.uint8),
        "num_dynamic_loop_steps": np.full((1, 1), fill, dtype=np.int32),
        "q_update_pred": np.full((num_tiles, 1), fill, dtype=np.uint8),
        "last_tile_indices": np.full((128, 2), fill, dtype=np.int32),
    }


def _decode_plan(fill: int, *, num_tiles: int = 1) -> dict[str, np.ndarray]:
    return {
        "tile_q_indices": np.full((num_tiles, 1), fill, dtype=np.int32),
        "tile_block_tables": np.full((num_tiles, 32), fill, dtype=np.int32),
        "tile_masks": np.full((128, num_tiles, 8), fill, dtype=np.uint8),
        "num_dynamic_loop_steps": np.full((1, 1), fill, dtype=np.int32),
        "q_update_pred": np.full((num_tiles, 1), fill, dtype=np.uint8),
        "last_tile_indices": np.full((128, 2), fill, dtype=np.int32),
    }


def _make_prepared_inputs() -> PreparedNkiStepInputs:
    decode_dummy_plan = _decode_plan(7, num_tiles=8)
    return PreparedNkiStepInputs(
        token_bucket=16,
        attn_bucket=128,
        max_num_prefill_tiles=1,
        max_num_decode_tiles=8,
        slot_mapping=_fake_tensor("slot_mapping", (16,), np.int32),
        slot_mapping_host=np.zeros((16,), dtype=np.int32),
        prefill_dummy_plan=_prefill_plan(9),
        decode_dummy_plan=decode_dummy_plan,
        prefill_plan_is_dummy=False,
        decode_plan_is_dummy=False,
        p_tqi=_fake_tensor("p_tqi", (1, 128), np.int32),
        p_tbt=_fake_tensor("p_tbt", (1, 32), np.int32),
        p_tm=_fake_tensor("p_tm", (128, 1, 1, 1024), np.uint8),
        p_ndls=_fake_tensor("p_ndls", (1, 1), np.int32),
        p_qup=_fake_tensor("p_qup", (1, 1), np.uint8),
        p_lti=_fake_tensor("p_lti", (128, 2), np.int32),
        d_tqi=_fake_tensor("d_tqi", (8, 1), np.int32),
        d_tbt=_fake_tensor("d_tbt", (8, 32), np.int32),
        d_tm=_fake_tensor("d_tm", (128, 8, 8), np.uint8),
        d_ndls=_fake_tensor("d_ndls", (1, 1), np.int32),
        d_qup=_fake_tensor("d_qup", (8, 1), np.uint8),
        d_lti=_fake_tensor("d_lti", (128, 2), np.int32),
        prefill_plan_host=build_dummy_prefill_tile_plan(
            max_num_prefill_tiles=1, block_size=32
        ),
        decode_plan_host=build_dummy_decode_tile_plan(
            max_num_decode_tiles=8, block_size=32
        ),
    )


def test_initialize_prepared_nki_step_inputs_uploads_dummy_plans() -> None:
    step_inputs = _make_prepared_inputs()
    writes: list[str] = []

    initialize_prepared_nki_step_inputs(
        step_inputs,
        lambda dst, src: writes.append(dst.name),
    )

    assert writes == [
        "p_tqi",
        "p_tbt",
        "p_tm",
        "p_ndls",
        "p_qup",
        "p_lti",
        "d_tqi",
        "d_tbt",
        "d_tm",
        "d_ndls",
        "d_qup",
        "d_lti",
    ]
    assert step_inputs.prefill_plan_is_dummy is True
    assert step_inputs.decode_plan_is_dummy is True


def test_prepare_prepared_nki_step_inputs_restores_inactive_side_to_dummy(
    monkeypatch,
) -> None:
    step_inputs = _make_prepared_inputs()
    step_inputs.prefill_plan_is_dummy = False
    step_inputs.decode_plan_is_dummy = False
    writes: list[str] = []

    monkeypatch.setattr(
        nki_step_inputs_module,
        "build_decode_tile_plan",
        lambda *args, **kwargs: pytest.fail(
            "decode-only fast path should bypass generic decode builder"
        ),
    )
    monkeypatch.setattr(
        nki_step_inputs_module,
        "build_unified_tile_plans",
        lambda *args, **kwargs: pytest.fail(
            "decode-only fast path should bypass unified builder"
        ),
    )

    decode_metadata = SimpleNamespace(
        query_start_loc=np.asarray([0, 1], dtype=np.int32),
        seq_lens=np.asarray([33], dtype=np.int32),
        block_tables=np.arange(64, dtype=np.int32).reshape((1, 64)),
        slot_mapping=np.asarray([21], dtype=np.int32),
    )

    prepared_inputs = prepare_prepared_nki_step_inputs(
        step_inputs,
        lambda dst, src: writes.append(dst.name),
        attn_metadata=decode_metadata,
        real_total_tokens=1,
        num_blocks=33,
        block_size=32,
    )

    assert prepared_inputs["slot_mapping"] is step_inputs.slot_mapping
    assert writes == [
        "slot_mapping",
        "d_tqi",
        "d_tbt",
        "d_tm",
        "d_ndls",
        "d_qup",
        "d_lti",
        "p_tqi",
        "p_tbt",
        "p_tm",
        "p_ndls",
        "p_qup",
        "p_lti",
    ]
    assert step_inputs.prefill_plan_is_dummy is True
    assert step_inputs.decode_plan_is_dummy is False
    assert step_inputs.decode_plan_host is not None
    assert step_inputs.decode_plan_host["tile_q_indices"][0, 0] == 0
    assert step_inputs.decode_plan_host["num_dynamic_loop_steps"][0, 0] == 1


def test_build_decode_tile_plan_inplace_fills_reusable_host_buffers() -> None:
    out = _decode_plan(0, num_tiles=8)
    metadata = SimpleNamespace(
        query_start_loc=np.asarray([0, 1, 2], dtype=np.int32),
        seq_lens=np.asarray([33, 1057], dtype=np.int32),
        block_tables=np.arange(128, dtype=np.int32).reshape((2, 64)),
    )

    plan = build_decode_tile_plan_inplace(
        metadata,
        max_num_decode_tiles=8,
        block_size=32,
        out=out,
    )

    assert plan is out
    assert plan["tile_q_indices"][:3, 0].tolist() == [0, 1, 1]
    assert plan["tile_block_tables"][0, :4].tolist() == [0, 1, 0, 0]
    assert plan["tile_block_tables"][1, :4].tolist() == [64, 65, 66, 67]
    assert plan["tile_block_tables"][2, :4].tolist() == [96, 97, 0, 0]
    assert int(plan["num_dynamic_loop_steps"][0, 0]) == 1
    assert plan["q_update_pred"][:3, 0].tolist() == [0, 1, 0]
    assert plan["last_tile_indices"][:2, :].tolist() == [[0, 0], [2, 1]]


def test_build_prefill_tile_plan_inplace_fills_reusable_host_buffers() -> None:
    out = _prefill_plan(0, num_tiles=2)
    metadata = SimpleNamespace(
        query_start_loc=np.asarray([0, 3, 5], dtype=np.int32),
        seq_lens=np.asarray([35, 41], dtype=np.int32),
        max_seq_len=41,
        block_tables=np.arange(128, dtype=np.int32).reshape((2, 64)),
    )

    plan = build_prefill_tile_plan_inplace(
        metadata,
        token_bucket=128,
        max_num_prefill_tiles=2,
        block_size=32,
        out=out,
    )

    assert plan is out
    assert plan["tile_q_indices"][0, :5].tolist() == [0, 1, 2, 1280, 1280]
    assert plan["tile_block_tables"][0, :4].tolist() == [0, 1, 0, 0]
    assert plan["tile_block_tables"][1, :4].tolist() == [2, 3, 0, 0]
    assert int(plan["num_dynamic_loop_steps"][0, 0]) == 1
    assert plan["q_update_pred"][0, 0] == 0
    assert plan["last_tile_indices"][0, :].tolist() == [0, 0]


def test_prepare_prepared_nki_step_inputs_reuses_slot_mapping_host_when_batch_shrinks(
    monkeypatch,
) -> None:
    step_inputs = _make_prepared_inputs()
    writes: list[np.ndarray] = []

    monkeypatch.setattr(
        nki_step_inputs_module,
        "build_decode_tile_plan",
        lambda *args, **kwargs: _decode_plan(3, num_tiles=8),
    )
    first = SimpleNamespace(
        query_start_loc=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        slot_mapping=np.asarray([11, 12, 13, 14], dtype=np.int32),
    )
    second = SimpleNamespace(
        query_start_loc=np.asarray([0, 1, 2], dtype=np.int32),
        slot_mapping=np.asarray([21, 22], dtype=np.int32),
    )

    def _capture(dst, src):
        if getattr(dst, "name", "") == "slot_mapping":
            writes.append(np.array(src, copy=True))

    prepare_prepared_nki_step_inputs(
        step_inputs,
        _capture,
        attn_metadata=first,
        real_total_tokens=4,
        num_blocks=33,
        block_size=32,
    )
    prepare_prepared_nki_step_inputs(
        step_inputs,
        _capture,
        attn_metadata=second,
        real_total_tokens=2,
        num_blocks=33,
        block_size=32,
    )

    scratch_slot = (33 - 1) * 32
    assert writes[0][:4].tolist() == [11, 12, 13, 14]
    assert writes[1][:4].tolist() == [21, 22, scratch_slot, scratch_slot]
    assert int(step_inputs.slot_mapping_active_tokens) == 2
    assert int(step_inputs.slot_mapping_scratch_slot) == scratch_slot
