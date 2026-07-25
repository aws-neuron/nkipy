from __future__ import annotations

import contextlib
import queue
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import numpy as np
import pytest

from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.config import RuntimeConfig
from nkipy_serving.runtime.worker_coordinator import (
    _CHECKPOINT_REQUEST_STATE_METADATA_DECODER,
    _CLEAR_REQUEST_STATE_METADATA_DECODER,
    _CMD_BLOCK_SIZE,
    _CMD_CHECKPOINT_REQUEST_STATE,
    _CMD_CLEAR_REQUEST_STATE,
    _CMD_FLUSH_CACHE,
    _CMD_FORWARD_STEP,
    _CMD_RELOAD_WEIGHTS,
    _CMD_RESTORE_REQUEST_STATE,
    _RELOAD_METADATA_DECODER,
    _RESTORE_REQUEST_STATE_METADATA_DECODER,
    _STATUS_ERROR,
    _STATUS_OK,
    WorkerCoordinator,
    _allocate_shared_buffers,
    _cmd_block_read,
    _cmd_block_write,
    _compute_ctrl_shm_layout,
    _decode_forward_batch_metadata,
    _encode_forward_batch_metadata,
    _output_slot_read,
    _output_slot_write_top1,
    _output_slot_write_topk,
    _read_batch_from_shm,
    _summarize_worker_startup,
    _worker_slot_write_done,
    _write_batch_to_shm,
)


def _test_runtime_config(**overrides) -> RuntimeConfig:
    defaults = dict(
        execution_backend="nkipy",
        tp_degree=2,
        ep_degree=1,
        request_buckets=(1, 2, 4),
        token_buckets=(32, 128),
        kv_pool_size=256,
        max_context_len=128,
    )
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


def _sample_forward_batch() -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=np.asarray([11, 12] + [0] * 30, dtype=np.int32),
        positions=np.asarray([5, 6] + [0] * 30, dtype=np.int32),
        seq_lens=np.asarray([5, 6], dtype=np.int64),
        slot_mapping=np.asarray([21, 22] + [0] * 30, dtype=np.int64),
        block_tables=np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        query_start_loc=np.asarray([0, 1, 2], dtype=np.int64),
        sample_mask=np.asarray([True, True], dtype=np.bool_),
        requested_topk=1,
        token_bucket=32,
        real_total_tokens=2,
    )


def _forward_batch_with(**overrides) -> ForwardBatch:
    return ForwardBatch(**{**_sample_forward_batch().__dict__, **overrides})


def _dp_attention_runtime_config(*, lanes: int = 2) -> RuntimeConfig:
    return _test_runtime_config(
        tp_degree=2,
        ep_degree=lanes,
        replica_degree=1,
        attention_dp_degree=lanes,
    )


def _dp_superstep_batch(*, lane_batch_sizes: tuple[int, ...] = (1, 1)) -> ForwardBatch:
    lane_batch_sizes_arr = np.asarray(lane_batch_sizes, dtype=np.int32)
    lane_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.cumsum(lane_batch_sizes_arr, dtype=np.int32),
        ]
    )
    return _forward_batch_with(
        attention_lane=-1,
        dp_attention_superstep=True,
        dp_attention_num_lanes=len(lane_batch_sizes),
        dp_attention_lane_token_counts=lane_batch_sizes_arr.copy(),
        dp_attention_lane_batch_sizes=lane_batch_sizes_arr,
        dp_attention_lane_token_offsets=lane_offsets,
        dp_attention_lane_batch_offsets=lane_offsets.copy(),
    )


@contextlib.contextmanager
def _fake_coordinator(
    runtime_config: RuntimeConfig | None = None,
    *,
    generation: int = 1,
    real_batch_shm: bool = False,
):
    runtime_config = runtime_config or _test_runtime_config()
    coordinator = WorkerCoordinator.__new__(WorkerCoordinator)
    coordinator.runtime_config = runtime_config
    coordinator.root_comm_id = ""
    coordinator.timeout_s = 1
    coordinator._shm_bufs = (
        _allocate_shared_buffers(runtime_config)
        if real_batch_shm
        else SimpleNamespace(close_and_unlink=lambda: None)
    )
    coordinator._ctx = None
    coordinator._result_queue = queue.Queue()
    coordinator._processes = {
        rank: SimpleNamespace(is_alive=lambda: True)
        for rank in range(runtime_config.total_workers)
    }
    coordinator._last_forward_output = None
    coordinator._prof_shm_write_dur = 0.0
    coordinator._prof_broadcast_dur = 0.0
    coordinator._prof_collect_dur = 0.0
    coordinator._prof_first_result_dur = 0.0
    coordinator._prof_combine_dur = 0.0
    coordinator._lane_metadata = {}
    coordinator._worker_startup_summaries = {}
    coordinator._tp_degree = runtime_config.tp_degree
    coordinator._attention_dp_degree = int(runtime_config.attention_dp_degree)
    coordinator._output_ranks = set(
        range(runtime_config.tp_degree * int(runtime_config.attention_dp_degree))
    )
    coordinator._active_forward_output_ranks = set(range(runtime_config.tp_degree))
    coordinator._forward_output_ranks_by_request = {}
    coordinator._active_forward_dp_lane_batch_sizes = None
    coordinator._forward_dp_lane_batch_sizes_by_request = {}
    coordinator._total_workers = runtime_config.total_workers
    ctrl_size, coordinator._ctrl_output_slot_size = _compute_ctrl_shm_layout(
        runtime_config
    )
    coordinator._ctrl_shm = SharedMemory(create=True, size=ctrl_size)
    coordinator._ctrl_shm.buf[:ctrl_size] = b"\x00" * ctrl_size
    coordinator._generation = generation

    try:
        yield coordinator
    finally:
        coordinator._shm_bufs.close_and_unlink()
        coordinator._ctrl_shm.close()
        coordinator._ctrl_shm.unlink()


def test_cmd_block_write_uses_explicit_metadata_length() -> None:
    shm = SharedMemory(create=True, size=_CMD_BLOCK_SIZE)
    try:
        shm.buf[:_CMD_BLOCK_SIZE] = b"\x00" * _CMD_BLOCK_SIZE
        _cmd_block_write(shm.buf, 1, _CMD_FORWARD_STEP, b'{"payload":"long"}')
        _cmd_block_write(shm.buf, 2, _CMD_FORWARD_STEP, b"{}")

        generation, cmd, metadata = _cmd_block_read(shm.buf)
        assert generation == 2
        assert cmd == _CMD_FORWARD_STEP
        assert metadata == b"{}"
    finally:
        shm.close()
        shm.unlink()


def test_worker_startup_summary_reports_slowest_rank_and_stage_max() -> None:
    per_rank = {
        0: {
            "rank": 0,
            "tp_rank": 0,
            "ep_rank": 0,
            "visible_core": 0,
            "total_elapsed_s": 2.0,
            "stages": [
                {"stage": "imports ready", "elapsed_s": 0.5, "total_elapsed_s": 0.8},
                {"stage": "warmup done", "elapsed_s": 1.0, "total_elapsed_s": 2.0},
            ],
        },
        1: {
            "rank": 1,
            "tp_rank": 1,
            "ep_rank": 0,
            "visible_core": 1,
            "total_elapsed_s": 3.0,
            "stages": [
                {"stage": "imports ready", "elapsed_s": 0.25, "total_elapsed_s": 0.7},
                {"stage": "warmup done", "elapsed_s": 2.0, "total_elapsed_s": 3.0},
            ],
        },
    }

    summary = _summarize_worker_startup(
        per_rank,
        total_workers=4,
        slowest_limit=1,
    )

    assert summary["total_workers"] == 4
    assert summary["ready_workers"] == 2
    assert summary["max_total_elapsed_s"] == 3.0
    assert summary["mean_total_elapsed_s"] == 2.5
    assert summary["slowest_ranks"][0]["rank"] == 1
    assert summary["slowest_ranks"][0]["stages"] == per_rank[1]["stages"]
    assert summary["stage_max_elapsed_s"]["warmup done"] == {
        "rank": 1,
        "elapsed_s": 2.0,
        "total_elapsed_s": 3.0,
    }


def test_worker_coordinator_startup_summary_uses_ready_payloads() -> None:
    with _fake_coordinator() as coordinator:
        coordinator._worker_startup_summaries = {
            0: {"rank": 0, "total_elapsed_s": 1.25, "stages": []},
            1: {"rank": 1, "total_elapsed_s": 1.5, "stages": []},
        }

        summary = coordinator.startup_summary()

    assert summary["total_workers"] == 2
    assert summary["ready_workers"] == 2
    assert summary["max_total_elapsed_s"] == 1.5


def test_control_commands_write_expected_metadata_to_shm() -> None:
    def _run_case(start_generation: int, op_name: str, cmd: int, invoke):
        with _fake_coordinator(generation=start_generation) as coordinator:
            expected_generation = start_generation + 1
            seen: dict[str, object] = {}

            def _fake_collect(generation: int, op_name: str) -> None:
                seen["generation"] = generation
                seen["op_name"] = op_name

            coordinator._collect_worker_command_shm = _fake_collect
            result = invoke(coordinator)
            generation, actual_cmd, metadata = _cmd_block_read(
                coordinator._ctrl_shm.buf
            )

        assert generation == expected_generation
        assert actual_cmd == cmd
        assert seen == {"generation": expected_generation, "op_name": op_name}
        return result, metadata

    _, metadata = _run_case(
        3,
        "reload_weights",
        _CMD_RELOAD_WEIGHTS,
        lambda coordinator: coordinator.reload_weights("/tmp/reload-target"),
    )
    decoded_reload = _RELOAD_METADATA_DECODER.decode(metadata)
    assert decoded_reload.model_path == "/tmp/reload-target"

    _, metadata = _run_case(
        8,
        "flush_cache",
        _CMD_FLUSH_CACHE,
        lambda coordinator: coordinator.flush_cache(),
    )
    assert metadata == b""

    _, metadata = _run_case(
        11,
        "clear_request_state",
        _CMD_CLEAR_REQUEST_STATE,
        lambda coordinator: coordinator.clear_request_state([3, 1, 3, -1]),
    )
    decoded_clear = _CLEAR_REQUEST_STATE_METADATA_DECODER.decode(metadata)
    assert decoded_clear.owner_ids == [1, 3]

    checkpoint_id, metadata = _run_case(
        20,
        "checkpoint_request_state",
        _CMD_CHECKPOINT_REQUEST_STATE,
        lambda coordinator: coordinator.checkpoint_request_state(
            checkpoint_id="cp-a",
            owner_id=3,
            seq_len=11,
            num_tokens=2,
        ),
    )
    decoded_checkpoint = _CHECKPOINT_REQUEST_STATE_METADATA_DECODER.decode(metadata)
    assert checkpoint_id == "cp-a"
    assert decoded_checkpoint.checkpoint_id == "cp-a"
    assert decoded_checkpoint.owner_id == 3
    assert decoded_checkpoint.seq_len == 11
    assert decoded_checkpoint.num_tokens == 2

    _, metadata = _run_case(
        30,
        "restore_request_state",
        _CMD_RESTORE_REQUEST_STATE,
        lambda coordinator: coordinator.restore_request_state("cp-a"),
    )
    decoded_restore = _RESTORE_REQUEST_STATE_METADATA_DECODER.decode(metadata)
    assert decoded_restore.checkpoint_id == "cp-a"


def test_forward_batch_metadata_msgpack_round_trip() -> None:
    runtime_config = _test_runtime_config()
    shm_bufs = _allocate_shared_buffers(runtime_config)
    try:
        batch = _forward_batch_with(
            attention_lane=3,
            state_owner_ids=np.asarray([8, 9], dtype=np.int32),
        )
        metadata = _write_batch_to_shm(batch, shm_bufs)
        payload = _encode_forward_batch_metadata(metadata)
        decoded = _decode_forward_batch_metadata(payload)
        restored = _read_batch_from_shm(
            decoded,
            {
                "input_ids": shm_bufs.input_ids,
                "positions": shm_bufs.positions,
                "slot_mapping": shm_bufs.slot_mapping,
                "seq_lens": shm_bufs.seq_lens,
                "block_tables": shm_bufs.block_tables,
                "query_start_loc": shm_bufs.query_start_loc,
                "sample_mask": shm_bufs.sample_mask,
                "state_owner_ids": shm_bufs.state_owner_ids,
            },
        )
        assert restored.forward_mode == batch.forward_mode
        assert restored.batch_size == batch.batch_size
        assert restored.token_bucket == batch.token_bucket
        assert restored.real_total_tokens == batch.real_total_tokens
        assert restored.requested_topk == batch.requested_topk
        assert restored.attention_lane == batch.attention_lane
        assert np.array_equal(restored.state_owner_ids, batch.state_owner_ids)
        assert np.array_equal(restored.input_ids, batch.input_ids)
        assert np.array_equal(restored.positions, batch.positions)
        assert np.array_equal(restored.seq_lens, batch.seq_lens)
        assert np.array_equal(restored.slot_mapping, batch.slot_mapping)
        assert np.array_equal(restored.block_tables, batch.block_tables)
        assert np.array_equal(restored.query_start_loc, batch.query_start_loc)
        assert np.array_equal(restored.sample_mask, batch.sample_mask)
    finally:
        shm_bufs.close_and_unlink()


def test_forward_batch_dp_attention_superstep_metadata_round_trip() -> None:
    runtime_config = _dp_attention_runtime_config()
    shm_bufs = _allocate_shared_buffers(runtime_config)
    try:
        batch = _dp_superstep_batch()

        metadata = _write_batch_to_shm(batch, shm_bufs)
        restored = _read_batch_from_shm(
            _decode_forward_batch_metadata(_encode_forward_batch_metadata(metadata)),
            {
                "input_ids": shm_bufs.input_ids,
                "positions": shm_bufs.positions,
                "slot_mapping": shm_bufs.slot_mapping,
                "seq_lens": shm_bufs.seq_lens,
                "block_tables": shm_bufs.block_tables,
                "query_start_loc": shm_bufs.query_start_loc,
                "sample_mask": shm_bufs.sample_mask,
                "temperatures": shm_bufs.temperatures,
                "top_ks": shm_bufs.top_ks,
                "top_ps": shm_bufs.top_ps,
                "min_ps": shm_bufs.min_ps,
                "uniform_u": shm_bufs.uniform_u,
                "state_owner_ids": shm_bufs.state_owner_ids,
                "dp_attention_lane_token_counts": shm_bufs.dp_attention_lane_token_counts,
                "dp_attention_lane_batch_sizes": shm_bufs.dp_attention_lane_batch_sizes,
                "dp_attention_lane_token_offsets": shm_bufs.dp_attention_lane_token_offsets,
                "dp_attention_lane_batch_offsets": shm_bufs.dp_attention_lane_batch_offsets,
            },
        )

        assert restored.dp_attention_superstep is True
        assert restored.dp_attention_num_lanes == 2
        assert restored.attention_lane == -1
        assert np.array_equal(
            restored.dp_attention_lane_token_counts,
            np.asarray([1, 1], dtype=np.int32),
        )
        assert np.array_equal(
            restored.dp_attention_lane_batch_sizes,
            np.asarray([1, 1], dtype=np.int32),
        )
        assert np.array_equal(
            restored.dp_attention_lane_token_offsets,
            np.asarray([0, 1, 2], dtype=np.int32),
        )
        assert np.array_equal(
            restored.dp_attention_lane_batch_offsets,
            np.asarray([0, 1, 2], dtype=np.int32),
        )
    finally:
        shm_bufs.close_and_unlink()


def test_dispatch_forward_step_dp_attention_contracts() -> None:
    runtime_config = _dp_attention_runtime_config()
    with _fake_coordinator(runtime_config=runtime_config) as coordinator:
        with pytest.raises(RuntimeError, match="requires a DP-attention superstep"):
            coordinator.dispatch_forward_step(_forward_batch_with(attention_lane=0))

    batch = _dp_superstep_batch()
    with _fake_coordinator(
        runtime_config=runtime_config,
        real_batch_shm=True,
    ) as coordinator:
        request_id = coordinator.dispatch_forward_step(batch)
        generation, cmd, payload = _cmd_block_read(coordinator._ctrl_shm.buf)
        metadata = _decode_forward_batch_metadata(payload)

        assert request_id in coordinator._forward_output_ranks_by_request
        assert generation == 2
        assert cmd == _CMD_FORWARD_STEP
        assert metadata.dp_attention_superstep == 1
        assert metadata.dp_attention_num_lanes == 2
        assert coordinator._active_forward_output_ranks == {0, 1, 2, 3}
        assert np.array_equal(
            coordinator._forward_dp_lane_batch_sizes_by_request[request_id],
            np.asarray([1, 1], dtype=np.int32),
        )


def test_write_batch_to_shm_rejects_unexpected_layout() -> None:
    runtime_config = _test_runtime_config()
    shm_bufs = _allocate_shared_buffers(runtime_config)
    try:
        batch = _sample_forward_batch()
        bad_batch = _forward_batch_with(positions=batch.positions.astype(np.int64))
        with pytest.raises(TypeError, match="Unexpected dtype for positions"):
            _write_batch_to_shm(bad_batch, shm_bufs)
    finally:
        shm_bufs.close_and_unlink()


def test_output_slot_read_contracts() -> None:
    runtime_config = _test_runtime_config(tp_degree=1, request_buckets=(1, 2, 8))
    ctrl_size, output_slot_size = _compute_ctrl_shm_layout(runtime_config)
    shm = SharedMemory(create=True, size=ctrl_size)
    try:
        shm.buf[:ctrl_size] = b"\x00" * ctrl_size
        _output_slot_write_top1(
            shm.buf,
            runtime_config.total_workers,
            0,
            output_slot_size,
            3,
            np.asarray([0.5, 0.7], dtype=np.float32),
            np.asarray([11, 12], dtype=np.int32),
            100,
        )

        current = _output_slot_read(
            shm.buf,
            runtime_config.total_workers,
            0,
            output_slot_size,
            3,
        )
        stale = _output_slot_read(
            shm.buf,
            runtime_config.total_workers,
            0,
            output_slot_size,
            4,
        )

        assert current["top1_values"].dtype == np.float32
        assert current["top1_indices"].dtype == np.int32
        assert int(current["vocab_offset"][0]) == 100
        assert stale == {}
    finally:
        shm.close()
        shm.unlink()

    runtime_config = _test_runtime_config(
        tp_degree=1,
        request_buckets=(1, 2, 8),
        dense_local_topk=3,
    )
    ctrl_size, output_slot_size = _compute_ctrl_shm_layout(runtime_config)
    shm = SharedMemory(create=True, size=ctrl_size)
    try:
        shm.buf[:ctrl_size] = b"\x00" * ctrl_size
        _output_slot_write_topk(
            shm.buf,
            runtime_config.total_workers,
            0,
            output_slot_size,
            21,
            np.asarray([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32),
            np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            256,
        )

        out = _output_slot_read(
            shm.buf,
            runtime_config.total_workers,
            0,
            output_slot_size,
            21,
        )

        assert np.array_equal(
            out["topk_indices"],
            np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        )
        assert np.array_equal(
            out["topk_values"],
            np.asarray([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32),
        )
        assert int(out["vocab_offset"][0]) == 256
    finally:
        shm.close()
        shm.unlink()


def test_collect_forward_step_combines_topk_outputs_from_shm() -> None:
    runtime_config = _test_runtime_config(
        tp_degree=2,
        request_buckets=(1, 2, 4),
        dense_local_topk=2,
    )
    with _fake_coordinator(runtime_config=runtime_config, generation=17) as coordinator:
        _output_slot_write_topk(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            0,
            coordinator._ctrl_output_slot_size,
            17,
            np.asarray([[0.7, 0.4], [0.6, 0.2]], dtype=np.float32),
            np.asarray([[2, 1], [3, 1]], dtype=np.int32),
            0,
        )
        _output_slot_write_topk(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            1,
            coordinator._ctrl_output_slot_size,
            17,
            np.asarray([[0.9, 0.1], [0.3, 0.8]], dtype=np.float32),
            np.asarray([[4, 5], [6, 7]], dtype=np.int32),
            100,
        )
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 0, 17, _STATUS_OK)
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 1, 17, _STATUS_OK)

        coordinator.collect_forward_step("req-17")

        assert np.array_equal(
            coordinator.last_forward_output["next_token_ids"],
            np.asarray([104, 107], dtype=np.int32),
        )


def test_collect_forward_step_reads_only_active_attention_lane() -> None:
    runtime_config = _dp_attention_runtime_config()
    with _fake_coordinator(runtime_config=runtime_config, generation=23) as coordinator:
        lane_outputs = coordinator._output_ranks_for_attention_lane(1)
        coordinator._active_forward_output_ranks = set(lane_outputs)
        coordinator._forward_output_ranks_by_request["req-lane-1"] = set(lane_outputs)

        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            2,
            coordinator._ctrl_output_slot_size,
            23,
            np.asarray([0.2, 0.9], dtype=np.float32),
            np.asarray([7, 8], dtype=np.int32),
            0,
        )
        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            3,
            coordinator._ctrl_output_slot_size,
            23,
            np.asarray([0.8, 0.1], dtype=np.float32),
            np.asarray([5, 6], dtype=np.int32),
            100,
        )
        for rank in range(runtime_config.total_workers):
            _worker_slot_write_done(coordinator._ctrl_shm.buf, rank, 23, _STATUS_OK)

        coordinator.collect_forward_step("req-lane-1")

        assert np.array_equal(
            coordinator.last_forward_output["next_token_ids"],
            np.asarray([105, 8], dtype=np.int32),
        )


def test_collect_forward_step_combines_dp_attention_superstep_outputs() -> None:
    runtime_config = _dp_attention_runtime_config(lanes=3)
    with _fake_coordinator(runtime_config=runtime_config, generation=31) as coordinator:
        lane_sizes = np.asarray([1, 0, 2], dtype=np.int32)
        expected_ranks = coordinator._output_ranks_for_dp_attention_superstep(
            lane_sizes
        )
        coordinator._active_forward_output_ranks = set(expected_ranks)
        coordinator._forward_output_ranks_by_request["req-dp"] = set(expected_ranks)
        coordinator._forward_dp_lane_batch_sizes_by_request["req-dp"] = lane_sizes

        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            0,
            coordinator._ctrl_output_slot_size,
            31,
            np.asarray([0.9], dtype=np.float32),
            np.asarray([10], dtype=np.int32),
            0,
        )
        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            1,
            coordinator._ctrl_output_slot_size,
            31,
            np.asarray([0.1], dtype=np.float32),
            np.asarray([1], dtype=np.int32),
            100,
        )
        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            4,
            coordinator._ctrl_output_slot_size,
            31,
            np.asarray([0.2, 0.8], dtype=np.float32),
            np.asarray([2, 3], dtype=np.int32),
            0,
        )
        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            5,
            coordinator._ctrl_output_slot_size,
            31,
            np.asarray([0.5, 0.1], dtype=np.float32),
            np.asarray([4, 5], dtype=np.int32),
            100,
        )
        for rank in range(runtime_config.total_workers):
            _worker_slot_write_done(coordinator._ctrl_shm.buf, rank, 31, _STATUS_OK)

        coordinator.collect_forward_step("req-dp")

        assert np.array_equal(
            coordinator.last_forward_output["next_token_ids"],
            np.asarray([10, 104, 3], dtype=np.int32),
        )


def test_collect_forward_step_failure_contracts() -> None:
    with _fake_coordinator(generation=5) as coordinator:
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 0, 5, _STATUS_OK)
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 1, 5, _STATUS_OK)

        with pytest.raises(RuntimeError, match="No SHM outputs published"):
            coordinator.collect_forward_step("req-5")

    with _fake_coordinator(generation=9) as coordinator:
        _output_slot_write_top1(
            coordinator._ctrl_shm.buf,
            coordinator._total_workers,
            0,
            coordinator._ctrl_output_slot_size,
            9,
            np.asarray([1.0], dtype=np.float32),
            np.asarray([7], dtype=np.int32),
            0,
        )
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 0, 9, _STATUS_OK)
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 1, 9, _STATUS_OK)

        with pytest.raises(RuntimeError, match="Missing SHM outputs"):
            coordinator.collect_forward_step("req-9")

    with _fake_coordinator(generation=11) as coordinator:
        _worker_slot_write_done(coordinator._ctrl_shm.buf, 0, 11, _STATUS_OK)
        _worker_slot_write_done(
            coordinator._ctrl_shm.buf,
            1,
            11,
            _STATUS_ERROR,
            "mock failure",
        )

        with pytest.raises(RuntimeError, match="mock failure"):
            coordinator.collect_forward_step("req-11")
