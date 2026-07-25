from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nkipy_serving.attention.base import FORWARD_MODE_DECODE, FORWARD_MODE_EXTEND
from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.model_executor.model_runner import ModelRunner
from nkipy_serving.runtime.precompile_paddings import PrecompilePaddings
from nkipy_serving.runtime.warmup import (
    SyntheticWarmupStep,
    build_standard_warmup_steps,
    build_synthetic_warmup_inputs,
)


def _sample_batch(
    *,
    sample_mask: np.ndarray,
    attention_lane: int = -1,
) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=3,
        input_ids=np.asarray([10, 11, 12, 13, 14], dtype=np.int32),
        positions=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        seq_lens=np.asarray([2, 3, 5], dtype=np.int64),
        slot_mapping=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        block_tables=np.asarray([[1, 0], [2, 0], [3, 4]], dtype=np.int64),
        query_start_loc=np.asarray([0, 2, 3, 5], dtype=np.int64),
        sample_mask=sample_mask,
        token_bucket=8,
        real_total_tokens=5,
        attention_lane=attention_lane,
    )


def test_model_runner_delegates_request_state_checkpoint_restore() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Executor:
        def checkpoint_request_state(self, **kwargs):
            calls.append(("checkpoint", kwargs))

        def restore_request_state(self, checkpoint_id):
            calls.append(("restore", {"checkpoint_id": checkpoint_id}))

    runner = ModelRunner.__new__(ModelRunner)
    runner._executor = _Executor()

    runner.checkpoint_request_state(
        checkpoint_id="cp-runner",
        owner_id=2,
        seq_len=7,
        num_tokens=3,
    )
    runner.restore_request_state("cp-runner")

    assert calls == [
        (
            "checkpoint",
            {
                "checkpoint_id": "cp-runner",
                "owner_id": 2,
                "seq_len": 7,
                "num_tokens": 3,
            },
        ),
        ("restore", {"checkpoint_id": "cp-runner"}),
    ]


def test_model_runner_forward_output_contracts() -> None:
    dense_logits = np.zeros((5, 8), dtype=np.float32)
    dense_logits[1, 6] = 1.0
    dense_logits[2, 3] = 2.0
    cases = [
        (
            "nkipy",
            _sample_batch(sample_mask=np.asarray([True, True, False], dtype=np.bool_)),
            dense_logits,
            np.asarray([6, 3, 0], dtype=np.int32),
            np.asarray([1.0, 2.0, -np.inf], dtype=np.float32),
            64,
        ),
        (
            "numpy",
            _sample_batch(sample_mask=np.asarray([True, True, True], dtype=np.bool_)),
            np.arange(40, dtype=np.float32).reshape(5, 8),
            np.asarray([7, 7, 7], dtype=np.int32),
            None,
            0,
        ),
    ]

    for (
        backend,
        batch,
        logits,
        expected_indices,
        expected_values,
        vocab_offset,
    ) in cases:
        runner = ModelRunner.__new__(ModelRunner)
        runner._runtime_config = SimpleNamespace(execution_backend=backend)
        runner._executor = SimpleNamespace(
            weights=SimpleNamespace(
                num_kv_heads=1,
                head_dim=1,
                num_hidden_layers=1,
                lm_head_vocab_offset=vocab_offset,
            ),
            kv_pool=SimpleNamespace(
                block_size=16,
                get_kv_cache=lambda layer_id: np.zeros((1,), dtype=np.float32),
            ),
            forward=lambda *args, **kwargs: logits,
        )

        out = runner.forward(batch)

        assert isinstance(out, dict)
        assert np.array_equal(out["top1_indices"], expected_indices)
        if expected_values is not None:
            assert np.array_equal(out["top1_values"], expected_values)
        assert np.array_equal(
            out["vocab_offset"], np.asarray([vocab_offset], dtype=np.int32)
        )

    for payload in [
        {
            "top1_values": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "top1_indices": np.asarray([4, 5, 6], dtype=np.int32),
            "vocab_offset": np.asarray([32], dtype=np.int32),
        },
        {
            "topk_values": np.asarray(
                [[3.0, 2.0], [2.5, 1.5], [1.0, 0.5]], dtype=np.float32
            ),
            "topk_indices": np.asarray([[4, 7], [5, 8], [6, 9]], dtype=np.int32),
            "vocab_offset": np.asarray([32], dtype=np.int32),
        },
    ]:
        runner = ModelRunner.__new__(ModelRunner)
        runner._runtime_config = SimpleNamespace(execution_backend="nkipy")
        runner._executor = SimpleNamespace(
            weights=SimpleNamespace(num_kv_heads=1, head_dim=1, num_hidden_layers=1),
            kv_pool=SimpleNamespace(
                block_size=16,
                get_kv_cache=lambda layer_id: np.zeros((1,), dtype=np.float32),
            ),
            forward=lambda *args, **kwargs: payload,
        )

        assert (
            runner.forward(
                _sample_batch(
                    sample_mask=np.asarray([True, True, True], dtype=np.bool_)
                )
            )
            is payload
        )


def test_model_runner_forwards_attention_lane() -> None:
    calls: list[dict[str, object]] = []
    payload = {
        "top1_values": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "top1_indices": np.asarray([4, 5, 6], dtype=np.int32),
    }

    def _forward(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        calls.append(kwargs)
        return payload

    runner = ModelRunner.__new__(ModelRunner)
    runner._runtime_config = SimpleNamespace(execution_backend="nkipy")
    runner._executor = SimpleNamespace(
        weights=SimpleNamespace(num_kv_heads=1, head_dim=1, num_hidden_layers=1),
        kv_pool=SimpleNamespace(
            block_size=16,
            get_kv_cache=lambda layer_id: np.zeros((1,), dtype=np.float32),
        ),
        forward=_forward,
    )

    assert (
        runner.forward(
            _sample_batch(
                sample_mask=np.asarray([True, True, True], dtype=np.bool_),
                attention_lane=3,
            )
        )
        is payload
    )
    assert calls[0]["attention_lane"] == 3


def test_build_synthetic_warmup_inputs_match_scheduler_shapes() -> None:
    decode_input_ids, decode_positions, decode_metadata = build_synthetic_warmup_inputs(
        SyntheticWarmupStep(
            name="decode_t8_b8",
            forward_mode=FORWARD_MODE_DECODE,
            input_token_bucket=8,
            batch_size=8,
        ),
        token_paddings=(128,),
        bs_paddings=(2, 8),
        num_blocks=64,
        block_size=32,
        num_kv_heads=1,
        head_dim=64,
    )

    assert decode_input_ids.shape == (8,)
    assert decode_positions.shape == (8,)
    assert decode_metadata.total_tokens == 8
    assert decode_metadata.batch_size == 8
    assert np.array_equal(decode_metadata.seq_lens, np.ones((8,), dtype=np.int64))
    assert np.array_equal(decode_positions[:8], np.zeros((8,), dtype=np.int32))
    assert np.array_equal(decode_metadata.query_start_loc, np.arange(9, dtype=np.int64))
    assert decode_metadata.block_tables.shape == (8, 1)

    extend_input_ids, extend_positions, extend_metadata = build_synthetic_warmup_inputs(
        SyntheticWarmupStep(
            name="extend_real6_t8_b2",
            forward_mode=FORWARD_MODE_EXTEND,
            input_token_bucket=8,
            batch_size=2,
            real_total_tokens=6,
        ),
        token_paddings=(8,),
        bs_paddings=(2,),
        num_blocks=64,
        block_size=32,
        num_kv_heads=1,
        head_dim=64,
    )

    assert extend_input_ids.shape == (8,)
    assert extend_positions.shape == (8,)
    assert extend_metadata.total_tokens == 6
    assert extend_metadata.batch_size == 2
    assert np.array_equal(extend_metadata.seq_lens, np.asarray([3, 3], dtype=np.int64))
    assert np.array_equal(extend_metadata.query_start_loc, np.asarray([0, 3, 6]))


def test_build_synthetic_warmup_inputs_rejects_kv_pool_overflow() -> None:
    with pytest.raises(RuntimeError, match="more KV cache blocks than available"):
        build_synthetic_warmup_inputs(
            SyntheticWarmupStep(
                name="extend_real65_t128_b1",
                forward_mode=FORWARD_MODE_EXTEND,
                input_token_bucket=128,
                batch_size=1,
                real_total_tokens=65,
            ),
            token_paddings=(128,),
            bs_paddings=(1,),
            num_blocks=2,
            block_size=32,
            num_kv_heads=1,
            head_dim=64,
        )


def test_standard_warmup_decode_keeps_padded_bucket_for_single_request() -> None:
    steps = build_standard_warmup_steps(
        PrecompilePaddings(
            token_paddings=(8,),
            bs_paddings=(2,),
            max_padded_num_tokens=8,
            max_padded_batch_size=1,
        )
    )

    assert [(s.name, s.input_token_bucket, s.batch_size) for s in steps] == [
        ("extend_t8_b1", 8, 1),
        ("decode_t2_b1", 2, 1),
    ]

    input_ids, positions, metadata = build_synthetic_warmup_inputs(
        steps[-1],
        token_paddings=(8,),
        bs_paddings=(2,),
        num_blocks=64,
        block_size=32,
        num_kv_heads=1,
        head_dim=64,
    )

    assert input_ids.shape == (2,)
    assert positions.shape == (2,)
    assert metadata.total_tokens == 1
    assert metadata.batch_size == 1
    assert metadata.seq_lens.shape == (1,)
    assert metadata.query_start_loc.shape == (2,)
