from __future__ import annotations

import numpy as np
import pytest

from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.runtime.shape_guard import validate_forward_batch_shape


def _sample_batch(
    *, forward_mode: ForwardMode, token_bucket: int, real_total_tokens: int
) -> ForwardBatch:
    batch_size = 2 if forward_mode == ForwardMode.DECODE else 1
    return ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=np.zeros((token_bucket,), dtype=np.int32),
        positions=np.zeros((token_bucket,), dtype=np.int32),
        seq_lens=np.ones((batch_size,), dtype=np.int64),
        slot_mapping=np.zeros((token_bucket,), dtype=np.int64),
        block_tables=np.zeros((batch_size, 1), dtype=np.int64),
        query_start_loc=np.arange(batch_size + 1, dtype=np.int64),
        sample_mask=np.ones((batch_size,), dtype=np.bool_),
        requested_topk=1,
        token_bucket=token_bucket,
        real_total_tokens=real_total_tokens,
    )


def test_validate_forward_batch_shape_accepts_configured_buckets() -> None:
    cases = [
        (ForwardMode.EXTEND, 128, 64),
        (ForwardMode.DECODE, 128, 2),
        (ForwardMode.DECODE, 8, 2),
    ]
    for forward_mode, token_bucket, real_total_tokens in cases:
        batch = _sample_batch(
            forward_mode=forward_mode,
            token_bucket=token_bucket,
            real_total_tokens=real_total_tokens,
        )
        validate_forward_batch_shape(batch, (32, 128), (2, 4, 8))


def test_validate_forward_batch_shape_rejects_unconfigured_bucket() -> None:
    batch = _sample_batch(
        forward_mode=ForwardMode.DECODE,
        token_bucket=16,
        real_total_tokens=2,
    )
    with pytest.raises(RuntimeError, match="allowed="):
        validate_forward_batch_shape(batch, (32, 128), (2, 4, 8))
