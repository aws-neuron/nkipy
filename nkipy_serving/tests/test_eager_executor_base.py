"""CPU-only tests for shared eager executor base helpers."""

from __future__ import annotations

import pytest

from nkipy_serving.models.common.eager_executor_base import EagerExecutorBase


def test_require_single_rank_passes_on_dense_shape():
    EagerExecutorBase._require_single_rank_for_forward_cpu(tp_degree=1)


def test_require_single_rank_passes_on_moe_shape():
    EagerExecutorBase._require_single_rank_for_forward_cpu(tp_degree=1, ep_degree=1)


def test_require_single_rank_raises_on_tp_sharded():
    with pytest.raises(ValueError, match="tp_degree == 1"):
        EagerExecutorBase._require_single_rank_for_forward_cpu(tp_degree=2)


def test_require_single_rank_raises_on_ep_sharded():
    with pytest.raises(ValueError, match="ep_degree == 1"):
        EagerExecutorBase._require_single_rank_for_forward_cpu(
            tp_degree=1,
            ep_degree=2,
        )


def test_require_single_rank_raises_on_both_sharded():
    with pytest.raises(ValueError, match="tp_degree"):
        EagerExecutorBase._require_single_rank_for_forward_cpu(
            tp_degree=4,
            ep_degree=2,
        )


def test_require_single_rank_mentions_rank_local_risk():
    """Error text must warn about silent partial output, not just shape."""
    with pytest.raises(ValueError, match="rank-local partial output"):
        EagerExecutorBase._require_single_rank_for_forward_cpu(tp_degree=2)
