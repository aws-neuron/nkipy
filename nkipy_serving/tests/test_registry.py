"""Contract tests for model registry routing."""

import pytest

from nkipy_serving.models.registry import (
    _is_qwen3_moe_model,
    get_model_config_defaults,
    resolve_model_spec,
)


def test_model_registry_qwen_routing_contracts():
    moe_spec = resolve_model_spec("qwen3-moe")
    dense_spec = resolve_model_spec("Qwen/Qwen3-0.6B")

    moe_detection_cases = [
        ("Qwen/Qwen3-30B-A3B-Thinking-2507", True),
        ("Qwen/Qwen3-235B-A22B", True),
        ("Qwen/Qwen3-30B-A3B", True),
        ("Qwen/Qwen3-32B", False),
        ("Qwen/Qwen3-8B", False),
        ("Qwen/Qwen3-0.6B", False),
        ("Qwen/Qwen3-4B-Thinking-2507", False),
    ]
    for model_id, expected in moe_detection_cases:
        assert _is_qwen3_moe_model(model_id) is expected

    routing_cases = [
        ("qwen3-moe", moe_spec),
        ("Qwen/Qwen3-30B-A3B-Thinking-2507", moe_spec),
        ("Qwen/Qwen3-32B", dense_spec),
        ("Qwen/Qwen3-4B-Thinking-2507", dense_spec),
    ]
    for model_id, expected_spec in routing_cases:
        assert resolve_model_spec(model_id) is expected_spec

    assert (
        get_model_config_defaults("Qwen/Qwen3-30B-A3B-Thinking-2507")[
            "attention_backend"
        ]
        == "NKIBlockSparseFlashAttention"
    )
    assert "model_graph_mode" not in get_model_config_defaults("Qwen/Qwen3-32B")


def test_model_registry_gpt_oss_routing_contracts():
    gpt_oss_spec = resolve_model_spec("gpt-oss")

    assert resolve_model_spec("unsloth/gpt-oss-120b-BF16") is gpt_oss_spec
    defaults = get_model_config_defaults("gpt-oss")
    assert defaults["execution_backend"] == "nkipy"
    assert "model_graph_mode" not in defaults


def test_resolve_unsupported_model_raises():
    assert get_model_config_defaults("unknown/model-123") == {}
    with pytest.raises(RuntimeError, match="Unsupported model_id"):
        resolve_model_spec("meta-llama/Llama-3.1-8B")


def test_dense_model_rejects_ep():
    from unittest.mock import MagicMock

    spec = resolve_model_spec("Qwen/Qwen3-0.6B")
    rc = MagicMock()
    rc.ep_degree = 2
    rc.model_id = "Qwen/Qwen3-0.6B"
    with pytest.raises(RuntimeError, match="ep_degree > 1 is only supported for MoE"):
        spec.build_config(rc, tp_rank=0, ep_rank=0)
