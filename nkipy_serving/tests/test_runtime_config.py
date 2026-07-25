import json
from pathlib import Path

import pytest

import nkipy_serving.config as config_mod
from nkipy_serving.attention.base import FORWARD_MODE_EXTEND
from nkipy_serving.config import (
    RuntimeConfig,
    configure_runtime_environment,
    load_runtime_config,
    validate_runtime_config,
)
from nkipy_serving.models.deepseek_v4.assembly.warmup_plan import (
    build_dsv4_warmup_plan,
)
from nkipy_serving.runtime.precompile_paddings import (
    PrecompilePaddings,
)


def test_runtime_config_valid_runtime_contracts():
    configs = [
        RuntimeConfig(),
        RuntimeConfig(
            execution_backend="numpy",
            attention_backend="VanillaPagedAttention",
            paged_attn_impl="vanilla_paged_attention_kv_cache",
        ),
        RuntimeConfig(execution_backend="nkipy", tp_degree=2),
        RuntimeConfig(execution_backend="nkipy", tp_degree=8, ep_degree=4),
    ]

    for cfg in configs:
        validate_runtime_config(cfg)


def test_runtime_config_invalid_runtime_contracts():
    cases = [
        (
            RuntimeConfig(attention_backend="something_else"),
            "Invalid attention backend",
        ),
        (RuntimeConfig(execution_backend="not_real"), "Invalid execution backend"),
        (
            RuntimeConfig(tokenizer_backend="not_real_tokenizer"),
            "Invalid tokenizer backend",
        ),
        (RuntimeConfig(dense_local_topk=0), "dense_local_topk must be > 0"),
        (
            RuntimeConfig(execution_backend="numpy", tp_degree=2),
            "tp_degree > 1 is only supported",
        ),
        (
            RuntimeConfig(enable_mixed_chunk=True, chunked_prefill_size=-1),
            "enable_mixed_chunk requires",
        ),
        (RuntimeConfig(kv_pool_size=0), "kv_pool_size must be > 0"),
        (
            RuntimeConfig(dsv4_warmup_execute_forwards=False),
            "first-touch forwards before readiness",
        ),
        (
            RuntimeConfig(dsv4_prepared_weight_prestage_workers=0),
            "dsv4_prepared_weight_prestage_workers must be > 0",
        ),
        (
            RuntimeConfig(
                execution_backend="numpy",
                attention_backend="VanillaPagedAttention",
                paged_attn_impl="vanilla_paged_attention_kv_cache",
                ep_degree=2,
            ),
            "ep_degree > 1 is only supported",
        ),
    ]

    for cfg, match in cases:
        with pytest.raises(RuntimeError, match=match):
            validate_runtime_config(cfg)


def test_runtime_config_load_from_json(tmp_path):
    config_file = tmp_path / "runtime.json"
    config_file.write_text(
        (
            "{"
            '"request_buckets": [1, 4, 8], '
            '"token_buckets": [32, 128], '
            '"dsv4_prepared_weight_prestage": true, '
            '"nkipy_build_dir": "/tmp/custom_build_root", '
            '"tokenizer_backend": "hf"'
            "}"
        ),
        encoding="utf-8",
    )
    cfg = load_runtime_config(config_path=str(config_file))
    assert cfg.request_buckets == (1, 4, 8)
    assert cfg.token_buckets == (32, 128)
    assert cfg.dsv4_prepared_weight_prestage is True
    assert cfg.nkipy_build_dir == "/tmp/custom_build_root"


def test_dsv4_warmup_plan_uses_bucket_shapes_without_live_prefill_counts() -> None:
    plan = build_dsv4_warmup_plan(
        PrecompilePaddings(
            token_paddings=(256, 1024),
            bs_paddings=(1, 16),
            max_padded_num_tokens=1024,
            max_padded_batch_size=16,
        ),
        product_warmup_enabled=True,
        has_compressed_layers=False,
    )

    extend_steps = {
        (
            int(step.forward_mode),
            int(step.input_token_bucket),
            int(step.batch_size),
            step.real_total_tokens,
        )
        for step in plan.steps
    }
    assert (int(FORWARD_MODE_EXTEND), 256, 1, None) in extend_steps
    assert (int(FORWARD_MODE_EXTEND), 256, 16, None) in extend_steps
    assert (int(FORWARD_MODE_EXTEND), 1024, 1, None) in extend_steps
    assert (int(FORWARD_MODE_EXTEND), 1024, 16, None) in extend_steps
    assert not any(
        int(step.forward_mode) == int(FORWARD_MODE_EXTEND)
        and step.real_total_tokens is not None
        for step in plan.steps
    )
    assert plan.state_write_buckets == (16, 32, 64)
    assert plan.decode_write_buckets == (1, 16)


def test_runtime_config_dsv4_prepared_weight_dir_contracts(tmp_path, monkeypatch):
    monkeypatch.delenv("NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR", raising=False)
    monkeypatch.delenv("NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR", raising=False)

    source = tmp_path / "prepared"
    source.mkdir()
    local = tmp_path / "local"
    config_file = tmp_path / "runtime_prepared.json"
    config_file.write_text(
        json.dumps(
            {
                "dsv4_prepared_weight_dir": str(source),
                "dsv4_prepared_weight_local_dir": str(local),
            }
        ),
        encoding="utf-8",
    )

    cfg = load_runtime_config(config_path=str(config_file))

    assert cfg.dsv4_prepared_weight_dir == str(source)
    assert cfg.dsv4_prepared_weight_local_dir == str(local)
    validate_runtime_config(cfg)

    monkeypatch.setenv("NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR", str(source))
    cfg = load_runtime_config()
    assert cfg.dsv4_prepared_weight_dir == str(source)
    validate_runtime_config(cfg)
    monkeypatch.delenv("NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR")

    invalid_cases = [
        (
            RuntimeConfig(dsv4_prepared_weight_local_dir="/tmp/dsv4-local"),
            "requires dsv4_prepared_weight_dir",
        ),
        (
            RuntimeConfig(dsv4_prepared_weight_dir=str(tmp_path / "missing")),
            "dsv4_prepared_weight_dir does not exist",
        ),
        (
            RuntimeConfig(
                dsv4_prepared_weight_dir=str(source),
                dsv4_prepared_weight_prestage=True,
            ),
            "dsv4_prepared_weight_prestage requires both",
        ),
    ]
    for invalid_cfg, match in invalid_cases:
        with pytest.raises(RuntimeError, match=match):
            validate_runtime_config(invalid_cfg)


def test_runtime_config_env_overrides(monkeypatch):
    cases = [
        ("NKIPY_SERVING_REQUEST_BUCKETS", "1,2,4", "request_buckets", (1, 2, 4)),
        ("NKIPY_SERVING_TP_DEGREE", "8", "tp_degree", 8),
        (
            "NKIPY_SERVING_BUILD_DIR",
            "/tmp/env_build_root",
            "nkipy_build_dir",
            "/tmp/env_build_root",
        ),
        ("NKIPY_SERVING_HF_LOCAL_FILES_ONLY", "0", "hf_local_files_only", False),
        (
            "NKIPY_SERVING_DSV4_PREPARED_WEIGHT_PRESTAGE",
            "1",
            "dsv4_prepared_weight_prestage",
            True,
        ),
        ("NKIPY_SERVING_DSV4_STATE_SIZE", "2048", "dsv4_state_size", 2048),
        ("NKIPY_SERVING_EP_DEGREE", "16", "ep_degree", 16),
    ]
    for env_name, raw_value, field_name, expected in cases:
        monkeypatch.setenv(env_name, raw_value)
        cfg = load_runtime_config()
        assert getattr(cfg, field_name) == expected
        monkeypatch.delenv(env_name)


def test_runtime_config_infers_tokenizer_model_id_from_hf_model_id(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("NKIPY_SERVING_TOKENIZER_MODEL_ID", raising=False)
    monkeypatch.delenv("NKIPY_SERVING_MODEL_ID", raising=False)
    config_file = tmp_path / "runtime_hf_model.json"
    config_file.write_text('{"model_id": "Qwen/Qwen3-0.6B"}', encoding="utf-8")
    cfg = load_runtime_config(config_path=str(config_file))
    assert cfg.model_id == "Qwen/Qwen3-0.6B"
    assert cfg.tokenizer_model_id == "Qwen/Qwen3-0.6B"


def test_runtime_config_fail_fast_on_unsupported_hf_model_id():
    cfg = RuntimeConfig(hf_model_id="meta-llama/Llama-3.1-8B-Instruct")
    with pytest.raises(
        RuntimeError, match="restricted to Qwen3, GPT-OSS, or DeepSeek-V4"
    ):
        validate_runtime_config(cfg)


def test_runtime_config_allows_supported_hf_model_overrides(tmp_path):
    supported = [
        RuntimeConfig(model_id="gpt-oss", hf_model_id="unsloth/gpt-oss-120b-BF16"),
        RuntimeConfig(model_id="gpt-oss", hf_model_id=str(tmp_path)),
        RuntimeConfig(model_id="qwen3-moe", hf_model_id=str(tmp_path)),
    ]
    for cfg in supported:
        validate_runtime_config(cfg)


def test_runtime_config_deepseek_v4_attention_contracts(tmp_path):
    valid_sources = ["deepseek-ai/DeepSeek-V4-Flash", str(tmp_path)]
    for source in valid_sources:
        validate_runtime_config(
            RuntimeConfig(
                model_id="deepseek-ai/DeepSeek-V4-Flash",
                hf_model_id=source,
                attention_backend="Dsv4SparseAttention",
                paged_attn_impl="dsv4_sparse_attention",
                dsv4_disable_mtp=True,
                dsv4_state_size=4096,
            )
        )

    invalid_cases = [
        (
            RuntimeConfig(
                model_id="deepseek-ai/DeepSeek-V4-Flash",
                hf_model_id="deepseek-ai/DeepSeek-V4-Flash",
                attention_backend="NKIBlockSparseFlashAttention",
                paged_attn_impl="nki_blocksparse_flash_attention",
                dsv4_disable_mtp=True,
            ),
            "DeepSeek-V4 requires attention_backend",
        ),
        (
            RuntimeConfig(
                model_id="deepseek-ai/DeepSeek-V4-Flash",
                attention_backend="Dsv4SparseAttention",
                paged_attn_impl="dsv4_sparse_attention",
                dsv4_disable_mtp=False,
            ),
            "DeepSeek-V4 serving requires dsv4_disable_mtp=true",
        ),
        (
            RuntimeConfig(
                model_id="Qwen/Qwen3-0.6B",
                attention_backend="Dsv4SparseAttention",
                paged_attn_impl="dsv4_sparse_attention",
            ),
            "only supported for DeepSeek-V4",
        ),
    ]
    for cfg, match in invalid_cases:
        with pytest.raises(RuntimeError, match=match):
            validate_runtime_config(cfg)


def test_runtime_config_deepseek_v4_state_size_contracts():
    base = dict(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        attention_backend="Dsv4SparseAttention",
        paged_attn_impl="dsv4_sparse_attention",
        dsv4_disable_mtp=True,
    )

    invalid_cases = [
        (
            RuntimeConfig(**base),
            "DeepSeek-V4 serving requires dsv4_state_size > 0",
        ),
        (
            RuntimeConfig(**base, dsv4_state_size=-1),
            "dsv4_state_size must be >= 0",
        ),
        (
            RuntimeConfig(
                **base,
                max_context_len=4096,
                token_buckets=(128,),
                dsv4_state_size=2048,
            ),
            "dsv4_state_size must cover max_context_len",
        ),
        (
            RuntimeConfig(
                **base,
                max_context_len=128,
                token_buckets=(256,),
                dsv4_state_size=128,
            ),
            "dsv4_state_size must cover the largest token bucket",
        ),
        (
            RuntimeConfig(
                **base,
                max_context_len=128,
                token_buckets=(128,),
                dsv4_state_size=130,
            ),
            "dsv4_state_size must be divisible by 128",
        ),
    ]
    for cfg, match in invalid_cases:
        with pytest.raises(RuntimeError, match=match):
            validate_runtime_config(cfg)

    validate_runtime_config(
        RuntimeConfig(
            **base,
            max_context_len=128,
            token_buckets=(128,),
            dsv4_state_size=128,
        )
    )


def test_runtime_config_deepseek_v4_r1_4k_bucket_config() -> None:
    config_path = (
        Path(__file__).resolve().parent
        / "runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json"
    )
    cfg = load_runtime_config(str(config_path))

    assert cfg.tp_degree == 8
    assert cfg.ep_degree == 8
    assert cfg.replica_degree == 1
    assert cfg.attention_dp_degree == 8
    assert cfg.max_context_len == 4096
    assert cfg.kv_pool_size == 4096
    assert cfg.dsv4_state_size == 4096
    assert cfg.request_buckets == (1, 8)
    assert cfg.token_buckets == (256, 1024, 2048, 4096)
    validate_runtime_config(cfg)


def test_runtime_config_fail_fast_on_unknown_config_fields(tmp_path):
    cases = [
        {"tp_rank": 1},
        {"unknown_runtime_knob": True},
    ]
    for idx, fields in enumerate(cases):
        unknown = next(iter(fields))
        match = f"Unknown runtime config field\\(s\\): {unknown}"
        with pytest.raises(RuntimeError, match=match):
            load_runtime_config(overrides=fields)

        config_file = tmp_path / f"runtime_unknown_{idx}.json"
        config_file.write_text(json.dumps(fields), encoding="utf-8")
        with pytest.raises(RuntimeError, match=match):
            load_runtime_config(config_path=str(config_file))


def test_runtime_config_defaults_and_basic_overrides(monkeypatch):
    cfg = RuntimeConfig()
    assert cfg.chunked_prefill_size == 4096
    assert cfg.enable_mixed_chunk is False
    assert cfg.kv_pool_size == 16384
    assert cfg.max_requests == 32  # max(default request_buckets=(1,2,4,8,16,32))
    assert cfg.max_context_len == 4096
    assert cfg.request_timeout_s == 600
    assert cfg.dsv4_state_size == 0

    custom = RuntimeConfig(
        kv_pool_size=16384,
        request_buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        max_context_len=8192,
        request_timeout_s=300,
    )
    validate_runtime_config(custom)
    assert custom.kv_pool_size == 16384
    assert custom.max_requests == 128
    assert custom.max_context_len == 8192
    assert custom.request_timeout_s == 300

    cfg = load_runtime_config(
        overrides={
            "execution_backend": "nkipy",
            "tp_degree": 4,
        }
    )
    assert cfg.tp_degree == 4

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = load_runtime_config()
    assert cfg.tp_degree == 1


def test_configure_runtime_environment_contracts(monkeypatch):
    monkeypatch.delenv("LOG_NKI_KERNEL_CALL", raising=False)

    configure_runtime_environment(RuntimeConfig())

    assert config_mod.os.environ["LOG_NKI_KERNEL_CALL"] == "0"

    monkeypatch.setenv("LOG_NKI_KERNEL_CALL", "1")
    configure_runtime_environment(RuntimeConfig())
    assert config_mod.os.environ["LOG_NKI_KERNEL_CALL"] == "1"


# -- Config hash --


def test_runtime_config_hash_contracts():
    base = RuntimeConfig()
    assert base.compute_config_hash() == base.compute_config_hash()

    equal_pairs = [
        (
            RuntimeConfig(dsv4_prepared_weight_dir="/mnt/cache/a"),
            RuntimeConfig(dsv4_prepared_weight_dir="/mnt/cache/b"),
        ),
    ]
    for cfg_a, cfg_b in equal_pairs:
        assert cfg_a.compute_config_hash() == cfg_b.compute_config_hash()

    different_pairs = [
        (RuntimeConfig(max_context_len=4096), RuntimeConfig(max_context_len=8192)),
        (RuntimeConfig(tp_degree=1), RuntimeConfig(tp_degree=2)),
        (
            RuntimeConfig(model_id="Qwen/Qwen3-0.6B"),
            RuntimeConfig(model_id="gpt-oss"),
        ),
    ]
    for cfg_a, cfg_b in different_pairs:
        assert cfg_a.compute_config_hash() != cfg_b.compute_config_hash()


# -- Per-model config defaults --


def test_runtime_config_model_defaults():
    cases = [
        ({}, "Qwen/Qwen3-0.6B", True, True, None),
        ({"model_id": "gpt-oss"}, "gpt-oss", True, True, 4096),
        ({"model_id": "qwen3-moe"}, None, True, False, None),
        ({"model_id": "some-unknown-model"}, None, False, False, None),
    ]
    for (
        overrides,
        expected_model_id,
        expect_attention,
        expect_paged,
        max_context,
    ) in cases:
        cfg = (
            load_runtime_config(overrides=overrides)
            if overrides
            else load_runtime_config()
        )
        if expected_model_id is not None:
            assert cfg.model_id == expected_model_id
        assert cfg.execution_backend == "nkipy"
        if expect_attention:
            assert cfg.attention_backend == "NKIBlockSparseFlashAttention"
        if expect_paged:
            assert cfg.paged_attn_impl == "nki_blocksparse_flash_attention"
        if max_context is not None:
            assert cfg.max_context_len == max_context


def test_runtime_config_model_max_context_len_derivation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod,
        "_derive_model_default_max_context_len",
        lambda model_id, data: 40960,
    )
    cfg = load_runtime_config(overrides={"model_id": "Qwen/Qwen3-0.6B"})
    assert cfg.max_context_len == 40960

    cfg = load_runtime_config(
        overrides={"model_id": "Qwen/Qwen3-0.6B", "max_context_len": 8192}
    )
    assert cfg.max_context_len == 8192


# -- EP (expert parallelism) --


def test_runtime_config_ep_contracts(tmp_path):
    cfg = RuntimeConfig()
    assert cfg.ep_degree == 1
    assert cfg.total_workers == 1

    cfg = RuntimeConfig(execution_backend="nkipy", tp_degree=8, ep_degree=16)
    validate_runtime_config(cfg)
    assert cfg.total_workers == 128

    config_file = tmp_path / "ep_config.json"
    config_file.write_text('{"tp_degree": 8, "ep_degree": 4, "device_offset": 16}')
    cfg = load_runtime_config(config_path=str(config_file))
    assert cfg.tp_degree == 8
    assert cfg.ep_degree == 4
    assert cfg.device_offset == 16
    assert cfg.total_workers == 32
