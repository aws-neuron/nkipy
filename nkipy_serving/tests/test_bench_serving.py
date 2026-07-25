from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from nkipy_serving.ops.moe import blockwise_index


def _load_bench_serving():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_serving.py"
    spec = importlib.util.spec_from_file_location(
        "bench_serving_under_test", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bench_serving.py from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "_under_test"), script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name} from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeHFTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return "<chat-template>"


class _FakeTokenizer:
    def __init__(self) -> None:
        self._hf_tokenizer = _FakeHFTokenizer()

    def encode(self, text: str) -> np.ndarray:
        return np.arange(max(1, len(text.split())), dtype=np.int32)

    @property
    def tokenizer(self):
        return self._hf_tokenizer


class _TripleTokenWordTokenizer:
    def encode(self, text: str) -> np.ndarray:
        return np.arange(max(1, len(text.split()) * 3), dtype=np.int32)


def test_render_chat_prompt_uses_gpt_oss_chat_template() -> None:
    bench_serving = _load_bench_serving()
    tokenizer = _FakeTokenizer()
    messages = [{"role": "user", "content": "Explain MoE routing."}]

    prompt = bench_serving._render_chat_prompt(
        tokenizer=tokenizer,
        model="unsloth/gpt-oss-120b-BF16",
        messages=messages,
        extra_request_body={
            "reasoning_effort": "high",
            "chat_template_kwargs": {"foo": "bar"},
        },
    )

    assert prompt == "<chat-template>"
    assert len(tokenizer.tokenizer.calls) == 1
    call = tokenizer.tokenizer.calls[0]
    assert call["messages"] == messages
    assert call["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": "high",
        "foo": "bar",
    }


def test_summarize_dsv4_layer_profile_groups_runtime_rows(tmp_path: Path) -> None:
    summarizer = _load_script("summarize_dsv4_layer_profile.py")
    profile = tmp_path / "dsv4_product_runtime_rank_0.jsonl"
    rows = [
        {
            "stage": "run_product_kernel",
            "name": "kernel_a",
            "layer_graph_key": "layer0_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.3,
            "call_s": 0.2,
            "load_s": 0.1,
        },
        {
            "stage": "run_product_kernel",
            "name": "kernel_b",
            "layer_graph_key": "layer0_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.4,
            "call_s": 0.4,
            "load_s": 0.0,
        },
        {
            "stage": "run_product_kernel",
            "name": "kernel_b",
            "layer_graph_key": "layer1_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.5,
            "call_s": 0.5,
            "load_s": 0.0,
        },
        {
            "stage": "run_product_kernel",
            "name": "kernel_c",
            "layer_graph_key": "layer2_variant_b",
            "layer_variant_key": "variant_b",
            "elapsed_s": 0.1,
            "call_s": 0.1,
            "load_s": 0.0,
        },
    ]
    profile.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarizer.summarize_layer_profile([str(profile)], top=10)

    assert summary["runtime_rows"] == 4
    assert summary["sort_by"] == "call_s"
    assert summary["layer_keys"] == 3
    assert summary["variant_keys"] == 2
    assert summary["unique_kernel_names"] == 3
    assert summary["rows_without_layer_key"] == 0
    assert summary["top_variants"][0]["key"] == "variant_a"
    assert summary["top_variants"][0]["calls"] == 3
    assert summary["top_variants"][0]["unique_kernel_names"] == 2
    assert summary["top_variants"][0]["layer_keys"] == 2
    assert summary["top_variants"][0]["call_s"] == 1.1
    assert summary["top_variants"][0]["call_avg_s"] == pytest.approx(1.1 / 3)
    assert summary["top_variants"][0]["call_min_s"] == 0.2
    assert summary["top_variants"][0]["call_p50_s"] == 0.4
    assert summary["top_variants"][0]["call_p95_s"] == 0.5
    assert summary["top_variants"][0]["call_max_s"] == 0.5
    assert summary["top_variants"][0]["call_first_s"] == 0.2
    assert summary["top_variants"][0]["call_steady_count"] == 2
    assert summary["top_variants"][0]["call_steady_s"] == 0.9
    assert summary["top_variants"][0]["call_steady_avg_s"] == 0.45
    assert summary["top_kernel_names"][0]["key"] == "kernel_b"
    assert summary["top_kernel_names"][0]["call_first_s"] == 0.4
    assert summary["top_kernel_names"][0]["call_steady_avg_s"] == 0.5

    steady_summary = summarizer.summarize_layer_profile(
        [str(profile)],
        top=10,
        sort_by="call_steady_avg_s",
    )

    assert steady_summary["sort_by"] == "call_steady_avg_s"
    assert steady_summary["top_kernel_names"][0]["key"] == "kernel_b"


def test_summarize_dsv4_layer_profile_groups_forward_rows(tmp_path: Path) -> None:
    summarizer = _load_script("summarize_dsv4_layer_profile.py")
    profile = tmp_path / "dsv4_product_forward_rank_0.jsonl"
    rows = [
        {
            "stage": "attention_replicated_postprocess",
            "layer_graph_key": "layer0_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.2,
        },
        {
            "stage": "attention_dp_all_reduce_post_pre",
            "layer_graph_key": "layer0_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.5,
        },
        {
            "stage": "attention",
            "layer_graph_key": "layer1_variant_a",
            "layer_variant_key": "variant_a",
            "elapsed_s": 0.1,
        },
        {
            "stage": "prepare_inputs",
            "elapsed_s": 0.9,
        },
    ]
    profile.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarizer.summarize_layer_profile([str(profile)], top=10)

    assert summary["runtime_rows"] == 0
    assert summary["forward_rows"] == 3
    assert summary["forward_layer_keys"] == 2
    assert summary["forward_variant_keys"] == 1
    assert summary["forward_stage_keys"] == 3
    assert summary["top_forward_variants"][0]["key"] == "variant_a"
    assert summary["top_forward_variants"][0]["calls"] == 3
    assert summary["top_forward_variants"][0]["unique_stages"] == 3
    assert summary["top_forward_variants"][0]["elapsed_s"] == 0.8
    assert summary["top_forward_layers"][0]["key"] == "layer0_variant_a"
    assert summary["top_forward_layers"][0]["elapsed_s"] == 0.7
    assert summary["top_forward_layers"][0]["top_stages"] == [
        {"key": "attention_replicated_postprocess", "count": 1},
        {"key": "attention_dp_all_reduce_post_pre", "count": 1},
    ]
    assert summary["top_forward_layers"][0]["top_stage_times"] == [
        {"key": "attention_dp_all_reduce_post_pre", "count": 1, "elapsed_s": 0.5},
        {"key": "attention_replicated_postprocess", "count": 1, "elapsed_s": 0.2},
    ]
    assert summary["top_forward_stages"][0]["key"] == (
        "attention_dp_all_reduce_post_pre"
    )

    elapsed_summary = summarizer.summarize_layer_profile(
        [str(profile)],
        top=10,
        sort_by="elapsed_avg_s",
    )
    assert elapsed_summary["sort_by"] == "elapsed_avg_s"
    assert elapsed_summary["top_forward_stages"][0]["key"] == (
        "attention_dp_all_reduce_post_pre"
    )


def test_summarize_dsv4_layer_profile_expands_profile_dirs(tmp_path: Path) -> None:
    summarizer = _load_script("summarize_dsv4_layer_profile.py")
    profile_root = tmp_path / "profile"
    runtime_dir = profile_root / "runtime"
    runtime_dir.mkdir(parents=True)
    runtime_profile = runtime_dir / "dsv4_product_runtime_rank_0.jsonl"
    forward_profile = runtime_dir / "dsv4_product_forward_rank_0.jsonl"
    runtime_profile.write_text(
        json.dumps(
            {
                "stage": "run_product_kernel",
                "name": "kernel_a",
                "layer_graph_key": "layer0_variant_a",
                "layer_variant_key": "variant_a",
                "call_s": 0.2,
            }
        )
        + "\n"
    )
    forward_profile.write_text(
        json.dumps(
            {
                "stage": "attention_dp_materialize",
                "layer_graph_key": "layer0_variant_a",
                "layer_variant_key": "variant_a",
                "elapsed_s": 0.1,
            }
        )
        + "\n"
    )

    summary = summarizer.summarize_layer_profile([str(profile_root)], top=10)

    assert summary["runtime_rows"] == 1
    assert summary["forward_rows"] == 1
    assert summary["top_kernel_names"][0]["key"] == "kernel_a"
    assert summary["top_forward_stages"][0]["key"] == "attention_dp_materialize"


def test_summarize_dsv4_layer_profile_filters_to_serving_window(
    tmp_path: Path,
) -> None:
    summarizer = _load_script("summarize_dsv4_layer_profile.py")
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    worker_profile = profile_root / "worker_0_steps.jsonl"
    runtime_profile = profile_root / "dsv4_product_runtime_rank_0.jsonl"
    forward_profile = profile_root / "dsv4_product_forward_rank_0.jsonl"
    worker_profile.write_text(
        json.dumps(
            {
                "step": 1,
                "rank": 0,
                "ts": 12.0,
                "t_total": 2.0,
                "t_model_forward": 1.8,
            }
        )
        + "\n"
    )
    runtime_profile.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "ts": 6.0,
                    "stage": "run_product_kernel",
                    "name": "warmup_kernel",
                    "layer_graph_key": "warmup_layer",
                    "layer_variant_key": "warmup_variant",
                    "call_s": 4.0,
                },
                {
                    "ts": 11.0,
                    "stage": "run_product_kernel",
                    "name": "serving_kernel",
                    "layer_graph_key": "serving_layer",
                    "layer_variant_key": "serving_variant",
                    "call_s": 0.1,
                },
            )
        )
        + "\n"
    )
    forward_profile.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "ts": 6.0,
                    "stage": "embedding_mhc_pre",
                    "layer_graph_key": "warmup_layer",
                    "layer_variant_key": "warmup_variant",
                    "elapsed_s": 4.0,
                },
                {
                    "ts": 11.2,
                    "stage": "attention",
                    "layer_graph_key": "serving_layer",
                    "layer_variant_key": "serving_variant",
                    "elapsed_s": 0.2,
                },
            )
        )
        + "\n"
    )

    summary = summarizer.summarize_layer_profile(
        [str(profile_root)],
        top=10,
        serving_window=True,
    )

    assert summary["time_filter"] == "serving_window"
    assert summary["worker_step_rows"] == 1
    assert summary["min_ts"] == 10.0
    assert summary["max_ts"] == 12.0
    assert summary["timestamp_filtered_rows"] == 2
    assert summary["runtime_rows"] == 1
    assert summary["forward_rows"] == 1
    assert summary["top_kernel_names"][0]["key"] == "serving_kernel"
    assert summary["top_forward_stages"][0]["key"] == "attention"


def test_render_chat_prompt_uses_deepseek_v4_native_template() -> None:
    bench_serving = _load_bench_serving()

    class _FailingTokenizer(_FakeTokenizer):
        @property
        def tokenizer(self):
            raise AssertionError(
                "DeepSeek-V4 prompt rendering must not load AutoTokenizer"
            )

    prompt = bench_serving._render_chat_prompt(
        tokenizer=_FailingTokenizer(),
        model="deepseek-ai/DeepSeek-V4-Flash",
        messages=[{"role": "user", "content": "Say hello."}],
        extra_request_body={
            "chat_template_kwargs": {"reasoning_effort": "high", "thinking": True}
        },
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>")
    assert "<｜User｜>Say hello." in prompt
    assert "<｜Assistant｜><think>" in prompt


def test_chat_payload_uses_max_completion_tokens_by_default() -> None:
    bench_serving = _load_bench_serving()

    row = bench_serving.DatasetRow(
        prompt="hello",
        prompt_len=5,
        output_len=42,
        messages=[{"role": "user", "content": "hello"}],
        prompt_for_strip=None,
    )
    req = bench_serving.RequestFuncInput(
        api_url="http://localhost/v1/chat/completions",
        model="unsloth/gpt-oss-120b-BF16",
        row=row,
        stream=True,
        temperature=0.0,
        timeout_s=30,
        extra_request_body={},
        headers={},
    )

    payload = bench_serving._build_generation_payload(req, is_chat=True)

    assert payload["messages"] == row.messages
    assert payload["max_completion_tokens"] == 42
    assert "max_tokens" not in payload


def test_stream_request_fails_when_done_marker_missing(monkeypatch) -> None:
    bench_serving = _load_bench_serving()

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            chunk = {
                "choices": [
                    {
                        "delta": {"content": "partial"},
                        "finish_reason": None,
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

    monkeypatch.setattr(
        bench_serving.urllib_request,
        "urlopen",
        lambda request, timeout: _Response(),
    )
    req = bench_serving.RequestFuncInput(
        api_url="http://localhost/v1/chat/completions",
        model="deepseek-ai/DeepSeek-V4-Flash",
        row=bench_serving.DatasetRow(
            prompt="hello",
            prompt_len=1,
            output_len=1,
            messages=[{"role": "user", "content": "hello"}],
        ),
        stream=True,
        temperature=0.0,
        timeout_s=30,
        extra_request_body={},
        headers={},
    )

    out = bench_serving._sync_request_openai(
        req,
        tokenizer=_FakeTokenizer(),
        is_chat=True,
    )

    assert out.success is False
    assert out.error == "stream ended before [DONE]"


def test_blockwise_index_source_hash_handles_missing_soabi(monkeypatch) -> None:
    original_get_config_var = blockwise_index.sysconfig.get_config_var

    def _fake_get_config_var(name: str):
        if name == "SOABI":
            return None
        if name == "EXT_SUFFIX":
            return ".test-so"
        return original_get_config_var(name)

    monkeypatch.setattr(
        blockwise_index.sysconfig, "get_config_var", _fake_get_config_var
    )

    digest = blockwise_index._source_hash()

    assert isinstance(digest, str)
    assert len(digest) == 16


def test_gen_prompt_for_target_len_does_not_overshoot_by_word_count() -> None:
    bench_serving = _load_bench_serving()
    tokenizer = _TripleTokenWordTokenizer()

    prompt, prompt_len = bench_serving._gen_prompt_for_target_len(
        tokenizer=tokenizer,
        target_len=128,
        rng=random.Random(0),
    )

    assert prompt
    assert prompt_len >= 128
    # Minimal word-count search should overshoot by at most one word's token
    # contribution for this tokenizer shape.
    assert prompt_len <= 130
