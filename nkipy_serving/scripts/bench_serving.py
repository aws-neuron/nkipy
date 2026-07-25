#!/usr/bin/env python3

"""Benchmark online serving with dynamic requests for nkipy-serving.

This is a trimmed benchmark flow adapted from sglang-jax/sglang bench_serving:
- OpenAI-style endpoints only (`/v1/completions`, `/v1/chat/completions`)
- Random/custom/ShareGPT prompt datasets
- Poisson request arrivals
- Optional max-concurrency cap
- Warmup, TTFT/ITL/E2E latency and throughput metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from nkipy_serving.conversation import (
    generate_chat_conv,
    generate_deepseek_v4_chat_conv,
)
from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer


def _parse_custom_headers(header_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in header_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def _build_headers(custom_headers: dict[str, str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = custom_headers.pop("_api_key_internal", None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(custom_headers)
    return headers


@dataclass(frozen=True)
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    messages: list[dict[str, str]] | None = None
    prompt_for_strip: str | None = None


@dataclass(frozen=True)
class RequestFuncInput:
    api_url: str
    model: str
    row: DatasetRow
    stream: bool
    temperature: float
    timeout_s: int
    extra_request_body: dict[str, Any]
    headers: dict[str, str]


@dataclass
class RequestFuncOutput:
    success: bool = False
    generated_text: str = ""
    output_len: int = 0
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""


@dataclass(frozen=True)
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


def _safe_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _is_valid_sharegpt_sequence(prompt_len: int, output_len: int) -> bool:
    if prompt_len < 4 or output_len < 4:
        return False
    if prompt_len > 1024:
        return False
    if prompt_len + output_len > 2048:
        return False
    return True


def _summarize_lengths(rows: list[DatasetRow]) -> dict[str, dict[str, float]]:
    def _stats(values: list[int]) -> dict[str, float]:
        vals = sorted(int(v) for v in values)
        if not vals:
            return {
                "count": 0.0,
                "min": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "max": 0.0,
                "mean": 0.0,
            }
        return {
            "count": float(len(vals)),
            "min": float(vals[0]),
            "p50": _safe_percentile(vals, 50.0),
            "p90": _safe_percentile(vals, 90.0),
            "p95": _safe_percentile(vals, 95.0),
            "max": float(vals[-1]),
            "mean": float(sum(vals) / len(vals)),
        }

    prompt = [row.prompt_len for row in rows]
    output = [row.output_len for row in rows]
    total = [row.prompt_len + row.output_len for row in rows]
    return {
        "prompt": _stats(prompt),
        "output": _stats(output),
        "total": _stats(total),
    }


def _strip_prompt_prefix(text: str, prompt_prefix: str | None) -> str:
    if not prompt_prefix:
        return text
    return text[len(prompt_prefix) :] if text.startswith(prompt_prefix) else text


def _is_gpt_oss_model(model_name: str | None) -> bool:
    return "gpt-oss" in str(model_name or "").lower()


def _is_deepseek_v4_model(model_name: str | None) -> bool:
    return "DeepSeek-V4" in str(model_name or "")


def _render_chat_prompt(
    *,
    tokenizer: HfTokenizer,
    model: str | None,
    messages: list[dict[str, str]],
    extra_request_body: dict[str, Any] | None = None,
) -> str:
    extra_request_body = extra_request_body or {}
    chat_template_kwargs = extra_request_body.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}

    if _is_deepseek_v4_model(model):
        dsv4_messages = [dict(message) for message in messages]
        if dsv4_messages and dsv4_messages[0].get("role") != "system":
            dsv4_messages.insert(0, {"role": "system", "content": ""})
        task = extra_request_body.get("task")
        if task is not None:
            for msg in reversed(dsv4_messages):
                if msg.get("role") == "user":
                    msg["task"] = task
                    break
            else:
                raise ValueError("DeepSeek-V4 task requires at least one user message")
        effort_source = chat_template_kwargs.get(
            "reasoning_effort", extra_request_body.get("reasoning_effort")
        )
        dsv4_reasoning_effort = (
            effort_source if effort_source in {"max", "high"} else None
        )
        return generate_deepseek_v4_chat_conv(
            dsv4_messages,
            thinking=bool(
                chat_template_kwargs.get("thinking")
                or chat_template_kwargs.get("enable_thinking")
            ),
            reasoning_effort=dsv4_reasoning_effort,
        )

    hf_tok = getattr(tokenizer, "tokenizer", None)
    if (
        _is_gpt_oss_model(model)
        and hf_tok is not None
        and hasattr(hf_tok, "apply_chat_template")
    ):
        return hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=str(extra_request_body.get("reasoning_effort", "medium")),
            **chat_template_kwargs,
        )

    return generate_chat_conv(
        messages,
        tokenizer=hf_tok,
        chat_template=chat_template_kwargs.get("chat_template"),
    )


def _make_chat_dataset_row(
    *,
    tokenizer: HfTokenizer,
    model: str | None,
    prompt: str,
    output_len: int,
    messages: list[dict[str, str]],
    extra_request_body: dict[str, Any] | None = None,
) -> DatasetRow:
    chat_prompt = _render_chat_prompt(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        extra_request_body=extra_request_body,
    )
    return DatasetRow(
        prompt=prompt,
        prompt_len=int(tokenizer.encode(chat_prompt).size),
        output_len=output_len,
        messages=messages,
        prompt_for_strip=None,
    )


def _build_generation_payload(req: RequestFuncInput, is_chat: bool) -> dict[str, Any]:
    payload: dict[str, Any]
    if is_chat:
        messages = req.row.messages
        if messages is None:
            messages = [{"role": "user", "content": req.row.prompt}]
        payload = {
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        if (
            "max_completion_tokens" not in req.extra_request_body
            and "max_tokens" not in req.extra_request_body
        ):
            payload["max_completion_tokens"] = req.row.output_len
    else:
        payload = {
            "model": req.model,
            "prompt": req.row.prompt,
            "max_tokens": req.row.output_len,
            "temperature": req.temperature,
            "stream": req.stream,
        }
    payload.update(req.extra_request_body)
    return payload


def _sync_request_openai(
    req: RequestFuncInput,
    tokenizer: HfTokenizer,
    is_chat: bool,
) -> RequestFuncOutput:
    out = RequestFuncOutput(prompt_len=req.row.prompt_len)
    payload = _build_generation_payload(req, is_chat)

    body_bytes = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        req.api_url,
        data=body_bytes,
        method="POST",
        headers=req.headers,
    )

    start = time.perf_counter()
    generated_text = ""
    ttft = 0.0
    last_token_ts = start
    itl: list[float] = []

    try:
        with urllib_request.urlopen(request, timeout=req.timeout_s) as resp:
            status = int(resp.status)
            if status != 200:
                out.error = (
                    f"HTTP {status}: {resp.read().decode('utf-8', errors='replace')}"
                )
                return out

            if req.stream:
                saw_done = False
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[len("data:") :].strip()
                    if payload_text == "[DONE]":
                        saw_done = True
                        break
                    data = json.loads(payload_text)
                    choice = data.get("choices", [{}])[0]
                    if is_chat:
                        delta = choice.get("delta", {})
                        piece = str(
                            delta.get("content") or delta.get("reasoning_content") or ""
                        )
                    else:
                        piece = str(choice.get("text", ""))
                    if not piece:
                        continue
                    now = time.perf_counter()
                    if ttft == 0.0:
                        ttft = now - start
                    else:
                        itl.append(now - last_token_ts)
                    last_token_ts = now
                    generated_text += piece
                if not saw_done:
                    out.error = "stream ended before [DONE]"
                    out.latency = time.perf_counter() - start
                    return out
            else:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0]
                if is_chat:
                    generated_text = str(choice.get("message", {}).get("content", ""))
                else:
                    generated_text = str(choice.get("text", ""))
                ttft = time.perf_counter() - start
                last_token_ts = time.perf_counter()

        latency = last_token_ts - start
        completion_text = _strip_prompt_prefix(generated_text, req.row.prompt_for_strip)
        completion_token_len = int(tokenizer.encode(completion_text).size)

        out.success = True
        out.generated_text = completion_text
        out.output_len = completion_token_len
        out.latency = latency
        out.ttft = ttft
        out.itl = itl
        return out
    except urllib_error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        out.error = f"HTTPError {exc.code}: {details}"
        out.latency = time.perf_counter() - start
        return out
    except Exception:
        out.error = "".join(traceback.format_exception(*sys.exc_info()))
        out.latency = time.perf_counter() - start
        return out


async def _async_request_openai(
    req: RequestFuncInput,
    tokenizer: HfTokenizer,
    is_chat: bool,
) -> RequestFuncOutput:
    return await asyncio.to_thread(_sync_request_openai, req, tokenizer, is_chat)


def _gen_prompt_for_target_len(
    tokenizer: HfTokenizer, target_len: int, rng: random.Random
) -> tuple[str, int]:
    target_len = max(1, int(target_len))
    # Search for the shortest random word list whose tokenized length reaches
    # the target. Starting with `target_len` words can overshoot real token
    # length by ~3x on GPT-OSS tokenization.
    cache: dict[int, tuple[str, int]] = {}

    def _render(num_words: int) -> tuple[str, int]:
        cached = cache.get(num_words)
        if cached is not None:
            return cached
        words = [f"w{rng.randrange(1_000_000)}" for _ in range(num_words)]
        text = " ".join(words)
        token_len = int(tokenizer.encode(text).size)
        cache[num_words] = (text, token_len)
        return text, token_len

    lo = 1
    hi = 1
    _, token_len = _render(hi)
    while token_len < target_len:
        lo = hi + 1
        hi *= 2
        _, token_len = _render(hi)

    best_text, best_len = _render(hi)
    left = max(1, lo)
    right = hi
    while left <= right:
        mid = (left + right) // 2
        text, token_len = _render(mid)
        if token_len >= target_len:
            best_text, best_len = text, token_len
            right = mid - 1
        else:
            left = mid + 1
    return best_text, best_len


def _compute_random_lens(
    base: int, ratio: float, num: int, rng: random.Random
) -> list[int]:
    base = max(1, int(base))
    if ratio <= 0:
        return [base for _ in range(num)]
    low = max(1, int(math.floor(base * (1.0 - ratio))))
    high = max(low, int(math.ceil(base * (1.0 + ratio))))
    return [rng.randint(low, high) for _ in range(num)]


def _sample_random_requests(
    tokenizer: HfTokenizer,
    model: str | None,
    num_prompts: int,
    random_input_len: int,
    random_output_len: int,
    random_range_ratio: float,
    prompt_suffix: str,
    as_chat: bool,
    seed: int,
    extra_request_body: dict[str, Any] | None = None,
) -> list[DatasetRow]:
    rng = random.Random(seed)
    input_lens = _compute_random_lens(
        base=random_input_len,
        ratio=random_range_ratio,
        num=num_prompts,
        rng=rng,
    )
    output_lens = _compute_random_lens(
        base=random_output_len,
        ratio=random_range_ratio,
        num=num_prompts,
        rng=rng,
    )

    rows: list[DatasetRow] = []
    for input_len, output_len in zip(input_lens, output_lens):
        prompt, prompt_len = _gen_prompt_for_target_len(tokenizer, input_len, rng=rng)
        if prompt_suffix:
            prompt = f"{prompt}{prompt_suffix}"
            prompt_len = int(tokenizer.encode(prompt).size)
        if as_chat:
            messages = [{"role": "user", "content": prompt}]
            rows.append(
                _make_chat_dataset_row(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    output_len=output_len,
                    messages=messages,
                    extra_request_body=extra_request_body,
                )
            )
        else:
            rows.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    messages=None,
                    prompt_for_strip=prompt,
                )
            )
    return rows


def _load_custom_requests(
    tokenizer: HfTokenizer,
    model: str | None,
    dataset_path: str,
    num_prompts: int,
    as_chat: bool,
    extra_request_body: dict[str, Any] | None = None,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            output_len = int(item.get("output_len", item.get("max_tokens", 16)))
            output_len = max(1, output_len)
            if as_chat:
                messages = item.get("messages")
                if messages is None:
                    prompt = str(item.get("prompt", ""))
                    messages = [{"role": "user", "content": prompt}]
                prompt = str(item.get("prompt", ""))
                if not prompt and messages:
                    prompt = str(messages[-1].get("content", ""))
                rows.append(
                    _make_chat_dataset_row(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                        output_len=output_len,
                        messages=messages,
                        extra_request_body=extra_request_body,
                    )
                )
            else:
                prompt = str(item.get("prompt", ""))
                rows.append(
                    DatasetRow(
                        prompt=prompt,
                        prompt_len=int(tokenizer.encode(prompt).size),
                        output_len=output_len,
                        messages=None,
                        prompt_for_strip=prompt,
                    )
                )
            if len(rows) >= num_prompts:
                break
    if not rows:
        raise RuntimeError(f"No rows loaded from custom dataset: {dataset_path}")
    return rows


def _load_sharegpt_requests(
    tokenizer: HfTokenizer,
    model: str | None,
    dataset_path: str,
    num_prompts: int,
    as_chat: bool,
    seed: int,
    sharegpt_output_len: int | None,
    extra_request_body: dict[str, Any] | None = None,
) -> list[DatasetRow]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [
        entry
        for entry in data
        if "conversations" in entry and len(entry["conversations"]) >= 2
    ]
    rng = random.Random(seed)
    rng.shuffle(filtered)

    rows: list[DatasetRow] = []
    for entry in filtered:
        if len(rows) >= num_prompts:
            break
        prompt = str(entry["conversations"][0]["value"])
        completion = str(entry["conversations"][1]["value"])
        prompt_len = int(tokenizer.encode(prompt).size)
        output_len = (
            int(sharegpt_output_len)
            if sharegpt_output_len is not None
            else int(tokenizer.encode(completion).size)
        )
        if not _is_valid_sharegpt_sequence(prompt_len, output_len):
            continue
        if as_chat:
            messages = [{"role": "user", "content": prompt}]
            rows.append(
                _make_chat_dataset_row(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    output_len=output_len,
                    messages=messages,
                    extra_request_body=extra_request_body,
                )
            )
        else:
            rows.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    messages=None,
                    prompt_for_strip=prompt,
                )
            )
    if len(rows) < num_prompts:
        raise RuntimeError(f"Only sampled {len(rows)} prompts, expected {num_prompts}")
    return rows


def _build_dataset(
    args: argparse.Namespace,
    tokenizer: HfTokenizer,
    as_chat: bool,
    model: str | None,
    extra_request_body: dict[str, Any] | None,
) -> list[DatasetRow]:
    if args.dataset_name == "random":
        return _sample_random_requests(
            tokenizer=tokenizer,
            model=model,
            num_prompts=args.num_prompts,
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            random_range_ratio=args.random_range_ratio,
            prompt_suffix=args.prompt_suffix,
            as_chat=as_chat,
            seed=args.seed,
            extra_request_body=extra_request_body,
        )
    if args.dataset_name == "custom":
        if not args.dataset_path:
            raise RuntimeError("--dataset-path is required when --dataset-name=custom")
        return _load_custom_requests(
            tokenizer=tokenizer,
            model=model,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            as_chat=as_chat,
            extra_request_body=extra_request_body,
        )
    if args.dataset_name == "sharegpt":
        if not args.dataset_path:
            raise RuntimeError(
                "--dataset-path is required when --dataset-name=sharegpt"
            )
        return _load_sharegpt_requests(
            tokenizer=tokenizer,
            model=model,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            as_chat=as_chat,
            seed=args.seed,
            sharegpt_output_len=args.sharegpt_output_len,
            extra_request_body=extra_request_body,
        )
    raise RuntimeError(f"Unsupported dataset_name={args.dataset_name}")


def _summarize_http_profile(
    *,
    path: str,
    submit_ts_min: float,
    submit_ts_max: float,
) -> dict[str, Any]:
    scheduled: list[float] = []
    queue_delay: list[float] = []
    total: list[float] = []
    matched = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") != "request_completed":
                continue
            submit_ts = item.get("submit_ts")
            if not isinstance(submit_ts, (int, float)):
                continue
            if float(submit_ts) < (submit_ts_min - 5.0) or float(submit_ts) > (
                submit_ts_max + 5.0
            ):
                continue
            matched += 1
            scheduled_ttft_ms = item.get("scheduled_ttft_ms")
            ttft_ms = item.get("ttft_ms")
            total_ms = item.get("total_ms")
            if isinstance(scheduled_ttft_ms, (int, float)):
                scheduled.append(float(scheduled_ttft_ms))
            if isinstance(ttft_ms, (int, float)) and isinstance(
                scheduled_ttft_ms, (int, float)
            ):
                queue_delay.append(float(ttft_ms) - float(scheduled_ttft_ms))
            if isinstance(total_ms, (int, float)):
                total.append(float(total_ms))
    return {
        "matched_completed_events": matched,
        "mean_scheduled_ttft_ms": float(statistics.mean(scheduled))
        if scheduled
        else 0.0,
        "median_scheduled_ttft_ms": float(statistics.median(scheduled))
        if scheduled
        else 0.0,
        "p90_scheduled_ttft_ms": _safe_percentile(scheduled, 90.0),
        "mean_queue_delay_ms": float(statistics.mean(queue_delay))
        if queue_delay
        else 0.0,
        "median_queue_delay_ms": float(statistics.median(queue_delay))
        if queue_delay
        else 0.0,
        "p90_queue_delay_ms": _safe_percentile(queue_delay, 90.0),
        "mean_server_total_ms": float(statistics.mean(total)) if total else 0.0,
        "median_server_total_ms": float(statistics.median(total)) if total else 0.0,
        "p90_server_total_ms": _safe_percentile(total, 90.0),
    }


async def _get_request(
    rows: list[DatasetRow],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    for row in rows:
        yield row
        if request_rate == float("inf"):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(float(interval))


def _calculate_metrics(
    input_rows: list[DatasetRow],
    outputs: list[RequestFuncOutput],
    duration_s: float,
) -> BenchmarkMetrics:
    total_input = 0
    total_output = 0
    ttfts: list[float] = []
    itls: list[float] = []
    e2e: list[float] = []
    completed = 0
    failed = 0

    for row, out in zip(input_rows, outputs):
        if out.success:
            completed += 1
            total_input += row.prompt_len
            total_output += out.output_len
            if out.ttft > 0:
                ttfts.append(out.ttft)
            itls.extend(out.itl)
            e2e.append(out.latency)
        else:
            failed += 1

    if duration_s <= 0:
        duration_s = 1e-9

    return BenchmarkMetrics(
        completed=completed,
        failed=failed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / duration_s,
        input_throughput=total_input / duration_s,
        output_throughput=total_output / duration_s,
        total_throughput=(total_input + total_output) / duration_s,
        mean_ttft_ms=float(np.mean(ttfts) * 1000 if ttfts else 0.0),
        median_ttft_ms=float(np.median(ttfts) * 1000 if ttfts else 0.0),
        p99_ttft_ms=_safe_percentile(ttfts, 99) * 1000,
        mean_itl_ms=float(np.mean(itls) * 1000 if itls else 0.0),
        median_itl_ms=float(np.median(itls) * 1000 if itls else 0.0),
        p95_itl_ms=_safe_percentile(itls, 95) * 1000,
        p99_itl_ms=_safe_percentile(itls, 99) * 1000,
        mean_e2e_latency_ms=float(np.mean(e2e) * 1000 if e2e else 0.0),
        median_e2e_latency_ms=float(np.median(e2e) * 1000 if e2e else 0.0),
        p99_e2e_latency_ms=_safe_percentile(e2e, 99) * 1000,
        concurrency=float(sum(e2e) / duration_s if e2e else 0.0),
    )


async def _benchmark(
    *,
    api_url: str,
    model: str,
    tokenizer: HfTokenizer,
    rows: list[DatasetRow],
    request_rate: float,
    max_concurrency: int | None,
    stream: bool,
    temperature: float,
    timeout_s: int,
    extra_request_body: dict[str, Any],
    headers: dict[str, str],
    warmup_requests: int,
    is_chat: bool,
) -> tuple[BenchmarkMetrics, list[RequestFuncOutput], float]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _run_one(row: DatasetRow) -> RequestFuncOutput:
        req = RequestFuncInput(
            api_url=api_url,
            model=model,
            row=row,
            stream=stream,
            temperature=temperature,
            timeout_s=timeout_s,
            extra_request_body=extra_request_body,
            headers=headers,
        )
        return await _async_request_openai(req, tokenizer=tokenizer, is_chat=is_chat)

    async def _run_one_limited(row: DatasetRow) -> RequestFuncOutput:
        if semaphore is None:
            return await _run_one(row)
        async with semaphore:
            return await _run_one(row)

    warmup_n = max(0, int(warmup_requests))
    if warmup_n > 0:
        warmup_row = rows[0]
        warmup_row = DatasetRow(
            prompt=warmup_row.prompt,
            prompt_len=warmup_row.prompt_len,
            output_len=min(warmup_row.output_len, 32),
            messages=warmup_row.messages,
            prompt_for_strip=warmup_row.prompt_for_strip,
        )
        warmup_outs = await asyncio.gather(
            *(_run_one(warmup_row) for _ in range(warmup_n))
        )
        if not any(item.success for item in warmup_outs):
            raise RuntimeError(
                f"Warmup failed. First error: {warmup_outs[0].error if warmup_outs else 'unknown'}"
            )

    tasks: list[asyncio.Task[RequestFuncOutput]] = []
    benchmark_start = time.perf_counter()
    async for row in _get_request(rows, request_rate=request_rate):
        tasks.append(asyncio.create_task(_run_one_limited(row)))
    outputs = await asyncio.gather(*tasks)
    duration_s = time.perf_counter() - benchmark_start
    metrics = _calculate_metrics(rows, outputs, duration_s)
    return metrics, outputs, duration_s


def _sync_get_json(url: str, headers: dict[str, str], timeout_s: int) -> dict[str, Any]:
    req = urllib_request.Request(url, method="GET", headers=headers)
    with urllib_request.urlopen(req, timeout=timeout_s) as resp:
        if int(resp.status) != 200:
            raise RuntimeError(f"GET {url} failed with status={resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def _resolve_model(args: argparse.Namespace, headers: dict[str, str]) -> str:
    if args.model:
        return args.model
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    model_url = f"{base_url}/v1/models"
    data = _sync_get_json(model_url, headers=headers, timeout_s=args.timeout_s)
    model_list = data.get("data", [])
    if not model_list:
        raise RuntimeError(f"No model returned by {model_url}")
    model_id = str(model_list[0].get("id", "")).strip()
    if not model_id:
        raise RuntimeError(f"Invalid model payload from {model_url}: {data}")
    return model_id


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark nkipy-serving online serving.")
    p.add_argument(
        "--backend",
        type=str,
        default="nkipy-serving-oai",
        choices=["nkipy-serving-oai", "nkipy-serving-oai-chat"],
    )
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "custom", "sharegpt"],
    )
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--http-profile-jsonl", type=str, default="")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--random-input-len", type=int, default=128)
    p.add_argument("--random-output-len", type=int, default=64)
    p.add_argument("--random-range-ratio", type=float, default=0.0)
    p.add_argument("--sharegpt-output-len", type=int, default=None)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--disable-stream", action="store_true")
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prompt-suffix", type=str, default="")
    p.add_argument(
        "--extra-request-body",
        type=str,
        default=None,
        help="JSON object string to merge into generation payload.",
    )
    p.add_argument("--header", type=str, nargs="*", default=[])
    p.add_argument("--output-file", type=str, default=None)
    p.add_argument("--output-details", action="store_true")
    return p


def _print_metrics(
    metrics: BenchmarkMetrics,
    duration_s: float,
    backend: str,
    http_profile: dict[str, Any] | None = None,
) -> None:
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=56, c="="))
    print("{:<42} {:<10}".format("Backend:", backend))
    print("{:<42} {:<10.2f}".format("Benchmark duration (s):", duration_s))
    print("{:<42} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<42} {:<10}".format("Failed requests:", metrics.failed))
    print("{:<42} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<42} {:<10}".format("Total output tokens:", metrics.total_output))
    print(
        "{:<42} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<42} {:<10.2f}".format("Input throughput (tok/s):", metrics.input_throughput)
    )
    print(
        "{:<42} {:<10.2f}".format(
            "Output throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<42} {:<10.2f}".format("Total throughput (tok/s):", metrics.total_throughput)
    )
    print("{:<42} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s=" Time to First Token ", n=56, c="-"))
    print("{:<42} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<42} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<42} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s=" Inter-Token Latency ", n=56, c="-"))
    print("{:<42} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<42} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<42} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<42} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{s:{c}^{n}}".format(s=" End-to-End Latency ", n=56, c="-"))
    print(
        "{:<42} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<42} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print(
        "{:<42} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms)
    )
    if http_profile:
        print("{s:{c}^{n}}".format(s=" Server TTFT ", n=56, c="-"))
        print(
            "{:<42} {:<10.2f}".format(
                "Mean scheduled TTFT (ms):",
                float(http_profile.get("mean_scheduled_ttft_ms", 0.0)),
            )
        )
        print(
            "{:<42} {:<10.2f}".format(
                "Median scheduled TTFT (ms):",
                float(http_profile.get("median_scheduled_ttft_ms", 0.0)),
            )
        )
        print(
            "{:<42} {:<10.2f}".format(
                "P90 scheduled TTFT (ms):",
                float(http_profile.get("p90_scheduled_ttft_ms", 0.0)),
            )
        )
        print(
            "{:<42} {:<10.2f}".format(
                "Mean queue delay (ms):",
                float(http_profile.get("mean_queue_delay_ms", 0.0)),
            )
        )
        print(
            "{:<42} {:<10.2f}".format(
                "Median queue delay (ms):",
                float(http_profile.get("median_queue_delay_ms", 0.0)),
            )
        )
    print("=" * 56)


def _default_output_file_name(args: argparse.Namespace) -> str:
    now = datetime.now().strftime("%m%d")
    if args.dataset_name == "random":
        return (
            f"{args.backend}_{now}_{args.num_prompts}_"
            f"{args.random_input_len}_{args.random_output_len}.jsonl"
        )
    if args.dataset_name == "sharegpt":
        return f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"
    return f"{args.backend}_{now}_{args.num_prompts}_custom.jsonl"


def main() -> None:
    args = _build_arg_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.request_rate <= 0 and args.request_rate != float("inf"):
        raise RuntimeError(f"request_rate must be > 0 or inf, got {args.request_rate}")
    if args.num_prompts <= 0:
        raise RuntimeError(f"num_prompts must be > 0, got {args.num_prompts}")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        raise RuntimeError(f"max_concurrency must be > 0, got {args.max_concurrency}")

    custom_headers = _parse_custom_headers(args.header)
    api_key = custom_headers.pop("Authorization", None)
    if api_key is None:
        env_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if env_api_key:
            custom_headers["_api_key_internal"] = env_api_key
    else:
        # If user passes `Authorization=Bearer ...`, keep raw header value.
        custom_headers["Authorization"] = api_key
    headers = _build_headers(custom_headers)

    model = _resolve_model(args, headers=headers)
    tokenizer_id = args.tokenizer or model
    tokenizer = HfTokenizer(model_id=tokenizer_id, local_files_only=True)

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    is_chat = args.backend == "nkipy-serving-oai-chat"
    api_url = (
        f"{base_url}/v1/chat/completions" if is_chat else f"{base_url}/v1/completions"
    )

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)
        if not isinstance(extra_request_body, dict):
            raise RuntimeError("--extra-request-body must be a JSON object")

    rows = _build_dataset(
        args=args,
        tokenizer=tokenizer,
        as_chat=is_chat,
        model=model,
        extra_request_body=extra_request_body,
    )
    length_summary = _summarize_lengths(rows)

    print(
        "benchmark_args="
        + json.dumps(
            {
                "backend": args.backend,
                "api_url": api_url,
                "model": model,
                "tokenizer": tokenizer_id,
                "dataset_name": args.dataset_name,
                "num_prompts": len(rows),
                "request_rate": args.request_rate,
                "max_concurrency": args.max_concurrency,
                "stream": not args.disable_stream,
                "warmup_requests": args.warmup_requests,
            },
            sort_keys=True,
        )
    )

    wall_start_ts = time.time()
    metrics, outputs, duration_s = asyncio.run(
        _benchmark(
            api_url=api_url,
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            stream=not args.disable_stream,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            extra_request_body=extra_request_body,
            headers=headers,
            warmup_requests=args.warmup_requests,
            is_chat=is_chat,
        )
    )
    wall_end_ts = time.time()

    http_profile = None
    if args.http_profile_jsonl:
        http_profile = _summarize_http_profile(
            path=args.http_profile_jsonl,
            submit_ts_min=wall_start_ts,
            submit_ts_max=wall_end_ts,
        )

    _print_metrics(
        metrics=metrics,
        duration_s=duration_s,
        backend=args.backend,
        http_profile=http_profile,
    )

    result: dict[str, Any] = {
        "backend": args.backend,
        "api_url": api_url,
        "dataset_name": args.dataset_name,
        "model": model,
        "duration": duration_s,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "total_throughput": metrics.total_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p95_itl_ms": metrics.p95_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "concurrency": metrics.concurrency,
        "length_summary": length_summary,
    }
    if http_profile is not None:
        result["http_profile"] = http_profile
        result.update(http_profile)
    if args.output_details:
        result["details"] = {
            "input_lens": [row.prompt_len for row in rows],
            "expected_output_lens": [row.output_len for row in rows],
            "output_lens": [out.output_len for out in outputs],
            "ttfts": [out.ttft for out in outputs],
            "itls": [out.itl for out in outputs],
            "errors": [out.error for out in outputs],
            "generated_texts": [out.generated_text for out in outputs],
        }

    output_file = args.output_file or _default_output_file_name(args)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"result_file={output_file}")

    if metrics.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
