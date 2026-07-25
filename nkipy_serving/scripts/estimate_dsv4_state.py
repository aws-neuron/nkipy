"""Estimate DeepSeek-V4 mutable KV/state memory.

This is a scheduler-side planning tool. It estimates only DSV4-owned mutable
attention state: SWA KV, compressor ring/cache, and c4 indexer ring/cache.
It does not estimate model weights, scratch buffers, compiler/runtime
overhead, fragmentation, or graph intermediates.

Usage with a runtime JSON:

    uv run python -m scripts.estimate_dsv4_state \
        --config tests/runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json

Usage with a raw HF config:

    uv run python -m scripts.estimate_dsv4_state \
        --config /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale/config.json \
        --max-context-len 4096 \
        --state-size 4096 \
        --max-requests 1 \
        --kv-pool-size 16384
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nkipy_serving.managers.scheduler import _Dsv4KVPressureModel


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config root must be an object: {path}")
    return data


def _read_raw_hf_config(config: Path) -> tuple[dict[str, Any], Path]:
    path = Path(config)
    if path.is_dir():
        path = path / "config.json"
    data = _read_json(path)
    if data.get("model_type") != "deepseek_v4":
        raise RuntimeError(
            f"Expected model_type='deepseek_v4' in {path}, got {data.get('model_type')!r}"
        )
    return data, path


def _resolve_hf_config_from_runtime_config(
    runtime_config: dict[str, Any],
    runtime_path: Path,
) -> tuple[dict[str, Any], Path]:
    candidates = [
        runtime_config.get("hf_model_id"),
        runtime_config.get("model_id"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = (runtime_path.parent / path).resolve()
        if path.is_dir():
            cfg_path = path / "config.json"
        else:
            cfg_path = path
        if not cfg_path.exists():
            continue
        try:
            return _read_raw_hf_config(cfg_path)
        except RuntimeError:
            continue
    raise RuntimeError(
        "Runtime config does not point at a local DeepSeek-V4 config.json via "
        f"hf_model_id or model_id: {runtime_path}"
    )


def _resolve_config(config: Path) -> tuple[dict[str, Any], dict[str, Any] | None, Path]:
    path = Path(config)
    if path.is_dir():
        hf_config, hf_path = _read_raw_hf_config(path)
        return hf_config, None, hf_path
    data = _read_json(path)
    if data.get("model_type") == "deepseek_v4":
        return data, None, path
    hf_config, hf_path = _resolve_hf_config_from_runtime_config(data, path)
    return hf_config, data, hf_path


def _infer_runtime_max_requests(runtime_config: dict[str, Any]) -> int:
    buckets = runtime_config.get("request_buckets")
    if isinstance(buckets, list) and buckets:
        return max(int(v) for v in buckets)
    return int(runtime_config.get("max_requests", 1))


def estimate_from_config(
    config: Path,
    *,
    max_context_len: int | None = None,
    state_size: int | None = None,
    max_requests: int | None = None,
    kv_pool_size: int | None = None,
    block_size: int | None = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    data, runtime_config, model_config_path = _resolve_config(config)
    if runtime_config is not None:
        if max_context_len is None:
            max_context_len = int(runtime_config["max_context_len"])
        if state_size is None:
            state_size = int(runtime_config.get("dsv4_state_size", 0))
        if max_requests is None:
            max_requests = _infer_runtime_max_requests(runtime_config)
        if kv_pool_size is None:
            kv_pool_size = int(runtime_config.get("kv_pool_size", 16384))
        if block_size is None:
            block_size = int(runtime_config.get("kv_cache_block_size", 32))
        if (
            num_layers is None
            and runtime_config.get("hf_num_hidden_layers") is not None
        ):
            num_layers = int(runtime_config["hf_num_hidden_layers"])
    elif max_context_len is None:
        raise RuntimeError(
            "--max-context-len is required for raw HF config.json inputs"
        )

    if max_context_len is None:
        raise RuntimeError("max_context_len could not be inferred")
    if state_size is None:
        raise RuntimeError("--state-size is required for raw HF config.json inputs")
    if state_size <= 0:
        raise RuntimeError(f"dsv4_state_size must be > 0, got {state_size}")
    if state_size < max_context_len:
        raise RuntimeError(
            "dsv4_state_size must cover max_context_len: "
            f"dsv4_state_size={state_size}, max_context_len={max_context_len}"
        )
    if max_requests is None:
        max_requests = 1
    if kv_pool_size is None:
        kv_pool_size = 16384
    if block_size is None:
        block_size = 32

    n_layers = int(data["num_hidden_layers"])
    if num_layers is not None:
        n_layers = min(n_layers, int(num_layers))
    ratios = tuple(int(v) for v in data["compress_ratios"][:n_layers])
    if len(ratios) != n_layers:
        raise RuntimeError(
            f"compress_ratios has {len(ratios)} entries, expected {n_layers}"
        )
    model = _Dsv4KVPressureModel(
        compress_ratios=ratios,
        sliding_window=int(data["sliding_window"]),
        head_dim=int(data["head_dim"]),
        index_head_dim=int(data.get("index_head_dim", data["head_dim"])),
        max_context_len=int(max_context_len),
    )
    owner_swa_slots = int(max_requests) * int(model.sliding_window)
    num_slots_per_layer = max(
        int(kv_pool_size) + 1,
        int(block_size) + 1,
        owner_swa_slots + 1,
    )
    out = model.summary(
        max_requests=int(max_requests),
        num_slots_per_layer=num_slots_per_layer,
        max_seq_len=int(state_size),
    )
    out["config"] = str(config)
    out["model_config"] = str(model_config_path)
    if runtime_config is not None:
        out["runtime_config"] = str(config)
    out["block_size"] = int(block_size)
    out["kv_pool_size"] = int(kv_pool_size)
    return out


def _format_bytes(value: int) -> str:
    return f"{int(value):,} bytes ({int(value) / (1024**2):.2f} MiB)"


def format_text(summary: dict[str, Any]) -> str:
    lines = [
        f"config: {summary['config']}",
        f"layers: {summary['num_layers']} {summary['layer_kind_counts']}",
        f"max_context_len: {summary['max_context_len']}",
        f"state_size: {summary['state_size']}",
        f"max_requests: {summary['max_requests']}",
        f"kv_pool_size: {summary['kv_pool_size']}",
        f"num_slots_per_layer: {summary['num_slots_per_layer']}",
        "estimated_static_state_bytes_per_worker: "
        + _format_bytes(summary["estimated_static_state_bytes_per_worker"]),
        "estimated_bytes_per_full_context_request: "
        + _format_bytes(summary["estimated_bytes_per_full_context_request"]),
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="runtime JSON, config.json, or snapshot dir",
    )
    parser.add_argument("--max-context-len", type=int)
    parser.add_argument("--state-size", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--kv-pool-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of text"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = estimate_from_config(
        args.config,
        max_context_len=args.max_context_len,
        state_size=args.state_size,
        max_requests=args.max_requests,
        kv_pool_size=args.kv_pool_size,
        block_size=args.block_size,
        num_layers=args.num_layers,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
