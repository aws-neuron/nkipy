"""Prepare rank-local DSV4 weights for faster runtime upload.

Input: a converted DeepSeek-V4 serving snapshot produced by
``scripts.convert_dsv4_checkpoint``.

Output: a prepared-weight root, commonly named for the topology, such as
``DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1`` for the current
TP8/EP8/R1 4k serving config. For TP8/EP8, run with ``--all-unique-ranks`` to
emit the 64 unique lane/TP rank directories.

The normal product loader still does CPU work at startup: attention FP8+scale
weights are dequantized to BF16 and per-expert routed MoE tensors are packed
into blockwise per-rank tensors. This script performs that work once and writes
streamable per-rank safetensors:

    <dst-root>/tp8_ep8_rep1/lane00_tp00/dense.safetensors
    <dst-root>/tp8_ep8_rep1/lane00_tp00/layer_000.safetensors
    ...

Runtime enables the cache with `dsv4_prepared_weight_dir`, or the equivalent
environment override:

    NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR=<dst-root>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from nkipy_serving.models.deepseek_v4.config import DeepseekV4ModelConfig
from nkipy_serving.models.deepseek_v4.device_weights import (
    _DENSE_PREPARED_KEYS,
    _DSV4_PREPARED_WEIGHT_CACHE_VERSION,
    _LAYER_PREPARED_KEYS,
    V4DeviceWeights,
    V4LoadPlan,
    _snapshot_is_fp8_noscale,
    _stage_upload,
    _stage_v4_dense_hc_weights,
    _stage_v4_layer_weights,
    _StagedUpload,
    prepared_weight_rank_dir,
)
from nkipy_serving.models.deepseek_v4.weight_reader import V4WeightReader
from nkipy_serving.models.deepseek_v4.weights import _load_deepseek_v4_weights


def _collect_staged(value: Any, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for key in keys:
        item = getattr(value, key, None)
        if isinstance(item, _StagedUpload):
            tensors[key] = np.ascontiguousarray(item.array)
    return tensors


def _count_bytes(value: Any) -> int:
    if isinstance(value, _StagedUpload):
        return int(value.array.nbytes)
    if isinstance(value, list):
        return sum(_count_bytes(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_count_bytes(getattr(value, item.name)) for item in fields(value))
    return 0


def _prepare_one_rank(
    *,
    src: Path,
    dst_root: Path,
    num_layers: int | None,
    tp_degree: int,
    tp_rank: int,
    ep_degree: int,
    replica_degree: int,
    attention_dp_degree: int,
    request_lane_rank: int,
    overwrite: bool,
) -> Path:
    cfg = DeepseekV4ModelConfig(
        hf_model_id=str(src),
        hf_local_files_only=True,
        hf_num_hidden_layers=num_layers,
        tp_degree=int(tp_degree),
        tp_rank=int(tp_rank),
        tp_world_size=int(tp_degree),
        ep_degree=int(ep_degree),
        replica_degree=int(replica_degree),
        attention_dp_degree=int(attention_dp_degree),
        request_lane_rank=int(request_lane_rank),
        request_lane_world_size=int(attention_dp_degree),
    )
    _, v4 = _load_deepseek_v4_weights(cfg)
    out_dir = prepared_weight_rank_dir(dst_root, v4)
    if out_dir.exists() and not overwrite:
        if (out_dir / "metadata.json").exists():
            print(f"[skip] path={out_dir}", flush=True)
            return out_dir
        raise RuntimeError(f"{out_dir} already exists but has no metadata.json")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    plan = V4LoadPlan.sampled_blockwise_fp8()
    fp8_noscale = _snapshot_is_fp8_noscale(src)
    if not fp8_noscale:
        raise RuntimeError(
            f"{src} is not a DSV4 no-scale FP8 snapshot "
            "(metadata dsv4_conversion='neuron_fp8_noscale')"
        )

    reader = V4WeightReader(src)
    bytes_written = 0
    last_log = t0
    try:
        dense = V4DeviceWeights(fp8_noscale_snapshot=True)
        _stage_v4_dense_hc_weights(
            reader,
            dense,
            v4,
            plan=plan,
            upload_fn=_stage_upload,
        )
        dense_tensors = _collect_staged(dense, _DENSE_PREPARED_KEYS)
        save_file(dense_tensors, str(out_dir / "dense.safetensors"))
        bytes_written += _count_bytes(dense)

        n_layers = int(v4.num_hidden_layers)
        for layer_id in range(n_layers):
            layer_t0 = time.perf_counter()
            layer = _stage_v4_layer_weights(
                reader,
                v4,
                plan=plan,
                layer_id=layer_id,
                fp8_noscale=fp8_noscale,
                upload_fn=_stage_upload,
            )
            layer_tensors = _collect_staged(layer, _LAYER_PREPARED_KEYS)
            save_file(
                layer_tensors,
                str(out_dir / f"layer_{int(layer.layer_id):03d}.safetensors"),
            )
            bytes_written += _count_bytes(layer)
            now = time.perf_counter()
            done = layer_id + 1
            if done == 1 or done == n_layers or now - last_log >= 30.0:
                print(
                    f"[rank-progress] path={out_dir} layers={done}/{n_layers} "
                    f"bytes={bytes_written / 1e9:.3f}GB "
                    f"layer_elapsed={now - layer_t0:.1f}s "
                    f"elapsed={now - t0:.1f}s",
                    flush=True,
                )
                last_log = now
            del layer_tensors, layer
    finally:
        reader.close()

    stage_elapsed = time.perf_counter() - t0

    metadata = {
        "version": _DSV4_PREPARED_WEIGHT_CACHE_VERSION,
        "source": str(src),
        "num_hidden_layers": int(v4.num_hidden_layers),
        "tp_degree": int(v4.tp_degree),
        "tp_rank": int(v4.tp_rank),
        "ep_degree": int(v4.ep_degree),
        "replica_degree": int(v4.replica_degree),
        "attention_lane": int(v4.attention_lane),
        "local_expert_ids": list(v4.local_expert_ids),
        "bytes": int(bytes_written),
        "stage_elapsed_s": round(stage_elapsed, 3),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(
        f"[prepared] path={out_dir} layers={int(v4.num_hidden_layers)} "
        f"bytes={metadata['bytes'] / 1e9:.3f}GB stage_elapsed={stage_elapsed:.1f}s",
        flush=True,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare rank-local DSV4 safetensors for fast runtime upload.",
    )
    ap.add_argument(
        "--src", type=Path, required=True, help="Converted DSV4 product snapshot."
    )
    ap.add_argument(
        "--dst-root", type=Path, required=True, help="Prepared-weight cache root."
    )
    ap.add_argument(
        "--num-layers", type=int, default=None, help="Limit layers for benchmarking."
    )
    ap.add_argument("--tp-degree", type=int, default=8)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--ep-degree", type=int, default=8)
    ap.add_argument("--replica-degree", type=int, default=2)
    ap.add_argument("--attention-dp-degree", type=int, default=16)
    ap.add_argument("--request-lane-rank", type=int, default=0)
    ap.add_argument(
        "--all-unique-ranks",
        action="store_true",
        help="Prepare all replica-unique rank caches: "
        "lanes 0..ep_degree-1 for every TP rank. "
        "Replica>0 runtime lanes reuse these directories.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel rank-preparation workers for --all-unique-ranks.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    total_t0 = time.perf_counter()
    if args.all_unique_ranks:
        tasks = [
            {
                "src": args.src,
                "dst_root": args.dst_root,
                "num_layers": args.num_layers,
                "tp_degree": int(args.tp_degree),
                "tp_rank": tp_rank,
                "ep_degree": int(args.ep_degree),
                "replica_degree": int(args.replica_degree),
                "attention_dp_degree": int(args.attention_dp_degree),
                "request_lane_rank": lane,
                "overwrite": bool(args.overwrite),
            }
            for lane in range(int(args.ep_degree))
            for tp_rank in range(int(args.tp_degree))
        ]
        count = 0
        jobs = max(1, int(args.jobs))
        if jobs == 1:
            for task in tasks:
                _prepare_one_rank(**task)
                count += 1
        else:
            print(
                f"[total] preparing_unique_ranks={len(tasks)} jobs={jobs}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(_prepare_one_rank, **task) for task in tasks]
                for fut in as_completed(futures):
                    fut.result()
                    count += 1
                    print(
                        f"[total-progress] prepared_unique_ranks={count}/{len(tasks)} "
                        f"elapsed={time.perf_counter() - total_t0:.1f}s",
                        flush=True,
                    )
        print(
            f"[total] prepared_unique_ranks={count} "
            f"elapsed={time.perf_counter() - total_t0:.1f}s",
            flush=True,
        )
        return 0

    _prepare_one_rank(
        src=args.src,
        dst_root=args.dst_root,
        num_layers=args.num_layers,
        tp_degree=int(args.tp_degree),
        tp_rank=int(args.tp_rank),
        ep_degree=int(args.ep_degree),
        replica_degree=int(args.replica_degree),
        attention_dp_degree=int(args.attention_dp_degree),
        request_lane_rank=int(args.request_lane_rank),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
