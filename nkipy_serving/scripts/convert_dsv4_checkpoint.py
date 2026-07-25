"""Pre-convert a DeepSeek-V4-Flash HF checkpoint for product serving.

Input: a raw HF snapshot containing ``model.safetensors.index.json`` and the
original MXFP4/FP8 shard files.

Output: a converted serving snapshot, commonly named
``DeepSeek-V4-Flash-neuron-fp8-noscale``. This directory is the ``--src`` input
for ``scripts.prepare_dsv4_rank_weights`` and can also be used as the runtime
checkpoint/tokenizer source.

What this does (offline, once per checkpoint):

  1. For every **routed expert** tensor (``layers.L.ffn.experts.E.{w1,w2,w3}``
     and ``mtp.M.ffn.experts.E.{w1,w2,w3}``), read MXFP4 I8 + UE8M0 scale
     and emit only ``.weight`` as Neuron-range FP8 E4M3 ``[out, in]``.

  2. For every **shared-expert** tensor
     (``*.ffn.shared_experts.{w1,w2,w3}``), read FP8 E4M3 + UE8M0/fp32
     scale and emit only ``.weight`` as BF16.

  3. Every other tensor (BF16, FP32, I64 ``tid2eid``, ``gate.bias``,
     ``hc_head_*``, attention/compressor FP8 pairs, ...) is byte-copied
     unchanged.

The product direct loader uses ``V4LoadPlan.sampled_blockwise_fp8()`` and
consumes these tensors without runtime MoE scales or duplicate expert formats.
MTP remains fail-closed on the direct path.

Parallelized per shard via ``ProcessPoolExecutor``: every input shard is
independent (weight and its paired scale always live in the same shard,
verified against the real HF checkpoint). Output file names match input
file names so callers can point directly at the new directory.

Usage
-----
Run from the ``nkipy_serving/`` package directory (``scripts/`` is a package there)::

    uv run python -m scripts.convert_dsv4_checkpoint \\
        --src  /path/to/hf/snapshots/<rev> \\
        --dst  /path/to/neuron-fp8-noscale-snapshot \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

from nkipy_serving.models.deepseek_v4.device_weights import (
    _DSV4_CONVERSION_FP8_NOSCALE,
    _cast_mxfp4_to_neuron_fp8_noscale,
)
from nkipy_serving.models.deepseek_v4.weight_reader import (
    _ShardReader,
    dequant_fp8_block,
)


def _conversion_tag() -> str:
    return str(_DSV4_CONVERSION_FP8_NOSCALE)


# -- Safetensors dtype strings we care about -------------------------------

_F8_E4M3_TAG = "F8_E4M3"
_I8_TAG = "I8"


# -------------------------------------------------------------------------


def _is_routed_expert_weight(name: str) -> bool:
    """Routed expert weight under ``{prefix}.ffn.experts.{E}.{w1,w2,w3}.weight``."""
    if not name.endswith(".weight"):
        return False
    parts = name.split(".")
    if ".ffn.experts." not in name:
        return False
    # Last 2 parts are "{w1,w2,w3}" and "weight".
    if len(parts) < 5:
        return False
    return parts[-2] in ("w1", "w2", "w3") and parts[-1] == "weight"


def _is_shared_expert_weight(name: str) -> bool:
    """Shared expert weight under ``{prefix}.ffn.shared_experts.{w1,w2,w3}.weight``."""
    if not name.endswith(".weight") or ".ffn.shared_experts." not in name:
        return False
    parts = name.split(".")
    return len(parts) >= 4 and parts[-2] in ("w1", "w2", "w3")


def _scale_key_for_weight(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"weight key must end with .weight: {weight_key}")
    return weight_key[: -len(".weight")] + ".scale"


def _weight_key_for_scale(scale_key: str) -> str:
    if not scale_key.endswith(".scale"):
        raise ValueError(f"scale key must end with .scale: {scale_key}")
    return scale_key[: -len(".scale")] + ".weight"


def _is_converted_scale_key(name: str, specs: dict[str, Any]) -> bool:
    if not name.endswith(".scale"):
        return False
    weight_key = _weight_key_for_scale(name)
    if weight_key not in specs:
        return False
    weight_tag = specs[weight_key].dtype_tag
    if _is_routed_expert_weight(weight_key):
        return weight_tag == _I8_TAG
    if _is_shared_expert_weight(weight_key):
        return weight_tag == _F8_E4M3_TAG
    return False


def _convert_shard(shard_path: Path, out_path: Path) -> dict[str, Any]:
    """Convert one safetensors shard.

    Returns a dict mapping tensor-name → dtype/shape info for the new index.
    """
    reader = _ShardReader(shard_path)
    specs = reader.specs()

    tensors_out: dict[str, np.ndarray] = {}
    handled_scale_keys: set[str] = set()

    stats = {"mxfp4_fp8": 0, "fp8_bf16": 0, "passthrough": 0}

    for name, spec in specs.items():
        if name in handled_scale_keys:
            continue
        if _is_converted_scale_key(name, specs):
            handled_scale_keys.add(name)
            continue
        scale_key = _scale_key_for_weight(name) if name.endswith(".weight") else None

        if _is_routed_expert_weight(name) and spec.dtype_tag == _I8_TAG:
            # MXFP4 routed expert. Fetch sibling scale and cast to no-scale
            # Neuron FP8. Runtime carries no scale tensor for routed MoE.
            if scale_key is None or scale_key not in specs:
                raise RuntimeError(
                    f"Missing sibling scale for MXFP4 expert {name!r} in {shard_path.name}"
                )
            w_i8 = np.ascontiguousarray(reader.raw(name))
            s_e8 = np.ascontiguousarray(reader.raw(scale_key))
            tensors_out[name] = np.ascontiguousarray(
                _cast_mxfp4_to_neuron_fp8_noscale(w_i8, s_e8)
            )
            handled_scale_keys.add(scale_key)
            stats["mxfp4_fp8"] += 1
            continue

        if _is_routed_expert_weight(name) and spec.dtype_tag == _F8_E4M3_TAG:
            if scale_key is not None and scale_key in specs:
                raise RuntimeError(
                    f"Routed expert {name!r} is FP8+scale in {shard_path.name}; "
                    "product conversion expects HF MXFP4 routed experts."
                )

        if _is_shared_expert_weight(name) and spec.dtype_tag == _F8_E4M3_TAG:
            # Shared experts are small enough to keep BF16 in product serving.
            if scale_key is None or scale_key not in specs:
                raise RuntimeError(
                    f"Missing sibling scale for FP8 weight {name!r} in {shard_path.name}"
                )
            w = np.ascontiguousarray(reader.raw(name))
            s = np.ascontiguousarray(reader.raw(scale_key))
            tensors_out[name] = np.ascontiguousarray(
                dequant_fp8_block(w, s).astype(ml_dtypes.bfloat16)
            )
            handled_scale_keys.add(scale_key)
            stats["fp8_bf16"] += 1
            continue

        # Everything else: byte-copy. This also catches standalone E8M0 scales
        # whose paired weight is missing (shouldn't happen on a clean HF
        # checkpoint, but safe to pass them through if they do).
        tensors_out[name] = np.ascontiguousarray(reader.raw(name))
        stats["passthrough"] += 1

    reader.close()

    # Metadata helps downstream tools identify the converted shard without
    # having to re-run the detection heuristic.
    metadata = {
        "format": "pt",
        "dsv4_conversion": _conversion_tag(),
    }
    save_file(tensors_out, str(out_path), metadata=metadata)

    # Build a per-tensor entry for the new index.
    new_entries: dict[str, Any] = {}
    with open(out_path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        new_entries[k] = {"dtype": v["dtype"], "shape": v["shape"]}
    return {"tensors": new_entries, "stats": stats}


def _convert_one(src_dst: tuple[Path, Path]) -> tuple[str, dict[str, Any], float]:
    src, dst = src_dst
    t0 = time.perf_counter()
    result = _convert_shard(src, dst)
    return src.name, result, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-convert DSV4 HF checkpoint MoE weights to product FP8/BF16.",
    )
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the HF snapshot (directory containing "
        "model.safetensors.index.json).",
    )
    ap.add_argument(
        "--dst", type=Path, required=True, help="Output directory. Created if missing."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of parallel worker processes.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="List shards and exit without writing."
    )
    args = ap.parse_args(argv)

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        print(f"--src {src} is not a directory", file=sys.stderr)
        return 2

    # Resolve both sides so ``a/./b`` vs ``a/b`` and relative-vs-absolute
    # paths compare correctly. Writing into the source directory would
    # destructively rewrite the original HF index/shards in place.
    try:
        src_resolved = src.resolve(strict=True)
    except FileNotFoundError:
        print(f"--src {src} does not exist", file=sys.stderr)
        return 2
    dst_resolved = dst.resolve(strict=False)
    if src_resolved == dst_resolved:
        print(
            f"--dst must differ from --src ({src_resolved}); refusing to "
            "overwrite the source snapshot in place.",
            file=sys.stderr,
        )
        return 2

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Missing {index_path}", file=sys.stderr)
        return 2

    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: dict[str, str] = idx["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    print(f"[src]  {src}")
    print(f"[dst]  {dst}")
    print(f"[num_shards]  {len(shard_names)}")
    print(f"[workers]  {args.workers}")
    if args.dry_run:
        for n in shard_names:
            print(f"  {n}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    # Copy sidecar files (config.json, tokenizer, README...) that aren't
    # safetensors shards. We never mutate these.
    sidecars = [
        p
        for p in src.iterdir()
        if p.is_file()
        and p.name != "model.safetensors.index.json"
        and not p.name.endswith(".safetensors")
    ]
    for side in sidecars:
        target = dst / side.name
        if not target.exists():
            shutil.copy2(side, target)
            print(f"[copy]  {side.name}")

    # Kick off shard conversion. Each shard is CPU-bound and holds its own
    # mmap of one input file, so parallelism scales with core count up to
    # memory bandwidth. When workers==1 we run inline — useful for tests
    # whose test runner can't pickle the worker function across processes.
    pairs = [(src / name, dst / name) for name in shard_names]
    new_weight_map: dict[str, str] = {}
    total_t0 = time.perf_counter()

    def _record(
        shard_name: str, result: dict[str, Any], elapsed: float, done: int
    ) -> None:
        s = result["stats"]
        print(
            f"[{done:>3d}/{len(pairs)}] {shard_name}  "
            f"mxfp4_fp8={s['mxfp4_fp8']:<4d} "
            f"fp8_bf16={s['fp8_bf16']:<4d} "
            f"passthrough={s['passthrough']:<4d} took {elapsed:.1f}s",
            flush=True,
        )
        for tensor_name in result["tensors"]:
            new_weight_map[tensor_name] = shard_name

    if args.workers <= 1:
        for i, pair in enumerate(pairs, start=1):
            shard_name, result, elapsed = _convert_one(pair)
            _record(shard_name, result, elapsed, i)
    else:
        # "fork" workers inherit the parent's imports; "spawn" workers must
        # re-import the script module, which fails when the script is loaded
        # via importlib (e.g., in tests). Linux defaults to "fork", but macOS
        # and Python 3.14+ default to "spawn" — force fork where available.
        mp_ctx = (
            multiprocessing.get_context("fork")
            if "fork" in multiprocessing.get_all_start_methods()
            else multiprocessing.get_context()
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_ctx,
        ) as pool:
            futures = {pool.submit(_convert_one, pair): pair for pair in pairs}
            done = 0
            for fut in as_completed(futures):
                shard_name, result, elapsed = fut.result()
                done += 1
                _record(shard_name, result, elapsed, done)

    total = time.perf_counter() - total_t0
    print(f"[total]  {total:.1f}s on {args.workers} worker(s)")

    # Rewrite the index.
    out_index = {
        "metadata": dict(idx.get("metadata", {})),
        "weight_map": new_weight_map,
    }
    out_index["metadata"]["dsv4_conversion"] = _conversion_tag()
    with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(out_index, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
