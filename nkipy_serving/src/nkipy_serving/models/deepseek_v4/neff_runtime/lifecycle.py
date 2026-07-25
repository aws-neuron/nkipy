"""Runtime utility helpers for DSV4 product execution."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping

import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.diagnostics import (
    rank_trace_allowed,
    warmup_trace_enabled,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _canonical_product_kernel_cache_key,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    _PRODUCT_KERNEL_CACHE_FIELDS,
    _PRODUCT_PREALLOCATED_OUTPUT_FIELDS,
    Dsv4ProductBucket,
)
from nkipy_serving.models.deepseek_v4.rank_layout import coord_for_rank
from nkipy_serving.runtime.device_tensor import (
    normalize_dtype as _normalize_dtype,
)
from nkipy_serving.runtime.device_tensor import (
    sample_like as _runtime_sample_like,
)

logger = logging.getLogger(__name__)


def _product_warmup_trace(coord: Any, message: str) -> None:
    if not warmup_trace_enabled():
        return
    rank = int(getattr(coord, "rank", -1))
    if not rank_trace_allowed(rank):
        return
    col = int(getattr(coord, "col", -1))
    ep = int(getattr(coord, "row_in_replica", -1))
    lane = int(getattr(coord, "attn_lane", -1))
    logger.info(
        "DSV4 product forward rank=%d tp=%d ep=%d lane=%d %s",
        rank,
        col,
        ep,
        lane,
        message,
    )
    # File mirror (survives a native worker crash that drops logger buffers; the
    # 2neff warmup dies via SIGABRT before logger.info is flushed). Mirrors the
    # barrier-trace file mechanism in collective_load.py. Line-buffered append +
    # flush so the last reached stage per rank is durable.
    if os.getenv("NKIPY_SERVING_DSV4_WARMUP_TRACE_FILE"):
        try:
            with open("/tmp/_dsv4_warmup_trace.log", "a") as _wt:
                _wt.write(f"rank={rank} tp={col} ep={ep} lane={lane} {message}\n")
                _wt.flush()
        except Exception:
            pass


def _product_executor_coord(executor: Any) -> Any:
    coord = getattr(executor, "_coord", None)
    if coord is not None:
        return coord
    runtime_surface = getattr(executor, "runtime_surface", None)
    model_config = getattr(runtime_surface, "model_config", None)
    runtime_config = getattr(model_config, "runtime_config", None)

    def _int_attr(owner: Any, name: str) -> int | None:
        if owner is None or not hasattr(owner, name):
            return None
        value = getattr(owner, name)
        if value is None:
            return None
        return int(value)

    v4 = getattr(runtime_surface, "v4", None)
    tp_degree = (
        _int_attr(runtime_config, "tp_degree")
        or _int_attr(model_config, "tp_degree")
        or _int_attr(v4, "tp_degree")
    )
    ep_degree = (
        _int_attr(runtime_config, "ep_degree")
        or _int_attr(model_config, "ep_degree")
        or _int_attr(v4, "ep_degree")
    )
    replica_degree = (
        _int_attr(runtime_config, "replica_degree")
        or _int_attr(model_config, "replica_degree")
        or _int_attr(v4, "replica_degree")
    )
    if tp_degree is None or ep_degree is None or replica_degree is None:
        return None
    request_lane = _int_attr(model_config, "request_lane_rank")
    tp_rank = _int_attr(model_config, "tp_rank")
    if request_lane is None:
        request_lane = _int_attr(v4, "attention_lane")
    if tp_rank is None:
        tp_rank = _int_attr(v4, "tp_rank")
    if request_lane is None or tp_rank is None:
        return None
    global_rank = int(request_lane) * int(tp_degree) + int(tp_rank)
    return coord_for_rank(
        rank=global_rank,
        tp_degree=int(tp_degree),
        ep_degree=int(ep_degree),
        replica_degree=int(replica_degree),
    )


def _blockwise_moe_ep_tp_groups(state: Any | None) -> tuple[tuple[int, ...], ...]:
    if (
        state is None
        or not hasattr(state, "ep_degree")
        or not hasattr(state, "tp_degree")
    ):
        return ()
    if (
        int(getattr(state, "ep_degree", 1) or 1) <= 1
        and int(getattr(state, "tp_degree", 1) or 1) <= 1
    ):
        return ()
    from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
        _moe_ep_tp_replica_groups_for_collective,
    )

    groups = _moe_ep_tp_replica_groups_for_collective(state)
    if not groups:
        return ()
    return tuple(tuple(int(rank) for rank in group) for group in groups)


def _is_device_value(value: Any) -> bool:
    module = type(value).__module__
    return (
        hasattr(value, "tensor_ref")
        and hasattr(value, "shape")
        and (module.startswith("nkipy.") or module.startswith("spike."))
    )


def _sample_array(value: Any, *, fallback_dtype: Any = np.float32) -> np.ndarray:
    """Build a host sample input for DeviceKernel tracing."""
    return _runtime_sample_like(value, fill="zeros", fallback_dtype=fallback_dtype)


def _as_product_device_input(value: Any, *, name: str) -> Any:
    """Return a DeviceTensor input for raw product DeviceKernel calls."""
    if _is_device_value(value):
        return value
    return _get_device_tensor_cls().from_numpy(
        np.ascontiguousarray(np.asarray(value)),
        name=name,
    )


def _require_product_device_value(value: Any, *, where: str) -> Any:
    if not _is_device_value(value):
        raise RuntimeError(
            f"DSV4 product {where} requires a DeviceTensor input, "
            f"got {type(value).__name__}"
        )
    return value


def _host_array_signature(value: Any) -> tuple[tuple[int, ...], str, int]:
    arr = np.asarray(value)
    return (tuple(int(dim) for dim in arr.shape), str(arr.dtype), id(value))


def _value_dtype(value: Any, *, fallback: Any = np.float32) -> Any:
    return _normalize_dtype(getattr(value, "dtype", None), fallback)


def _warmup_tensor_signature(value: Any) -> tuple[tuple[int, ...], str] | None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    return (
        tuple(int(dim) for dim in getattr(value, "shape", ())),
        str(getattr(value, "dtype", "unknown")),
    )


def _warmup_object_tensor_signatures(obj: Any) -> tuple[tuple[str, Any], ...]:
    """Return shape/dtype signatures for direct tensor fields on an object."""
    try:
        items = vars(obj).items()
    except TypeError:
        return ()
    sigs: list[tuple[str, Any]] = []
    for name, value in items:
        sig = _warmup_tensor_signature(value)
        if sig is not None:
            sigs.append((str(name), sig))
    return tuple(sorted(sigs))


def _warmup_object_scalar_signatures(
    obj: Any,
    names: tuple[str, ...],
) -> tuple[tuple[str, Any], ...]:
    sigs: list[tuple[str, Any]] = []
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, (bool, int, float, str, type(None))):
            sigs.append((name, value))
    return tuple(sigs)


class Dsv4ProductManifestMixin:
    def product_compile_manifest(self) -> dict[str, Any]:
        warmup_stats = getattr(
            self,
            "_product_warmup_stats",
            getattr(self, "_product_last_warmup_stats", None),
        )
        return build_product_compile_manifest(
            bucket_registry=self._product_bucket_registry(),
            warmup_stats=warmup_stats,
            sealed=bool(getattr(self, "_product_manifest_sealed", False)),
        )

    def seal_product_compile_manifest(self) -> dict[str, Any]:
        """Freeze product warmup coverage and reject future late compiles."""
        self._product_manifest_sealed = True
        self._product_manifest_snapshot = self.product_compile_manifest()
        rank_id = self._product_manifest_log_rank()
        if rank_id in (None, 0):
            summary = self._product_manifest_snapshot.get("summary", {})
            logger.info(
                "DSV4 product compile manifest sealed: token_buckets=%s "
                "warmup_dedup=%s kernel_cache_total=%s scratch_outputs=%s",
                sorted(self._product_manifest_snapshot["token_buckets"]),
                self._product_manifest_snapshot.get("warmup_dedup", {}),
                summary.get("kernel_cache_total", 0),
                summary.get("scratch_outputs", 0),
            )
        return self._product_manifest_snapshot

    def _product_manifest_log_rank(self) -> int | None:
        return product_manifest_log_rank(getattr(self, "graph", {}))

    def _collective_graph_metadata(
        self,
        graph_key: str,
        *,
        where: str,
        base: Any | None = None,
    ) -> tuple[int, int]:
        if base is None:
            graph = getattr(self, "graph", {})
            base = graph.get(graph_key) if isinstance(graph, dict) else None
        rank_id = getattr(base, "_rank_id", None)
        world_size = getattr(base, "_world_size", None)
        if rank_id is None or world_size is None:
            raise RuntimeError(
                f"DSV4 product {where} collective requires graph "
                f"{graph_key} rank_id/world_size for DeviceKernel load"
            )
        return int(rank_id), int(world_size)

    def _require_unsealed_product_manifest(
        self,
        *,
        bucket: Dsv4ProductBucket,
        cache_name: str,
        key: tuple[Any, ...],
    ) -> None:
        require_unsealed_product_manifest(
            sealed=bool(getattr(self, "_product_manifest_sealed", False)),
            bucket=bucket,
            cache_name=cache_name,
            key=key,
        )

    def _cached_product_kernel(
        self,
        *,
        bucket: Dsv4ProductBucket,
        cache_name: str,
        key: tuple[Any, ...],
        compile_kernel: Callable[[], Any],
    ) -> Any:
        """Memoize a compiled product kernel on ``bucket.kernel_caches`` by ``key``.

        Wraps the get → manifest-unseal-guard → compile → store sequence shared by
        every ``_*_kernel_for`` builder. ``key`` is passed through opaquely (the
        caller owns its exact shape so the compile cache stays correct), and
        ``compile_kernel`` is only invoked on a miss.
        """
        cache = bucket.kernel_caches[cache_name]
        cached = cache.get(key)
        if cached is not None:
            return cached
        self._require_unsealed_product_manifest(
            bucket=bucket,
            cache_name=cache_name,
            key=key,
        )
        kernel = compile_kernel()
        cache[key] = kernel
        return kernel


def build_product_compile_manifest(
    *,
    bucket_registry: Mapping[int, Dsv4ProductBucket],
    warmup_stats: Mapping[str, Any] | None,
    sealed: bool,
) -> dict[str, Any]:
    """Return the product bucket/kernel cache manifest."""
    buckets: dict[str, Any] = {}
    for token_bucket, bucket in sorted(bucket_registry.items()):
        kernel_counts = {
            name: len(bucket.kernel_caches[name])
            for name in _PRODUCT_KERNEL_CACHE_FIELDS
        }
        preallocated_counts = {
            name: len(tuple(getattr(bucket, name)))
            for name in _PRODUCT_PREALLOCATED_OUTPUT_FIELDS
        }
        bucket_manifest = dict(kernel_counts)
        bucket_manifest.update(
            {
                "token_bucket": int(token_bucket),
                "max_requests": int(bucket.max_requests),
                "kernel_cache_total": int(sum(kernel_counts.values())),
                "kernel_cache_nonzero": {
                    name: int(count)
                    for name, count in kernel_counts.items()
                    if int(count) > 0
                },
                "scratch_outputs": int(len(bucket.scratch_outputs)),
                "preallocated_outputs": preallocated_counts,
                "preallocated_output_total": int(sum(preallocated_counts.values())),
            }
        )
        buckets[str(int(token_bucket))] = bucket_manifest

    summary = {
        "bucket_count": len(buckets),
        "kernel_cache_total": int(
            sum(int(bucket.get("kernel_cache_total", 0)) for bucket in buckets.values())
        ),
        "scratch_outputs": int(
            sum(int(bucket.get("scratch_outputs", 0)) for bucket in buckets.values())
        ),
        "preallocated_output_total": int(
            sum(
                int(bucket.get("preallocated_output_total", 0))
                for bucket in buckets.values()
            )
        ),
    }
    return {
        "sealed": bool(sealed),
        "token_buckets": buckets,
        "summary": summary,
        "warmup_dedup": dict(warmup_stats) if warmup_stats is not None else {},
    }


def product_manifest_log_rank(graph: Any) -> int | None:
    for value in getattr(graph, "values", lambda: ())():
        rank_id = getattr(value, "_rank_id", None)
        if rank_id is not None:
            return int(rank_id)
    return None


def require_unsealed_product_manifest(
    *,
    sealed: bool,
    bucket: Dsv4ProductBucket,
    cache_name: str,
    key: tuple[Any, ...],
) -> None:
    key = _canonical_product_kernel_cache_key(key)
    if not bool(sealed):
        return
    key_s = repr(key)
    if len(key_s) > 512:
        key_s = key_s[:509] + "..."
    cache = getattr(bucket, cache_name, None)
    nearest_s = ""
    if isinstance(cache, dict) and cache:
        nearest_key = None
        nearest_score: int | None = None
        for candidate in cache.keys():
            if not isinstance(candidate, tuple):
                continue
            shared = min(len(candidate), len(key))
            mismatches = sum(1 for idx in range(shared) if candidate[idx] != key[idx])
            score = mismatches + abs(len(candidate) - len(key))
            if nearest_score is None or score < nearest_score:
                nearest_score = score
                nearest_key = candidate
        if nearest_key is not None:
            diffs: list[str] = []
            shared = min(len(nearest_key), len(key))
            for idx in range(shared):
                if nearest_key[idx] != key[idx]:
                    want_s = repr(key[idx])
                    have_s = repr(nearest_key[idx])
                    # Truncating at fixed width hides differences deep inside a
                    # long nested tuple — slice to a window around the FIRST
                    # differing character so the diff is always visible.
                    pos = next(
                        (
                            j
                            for j in range(min(len(want_s), len(have_s)))
                            if want_s[j] != have_s[j]
                        ),
                        min(len(want_s), len(have_s)),
                    )
                    lo = max(0, pos - 40)
                    diffs.append(
                        f"{idx}:want=..{want_s[lo : pos + 80]} "
                        f"have=..{have_s[lo : pos + 80]}"
                    )
                    if len(diffs) >= 12:
                        break
            if len(nearest_key) != len(key):
                diffs.append(f"len:want={len(key)} have={len(nearest_key)}")
            nearest_s = (
                f", cache_size={len(cache)}, nearest_score={nearest_score}, "
                f"nearest_diffs=[{'; '.join(diffs)}]"
            )
    raise RuntimeError(
        "DSV4 product late kernel compile blocked after warmup seal: "
        f"token_bucket={int(bucket.token_bucket)}, cache={cache_name}, "
        f"key={key_s}{nearest_s}"
    )


class Dsv4ProductWarmupMixin:
    def needs_dp_attention_collective_warmup_prepass(self) -> bool:
        """Product collective kernels use explicit load barriers, not full prepass."""
        return False

    def begin_product_warmup(self) -> None:
        """Enable representative-layer warmup deduplication.

        Product kernel caches are keyed by tensor shapes and compile-time
        constants, not by layer id. Full V4 repeats the same few layer shapes
        many times, so startup only needs to execute one synthetic layer per
        distinct signature to populate the manifest.
        """
        begin_product_warmup_state(self)

    def end_product_warmup(self) -> None:
        """Disable representative-layer warmup deduplication."""
        end_product_warmup_state(self)

    def _should_skip_hc_layer_for_warmup(
        self,
        *,
        layer_id: int,
        block: Any,
        h: Any,
        metadata: Any | None,
        start_pos: int,
        token_bucket: int | None,
        is_decode: bool,
    ) -> bool:
        return should_skip_hc_layer_for_warmup(
            self,
            layer_id=int(layer_id),
            block=block,
            h=h,
            metadata=metadata,
            start_pos=int(start_pos),
            token_bucket=token_bucket,
            is_decode=bool(is_decode),
        )


def begin_product_warmup_state(executor: Any) -> None:
    """Enable representative-layer warmup deduplication."""
    executor._product_warmup_seen_layer_signatures = set()
    executor._product_warmup_stats = {
        "enabled": True,
        "executed_layers": 0,
        "skipped_layers": 0,
        "signatures": 0,
    }


def end_product_warmup_state(executor: Any) -> None:
    """Disable representative-layer warmup deduplication."""
    seen = getattr(executor, "_product_warmup_seen_layer_signatures", None)
    stats = getattr(executor, "_product_warmup_stats", None)
    if stats is not None:
        last_stats = dict(stats)
        last_stats["signatures"] = len(seen) if seen is not None else 0
        executor._product_last_warmup_stats = last_stats
    if hasattr(executor, "_product_warmup_seen_layer_signatures"):
        delattr(executor, "_product_warmup_seen_layer_signatures")
    if hasattr(executor, "_product_warmup_stats"):
        delattr(executor, "_product_warmup_stats")


def _product_warmup_layer_signature(
    executor: Any,
    *,
    layer_id: int,
    block: Any,
    h: Any,
    metadata: Any | None,
    start_pos: int,
    token_bucket: int | None,
    is_decode: bool,
) -> tuple[Any, ...]:
    base = executor._base_metadata(metadata)
    batch_size = (
        int(getattr(base, "batch_size"))
        if base is not None and hasattr(base, "batch_size")
        else None
    )
    total_tokens = (
        int(getattr(base, "total_tokens"))
        if base is not None and hasattr(base, "total_tokens")
        else None
    )
    query_lens = None
    if base is not None and hasattr(base, "query_start_loc"):
        qsl = np.asarray(getattr(base, "query_start_loc"), dtype=np.int64).reshape(-1)
        if batch_size is not None and qsl.shape[0] >= batch_size + 1:
            query_lens = tuple(
                int(v) for v in (qsl[1 : batch_size + 1] - qsl[:batch_size])
            )
    args = getattr(getattr(executor, "runtime_surface", None), "args", None)
    attn = getattr(block, "attn", None)
    compressor = getattr(attn, "compressor", None)
    indexer = getattr(attn, "indexer", None)
    indexer_compressor = getattr(indexer, "compressor", None)
    ffn = getattr(block, "ffn", None)
    gate = getattr(ffn, "gate", None)
    shared = getattr(ffn, "shared", None)
    experts = tuple(getattr(ffn, "experts", ()) or ())
    expert_sigs = tuple(
        sorted(
            {
                (
                    _warmup_object_scalar_signatures(
                        expert,
                        ("swiglu_limit",),
                    ),
                    _warmup_object_tensor_signatures(expert),
                )
                for expert in experts
            }
        )
    )
    blockwise_layer = None
    blockwise_state = getattr(executor, "blockwise_moe_state", None)
    blockwise_layers = tuple(getattr(blockwise_state, "layers", ()) or ())
    if 0 <= int(layer_id) < len(blockwise_layers):
        blockwise_layer = blockwise_layers[int(layer_id)]
    return (
        ("token_bucket", int(token_bucket) if token_bucket is not None else None),
        ("decode", bool(is_decode)),
        ("start_pos", int(start_pos)),
        ("batch_size", batch_size),
        ("total_tokens", total_tokens),
        ("query_lens", query_lens),
        ("h", _warmup_tensor_signature(h)),
        (
            "args",
            _warmup_object_scalar_signatures(
                args,
                (
                    "dim",
                    "n_heads",
                    "n_hash_layers",
                    "n_routed_experts",
                    "n_activated_experts",
                    "n_shared_experts",
                    "q_lora_rank",
                    "o_lora_rank",
                    "o_groups",
                    "head_dim",
                    "rope_head_dim",
                    "window_size",
                    "index_n_heads",
                    "index_head_dim",
                    "index_topk",
                    "hc_mult",
                    "hc_sinkhorn_iters",
                    "hc_eps",
                    "norm_eps",
                    "swiglu_limit",
                    "routed_scaling_factor",
                    "scoring_func",
                    "topk_method",
                    "moe_inter_dim",
                ),
            ),
        ),
        (
            "block",
            type(block).__name__,
            _warmup_object_tensor_signatures(block),
        ),
        (
            "attn",
            type(attn).__name__,
            _warmup_object_scalar_signatures(
                attn,
                (
                    "n_heads",
                    "head_dim",
                    "rope_head_dim",
                    "n_groups",
                    "window_size",
                    "compress_ratio",
                    "eps",
                    "softmax_scale",
                ),
            ),
            _warmup_object_tensor_signatures(attn),
        ),
        (
            "compressor",
            compressor is not None,
            type(compressor).__name__,
            _warmup_object_scalar_signatures(
                compressor,
                (
                    "dim",
                    "head_dim",
                    "rope_head_dim",
                    "compress_ratio",
                    "overlap",
                    "rotate",
                    "eps",
                ),
            ),
            _warmup_object_tensor_signatures(compressor),
        ),
        (
            "indexer",
            indexer is not None,
            type(indexer).__name__,
            _warmup_object_scalar_signatures(
                indexer,
                (
                    "n_heads",
                    "head_dim",
                    "rope_head_dim",
                    "compress_ratio",
                    "index_topk",
                    "softmax_scale",
                ),
            ),
            _warmup_object_tensor_signatures(indexer),
        ),
        (
            "indexer_compressor",
            indexer_compressor is not None,
            type(indexer_compressor).__name__,
            _warmup_object_scalar_signatures(
                indexer_compressor,
                (
                    "dim",
                    "head_dim",
                    "rope_head_dim",
                    "compress_ratio",
                    "overlap",
                    "rotate",
                    "eps",
                ),
            ),
            _warmup_object_tensor_signatures(indexer_compressor),
        ),
        (
            "ffn",
            type(ffn).__name__,
            _warmup_object_scalar_signatures(
                ffn,
                (
                    "dim",
                    "n_routed_experts",
                    "n_shared_experts",
                    "moe_inter_dim",
                ),
            ),
            _warmup_object_tensor_signatures(ffn),
        ),
        (
            "gate",
            type(gate).__name__,
            _warmup_object_scalar_signatures(
                gate,
                ("is_hash", "topk", "route_scale", "score_func"),
            ),
            _warmup_object_tensor_signatures(gate),
        ),
        (
            "shared",
            type(shared).__name__,
            _warmup_object_scalar_signatures(shared, ("swiglu_limit",)),
            _warmup_object_tensor_signatures(shared),
        ),
        ("experts", len(experts), expert_sigs),
        (
            "blockwise_state",
            _warmup_object_scalar_signatures(
                blockwise_state,
                (
                    "hidden_size",
                    "intermediate_size",
                    "n_local_experts",
                    "experts_per_token",
                    "ep_degree",
                    "tp_degree",
                    "swiglu_limit",
                ),
            ),
        ),
        (
            "blockwise_layer",
            type(blockwise_layer).__name__,
            _warmup_object_scalar_signatures(
                blockwise_layer,
                ("n_local_experts", "hidden_size", "intermediate_size"),
            ),
            _warmup_object_tensor_signatures(blockwise_layer),
        ),
    )


def should_skip_hc_layer_for_warmup(
    executor: Any,
    *,
    layer_id: int,
    block: Any,
    h: Any,
    metadata: Any | None,
    start_pos: int,
    token_bucket: int | None,
    is_decode: bool,
) -> bool:
    seen = getattr(executor, "_product_warmup_seen_layer_signatures", None)
    if seen is None:
        return False
    signature = _product_warmup_layer_signature(
        executor,
        layer_id=int(layer_id),
        block=block,
        h=h,
        metadata=metadata,
        start_pos=int(start_pos),
        token_bucket=token_bucket,
        is_decode=bool(is_decode),
    )
    stats = getattr(executor, "_product_warmup_stats", None)
    if signature in seen:
        if stats is not None:
            stats["skipped_layers"] = int(stats.get("skipped_layers", 0)) + 1
            stats["signatures"] = len(seen)
        return True
    seen.add(signature)
    if stats is not None:
        stats["executed_layers"] = int(stats.get("executed_layers", 0)) + 1
        stats["signatures"] = len(seen)
    return False
