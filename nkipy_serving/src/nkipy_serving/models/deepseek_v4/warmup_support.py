"""Warmup and startup support helpers for the DSV4 executor."""

from __future__ import annotations

import logging
import math
import os
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.diagnostics import (
    current_rank,
    rank_trace_allowed,
    warmup_trace_enabled,
)
from nkipy_serving.profiling import StartupProfiler
from nkipy_serving.runtime.kernel_compile import seal_kernel_compile_namespace

logger = logging.getLogger(__name__)

_DSV4_SUPPORT_KERNEL_NAMESPACES = (
    "dsv4_attention_kernels",
    "dsv4_compressor_kernels",
    "dsv4_indexer_kernels",
    "dsv4_state_kernels",
    "fragment_jit",
    "fragment_jit_collective",
    "moe_schedule",
)


def _seal_dsv4_support_kernel_namespaces() -> None:
    for namespace in _DSV4_SUPPORT_KERNEL_NAMESPACES:
        seal_kernel_compile_namespace(namespace, reason="DSV4 warmup complete")


def _warmup_trace(message: str) -> None:
    if not warmup_trace_enabled():
        return
    if not rank_trace_allowed(current_rank()):
        return
    logger.info("DSV4 warmup %s", message)
    # File mirror survives a native worker crash that drops logger buffers.
    if os.getenv("NKIPY_SERVING_DSV4_WARMUP_TRACE_FILE"):
        try:
            with open("/tmp/_dsv4_warmup_trace.log", "a") as _wt:
                _wt.write(f"rank={int(current_rank())} DSV4 warmup {message}\n")
                _wt.flush()
        except Exception:
            pass


_Dsv4WarmupCompileKey = tuple[Any, ...]
_Dsv4WarmupCompileEntry = tuple[
    _Dsv4WarmupCompileKey,
    str,
    Callable[[], None],
    dict[str, Any],
]

_DSV4_DP_WARMUP_METHODS = (
    "precompile_dp_attention_reduce_paths",
    "precompile_first_layer_embedding_mhc_shapes",
    "precompile_lane_moe_helpers",
    "precompile_shared_expert_restore_post_pre_helpers",
    "precompile_lane_head_helpers",
    "precompile_lane_dp_attention_helpers",
    "precompile_lane_dp_attention_decode_continuation_helpers",
)


def _dsv4_warmup_compile_key(
    family: str,
    name: str,
    *,
    mode: int | str | None = None,
    token_bucket: int | None = None,
    request_bucket: int | None = None,
    metadata_key: tuple[Any, ...] = (),
) -> _Dsv4WarmupCompileKey:
    return (
        str(family),
        str(name),
        None if mode is None else str(mode),
        None if token_bucket is None else int(token_bucket),
        None if request_bucket is None else int(request_bucket),
        tuple(metadata_key),
    )


def _append_dsv4_warmup_compile_entry(
    entries: list[_Dsv4WarmupCompileEntry],
    *,
    family: str,
    name: str,
    stage: str,
    compile_fn: Callable[[], None],
    mode: int | str | None = None,
    token_bucket: int | None = None,
    request_bucket: int | None = None,
    metadata_key: tuple[Any, ...] = (),
    **record_fields: Any,
) -> None:
    key = _dsv4_warmup_compile_key(
        family,
        name,
        mode=mode,
        token_bucket=token_bucket,
        request_bucket=request_bucket,
        metadata_key=metadata_key,
    )
    fields = dict(record_fields)
    fields.setdefault("compile_family", str(family))
    fields.setdefault("compile_name", str(name))
    if token_bucket is not None:
        fields.setdefault("compile_token_bucket", int(token_bucket))
    if request_bucket is not None:
        fields.setdefault("compile_request_bucket", int(request_bucket))
    fields.setdefault("compile_key", str(key))
    entries.append((key, str(stage), compile_fn, fields))


def _run_dsv4_warmup_compile_manifest(
    entries: list[_Dsv4WarmupCompileEntry],
    *,
    manifest_name: str,
    rank_msg: str,
    record_warmup: Callable[..., None],
) -> None:
    if not entries:
        return

    manifest_t0 = time.perf_counter()
    seen: set[_Dsv4WarmupCompileKey] = set()
    compiled = 0
    skipped = 0
    _warmup_trace(
        f"{rank_msg} compile manifest {manifest_name} start entries={len(entries)}"
    )
    for key, stage, compile_fn, fields in entries:
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        stage_t0 = time.perf_counter()
        _warmup_trace(
            f"{rank_msg} compile manifest {manifest_name} entry start "
            f"stage={stage} key={key}"
        )
        try:
            compile_fn()
        except Exception as exc:
            record_warmup(
                f"{stage} failed",
                start=stage_t0,
                error=repr(exc),
                **fields,
            )
            raise
        record_warmup(stage, start=stage_t0, **fields)
        compiled += 1
        _warmup_trace(
            f"{rank_msg} compile manifest {manifest_name} entry done "
            f"stage={stage} key={key}"
        )
    record_warmup(
        f"compile manifest {manifest_name}",
        start=manifest_t0,
        entries=len(entries),
        compiled=compiled,
        skipped_duplicates=skipped,
    )
    _warmup_trace(
        f"{rank_msg} compile manifest {manifest_name} done "
        f"compiled={compiled} skipped_duplicates={skipped}"
    )


def _has_dsv4_dp_warmup_precompile(owner: Any) -> bool:
    return any(
        callable(getattr(owner, method_name, None))
        for method_name in _DSV4_DP_WARMUP_METHODS
    )


class Dsv4WarmupRecorder:
    """Record DSV4 warmup phases with common step fields."""

    def __init__(
        self,
        *,
        rank: int,
        token_paddings: tuple[int, ...],
        bs_paddings: tuple[int, ...],
    ) -> None:
        self.profiler = StartupProfiler(
            "dsv4_warmup",
            rank=int(rank),
            token_buckets=str(tuple(int(v) for v in token_paddings)),
            bs_buckets=str(tuple(int(v) for v in bs_paddings)),
        )

    def record(
        self,
        stage: str,
        *,
        start: float | None = None,
        step: Any | None = None,
        **fields: Any,
    ) -> None:
        if step is not None:
            fields.update(
                step=str(getattr(step, "name", "<unnamed>")),
                forward_mode=int(getattr(step, "forward_mode")),
                token_bucket=int(getattr(step, "input_token_bucket")),
                batch_size=int(getattr(step, "batch_size")),
                real_total_tokens=(
                    None
                    if getattr(step, "real_total_tokens", None) is None
                    else int(getattr(step, "real_total_tokens"))
                ),
                decode_target_pos=(
                    None
                    if getattr(step, "decode_target_pos", None) is None
                    else int(getattr(step, "decode_target_pos"))
                ),
                execute_forward=bool(getattr(step, "execute_forward", True)),
            )
        elapsed = None if start is None else time.perf_counter() - start
        self.profiler.record(stage, elapsed_s=elapsed, **fields)


def _warmup_rank_message(coord: Any) -> str:
    return (
        f"rank={int(getattr(coord, 'rank', -1))} "
        f"tp={int(getattr(coord, 'col', -1))} "
        f"ep={int(getattr(coord, 'row_in_replica', -1))} "
        f"lane={int(getattr(coord, 'attn_lane', -1))}"
    )


def _warmup_kv_pool_shape(kv_pool: Any, runtime_config: Any) -> tuple[int, int]:
    block_size = int(
        getattr(
            kv_pool,
            "block_size",
            getattr(runtime_config, "kv_cache_block_size", 32),
        )
    )
    num_blocks = int(
        getattr(
            kv_pool,
            "num_blocks",
            max(1, int(getattr(kv_pool, "size", block_size)) // block_size),
        )
    )
    return block_size, num_blocks


def _sampled_rectangular_warmup_shape(
    attn_metadata: Any,
) -> tuple[int, int] | None:
    """Shape consumed by `_prepare_sampled_input` after scheduler padding."""
    batch_i = int(getattr(attn_metadata, "batch_size", 0) or 0)
    total_i = int(getattr(attn_metadata, "total_tokens", 0) or 0)
    if batch_i <= 0 or total_i <= 0:
        return None
    if batch_i == 1:
        return 1, int(total_i)
    if total_i % batch_i != 0:
        return None
    return batch_i, total_i // batch_i


def _sampled_warmup_keys(
    step: Any,
    attn_metadata: Any,
    sampled_shape: tuple[int, int] | None,
) -> list[tuple[int, int, int]]:
    sampled_keys: list[tuple[int, int, int]] = []
    if sampled_shape is not None:
        sampled_batch, sampled_seqlen = sampled_shape
        sampled_keys.append(
            (
                int(step.input_token_bucket),
                int(sampled_batch),
                int(sampled_seqlen),
            )
        )
    metadata_batch = int(attn_metadata.batch_size)
    metadata_tokens = int(attn_metadata.total_tokens)
    if (
        metadata_batch > 0
        and metadata_tokens > 0
        and metadata_tokens % metadata_batch == 0
    ):
        metadata_key = (
            int(step.input_token_bucket),
            metadata_batch,
            metadata_tokens // metadata_batch,
        )
        if metadata_key not in sampled_keys:
            sampled_keys.append(metadata_key)
    return sampled_keys


def _decode_history_len(attn_metadata: Any) -> int:
    decode_history_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
    if decode_history_len > 0:
        return decode_history_len
    try:
        seq_lens = np.asarray(
            getattr(attn_metadata, "seq_lens"),
            dtype=np.int64,
        ).reshape(-1)
        return int(seq_lens.max()) if seq_lens.size > 0 else 0
    except (AttributeError, TypeError, ValueError):
        return 0


class Dsv4DpWarmupPrecompiler:
    """Deduplicates DP-superstep precompile coverage across warmup steps."""

    def __init__(
        self,
        owner: Any,
        *,
        token_buckets: tuple[int, ...],
        decode_continuation_bucket: int,
        forward_mode_extend: int,
        rank_msg: str,
    ) -> None:
        self.owner = owner
        self.token_buckets = tuple(
            sorted({int(v) for v in token_buckets if int(v) > 0})
        )
        self.decode_continuation_bucket = int(decode_continuation_bucket)
        self.forward_mode_extend = int(forward_mode_extend)
        self.rank_msg = str(rank_msg)
        self.dp_reduce_paths_precompiled: set[tuple[int, int, int, bool]] = set()
        self.first_embedding_mhc_precompiled: set[tuple[int, int, int, bool]] = set()
        self.lane_dp_helpers_precompiled: set[tuple[int, int, int, bool]] = set()
        self.lane_dp_decode_continuation_precompiled: set[tuple[int, int, int]] = set()
        self.lane_moe_precompiled: set[tuple[int, int, int, bool]] = set()
        self.shared_restore_post_pre_precompiled: set[tuple[int, int, int, bool]] = (
            set()
        )
        self.lane_head_precompiled: set[tuple[int, int, int, bool]] = set()

    def precompile_step(
        self,
        *,
        step: Any,
        attn_metadata: Any,
        forward_batch: Any,
        sampled_shape: tuple[int, int] | None,
    ) -> None:
        step_name = str(getattr(step, "name", "<unnamed>"))
        is_decode = int(step.forward_mode) != self.forward_mode_extend
        self._precompile_full_dp_reduce(
            step=step,
            attn_metadata=attn_metadata,
            sampled_shape=sampled_shape,
            is_decode=is_decode,
            step_name=step_name,
        )
        sampled_keys = _sampled_warmup_keys(step, attn_metadata, sampled_shape)
        self._precompile_sampled_lane_helpers(
            step=step,
            sampled_keys=sampled_keys,
            is_decode=is_decode,
            step_name=step_name,
        )
        self._precompile_dp_lane_helpers(
            step=step,
            attn_metadata=attn_metadata,
            forward_batch=forward_batch,
            is_decode=is_decode,
            step_name=step_name,
        )

    def _precompile_full_dp_reduce(
        self,
        *,
        step: Any,
        attn_metadata: Any,
        sampled_shape: tuple[int, int] | None,
        is_decode: bool,
        step_name: str,
    ) -> None:
        precompile_dp_reduce_paths = getattr(
            self.owner,
            "precompile_dp_attention_reduce_paths",
            None,
        )
        if not callable(precompile_dp_reduce_paths):
            return
        dp_reduce_keys = [
            (
                int(step.input_token_bucket),
                int(attn_metadata.batch_size),
                int(attn_metadata.total_tokens),
                bool(is_decode),
            )
        ]
        if sampled_shape is not None:
            sampled_batch, sampled_seqlen = sampled_shape
            sampled_reduce_key = (
                int(step.input_token_bucket),
                int(sampled_batch),
                int(sampled_batch) * int(sampled_seqlen),
                bool(is_decode),
            )
            if sampled_reduce_key not in dp_reduce_keys:
                dp_reduce_keys.append(sampled_reduce_key)
        for dp_reduce_key in dp_reduce_keys:
            if dp_reduce_key in self.dp_reduce_paths_precompiled:
                continue
            _warmup_trace(
                f"{self.rank_msg} precompile dp-reduce full start "
                f"step={step_name} key={dp_reduce_key}"
            )
            precompile_dp_reduce_paths(
                dp_reduce_key[0],
                batch_size=dp_reduce_key[1],
                total_tokens=dp_reduce_key[2],
                is_decode=bool(dp_reduce_key[3]),
            )
            _warmup_trace(
                f"{self.rank_msg} precompile dp-reduce full done "
                f"step={step_name} key={dp_reduce_key}"
            )
            self.dp_reduce_paths_precompiled.add(dp_reduce_key)

    def _precompile_sampled_lane_helpers(
        self,
        *,
        step: Any,
        sampled_keys: list[tuple[int, int, int]],
        is_decode: bool,
        step_name: str,
    ) -> None:
        precompile_first_embedding_mhc = getattr(
            self.owner,
            "precompile_first_layer_embedding_mhc_shapes",
            None,
        )
        precompile_lane_moe = getattr(self.owner, "precompile_lane_moe_helpers", None)
        precompile_shared_restore_post_pre = getattr(
            self.owner,
            "precompile_shared_expert_restore_post_pre_helpers",
            None,
        )
        precompile_lane_head = getattr(
            self.owner,
            "precompile_lane_head_helpers",
            None,
        )
        if callable(precompile_first_embedding_mhc):
            self._precompile_sampled_keys(
                sampled_keys=sampled_keys,
                is_decode=is_decode,
                cache=self.first_embedding_mhc_precompiled,
                label="embedding-mhc",
                step_name=step_name,
                precompile=precompile_first_embedding_mhc,
            )
        if callable(precompile_lane_moe):
            self._precompile_sampled_keys(
                sampled_keys=sampled_keys,
                is_decode=is_decode,
                cache=self.lane_moe_precompiled,
                label="lane-moe",
                step_name=step_name,
                precompile=precompile_lane_moe,
            )
            if not bool(is_decode):
                self._precompile_fixed_sampled_key(
                    key=(int(step.input_token_bucket), 1, 1),
                    is_decode=False,
                    cache=self.lane_moe_precompiled,
                    label="lane-moe live-single",
                    step_name=step_name,
                    precompile=precompile_lane_moe,
                )
                self._precompile_fixed_sampled_key(
                    key=(int(step.input_token_bucket), 1, int(step.input_token_bucket)),
                    is_decode=False,
                    cache=self.lane_moe_precompiled,
                    label="lane-moe padded-single",
                    step_name=step_name,
                    precompile=precompile_lane_moe,
                )
        if callable(precompile_shared_restore_post_pre):
            self._precompile_sampled_keys(
                sampled_keys=sampled_keys,
                is_decode=is_decode,
                cache=self.shared_restore_post_pre_precompiled,
                label="shared-restore",
                step_name=step_name,
                precompile=precompile_shared_restore_post_pre,
            )
        if callable(precompile_lane_head):
            self._precompile_lane_head_keys(
                step=step,
                sampled_keys=sampled_keys,
                is_decode=is_decode,
                step_name=step_name,
                precompile_lane_head=precompile_lane_head,
            )

    def _precompile_sampled_keys(
        self,
        *,
        sampled_keys: list[tuple[int, int, int]],
        is_decode: bool,
        cache: set[tuple[int, int, int, bool]],
        label: str,
        step_name: str,
        precompile: Callable[..., None],
    ) -> None:
        for key in sampled_keys:
            mode_key = (key[0], key[1], key[2], bool(is_decode))
            if mode_key in cache:
                continue
            _warmup_trace(
                f"{self.rank_msg} precompile {label} start step={step_name} key={key}"
            )
            precompile(
                key[0],
                batch_size=key[1],
                seqlen=key[2],
                is_decode=bool(is_decode),
            )
            _warmup_trace(
                f"{self.rank_msg} precompile {label} done step={step_name} key={key}"
            )
            cache.add(mode_key)

    def _precompile_fixed_sampled_key(
        self,
        *,
        key: tuple[int, int, int],
        is_decode: bool,
        cache: set[tuple[int, int, int, bool]],
        label: str,
        step_name: str,
        precompile: Callable[..., None],
    ) -> None:
        mode_key = (key[0], key[1], key[2], bool(is_decode))
        if mode_key in cache:
            return
        _warmup_trace(
            f"{self.rank_msg} precompile {label} start step={step_name} key={key}"
        )
        precompile(
            key[0],
            batch_size=key[1],
            seqlen=key[2],
            is_decode=bool(is_decode),
        )
        _warmup_trace(
            f"{self.rank_msg} precompile {label} done step={step_name} key={key}"
        )
        cache.add(mode_key)

    def _precompile_lane_head_keys(
        self,
        *,
        step: Any,
        sampled_keys: list[tuple[int, int, int]],
        is_decode: bool,
        step_name: str,
        precompile_lane_head: Callable[..., None],
    ) -> None:
        for full_head_key in sampled_keys:
            mode_key = (
                full_head_key[0],
                full_head_key[1],
                full_head_key[2],
                bool(is_decode),
            )
            if mode_key not in self.lane_head_precompiled:
                _warmup_trace(
                    f"{self.rank_msg} precompile lane-head start "
                    f"step={step_name} key={full_head_key}"
                )
                precompile_lane_head(
                    full_head_key[0],
                    batch_size=full_head_key[1],
                    seqlen=full_head_key[2],
                    is_decode=bool(is_decode),
                )
                _warmup_trace(
                    f"{self.rank_msg} precompile lane-head done "
                    f"step={step_name} key={full_head_key}"
                )
                self.lane_head_precompiled.add(mode_key)
            if not bool(is_decode):
                self._precompile_fixed_lane_head_key(
                    key=(
                        int(step.input_token_bucket),
                        1,
                        int(step.input_token_bucket),
                        False,
                    ),
                    label="padded-single",
                    step_name=step_name,
                    precompile_lane_head=precompile_lane_head,
                )
            self._precompile_fixed_lane_head_key(
                key=(int(step.input_token_bucket), 1, 1, True),
                label="decode-single",
                step_name=step_name,
                precompile_lane_head=precompile_lane_head,
            )

    def _precompile_fixed_lane_head_key(
        self,
        *,
        key: tuple[int, int, int, bool],
        label: str,
        step_name: str,
        precompile_lane_head: Callable[..., None],
    ) -> None:
        if key in self.lane_head_precompiled:
            return
        _warmup_trace(
            f"{self.rank_msg} precompile lane-head {label} start "
            f"step={step_name} key={key}"
        )
        precompile_lane_head(
            key[0],
            batch_size=key[1],
            seqlen=key[2],
            is_decode=bool(key[3]),
        )
        _warmup_trace(
            f"{self.rank_msg} precompile lane-head {label} done "
            f"step={step_name} key={key}"
        )
        self.lane_head_precompiled.add(key)

    def _precompile_dp_lane_helpers(
        self,
        *,
        step: Any,
        attn_metadata: Any,
        forward_batch: Any,
        is_decode: bool,
        step_name: str,
    ) -> None:
        precompile_dp_reduce_paths = getattr(
            self.owner,
            "precompile_dp_attention_reduce_paths",
            None,
        )
        precompile_lane_dp_helpers = getattr(
            self.owner,
            "precompile_lane_dp_attention_helpers",
            None,
        )
        precompile_lane_dp_decode_continuation = getattr(
            self.owner,
            "precompile_lane_dp_attention_decode_continuation_helpers",
            None,
        )
        if not (
            callable(precompile_dp_reduce_paths)
            or callable(precompile_lane_dp_helpers)
            or callable(precompile_lane_dp_decode_continuation)
        ):
            return
        lane_counts = np.asarray(
            getattr(forward_batch, "dp_attention_lane_token_counts"),
            dtype=np.int32,
        ).reshape(-1)
        lane_batches = np.asarray(
            getattr(forward_batch, "dp_attention_lane_batch_sizes"),
            dtype=np.int32,
        ).reshape(-1)
        for lane_tokens, lane_batch in zip(lane_counts, lane_batches, strict=False):
            tokens_i = int(lane_tokens)
            batch_i = int(lane_batch)
            if tokens_i <= 0 or batch_i <= 0 or tokens_i % batch_i != 0:
                continue
            lane_query_seqlen = tokens_i // batch_i
            lane_helper_seqlen = int(lane_query_seqlen)
            if is_decode:
                lane_helper_seqlen = max(
                    int(lane_query_seqlen),
                    int(_decode_history_len(attn_metadata)),
                )
            unpad_key = (
                int(step.input_token_bucket),
                batch_i,
                int(lane_helper_seqlen),
                bool(is_decode),
            )
            lane_reduce_key = (
                int(step.input_token_bucket),
                batch_i,
                tokens_i,
                bool(is_decode),
            )
            if callable(precompile_dp_reduce_paths):
                self._precompile_lane_reduce_key(
                    key=lane_reduce_key,
                    step_name=step_name,
                    precompile_dp_reduce_paths=precompile_dp_reduce_paths,
                )
            if callable(precompile_lane_dp_helpers):
                self._precompile_lane_dp_helper_key(
                    key=unpad_key,
                    label="helper",
                    step_name=step_name,
                    precompile_lane_dp_helpers=precompile_lane_dp_helpers,
                )
                if bool(is_decode) and int(lane_helper_seqlen) > 1:
                    total_prefill_tokens = int(step.batch_size) * int(
                        lane_helper_seqlen
                    )
                    prefill_bucket = next(
                        (
                            int(bucket)
                            for bucket in self.token_buckets
                            if int(total_prefill_tokens) <= int(bucket)
                        ),
                        int(self.token_buckets[-1]) if self.token_buckets else 0,
                    )
                    if prefill_bucket > 0:
                        self._precompile_lane_dp_helper_key(
                            key=(
                                int(prefill_bucket),
                                batch_i,
                                int(lane_helper_seqlen),
                                False,
                            ),
                            label="helper decode-history-prefill",
                            step_name=step_name,
                            precompile_lane_dp_helpers=precompile_lane_dp_helpers,
                        )
            if not bool(is_decode) and callable(precompile_lane_dp_decode_continuation):
                self._precompile_lane_dp_continuation_key(
                    key=(
                        int(self.decode_continuation_bucket),
                        batch_i,
                        tokens_i // batch_i,
                    ),
                    step_name=step_name,
                    precompile_lane_dp_decode_continuation=(
                        precompile_lane_dp_decode_continuation
                    ),
                )
        if not bool(is_decode) and callable(precompile_lane_dp_helpers):
            self._precompile_lane_dp_helper_key(
                key=(
                    int(step.input_token_bucket),
                    1,
                    int(step.input_token_bucket),
                    False,
                ),
                label="helper padded-single",
                step_name=step_name,
                precompile_lane_dp_helpers=precompile_lane_dp_helpers,
            )

    def _precompile_lane_reduce_key(
        self,
        *,
        key: tuple[int, int, int, bool],
        step_name: str,
        precompile_dp_reduce_paths: Callable[..., None],
    ) -> None:
        if key in self.dp_reduce_paths_precompiled:
            return
        _warmup_trace(
            f"{self.rank_msg} precompile dp-reduce lane start "
            f"step={step_name} key={key}"
        )
        precompile_dp_reduce_paths(
            key[0],
            batch_size=key[1],
            total_tokens=key[2],
            is_decode=bool(key[3]),
        )
        _warmup_trace(
            f"{self.rank_msg} precompile dp-reduce lane done step={step_name} key={key}"
        )
        self.dp_reduce_paths_precompiled.add(key)

    def _precompile_lane_dp_helper_key(
        self,
        *,
        key: tuple[int, int, int, bool],
        label: str,
        step_name: str,
        precompile_lane_dp_helpers: Callable[..., None],
    ) -> None:
        if key in self.lane_dp_helpers_precompiled:
            return
        _warmup_trace(
            f"{self.rank_msg} precompile lane-dp {label} start "
            f"step={step_name} key={key}"
        )
        precompile_lane_dp_helpers(
            key[0],
            batch_size=key[1],
            seqlen=key[2],
            is_decode=bool(key[3]),
        )
        _warmup_trace(
            f"{self.rank_msg} precompile lane-dp {label} done "
            f"step={step_name} key={key}"
        )
        self.lane_dp_helpers_precompiled.add(key)

    def _precompile_lane_dp_continuation_key(
        self,
        *,
        key: tuple[int, int, int],
        step_name: str,
        precompile_lane_dp_decode_continuation: Callable[..., None],
    ) -> None:
        if key in self.lane_dp_decode_continuation_precompiled:
            return
        _warmup_trace(
            f"{self.rank_msg} precompile lane-dp decode-continuation "
            f"start step={step_name} key={key}"
        )
        precompile_lane_dp_decode_continuation(
            key[0],
            batch_size=key[1],
            seqlen=key[2],
        )
        _warmup_trace(
            f"{self.rank_msg} precompile lane-dp decode-continuation "
            f"done step={step_name} key={key}"
        )
        self.lane_dp_decode_continuation_precompiled.add(key)


def _synthetic_warmup_forward_batch(
    attn_metadata: Any,
    *,
    runtime_config: Any,
    coord: Any,
    split_dp_lanes: bool = True,
) -> Any:
    """Minimal ForwardBatch-like object for DSV4 metadata warmup."""
    batch_size = int(getattr(attn_metadata, "batch_size", 1))
    total_tokens = int(getattr(attn_metadata, "total_tokens", batch_size))
    lanes = max(1, int(getattr(runtime_config, "attention_dp_degree", 1)))
    lane = int(getattr(coord, "attn_lane", 0))
    use_dp_superstep = lanes > 1 and 0 <= lane < lanes

    lane_token_counts = np.zeros((lanes,), dtype=np.int32)
    lane_batch_sizes = np.zeros((lanes,), dtype=np.int32)
    if use_dp_superstep and split_dp_lanes:
        qsl = np.asarray(
            getattr(attn_metadata, "query_start_loc"),
            dtype=np.int64,
        ).reshape(-1)
        if qsl.shape[0] < batch_size + 1:
            raise RuntimeError(
                "DSV4 synthetic DP-attention warmup requires "
                f"query_start_loc with at least {batch_size + 1} entries, "
                f"got {qsl.shape}"
            )
        q_lens = qsl[1 : batch_size + 1] - qsl[:batch_size]
        if int(q_lens.sum()) != total_tokens:
            raise RuntimeError(
                "DSV4 synthetic DP-attention warmup token mismatch: "
                f"query_lens_sum={int(q_lens.sum())}, total={total_tokens}"
            )
        active_lanes = min(int(batch_size), lanes)
        base, rem = divmod(int(batch_size), active_lanes)
        req_start = 0
        for lane_i in range(active_lanes):
            req_count = base + (1 if lane_i < rem else 0)
            req_end = req_start + req_count
            lane_batch_sizes[lane_i] = np.int32(req_count)
            lane_token_counts[lane_i] = np.int32(int(q_lens[req_start:req_end].sum()))
            req_start = req_end
    elif use_dp_superstep:
        lane_token_counts[lane] = np.int32(total_tokens)
        lane_batch_sizes[lane] = np.int32(batch_size)
    else:
        lane_token_counts[0] = np.int32(total_tokens)
        lane_batch_sizes[0] = np.int32(batch_size)
    lane_token_offsets = np.zeros((lanes + 1,), dtype=np.int32)
    lane_token_offsets[1:] = np.cumsum(lane_token_counts, dtype=np.int32)
    lane_batch_offsets = np.zeros((lanes + 1,), dtype=np.int32)
    lane_batch_offsets[1:] = np.cumsum(lane_batch_sizes, dtype=np.int32)
    return SimpleNamespace(
        state_owner_ids=np.arange(batch_size, dtype=np.int32),
        dp_attention_superstep=bool(use_dp_superstep),
        dp_attention_num_lanes=lanes,
        dp_attention_lane_token_counts=lane_token_counts,
        dp_attention_lane_batch_sizes=lane_batch_sizes,
        dp_attention_lane_token_offsets=lane_token_offsets,
        dp_attention_lane_batch_offsets=lane_batch_offsets,
    )


def _seal_dsv4_warmup_manifests(
    owner: Any,
    record_warmup: Callable[..., None],
) -> None:
    seal_manifest = getattr(owner, "seal_product_compile_manifest", None)
    seal_logits = getattr(owner, "seal_logits_processor_precompiled_kernels", None)
    seal_blockwise_moe = getattr(
        owner,
        "seal_blockwise_moe_precompiled_kernels",
        None,
    )
    stage_t0 = time.perf_counter()
    if callable(seal_manifest):
        seal_manifest()
    record_warmup("seal manifest", start=stage_t0)
    stage_t0 = time.perf_counter()
    if callable(seal_blockwise_moe):
        seal_blockwise_moe()
    record_warmup("seal blockwise moe", start=stage_t0)
    stage_t0 = time.perf_counter()
    if callable(seal_logits):
        seal_logits()
    record_warmup("seal logits", start=stage_t0)


def _run_dsv4_warmup_step_barrier(
    *,
    runtime_config: Any,
    coord: Any,
    step: Any,
) -> None:
    # Warmup steps run free across ranks; without a step barrier the replicas
    # can drift into different collective NEFFs. Real serving is
    # superstep-lockstep, so this is warmup-only.
    if int(getattr(runtime_config, "total_workers", 1)) <= 1:
        return

    from nkipy_serving.models.deepseek_v4.neff_compiler import (
        _product_shared_build_dir,
    )
    from nkipy_serving.runtime.collective_load import collective_load_barrier

    collective_load_barrier(
        build_dir=_product_shared_build_dir(str(runtime_config.config_build_dir())),
        name=f"warmup_step_{getattr(step, 'name', 'step')}",
        rank_id=int(getattr(coord, "rank")),
        world_size=int(runtime_config.total_workers),
    )


def _collect_dsv4_support_compile_manifest(
    owner: Any,
    *,
    warmup_plan: Any,
    rank_msg: str | None = None,
) -> list[_Dsv4WarmupCompileEntry]:
    del rank_msg
    entries: list[_Dsv4WarmupCompileEntry] = []

    token_buckets = tuple(int(bucket) for bucket in warmup_plan.token_paddings)
    state_write_buckets = tuple(
        int(bucket) for bucket in warmup_plan.state_write_buckets
    )
    decode_write_buckets = tuple(
        int(bucket) for bucket in warmup_plan.decode_write_buckets
    )

    precompile_swa = getattr(
        owner,
        "precompile_bucketed_prefill_swa_attention",
        None,
    )
    if callable(precompile_swa) and token_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="support_attention",
            name="bucketed_swa",
            stage="precompile bucketed swa attention",
            mode="extend",
            metadata_key=(token_buckets,),
            token_buckets=str(token_buckets),
            compile_fn=lambda buckets=token_buckets: precompile_swa(buckets),
        )

    precompile_two_source = getattr(
        owner,
        "precompile_bucketed_prefill_two_source_attention",
        None,
    )
    if callable(precompile_two_source) and token_buckets:
        # Compressed attention intentionally keeps an internal minimum of two
        # query rows, even when the request bucket is one. Warm that exact
        # short-prefill backend shape without expanding the support-kernel
        # manifest to every live request count.
        two_source_exact_rows = (2,)
        _append_dsv4_warmup_compile_entry(
            entries,
            family="support_attention",
            name="bucketed_two_source",
            stage="precompile bucketed two-source attention",
            mode="extend",
            metadata_key=(token_buckets, two_source_exact_rows),
            token_buckets=str(token_buckets),
            exact_query_rows=str(two_source_exact_rows),
            compile_fn=(
                lambda buckets=token_buckets, exact_rows=two_source_exact_rows: (
                    precompile_two_source(
                        buckets,
                        exact_query_rows=exact_rows,
                    )
                )
            ),
        )

    precompile_swa_owner_window = getattr(
        owner,
        "precompile_swa_owner_window_write_buckets",
        None,
    )
    if callable(precompile_swa_owner_window) and state_write_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="swa_owner_window",
            stage="precompile swa owner-window writes",
            metadata_key=(state_write_buckets,),
            buckets=str(state_write_buckets),
            compile_fn=(
                lambda buckets=state_write_buckets: precompile_swa_owner_window(buckets)
            ),
        )

    precompile_compressor_state = getattr(
        owner,
        "precompile_compressor_state_write_buckets",
        None,
    )
    if callable(precompile_compressor_state) and state_write_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="compressor_state",
            stage="precompile compressor state writes",
            metadata_key=(state_write_buckets,),
            buckets=str(state_write_buckets),
            compile_fn=(
                lambda buckets=state_write_buckets: precompile_compressor_state(buckets)
            ),
        )

    precompile_compressor_prefill_pool = getattr(
        owner,
        "precompile_compressor_prefill_pool_buckets",
        None,
    )
    if callable(precompile_compressor_prefill_pool) and token_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="compressor_prefill_pool",
            stage="precompile compressor prefill pool",
            mode="extend",
            metadata_key=(token_buckets,),
            buckets=str(token_buckets),
            compile_fn=(
                lambda buckets=token_buckets: precompile_compressor_prefill_pool(
                    buckets
                )
            ),
        )

    precompile_compressor_slot_write = getattr(
        owner,
        "precompile_compressor_slot_write_buckets",
        None,
    )
    if callable(precompile_compressor_slot_write) and token_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="compressor_slot_write",
            stage="precompile compressor slot writes",
            mode="extend",
            metadata_key=(token_buckets,),
            buckets=str(token_buckets),
            compile_fn=(
                lambda buckets=token_buckets: precompile_compressor_slot_write(buckets)
            ),
        )

    precompile_dual_state_swa = getattr(
        owner,
        "precompile_dual_state_swa_write_buckets",
        None,
    )
    if callable(precompile_dual_state_swa) and decode_write_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="dual_state_swa",
            stage="precompile dual-state swa writes",
            mode="decode",
            metadata_key=(decode_write_buckets,),
            buckets=str(decode_write_buckets),
            compile_fn=(
                lambda buckets=decode_write_buckets: precompile_dual_state_swa(buckets)
            ),
        )

    precompile_bucketed_single_state_swa_cache = getattr(
        owner,
        "precompile_bucketed_single_state_swa_cache_write_buckets",
        None,
    )
    if callable(precompile_bucketed_single_state_swa_cache) and token_buckets:
        _append_dsv4_warmup_compile_entry(
            entries,
            family="state_write",
            name="single_state_swa_cache",
            stage="precompile bucketed single-state swa/cache writes",
            mode="extend",
            metadata_key=(token_buckets,),
            buckets=str(token_buckets),
            compile_fn=(
                lambda buckets=token_buckets: (
                    precompile_bucketed_single_state_swa_cache(buckets)
                )
            ),
        )

    return entries


def _collect_dsv4_step_compile_manifest(
    owner: Any,
    *,
    step: Any,
    attn_metadata: Any,
    forward_batch: Any,
    sampled_shape: tuple[int, int] | None,
    dp_precompiler: Dsv4DpWarmupPrecompiler,
) -> list[_Dsv4WarmupCompileEntry]:
    entries: list[_Dsv4WarmupCompileEntry] = []
    token_bucket = int(step.input_token_bucket)
    batch_size = int(step.batch_size)
    forward_mode = int(step.forward_mode)
    step_name = str(getattr(step, "name", "<unnamed>"))

    precompile_logits = getattr(
        owner,
        "precompile_logits_processor_bucket",
        None,
    )
    if callable(precompile_logits):
        _append_dsv4_warmup_compile_entry(
            entries,
            family="logits",
            name="bucket",
            stage="precompile logits",
            mode=forward_mode,
            token_bucket=token_bucket,
            step=step,
            compile_fn=lambda bucket=token_bucket: precompile_logits(bucket),
        )

    precompile_bucket = getattr(
        owner,
        "precompile_token_bucket",
        None,
    )
    if callable(precompile_bucket):
        _append_dsv4_warmup_compile_entry(
            entries,
            family="product",
            name="token_bucket",
            stage="precompile bucket",
            mode=forward_mode,
            token_bucket=token_bucket,
            step=step,
            compile_fn=lambda bucket=token_bucket: precompile_bucket(bucket),
        )

    if _has_dsv4_dp_warmup_precompile(owner):
        lane_counts = tuple(
            int(v)
            for v in np.asarray(
                getattr(forward_batch, "dp_attention_lane_token_counts", ()),
                dtype=np.int32,
            ).reshape(-1)
        )
        lane_batches = tuple(
            int(v)
            for v in np.asarray(
                getattr(forward_batch, "dp_attention_lane_batch_sizes", ()),
                dtype=np.int32,
            ).reshape(-1)
        )
        metadata_key = (
            step_name,
            int(getattr(attn_metadata, "total_tokens", 0) or 0),
            int(getattr(attn_metadata, "max_seq_len", 0) or 0),
            None if sampled_shape is None else tuple(int(v) for v in sampled_shape),
            lane_counts,
            lane_batches,
        )
        _append_dsv4_warmup_compile_entry(
            entries,
            family="dp_helpers",
            name="step",
            stage="precompile dp helpers",
            mode=forward_mode,
            token_bucket=token_bucket,
            request_bucket=batch_size,
            metadata_key=metadata_key,
            step=step,
            total_tokens=int(getattr(attn_metadata, "total_tokens", 0) or 0),
            compile_fn=(
                lambda step=step,
                attn_metadata=attn_metadata,
                forward_batch=forward_batch,
                sampled_shape=sampled_shape: (
                    dp_precompiler.precompile_step(
                        step=step,
                        attn_metadata=attn_metadata,
                        forward_batch=forward_batch,
                        sampled_shape=sampled_shape,
                    )
                )
            ),
        )

    return entries


def run_dsv4_executor_warmup(executor: Any, paddings: Any = None) -> None:
    """Compile and first-touch sampled DSV4 bucket paths."""
    self = executor
    if paddings is None or not self._neff_runtime_ready:
        return None

    from nkipy_serving.attention.base import (
        FORWARD_MODE_DECODE,
        FORWARD_MODE_EXTEND,
    )
    from nkipy_serving.models.deepseek_v4.assembly.warmup_plan import (
        build_dsv4_warmup_plan,
    )
    from nkipy_serving.runtime.warmup import (
        build_synthetic_warmup_inputs,
    )

    token_paddings = tuple(int(bucket) for bucket in paddings.token_paddings)
    bs_paddings = tuple(int(bucket) for bucket in paddings.bs_paddings)
    block_size, num_blocks = _warmup_kv_pool_shape(
        self._kv_pool,
        self._runtime_config,
    )
    coord = getattr(self, "_coord", None)
    rank_msg = _warmup_rank_message(coord)
    warmup_start = time.monotonic()
    record_warmup = Dsv4WarmupRecorder(
        rank=int(getattr(coord, "rank", -1)),
        token_paddings=token_paddings,
        bs_paddings=bs_paddings,
    ).record

    record_warmup("start")
    _warmup_trace(
        f"{rank_msg} start token_buckets={token_paddings} bs_buckets={bs_paddings}"
    )
    try:
        begin_product_warmup = getattr(
            self,
            "begin_product_warmup",
            None,
        )
        end_product_warmup = getattr(
            self,
            "end_product_warmup",
            None,
        )
        if callable(begin_product_warmup):
            begin_product_warmup()
        has_compressed_layers = bool(
            getattr(
                self,
                "has_compressed_layers",
                False,
            )
        )
        compressed_boundary_pos = (
            _compressed_decode_boundary_warmup_target_pos(self)
            if callable(begin_product_warmup) and has_compressed_layers
            else None
        )
        warmup_plan = build_dsv4_warmup_plan(
            paddings,
            product_warmup_enabled=callable(begin_product_warmup),
            has_compressed_layers=has_compressed_layers,
            compressed_boundary_pos=compressed_boundary_pos,
        )
        token_paddings = warmup_plan.token_paddings
        bs_paddings = warmup_plan.bs_paddings
        steps = list(warmup_plan.steps)
        compressed_nonboundary_pos = (
            _compressed_decode_nonboundary_warmup_target_pos(self)
            if has_compressed_layers
            else None
        )

        _warmup_trace(
            f"{rank_msg} built steps="
            f"{[getattr(step, 'name', '<unnamed>') for step in steps]}"
        )
        record_warmup(
            "steps built",
            step_count=len(steps),
            execute_forwards=True,
        )

        warmup_inputs: list[tuple[Any, Any, Any, Any, bool, Any]] = []
        step_compile_manifest: list[_Dsv4WarmupCompileEntry] = []
        decode_continuation_bucket = next(
            (int(bucket) for bucket in bs_paddings if int(bucket) > 0),
            1,
        )
        dp_precompiler = Dsv4DpWarmupPrecompiler(
            self,
            token_buckets=tuple(int(bucket) for bucket in token_paddings),
            decode_continuation_bucket=decode_continuation_bucket,
            forward_mode_extend=int(FORWARD_MODE_EXTEND),
            rank_msg=rank_msg,
        )

        for step in steps:
            collect_step_t0 = time.perf_counter()
            step_name = str(getattr(step, "name", "<unnamed>"))
            _warmup_trace(
                f"{rank_msg} collect compile start step={step_name} "
                f"mode={int(step.forward_mode)} bucket={int(step.input_token_bucket)} "
                f"batch={int(step.batch_size)}"
            )
            _warmup_trace(f"{rank_msg} build inputs start step={step_name}")
            stage_t0 = time.perf_counter()
            input_ids, positions, attn_metadata = build_synthetic_warmup_inputs(
                step,
                token_paddings=token_paddings,
                bs_paddings=bs_paddings,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=int(self._weights.num_kv_heads),
                head_dim=int(self._weights.head_dim),
            )
            record_warmup(
                "build inputs",
                start=stage_t0,
                step=step,
                total_tokens=int(attn_metadata.total_tokens),
            )
            _warmup_trace(f"{rank_msg} build inputs done step={step_name}")
            if int(step.forward_mode) == int(FORWARD_MODE_DECODE) and bool(
                getattr(
                    self,
                    "has_compressed_layers",
                    False,
                )
            ):
                decode_target_pos = (
                    (
                        1
                        if compressed_nonboundary_pos is None
                        else int(compressed_nonboundary_pos)
                    )
                    if step.decode_target_pos is None
                    else int(step.decode_target_pos)
                )
                _warmup_trace(f"{rank_msg} retarget decode start step={step_name}")
                stage_t0 = time.perf_counter()
                positions, attn_metadata = _retarget_compressed_decode_warmup(
                    positions,
                    attn_metadata,
                    target_pos=int(decode_target_pos),
                )
                record_warmup(
                    "retarget decode",
                    start=stage_t0,
                    step=step,
                    target_pos=int(decode_target_pos),
                )
                _warmup_trace(f"{rank_msg} retarget decode done step={step_name}")
            _warmup_trace(f"{rank_msg} synthetic batch start step={step_name}")
            stage_t0 = time.perf_counter()
            forward_batch = _synthetic_warmup_forward_batch(
                attn_metadata,
                runtime_config=self._runtime_config,
                coord=self._coord,
            )
            record_warmup("synthetic batch", start=stage_t0, step=step)
            _warmup_trace(f"{rank_msg} synthetic batch done step={step_name}")
            uses_dp_superstep = bool(
                getattr(forward_batch, "dp_attention_superstep", False)
            )
            sampled_shape = _sampled_rectangular_warmup_shape(attn_metadata)
            step_compile_manifest.extend(
                _collect_dsv4_step_compile_manifest(
                    self,
                    step=step,
                    attn_metadata=attn_metadata,
                    forward_batch=forward_batch,
                    sampled_shape=sampled_shape,
                    dp_precompiler=dp_precompiler,
                )
            )
            if bool(getattr(step, "execute_forward", True)):
                warmup_inputs.append(
                    (
                        step,
                        input_ids,
                        positions,
                        attn_metadata,
                        uses_dp_superstep,
                        forward_batch,
                    )
                )
            else:
                _warmup_trace(
                    f"{rank_msg} precompile-only step skip forward step={step_name}"
                )
            _warmup_trace(
                f"{rank_msg} collect compile done step={step_name} "
                f"total_tokens={int(attn_metadata.total_tokens)} "
                f"dp_superstep={uses_dp_superstep}"
            )
            record_warmup(
                "collect compile step",
                start=collect_step_t0,
                step=step,
                total_tokens=int(attn_metadata.total_tokens),
                dp_superstep=bool(uses_dp_superstep),
            )

        _run_dsv4_warmup_compile_manifest(
            step_compile_manifest,
            manifest_name="steps",
            rank_msg=rank_msg,
            record_warmup=record_warmup,
        )

        for (
            step,
            input_ids,
            positions,
            attn_metadata,
            uses_dp_superstep,
            forward_batch,
        ) in warmup_inputs:
            forward_step_t0 = time.perf_counter()
            _warmup_trace(
                f"{rank_msg} forward start step={getattr(step, 'name', '<unnamed>')} "
                f"mode={int(step.forward_mode)} bucket={int(step.input_token_bucket)} "
                f"total_tokens={int(attn_metadata.total_tokens)} "
                f"dp_superstep={uses_dp_superstep}"
            )
            if uses_dp_superstep:
                needs_collective_prepass = getattr(
                    self,
                    "needs_dp_attention_collective_warmup_prepass",
                    None,
                )
                if callable(needs_collective_prepass) and bool(
                    needs_collective_prepass()
                ):
                    stage_t0 = time.perf_counter()
                    preload_batch = _synthetic_warmup_forward_batch(
                        attn_metadata,
                        runtime_config=self._runtime_config,
                        coord=self._coord,
                        split_dp_lanes=False,
                    )
                    preload = self.prepare_attention_metadata(
                        attn_metadata,
                        positions=positions,
                        token_bucket=int(step.input_token_bucket),
                        forward_batch=preload_batch,
                    )
                    self.forward(
                        input_ids,
                        positions,
                        [],
                        preload,
                        token_bucket=int(step.input_token_bucket),
                        real_total_tokens=int(attn_metadata.total_tokens),
                        sampling_batch=None,
                        attention_lane=int(self._coord.attn_lane),
                    )
                    record_warmup(
                        "collective prepass",
                        start=stage_t0,
                        step=step,
                        total_tokens=int(attn_metadata.total_tokens),
                    )
                    _warmup_trace(
                        f"{rank_msg} collective prepass done "
                        f"step={getattr(step, 'name', '<unnamed>')}"
                    )
            stage_t0 = time.perf_counter()
            prepared = self.prepare_attention_metadata(
                attn_metadata,
                positions=positions,
                token_bucket=int(step.input_token_bucket),
                forward_batch=forward_batch,
            )
            record_warmup(
                "prepare attention metadata",
                start=stage_t0,
                step=step,
                total_tokens=int(attn_metadata.total_tokens),
            )
            stage_t0 = time.perf_counter()
            self.forward(
                input_ids,
                positions,
                [],
                prepared,
                token_bucket=int(step.input_token_bucket),
                real_total_tokens=int(attn_metadata.total_tokens),
                sampling_batch=None,
                attention_lane=int(self._coord.attn_lane),
            )
            record_warmup(
                "forward body",
                start=stage_t0,
                step=step,
                total_tokens=int(attn_metadata.total_tokens),
                dp_superstep=bool(uses_dp_superstep),
            )
            _run_dsv4_warmup_step_barrier(
                runtime_config=self._runtime_config,
                coord=self._coord,
                step=step,
            )
            _warmup_trace(
                f"{rank_msg} forward done step={getattr(step, 'name', '<unnamed>')}"
            )
            record_warmup(
                "forward step",
                start=forward_step_t0,
                step=step,
                total_tokens=int(attn_metadata.total_tokens),
                dp_superstep=bool(uses_dp_superstep),
            )
        support_compile_manifest = _collect_dsv4_support_compile_manifest(
            self,
            warmup_plan=warmup_plan,
            rank_msg=rank_msg,
        )
        _run_dsv4_warmup_compile_manifest(
            support_compile_manifest,
            manifest_name="support",
            rank_msg=rank_msg,
            record_warmup=record_warmup,
        )
        _seal_dsv4_warmup_manifests(self, record_warmup)
        stage_t0 = time.perf_counter()
        _seal_dsv4_support_kernel_namespaces()
        record_warmup("seal support kernels", start=stage_t0)
        _warmup_trace(
            f"{rank_msg} done elapsed_s={time.monotonic() - warmup_start:.1f}"
        )
        record_warmup("done")
    finally:
        if "end_product_warmup" in locals() and callable(end_product_warmup):
            end_product_warmup()
        self.flush_cache()
        _warmup_trace(
            f"{rank_msg} cleanup elapsed_s={time.monotonic() - warmup_start:.1f}"
        )
        record_warmup("cleanup")
    return None


def _device_zero_allocator(shape: tuple[int, ...], dtype: Any, *, name: str) -> Any:
    """Allocate a zero-filled DeviceTensor for DSV4 runtime state."""
    arr = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    return _get_device_tensor_cls().from_numpy(np.ascontiguousarray(arr), name=name)


def _dtype_itemsize(dtype: Any) -> int:
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError:
        itemsize = getattr(dtype, "itemsize", None)
        return int(itemsize) if itemsize is not None else 0


def _profiled_device_allocator(
    allocator: Any,
    profiler: StartupProfiler,
    *,
    role: str,
) -> Any:
    if not profiler.enabled:
        return allocator

    def wrapped(shape: tuple[int, ...], dtype: Any, *, name: str) -> Any:
        shape_tuple = tuple(int(s) for s in shape)
        t0 = time.perf_counter()
        obj = allocator(shape_tuple, dtype, name=name)
        elapsed = time.perf_counter() - t0
        elements = int(np.prod(shape_tuple, dtype=np.int64)) if shape_tuple else 1
        profiler.record(
            "allocated",
            elapsed_s=elapsed,
            role=str(role),
            name=str(name),
            shape=list(shape_tuple),
            dtype=str(dtype),
            elements=elements,
            nbytes=elements * _dtype_itemsize(dtype),
        )
        return obj

    return wrapped


def _retarget_compressed_decode_warmup(
    positions: np.ndarray,
    metadata: Any,
    *,
    target_pos: int = 1,
) -> tuple[np.ndarray, Any]:
    """Warm compressed decode at a real cached-token position.

    Generic synthetic decode uses position 0 because it has no model-specific
    state assumptions. DSV4 compressed attention treats decode as a cached
    continuation, so warming position 0 creates static one-token NEFFs that
    serving does not use. Retarget to position 1 while keeping one query token
    per request.
    """
    from nkipy_serving.attention.base import AttentionMetadata

    base = metadata
    batch_size = int(getattr(base, "batch_size", 0))
    total_tokens = int(getattr(base, "total_tokens", 0))
    block_size = int(getattr(base, "block_size", 0))
    if batch_size <= 0 or total_tokens != batch_size or block_size <= 0:
        return positions, metadata

    pos_i = max(0, int(target_pos))
    if pos_i == 0:
        return positions, metadata

    block_tables = np.asarray(base.block_tables, dtype=np.int64).copy()
    block_col = pos_i // block_size
    if block_tables.ndim != 2 or block_tables.shape[0] < batch_size:
        return positions, metadata
    if block_tables.shape[1] <= block_col:
        extra_cols = block_col + 1 - int(block_tables.shape[1])
        next_block = int(block_tables.max()) + 1 if block_tables.size else 0
        extra = np.arange(
            next_block,
            next_block + batch_size * extra_cols,
            dtype=np.int64,
        ).reshape(batch_size, extra_cols)
        block_tables = np.concatenate([block_tables[:batch_size], extra], axis=1)
    else:
        block_tables = block_tables[:batch_size]

    live_positions = np.asarray(positions).copy()
    if live_positions.size < total_tokens:
        return positions, metadata
    live_positions[:total_tokens] = np.asarray(pos_i, dtype=live_positions.dtype)
    block_ids = block_tables[:, block_col]
    slot_mapping = (
        block_ids * np.int64(block_size) + np.int64(pos_i % block_size)
    ).astype(np.int64, copy=False)
    seq_len = pos_i + 1
    retargeted = AttentionMetadata(
        forward_mode=int(base.forward_mode),
        seq_lens=np.full((batch_size,), seq_len, dtype=np.int64),
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        query_start_loc=np.arange(batch_size + 1, dtype=np.int64),
        total_tokens=total_tokens,
        batch_size=batch_size,
        max_seq_len=seq_len,
        num_kv_heads=int(base.num_kv_heads),
        head_dim=int(base.head_dim),
        block_size=block_size,
    )
    return live_positions, retargeted


def _compressed_decode_boundary_warmup_target_pos(sampled: Any) -> int | None:
    """Return a decode position that hits all known compressor boundaries."""
    explicit = getattr(
        sampled,
        "compressed_decode_boundary_warmup_target_pos",
        None,
    )
    if explicit is not None:
        return max(0, int(explicit))

    ratios: set[int] = set()
    runtime_surface = getattr(sampled, "runtime_surface", None)
    for block in getattr(runtime_surface, "blocks", ()) or ():
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for owner in (
            attn,
            getattr(attn, "compressor", None),
            getattr(getattr(attn, "indexer", None), "compressor", None),
        ):
            ratio = getattr(owner, "compress_ratio", None)
            if ratio is None:
                continue
            ratio_i = int(ratio)
            if ratio_i > 1:
                ratios.add(ratio_i)
    if not ratios:
        return None
    boundary = 1
    for ratio in sorted(ratios):
        boundary = math.lcm(boundary, int(ratio))
    return max(1, int(boundary)) - 1


def _compressed_decode_nonboundary_warmup_target_pos(sampled: Any) -> int | None:
    """Return a non-boundary decode position with compressed indexer history."""
    explicit = getattr(
        sampled,
        "compressed_decode_nonboundary_warmup_target_pos",
        None,
    )
    if explicit is not None:
        return max(1, int(explicit))

    ratios: set[int] = set()
    indexer_ratios: set[int] = set()
    runtime_surface = getattr(sampled, "runtime_surface", None)
    for block in getattr(runtime_surface, "blocks", ()) or ():
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        indexer = getattr(attn, "indexer", None)
        owners = (
            attn,
            getattr(attn, "compressor", None),
            indexer,
            getattr(indexer, "compressor", None),
        )
        layer_ratios = {
            int(ratio)
            for owner in owners
            if owner is not None
            for ratio in (getattr(owner, "compress_ratio", None),)
            if ratio is not None and int(ratio) > 1
        }
        ratios.update(layer_ratios)
        if indexer is not None:
            indexer_ratio = int(
                getattr(indexer, "compress_ratio", None)
                or getattr(attn, "compress_ratio", None)
                or 0
            )
            if indexer_ratio > 1:
                indexer_ratios.add(indexer_ratio)

    if not indexer_ratios:
        return None

    target = max(indexer_ratios)
    relevant_ratios = ratios or indexer_ratios
    while any((target + 1) % ratio == 0 for ratio in relevant_ratios):
        target += 1
    return max(1, int(target))
