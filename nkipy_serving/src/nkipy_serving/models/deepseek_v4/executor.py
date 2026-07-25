"""DeepSeek-V4-Flash runtime executor.

Production forward runs the sampled NKIPy path. Without an installed sampled
executor, serving fails closed instead of silently returning synthetic zero
logits or falling back to CPU/eager compute.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.config import DeepseekV4ModelConfig
from nkipy_serving.models.deepseek_v4.neff_runtime.runtime import (
    Dsv4NeffRuntimeMixin,
)
from nkipy_serving.models.deepseek_v4.rank_layout import (
    build_attention_dp_lane_routes,
    build_moe_ep_row_groups,
    build_replica_groups,
    build_tp_row_groups,
    coord_for_rank,
    validate_v4_rank_layout,
)
from nkipy_serving.models.deepseek_v4.warmup_support import (
    _device_zero_allocator,
    _profiled_device_allocator,
    _warmup_trace,
    run_dsv4_executor_warmup,
)
from nkipy_serving.models.deepseek_v4.weights import _load_deepseek_v4_weights
from nkipy_serving.profiling import StartupProfiler


class DeepseekV4Executor(Dsv4NeffRuntimeMixin):
    """DSV4 executor facade used by ModelRunner.

    Populates lane/group metadata during startup, owns optional DSV4
    attention metadata preparation, and dispatches to an installed sampled
    forward. The no-forward path intentionally raises.
    """

    def __init__(
        self,
        model_config: DeepseekV4ModelConfig,
        kv_pool: Any,
        runtime_config: Any,
    ) -> None:
        startup_t0 = time.monotonic()
        startup_profiler = StartupProfiler("dsv4_executor_startup")

        def trace_stage(stage: str) -> None:
            startup_profiler.record(stage)
            _warmup_trace(
                f"executor init {stage} elapsed_s={time.monotonic() - startup_t0:.1f}"
            )

        trace_stage("start")
        self._model_config = model_config
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config

        if runtime_config.execution_backend != "nkipy":
            raise RuntimeError(
                "deepseek-v4 executor requires execution_backend='nkipy'"
            )

        # Load HF config + per-rank metadata. No tensor materialization.
        trace_stage("metadata load start")
        _, self._weights = _load_deepseek_v4_weights(model_config)
        trace_stage("metadata load done")

        # Rank layout sanity: tp*ep*replica == total_workers.
        validate_v4_rank_layout(
            tp_degree=runtime_config.tp_degree,
            ep_degree=runtime_config.ep_degree,
            replica_degree=runtime_config.replica_degree,
            world_size=runtime_config.total_workers,
        )
        # `request_lane_rank` identifies the TP row globally (0..rows_total-1);
        # it is the authoritative source for rank-to-lane mapping because
        # `ep_rank` is row-in-replica when replica_degree > 1.
        global_rank = (
            model_config.request_lane_rank * model_config.tp_degree
            + model_config.tp_rank
        )
        self._coord = coord_for_rank(
            rank=global_rank,
            tp_degree=runtime_config.tp_degree,
            ep_degree=runtime_config.ep_degree,
            replica_degree=runtime_config.replica_degree,
        )
        tp_rows = build_tp_row_groups(
            runtime_config.tp_degree,
            runtime_config.ep_degree,
            runtime_config.replica_degree,
        )
        ep_rows = build_moe_ep_row_groups(
            runtime_config.tp_degree,
            runtime_config.ep_degree,
            runtime_config.replica_degree,
        )
        rep_groups = build_replica_groups(
            runtime_config.tp_degree,
            runtime_config.ep_degree,
            runtime_config.replica_degree,
        )
        self._tp_replica_groups = tuple(tuple(group) for group in tp_rows)
        self._moe_ep_replica_groups = tuple(tuple(group) for group in ep_rows)
        self._tp_group = tuple(tp_rows[self._coord.row])
        moe_ep_group_idx = (
            self._coord.replica * runtime_config.tp_degree + self._coord.col
        )
        self._moe_ep_group = tuple(ep_rows[moe_ep_group_idx])
        self._replica_group = tuple(rep_groups[self._coord.replica])
        self._lane_route = build_attention_dp_lane_routes(
            runtime_config.tp_degree,
            runtime_config.ep_degree,
            runtime_config.replica_degree,
        )[self._coord.attn_lane]
        self._neff_runtime_ready = False
        self._attention_backend: Any | None = None
        self._device_state: Any | None = None
        self._device_weights: Any | None = None
        self._request_state_checkpoints: dict[str, Any] = {}
        trace_stage("layout ready; sampled-forward install start")
        self.install_sampled_forward_from_weights()
        trace_stage("done")

    # -- ModelRunner surface ----------------------------------------------

    @property
    def weights(self):
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    def install_sampled_forward_from_weights(
        self,
        *,
        artifacts_dir: Any = None,
        index_construction_max_c_len: int = 0,
        device_weights: Any | None = None,
        load_plan: Any | None = None,
        use_blockwise_moe: bool = True,
    ) -> Any:
        """Install sampled forward from executor-owned sampled weights.

        When weights are not already loaded, the default plan includes the
        no-scale FP8 blockwise-MoE path if ``use_blockwise_moe`` is enabled.
        """
        install_t0 = time.monotonic()
        install_profiler = StartupProfiler(
            "dsv4_sampled_forward_install",
            rank=int(self._coord.rank),
        )

        def trace_stage(stage: str) -> None:
            install_profiler.record(stage)
            _warmup_trace(
                "sampled-forward install "
                f"{stage} elapsed_s={time.monotonic() - install_t0:.1f}"
            )

        trace_stage("start")
        backend = self._attention_backend
        if backend is None:
            trace_stage("attention-runtime start")
            backend = self.initialize_attention_runtime()
            trace_stage("attention-runtime done")
        device_state = self._device_state or getattr(backend, "_device_state", None)
        if device_state is None:
            raise RuntimeError("DSV4 sampled forward requires Dsv4DeviceState")
        if device_weights is None:
            if self._device_weights is None:
                if load_plan is None:
                    from nkipy_serving.models.deepseek_v4.device_weights import (
                        V4LoadPlan,
                    )

                    load_plan = (
                        V4LoadPlan.sampled_blockwise_fp8()
                        if bool(use_blockwise_moe)
                        else V4LoadPlan.sampled()
                    )
                trace_stage("device-weight load start")
                device_weights = self.load_sampled_weights(plan=load_plan)
                trace_stage("device-weight load done")
            else:
                device_weights = self._device_weights
        build_dir = (
            artifacts_dir
            if artifacts_dir is not None
            else self._runtime_config.config_build_dir()
        )

        from nkipy_serving.models.deepseek_v4.assembly.install import (
            build_dsv4_runtime_components_from_weights,
        )

        trace_stage("install sampled-forward start")
        components = build_dsv4_runtime_components_from_weights(
            model_config=self._model_config,
            v4_weights=self._weights,
            device_weights=device_weights,
            max_batch_size=int(getattr(self._runtime_config, "max_requests", 1)),
            max_seq_len=int(getattr(self._runtime_config, "max_context_len", 4096)),
            build_dir=build_dir,
            compiler_args=getattr(self._runtime_config, "nkipy_compiler_args", ""),
            index_construction_max_c_len=index_construction_max_c_len,
            attention_backend=backend,
            device_state=device_state,
            use_blockwise_moe=use_blockwise_moe,
            blockwise_moe_ep_degree=int(self._runtime_config.ep_degree),
            blockwise_moe_ep_rank=int(self._coord.row_in_replica),
            blockwise_moe_ep_replica_groups=self._moe_ep_replica_groups,
            blockwise_moe_tp_degree=int(self._runtime_config.tp_degree),
            blockwise_moe_tp_rank=int(self._coord.col),
            blockwise_moe_tp_replica_groups=self._tp_replica_groups,
            dense_local_topk=int(getattr(self._runtime_config, "dense_local_topk", 1)),
            max_requests_per_step=int(getattr(self._runtime_config, "max_requests", 1)),
            product_prefill_moe_blockwise_fusion_max_rows=int(
                getattr(
                    self._runtime_config,
                    "dsv4_product_prefill_moe_blockwise_fusion_max_rows",
                    0,
                )
            ),
            product_prefill_moe_dispatch_fusion_max_rows=int(
                getattr(
                    self._runtime_config,
                    "dsv4_product_prefill_moe_dispatch_fusion_max_rows",
                    0,
                )
            ),
            product_prefill_dp_attention_post_pre_fusion_max_rows=int(
                getattr(
                    self._runtime_config,
                    "dsv4_product_prefill_dp_attention_post_pre_fusion_max_rows",
                    0,
                )
            ),
        )
        self._attention_backend = backend
        self._device_state = device_state
        self._device_weights = device_weights
        self._init_neff_runtime(components)
        self._neff_runtime_ready = True
        trace_stage("install sampled-forward done")
        return self

    @property
    def lane_metadata(self) -> dict[str, Any]:
        """Return the per-worker lane/group metadata."""
        c = self._coord
        return {
            "rank": c.rank,
            "row": c.row,
            "col": c.col,
            "replica": c.replica,
            "row_in_replica": c.row_in_replica,
            "attn_lane": c.attn_lane,
            "tp_group": list(self._tp_group),
            "moe_ep_group": list(self._moe_ep_group),
            "replica_group": list(self._replica_group),
            "lane_route": self._lane_route.to_dict(),
            "kv_owner_lane": c.attn_lane,
            "kv_replicated": self._lane_route.kv_replicated,
            "model_replica_count": self._runtime_config.replica_degree,
            "local_expert_ids": list(self._weights.local_expert_ids),
        }

    @property
    def device_weights(self) -> Any | None:
        return self._device_weights

    @property
    def attention_backend(self) -> Any | None:
        return self._attention_backend

    @attention_backend.setter
    def attention_backend(self, value: Any | None) -> None:
        self._attention_backend = value

    @property
    def device_state(self) -> Any | None:
        return self._device_state

    @device_state.setter
    def device_state(self, value: Any | None) -> None:
        self._device_state = value

    def load_sampled_weights(self, *, plan: Any | None = None) -> Any:
        """Load and retain sampled V4 device weights for this rank."""
        if self._device_weights is not None:
            return self._device_weights
        load_t0 = time.monotonic()
        from nkipy_serving.models.deepseek_v4.device_weights import (
            V4LoadPlan,
            load_v4_device_weights,
        )

        load_plan = plan if plan is not None else V4LoadPlan.sampled()
        load_profiler = StartupProfiler(
            "dsv4_weight_load",
            rank=int(self._coord.rank),
            plan=str(load_plan),
        )
        load_profiler.record("start")
        _warmup_trace(
            "device-weight load start "
            f"plan={load_plan} elapsed_s={time.monotonic() - load_t0:.1f}"
        )
        self._device_weights = load_v4_device_weights(
            self._model_config,
            self._weights,
            plan=load_plan,
        )
        _warmup_trace(
            "device-weight load done "
            f"layers={len(getattr(self._device_weights, 'layers', ()))} "
            f"elapsed_s={time.monotonic() - load_t0:.1f}"
        )
        load_profiler.record(
            "done",
            layers=len(getattr(self._device_weights, "layers", ())),
        )
        return self._device_weights

    def flush_cache(self) -> None:
        """Clear scheduler KV bookkeeping and DSV4-owned device state."""
        if hasattr(self._kv_pool, "clear"):
            self._kv_pool.clear()
        self._request_state_checkpoints.clear()
        if self._device_state is not None:
            from nkipy_serving.attention.deepseek_v4.state import (
                reset_dsv4_device_state,
            )

            reset_dsv4_device_state(self._device_state)

    def clear_request_state(self, owner_ids: list[int] | tuple[int, ...]) -> None:
        """Clear DSV4-owned persistent rows for completed request owners."""
        owners = {int(v) for v in owner_ids if int(v) >= 0}
        if owners:
            self._request_state_checkpoints = {
                key: checkpoint
                for key, checkpoint in self._request_state_checkpoints.items()
                if int(checkpoint.owner_id) not in owners
            }
        if self._device_state is None:
            return
        from nkipy_serving.attention.deepseek_v4.state import (
            clear_dsv4_device_state_owners,
        )

        clear_dsv4_device_state_owners(
            self._device_state,
            owners,
            artifacts_dir=str(self._runtime_config.config_build_dir()),
        )

    def checkpoint_request_state(
        self,
        *,
        checkpoint_id: str,
        owner_id: int,
        seq_len: int,
        num_tokens: int,
    ) -> None:
        """Checkpoint bounded DSV4 owner-local rows for speculative rollback."""
        if self._device_state is None:
            raise RuntimeError("DSV4 device state is not initialized")
        if not str(checkpoint_id):
            raise ValueError("checkpoint_id must be non-empty")
        from nkipy_serving.attention.deepseek_v4.state import (
            checkpoint_dsv4_device_state_owner,
        )

        self._request_state_checkpoints[str(checkpoint_id)] = (
            checkpoint_dsv4_device_state_owner(
                self._device_state,
                int(owner_id),
                seq_len=int(seq_len),
                num_tokens=int(num_tokens),
                artifacts_dir=str(self._runtime_config.config_build_dir()),
            )
        )

    def restore_request_state(self, checkpoint_id: str) -> None:
        """Restore and consume a previously checkpointed DSV4 owner state."""
        if self._device_state is None:
            raise RuntimeError("DSV4 device state is not initialized")
        key = str(checkpoint_id)
        try:
            checkpoint = self._request_state_checkpoints[key]
        except KeyError as exc:
            raise KeyError(f"unknown DSV4 request-state checkpoint: {key}") from exc
        from nkipy_serving.attention.deepseek_v4.state import (
            restore_dsv4_device_state_owner,
        )

        restore_dsv4_device_state_owner(
            self._device_state,
            checkpoint,
            artifacts_dir=str(self._runtime_config.config_build_dir()),
        )
        del self._request_state_checkpoints[key]

    def initialize_attention_runtime(
        self,
        *,
        alloc_device_cache: Any | None = None,
        alloc_device_scratch: Any | None = None,
        vanilla_mode: bool = False,
    ) -> Any:
        """Allocate DSV4 device state and bind the sparse-attention backend.

        This is intentionally separate from sampled-weight installation so
        runtime state and device weights can be profiled independently.
        """
        if self._attention_backend is not None:
            return self._attention_backend
        runtime_t0 = time.monotonic()
        runtime_profiler = StartupProfiler(
            "dsv4_attention_runtime",
            rank=int(self._coord.rank),
        )
        runtime_profiler.record("start")
        _warmup_trace("attention-runtime start elapsed_s=0.0")

        from nkipy_serving.attention.deepseek_v4.backend import (
            Dsv4SparseAttentionBackend,
        )
        from nkipy_serving.attention.deepseek_v4.state import (
            allocate_dsv4_device_state,
        )

        alloc_profiler = StartupProfiler(
            "dsv4_attention_alloc",
            rank=int(self._coord.rank),
        )
        alloc_cache = _profiled_device_allocator(
            alloc_device_cache or _device_zero_allocator,
            alloc_profiler,
            role="cache",
        )
        alloc_scratch = _profiled_device_allocator(
            alloc_device_scratch or _device_zero_allocator,
            alloc_profiler,
            role="scratch",
        )
        block_size = int(
            getattr(
                self._kv_pool,
                "block_size",
                getattr(self._runtime_config, "kv_cache_block_size", 32),
            )
        )
        max_context_len = int(getattr(self._runtime_config, "max_context_len", 4096))
        kv_slots = int(
            getattr(
                self._kv_pool,
                "size",
                getattr(self._runtime_config, "kv_pool_size", 16384),
            )
        )
        state_size = int(getattr(self._runtime_config, "dsv4_state_size", 0))
        if state_size <= 0:
            raise RuntimeError("DeepSeek-V4 runtime requires dsv4_state_size > 0")
        if state_size < max_context_len:
            raise RuntimeError(
                "dsv4_state_size must cover max_context_len: "
                f"dsv4_state_size={state_size}, max_context_len={max_context_len}"
            )
        if state_size % 128 != 0:
            raise RuntimeError(
                f"dsv4_state_size must be divisible by 128, got {state_size}"
            )
        # Reserve the backend padding sink. The legacy layout uses a single
        # extra slot; the bucketed-prefill write additionally needs a full guard
        # OWNER window block (the masked NKI write redirects padding rows to
        # owner == max_requests), so size SWA for (max_requests + 1) owners only
        # when bucketing is enabled. Real owners stay in [0, max_requests).
        # See dsv4_nki_writeswa_plan.
        _max_requests = int(getattr(self._runtime_config, "max_requests", 1))
        _swa_owners = _max_requests + 1
        owner_swa_slots = _swa_owners * int(self._weights.sliding_window)
        num_slots = max(kv_slots + 1, block_size + 1, owner_swa_slots + 1)
        # Prefill pads on token count (token_buckets); decode pads on batch
        # (request_buckets). Derive both ladders from the SAME normalization the
        # warmup uses (build_precompile_paddings) so the backend's runtime bucket
        # selection exactly matches the buckets that get precompiled -- e.g. the
        # _MIN_BUCKET floor (request_buckets [1] -> bs_paddings [2]) and the
        # chunked-prefill cap. Using raw config tuples here would let decode pick
        # a bucket (e.g. 1) that warmup never compiled -> post-seal late fault.
        from nkipy_serving.runtime.precompile_paddings import (
            build_precompile_paddings,
        )

        _paddings = build_precompile_paddings(self._runtime_config)
        prefill_buckets = tuple(int(v) for v in _paddings.token_paddings)
        decode_buckets = tuple(int(v) for v in _paddings.bs_paddings)
        token_bucket = max((*prefill_buckets, *decode_buckets))
        max_requests = int(getattr(self._runtime_config, "max_requests", 1))
        max_blocks_per_request = max(
            1,
            (max_context_len + block_size - 1) // block_size,
        )
        ratios = tuple(int(r) for r in self._weights.compress_ratios)
        has_indexer = tuple(r == 4 for r in ratios)

        device_state = allocate_dsv4_device_state(
            alloc_cache,
            layer_compress_ratios=ratios,
            layer_has_indexer=has_indexer,
            num_slots_per_layer=num_slots,
            reserve_guard_owner=True,
            head_dim=int(self._weights.head_dim),
            indexer_head_dim=int(self._weights.index_head_dim),
            window_size=int(self._weights.sliding_window),
            max_seq_len=state_size,
            max_batch_size=max_requests,
            prefix=f"dsv4_r{self._coord.rank}",
        )
        _warmup_trace(
            "attention-runtime state allocated "
            f"layers={len(ratios)} num_slots={int(num_slots)} "
            f"state_size={int(state_size)} "
            f"elapsed_s={time.monotonic() - runtime_t0:.1f}"
        )
        runtime_profiler.record(
            "state allocated",
            layers=len(ratios),
            num_slots=int(num_slots),
            state_max_seq_len=int(state_size),
        )
        max_k = int(self._weights.sliding_window)
        if not vanilla_mode:
            from nkipy_serving.models.deepseek_v4.constants import K_TILE

            max_k = ((max_k + int(K_TILE) - 1) // int(K_TILE)) * int(K_TILE)
        backend = Dsv4SparseAttentionBackend(
            num_layers=int(self._weights.num_hidden_layers),
            num_slots_per_layer=num_slots,
            head_dim=int(self._weights.head_dim),
            block_size=block_size,
            window_size=int(self._weights.sliding_window),
            max_k=max_k,
            token_bucket=token_bucket,
            max_requests=max_requests,
            max_blocks_per_request=max_blocks_per_request,
            alloc_device_scratch=None if vanilla_mode else alloc_scratch,
            device_state=device_state,
            artifacts_dir=str(self._runtime_config.config_build_dir()),
            vanilla_mode=bool(vanilla_mode),
            bucket_ladder=prefill_buckets,
            decode_bucket_ladder=decode_buckets,
            fuse_swa_slots_in_attention=not bool(vanilla_mode),
        )
        self._device_state = device_state
        self._attention_backend = backend
        _warmup_trace(
            f"attention-runtime done elapsed_s={time.monotonic() - runtime_t0:.1f}"
        )
        runtime_profiler.record("done")
        return backend

    def prepare_attention_metadata(
        self,
        attn_metadata: Any,
        *,
        positions: np.ndarray,
        **_extra: Any,
    ) -> Any:
        """Wrap generic runtime metadata for the DSV4 backend if installed."""
        backend = self._attention_backend
        if backend is None:
            return attn_metadata

        from nkipy_serving.attention.deepseek_v4.metadata import (
            SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
            SparseAttentionMetadata,
        )
        from nkipy_serving.attention.deepseek_v4.types import (
            Dsv4AttentionMetadata,
            Dsv4DpAttentionSuperstepMetadata,
        )

        total = int(getattr(attn_metadata, "total_tokens", 0))
        max_k = int(getattr(backend, "max_k", 0))
        if total < 0:
            raise RuntimeError(f"DSV4 metadata total_tokens must be >= 0, got {total}")
        if max_k <= 0:
            raise RuntimeError(
                "DSV4 attention backend must expose positive max_k for metadata "
                f"preparation, got {max_k}"
            )
        sparse = SparseAttentionMetadata(
            topk_indices=np.zeros((total, max_k), dtype=np.int32),
            topk_lens=np.zeros((total,), dtype=np.int32),
            num_kv_positions=int(getattr(backend, "num_slots_per_layer", 0)),
            window_size=int(getattr(backend, "window_size", 0)),
            index_topk=0,
            index_space=SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
        )
        pos = np.asarray(positions[:total], dtype=np.int32).reshape(-1)
        state_owner_ids = None
        forward_batch = _extra.get("forward_batch")
        if forward_batch is not None and hasattr(forward_batch, "state_owner_ids"):
            owners = np.asarray(
                getattr(forward_batch, "state_owner_ids"),
                dtype=np.int32,
            ).reshape(-1)
            batch_size = int(getattr(attn_metadata, "batch_size", owners.shape[0]))
            if owners.shape != (batch_size,):
                raise RuntimeError(
                    "DSV4 state_owner_ids must be [batch_size], got "
                    f"{owners.shape} for batch={batch_size}"
                )
            qsl = np.asarray(
                getattr(attn_metadata, "query_start_loc"),
                dtype=np.int64,
            ).reshape(-1)
            if qsl.shape != (batch_size + 1,):
                raise RuntimeError(
                    "DSV4 query_start_loc shape is inconsistent with batch: "
                    f"shape={qsl.shape}, batch={batch_size}"
                )
            state_owner_ids = np.zeros((total,), dtype=np.int32)
            for req_idx in range(batch_size):
                start = int(qsl[req_idx])
                end = int(qsl[req_idx + 1])
                state_owner_ids[start:end] = owners[req_idx]
        dp_superstep = None
        if forward_batch is not None and bool(
            getattr(forward_batch, "dp_attention_superstep", False)
        ):
            dp_superstep = Dsv4DpAttentionSuperstepMetadata(
                num_lanes=int(getattr(forward_batch, "dp_attention_num_lanes")),
                lane_token_counts=np.asarray(
                    getattr(forward_batch, "dp_attention_lane_token_counts"),
                    dtype=np.int32,
                ),
                lane_batch_sizes=np.asarray(
                    getattr(forward_batch, "dp_attention_lane_batch_sizes"),
                    dtype=np.int32,
                ),
                lane_token_offsets=np.asarray(
                    getattr(forward_batch, "dp_attention_lane_token_offsets"),
                    dtype=np.int32,
                ),
                lane_batch_offsets=np.asarray(
                    getattr(forward_batch, "dp_attention_lane_batch_offsets"),
                    dtype=np.int32,
                ),
            )
        metadata = Dsv4AttentionMetadata(
            base=attn_metadata,
            sparse=sparse,
            positions=pos,
            state_owner_ids=state_owner_ids,
            dp_superstep=dp_superstep,
        )
        if dp_superstep is not None:
            # Lane-local sampled attention prepares backend scratch once per
            # stack after the rank slices metadata to its lane. Preparing the
            # full superstep here would reintroduce full-batch scratch/kernels.
            return metadata
        return backend.prepare(metadata)

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_caches: list[Any],
        attn_metadata: Any,
        *,
        token_bucket: int | None = None,
        real_total_tokens: int | None = None,
        sampling_batch: Any | None = None,
        **_extra: Any,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Run one model step.

        Sampled NKIPy returns sampled-token candidates from ``LogitsProcessor``.
        """
        n = (
            int(real_total_tokens)
            if real_total_tokens is not None
            else int(input_ids.size)
        )
        if self._neff_runtime_ready:
            return self._forward_sampled(
                input_ids,
                positions,
                n,
                attn_metadata=attn_metadata,
                token_bucket=token_bucket,
                sampling_batch=sampling_batch,
            )
        raise RuntimeError(
            "DeepSeek-V4 sampled forward is not installed. Production serving "
            "must initialize the DSV4 sampled path."
        )

    def _forward_sampled(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        n: int,
        *,
        attn_metadata: Any,
        token_bucket: int | None,
        sampling_batch: Any | None,
    ) -> dict[str, np.ndarray]:
        """Run sampled DSV4 through LogitsProcessor; never download full logits."""
        if getattr(self, "logits_processor", None) is None:
            raise RuntimeError("DSV4 sampled forward missing LogitsProcessor")
        if token_bucket is None:
            raise RuntimeError("DSV4 sampled forward requires a scheduler token_bucket")
        bucket = max(int(token_bucket), int(n))
        ids, start_pos, sampled_metadata = self._prepare_sampled_input(
            input_ids,
            positions,
            n,
            attn_metadata=attn_metadata,
            token_bucket=bucket,
        )
        return self.forward_sampled(
            ids,
            start_pos=start_pos,
            metadata=sampled_metadata,
            token_bucket=bucket,
            sampling_batch=sampling_batch,
        )

    def _prepare_sampled_input(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        n: int,
        *,
        attn_metadata: Any,
        token_bucket: int | None = None,
    ) -> tuple[np.ndarray, int, Any | None]:
        """Convert packed scheduler input to the DSV4 generation rectangle.

        Sampled DSV4 still consumes ``[batch, q_len]``. This helper allows
        packed batches only when every request contributes the same query
        length. Compressed-layer state math still takes a scalar ``start_pos``,
        so compressed multi-request batches also require the same start
        position across requests.
        """
        if int(n) <= 0:
            raise RuntimeError("DSV4 sampled forward received no tokens")

        base = getattr(attn_metadata, "base", attn_metadata)
        sampled_metadata = (
            attn_metadata
            if hasattr(attn_metadata, "base") and hasattr(attn_metadata, "sparse")
            else None
        )
        batch_size = int(getattr(base, "batch_size", 1))
        flat_ids = np.asarray(input_ids[:n], dtype=np.int64)
        pos = np.asarray(positions[:n], dtype=np.int64)
        has_compressed = bool(getattr(self, "has_compressed_layers", False))

        if batch_size == 1:
            start_pos = int(pos[0]) if pos.size else 0
            if has_compressed and pos.size:
                expected = np.arange(
                    start_pos,
                    start_pos + int(n),
                    dtype=np.int64,
                )
                if not np.array_equal(pos, expected):
                    raise RuntimeError(
                        "DSV4 compressed sampled path requires "
                        "contiguous positions for single-request chunks; "
                        f"got {pos.tolist()}"
                    )
            return flat_ids.reshape(1, int(n)), start_pos, sampled_metadata

        if base is None or not hasattr(base, "query_start_loc"):
            raise RuntimeError(
                "DSV4 sampled multi-request batches require "
                "attention metadata with query_start_loc"
            )
        qsl = np.asarray(base.query_start_loc, dtype=np.int64).reshape(-1)
        if qsl.shape[0] < batch_size + 1:
            raise RuntimeError(
                "DSV4 sampled query_start_loc too short: "
                f"shape={qsl.shape}, batch={batch_size}"
            )
        q_lens = qsl[1 : batch_size + 1] - qsl[:batch_size]
        if int(q_lens.sum()) != int(n):
            raise RuntimeError(
                "DSV4 sampled token count mismatch: "
                f"query_lens_sum={int(q_lens.sum())}, n={int(n)}"
            )
        if np.any(q_lens <= 0):
            raise RuntimeError(f"DSV4 sampled empty query lens: {q_lens}")
        if not np.all(q_lens == q_lens[0]):
            raise RuntimeError(
                "DSV4 sampled currently requires rectangular packed "
                f"query lengths; got {q_lens.tolist()}"
            )

        q_len = int(q_lens[0])
        starts = (
            pos[qsl[:batch_size]] if pos.size else np.zeros(batch_size, dtype=np.int64)
        )
        if has_compressed:
            if not np.all(starts == starts[0]):
                raise RuntimeError(
                    "DSV4 compressed sampled path requires a shared "
                    f"start_pos across requests; got {starts.tolist()}"
                )
            if pos.size:
                pos_rect = pos.reshape(batch_size, q_len)
                expected = starts[:, None] + np.arange(q_len, dtype=np.int64)[None, :]
                if not np.array_equal(pos_rect, expected):
                    raise RuntimeError(
                        "DSV4 compressed sampled path requires "
                        "contiguous rectangular positions; got "
                        f"{pos_rect.tolist()}"
                    )
        ids = flat_ids.reshape(batch_size, q_len)
        return ids, int(starts[0]) if starts.size else 0, sampled_metadata

    def warmup(self, paddings: Any = None) -> None:
        """Compile and first-touch sampled DSV4 bucket paths."""
        return run_dsv4_executor_warmup(self, paddings)
