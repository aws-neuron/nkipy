"""GPT-OSS executor: device compilation, kernel management, and forward pass."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import (
    FORWARD_MODE_DECODE,
    FORWARD_MODE_EXTEND,
    AttentionMetadata,
)
from nkipy_serving.attention.nki_step_inputs import (
    PreparedNkiStepInputs,
    allocate_prepared_nki_step_inputs,
    initialize_prepared_nki_step_inputs,
    prepare_prepared_nki_step_inputs,
)
from nkipy_serving.models._device_utils import (
    _get_device_kernel_cls,
    _get_device_tensor_cls,
)
from nkipy_serving.models._device_utils import (
    alloc_device_scratch as _alloc_device_scratch,
)
from nkipy_serving.models._device_utils import (
    allocate_device_kv_cache as _allocate_device_kv_cache_shared,
)
from nkipy_serving.models._device_utils import (
    flush_device_kv_cache as _flush_device_kv_cache,
)
from nkipy_serving.models._device_utils import (
    join_compiler_args as _join_compiler_args,
)
from nkipy_serving.models._device_utils import (
    load_generated_kernel_fn as _load_generated_kernel_fn,
)
from nkipy_serving.models._device_utils import (
    nki_tile_plan_sample_arrays as _nki_tile_plan_sample_arrays,
)
from nkipy_serving.models._device_utils import (
    pre_allocate_kv_cache_zeros as _pre_allocate_kv_cache_zeros,
)
from nkipy_serving.models.gpt_oss.codegen import (
    generate_full_decode_kernel_source as _generate_full_decode_kernel_source,
)
from nkipy_serving.models.gpt_oss.config import GptOssModelConfig, GptOssWeights
from nkipy_serving.models.gpt_oss.graph_fns import (
    _build_rope_cache_for_positions_yarn,
    embedding_fn,
    prefill_layer_post_moe_fn,
    prefill_layer_pre_moe_nki_fn,
    tp_reduce_scatter_hidden_fn,
)
from nkipy_serving.models.gpt_oss.weights import (
    _load_gpt_oss_weights,
    _SafeTensorReader,
)
from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor as _overwrite_device_tensor,
)
from nkipy_serving.models.reload_utils import (
    resolve_model_snapshot_path,
    upsert_device_tensor,
)
from nkipy_serving.ops.moe.blockwise_index import (
    BLOCK_SIZE as MOE_BLOCK_SIZE,
)
from nkipy_serving.ops.moe.blockwise_index import (
    get_n_blocks as moe_get_n_blocks,
)
from nkipy_serving.ops.moe.blockwise_index import (
    preload_compiled_impl as preload_blockwise_index_ext,
)
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule
from nkipy_serving.ops.vocab_parallel_embedding import get_vocab_parallel_shard
from nkipy_serving.profiling import PROFILING_ENABLED, ProfileWriter
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge
from nkipy_serving.runtime.warmup import (
    SyntheticWarmupStep,
    build_standard_warmup_steps,
    build_synthetic_warmup_inputs,
    run_synthetic_warmup_steps,
)
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessor

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_NKI_MIN_Q_SEQLEN = 128


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _lm_head_vocab_range(weights) -> tuple[int, int]:
    local_vocab = int(weights.local_vocab_size)
    start = int(
        getattr(weights, "lm_head_vocab_offset", int(weights.tp_rank) * local_vocab)
    )
    return start, start + local_vocab


class _GptOssModelProfiler:
    """Lightweight GPT-OSS model-internal profiler."""

    def __init__(self, global_rank: int):
        self._rank = int(global_rank)
        self._step = 0
        self._step_writer: ProfileWriter | None = None
        self._layer_writer: ProfileWriter | None = None
        if PROFILING_ENABLED:
            self._step_writer = ProfileWriter(f"gpt_oss_model_rank{self._rank}_steps")
            if _env_flag("NKIPY_SERVING_PROFILE_GPT_OSS_LAYERS"):
                self._layer_writer = ProfileWriter(
                    f"gpt_oss_model_rank{self._rank}_layers"
                )

    @property
    def enabled(self) -> bool:
        return self._step_writer is not None

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def layer_enabled(self) -> bool:
        return self._layer_writer is not None

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def write_layer(self, record: dict[str, object]) -> None:
        if self._layer_writer is not None:
            self._layer_writer.write(record)

    def write_step(self, record: dict[str, object]) -> None:
        if self._step_writer is not None:
            self._step_writer.write(record)
            self._step_writer.flush()
            if self._layer_writer is not None:
                self._layer_writer.flush()


class GptOssExecutor:
    """GPT-OSS executor (device-only, per-layer prefill + full decode graphs)."""

    def __init__(self, model_config: GptOssModelConfig, kv_pool, runtime_config):
        from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

        self._model_config = model_config
        if runtime_config.execution_backend != "nkipy":
            raise RuntimeError(
                "gpt-oss executor currently requires execution_backend='nkipy'"
            )
        if runtime_config.attention_backend != "NKIBlockSparseFlashAttention":
            raise RuntimeError(
                "gpt-oss executor requires attention_backend='NKIBlockSparseFlashAttention'"
            )
        try:
            _get_device_tensor_cls()
        except ImportError:
            raise RuntimeError("nkipy runtime not available")

        snapshot_path, self._weights = _load_gpt_oss_weights(model_config)
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config
        if int(self._weights.num_hidden_layers) % 4 != 0:
            raise RuntimeError(
                "gpt-oss requires num_hidden_layers divisible by 4. "
                f"Got {self._weights.num_hidden_layers}."
            )
        self._compiler_args = runtime_config.nkipy_compiler_args
        self._sampled_local_topk = int(runtime_config.dense_local_topk)
        self._precompile_paddings = build_precompile_paddings(runtime_config)
        self._max_requests_per_step = int(
            self._precompile_paddings.max_padded_batch_size
        )
        _global_rank = (
            self._weights.ep_rank * self._weights.tp_degree + self._weights.tp_rank
        )
        self._build_dir = f"{runtime_config.config_build_dir()}/rank{_global_rank}"
        self._model_profiler = _GptOssModelProfiler(_global_rank)
        self._kernel_init_lock = threading.Lock()
        self._profile_suppression_depth = 0
        self._log_startup_progress = _env_flag("NKIPY_SERVING_LOG_STARTUP_PROGRESS")
        self._startup_log_path = (
            Path(self._build_dir) / "gpt_oss_startup.log"
            if self._log_startup_progress and int(_global_rank) == 0
            else None
        )

        self._attn_softmax_scale = 1.0 / (float(self._weights.head_dim) ** 0.5)
        if int(self._weights.head_dim) > 128:
            raise RuntimeError(
                "NKI blocksparse attention requires head_dim <= 128. "
                f"Got head_dim={self._weights.head_dim}."
            )

        # Device KV cache for NKI attention backend (+1 scratch block for padded slots).
        ensure_nki_bridge()

        self._nki_num_blocks = self._kv_pool.num_blocks + 1
        self._kv_cache_dev = _allocate_device_kv_cache_shared(
            num_hidden_layers=self._weights.num_hidden_layers,
            num_kv_heads=self._weights.num_kv_heads,
            head_dim=int(self._weights.head_dim),
            block_size=self._kv_pool.block_size,
            num_blocks=self._nki_num_blocks,
            dtype=self._weights.dtype,
        )
        self._kv_cache_zeros = _pre_allocate_kv_cache_zeros(
            num_blocks=self._nki_num_blocks,
            num_kv_heads=self._weights.num_kv_heads,
            block_size=self._kv_pool.block_size,
            head_dim=int(self._weights.head_dim),
            dtype=self._weights.dtype,
        )

        # Load + upload weights (device tensors).
        self._shared_tensors, self._layer_tensors = self._upload_all_weights(
            model_config,
            weights=self._weights,
            snapshot_path=snapshot_path,
        )
        if self._sampled_local_topk > int(self._weights.local_vocab_size):
            raise RuntimeError(
                "dense_local_topk exceeds local LM-head vocab shard size: "
                f"{self._sampled_local_topk} > {int(self._weights.local_vocab_size)}"
            )

        # Per-token-bucket compiled kernels.
        self._kernels_by_bucket: dict[int, object] = {}
        # NKIPy may squeeze leading dim-1 outputs; normalize bs buckets to >= 2.
        self._bs_buckets = tuple(
            sorted({max(int(b), 2) for b in self._precompile_paddings.bs_paddings})
        )
        self._prefill_token_buckets = self._normalize_prefill_token_buckets(
            self._precompile_paddings.token_paddings,
            int(self._weights.tp_degree),
        )
        self._decode_token_buckets = self._normalize_full_decode_token_buckets(
            self._precompile_paddings.bs_paddings,
        )

        # LogitsProcessors for LM-head sampling.
        # GPT-OSS has two variants: with seq-parallel gather_hidden (prefill)
        # and without (no-SP decode path).
        from nkipy_serving.runtime.parallel_groups import build_tp_replica_groups

        w = self._weights
        global_rank = int(w.ep_rank) * int(w.tp_degree) + int(w.tp_rank)
        total_workers = int(w.tp_degree) * int(w.ep_degree)
        tp_groups = build_tp_replica_groups(int(w.tp_degree), int(w.ep_degree))
        tp_groups_tuple = tuple(tuple(g) for g in tp_groups)
        self._logits_processor_sp = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=tp_groups_tuple,
            collective_rank=global_rank,
            collective_world_size=total_workers,
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._sampled_local_topk,
            gather_hidden=True,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=max(self._bs_buckets),
        )
        self._logits_processor_no_sp = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=tp_groups_tuple,
            collective_rank=global_rank,
            collective_world_size=total_workers,
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._sampled_local_topk,
            gather_hidden=False,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=max(self._bs_buckets),
        )

    @property
    def weights(self) -> GptOssWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- Weight upload --------------------------------------------------------

    def _upload_all_weights(
        self,
        model_config: GptOssModelConfig,
        *,
        weights: GptOssWeights | None = None,
        snapshot_path: Path | None = None,
        existing_shared: dict[str, object] | None = None,
        existing_layers: list[dict[str, object]] | None = None,
    ):
        w = self._weights if weights is None else weights
        if snapshot_path is None:
            snapshot_path = resolve_model_snapshot_path(
                model_config.hf_model_id,
                revision=model_config.hf_revision,
                local_files_only=model_config.hf_local_files_only,
            )
        reader = _SafeTensorReader(snapshot_path)
        try:
            shared: dict[str, object] = {}
            layers: list[dict[str, object]] = []

            # --- Shared weights ---
            emb_shard = get_vocab_parallel_shard(
                vocab_size=int(w.vocab_size),
                rank=int(w.tp_rank),
                world_size=int(w.tp_degree),
            )
            emb_slice = reader.get_slice("model.embed_tokens.weight")[
                int(emb_shard.vocab_start_index) : int(emb_shard.vocab_end_index), :
            ]
            emb = np.asarray(emb_slice, dtype=w.dtype)
            shared["embeddings"] = upsert_device_tensor(
                _get_device_tensor_cls(),
                emb,
                name="embeddings",
                existing=None
                if existing_shared is None
                else existing_shared["embeddings"],
            )
            del emb, emb_slice

            fn = np.asarray(reader.get_tensor("model.norm.weight"), dtype=w.dtype)
            shared["final_norm"] = upsert_device_tensor(
                _get_device_tensor_cls(),
                fn,
                name="final_norm",
                existing=None
                if existing_shared is None
                else existing_shared["final_norm"],
            )
            del fn

            # LM head: shard vocab across TP.
            v0, v1 = _lm_head_vocab_range(w)
            lm_slice = reader.get_slice("lm_head.weight")[v0:v1, :]
            lm = np.asarray(lm_slice, dtype=w.dtype)
            shared["lm_head"] = upsert_device_tensor(
                _get_device_tensor_cls(),
                lm,
                name="lm_head",
                existing=None
                if existing_shared is None
                else existing_shared["lm_head"],
            )
            del lm, lm_slice

            # --- Per-layer weights ---
            local_num_heads = w.num_heads
            local_num_kv = w.num_kv_heads
            head_dim = w.head_dim
            hidden = w.hidden_size
            tp_degree = w.tp_degree
            tp_rank = w.tp_rank

            q_out = local_num_heads * head_dim
            kv_out = local_num_kv * head_dim
            q_row0 = (tp_rank * local_num_heads) * head_dim
            q_row1 = q_row0 + q_out
            kv_row0 = tp_rank * head_dim
            kv_row1 = kv_row0 + kv_out

            # Bias sharding trick (output bias is applied before reduce-scatter/all-reduce).
            hidden_shard = hidden // tp_degree
            h0 = tp_rank * hidden_shard
            h1 = h0 + hidden_shard

            # MoE TP shard over intermediate dimension.
            I_local = w.local_intermediate_size
            i0 = tp_rank * I_local
            i1 = i0 + I_local

            # EP shard over expert dimension.
            E_local = w.local_num_experts
            e0 = w.ep_rank * E_local
            e1 = e0 + E_local

            for layer_idx in range(w.num_hidden_layers):
                existing_layer = (
                    None if existing_layers is None else existing_layers[layer_idx]
                )
                prefix = f"model.layers.{layer_idx}"

                # Norms.
                in_norm = np.asarray(
                    reader.get_tensor(f"{prefix}.input_layernorm.weight"), dtype=w.dtype
                )
                post_norm = np.asarray(
                    reader.get_tensor(f"{prefix}.post_attention_layernorm.weight"),
                    dtype=w.dtype,
                )

                # Attention weights: slice in HF orientation first (row slices), then transpose.
                q_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.q_proj.weight")[
                        q_row0:q_row1, :
                    ],
                    dtype=w.dtype,
                ).T
                k_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.k_proj.weight")[
                        kv_row0:kv_row1, :
                    ],
                    dtype=w.dtype,
                ).T
                v_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.v_proj.weight")[
                        kv_row0:kv_row1, :
                    ],
                    dtype=w.dtype,
                ).T

                q_b = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.q_proj.bias")[q_row0:q_row1],
                    dtype=w.dtype,
                )
                k_b = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.k_proj.bias")[
                        kv_row0:kv_row1
                    ],
                    dtype=w.dtype,
                )
                v_b = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.v_proj.bias")[
                        kv_row0:kv_row1
                    ],
                    dtype=w.dtype,
                )

                # Output projection: slice input columns and transpose.
                o_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.o_proj.weight")[
                        :, q_row0:q_row1
                    ],
                    dtype=w.dtype,
                ).T
                o_b_full = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.o_proj.bias"), dtype=w.dtype
                )
                o_b_full[:h0] = 0
                o_b_full[h1:] = 0

                sinks_full = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.sinks"), dtype=w.dtype
                ).reshape((-1,))
                sinks = sinks_full[
                    tp_rank * local_num_heads : (tp_rank + 1) * local_num_heads
                ].reshape((local_num_heads, 1))

                # Router.
                router_w = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.router.weight"), dtype=w.dtype
                ).T
                router_b = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.router.bias"), dtype=w.dtype
                )

                # Experts (shard I, keep experts intact). Cast weights to float8 for memory.
                #
                # GPT-OSS stores gate+up projections in an interleaved layout:
                #   [gate_0, up_0, gate_1, up_1, ...]
                # (see HF `GptOssExperts._apply_gate`: gate_up[..., ::2] / gate_up[..., 1::2]).
                # Our NKI blockwise kernel expects a separated layout [gate, up] with
                # shape [E, H, 2, I_local], so we slice a contiguous interleaved chunk
                # [2*i0 : 2*i1] and reshape/transpose.
                gup_sl = reader.get_slice(f"{prefix}.mlp.experts.gate_up_proj")
                gup_chunk = np.asarray(
                    gup_sl[:, :, 2 * i0 : 2 * i1], dtype=w.dtype
                )  # [E, H, 2*I_local]
                gup_chunk = gup_chunk.reshape(
                    (gup_chunk.shape[0], gup_chunk.shape[1], I_local, 2)
                )
                gup_chunk = np.transpose(gup_chunk, (0, 1, 3, 2))  # [E, H, 2, I_local]
                # EP slice: keep only local experts.
                gup_chunk = gup_chunk[e0:e1]  # [E_local, H, 2, I_local]
                gup_w = gup_chunk.astype(ml_dtypes.float8_e5m2)
                del gup_chunk, gup_sl

                gupb_sl = reader.get_slice(f"{prefix}.mlp.experts.gate_up_proj_bias")
                gupb_chunk = np.asarray(
                    gupb_sl[:, 2 * i0 : 2 * i1], dtype=np.float32
                )  # [E, 2*I_local]
                gup_bias = gupb_chunk.reshape(
                    (gupb_chunk.shape[0], I_local, 2)
                )  # [E, I_local, 2]
                # Pre-apply +1 to the "up" bias so the kernel can clamp in shifted space.
                gup_bias[:, :, 1] = gup_bias[:, :, 1] + np.float32(1.0)
                # EP slice.
                gup_bias = gup_bias[e0:e1]  # [E_local, I_local, 2]
                del gupb_chunk, gupb_sl

                down_sl = reader.get_slice(f"{prefix}.mlp.experts.down_proj")
                down_w = np.asarray(down_sl[:, i0:i1, :], dtype=w.dtype)
                # EP slice.
                down_w = down_w[e0:e1].astype(
                    ml_dtypes.float8_e5m2
                )  # [E_local, I_local, H]
                del down_sl

                down_b = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.experts.down_proj_bias"),
                    dtype=w.dtype,
                )
                down_b[:, :h0] = 0
                down_b[:, h1:] = 0
                # EP slice.
                down_b = down_b[e0:e1]  # [E_local, H]
                down_bias_bc = np.broadcast_to(
                    down_b[:, None, :],
                    (down_b.shape[0], MOE_BLOCK_SIZE, down_b.shape[1]),
                ).copy()
                del down_b

                # Upload to device.
                lt: dict[str, object] = {
                    "input_norm": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        in_norm,
                        name=f"in_norm_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["input_norm"],
                    ),
                    "post_attn_norm": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        post_norm,
                        name=f"post_norm_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["post_attn_norm"],
                    ),
                    "w_q": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        q_w,
                        name=f"wq_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["w_q"],
                    ),
                    "w_k": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        k_w,
                        name=f"wk_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["w_k"],
                    ),
                    "w_v": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        v_w,
                        name=f"wv_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["w_v"],
                    ),
                    "b_q": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        q_b,
                        name=f"bq_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["b_q"],
                    ),
                    "b_k": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        k_b,
                        name=f"bk_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["b_k"],
                    ),
                    "b_v": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        v_b,
                        name=f"bv_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["b_v"],
                    ),
                    "w_o": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        o_w,
                        name=f"wo_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["w_o"],
                    ),
                    "b_o": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        o_b_full,
                        name=f"bo_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["b_o"],
                    ),
                    "sink": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        sinks,
                        name=f"sink_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["sink"],
                    ),
                    "router_w": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        router_w,
                        name=f"router_w_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["router_w"],
                    ),
                    "router_b": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        router_b,
                        name=f"router_b_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["router_b"],
                    ),
                    "gup_w": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        gup_w,
                        name=f"gup_w_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["gup_w"],
                    ),
                    "gup_bias": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        gup_bias,
                        name=f"gup_bias_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["gup_bias"],
                    ),
                    "down_w": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        down_w,
                        name=f"down_w_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["down_w"],
                    ),
                    "down_bias_bc": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        down_bias_bc,
                        name=f"down_bias_bc_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["down_bias_bc"],
                    ),
                }
                layers.append(lt)

                # Free CPU intermediates eagerly.
                del in_norm, post_norm
                del (
                    q_w,
                    k_w,
                    v_w,
                    q_b,
                    k_b,
                    v_b,
                    o_w,
                    o_b_full,
                    sinks_full,
                    sinks,
                    router_w,
                    router_b,
                )
                del gup_w, gup_bias, down_w, down_bias_bc

            return shared, layers
        finally:
            reader.close()

    @staticmethod
    def _validate_reload_compatibility(
        current: GptOssWeights,
        new: GptOssWeights,
    ) -> None:
        fields = (
            "vocab_size",
            "hidden_size",
            "head_dim",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "num_experts",
            "experts_per_token",
            "rms_norm_eps",
            "rope_theta",
            "yarn_factor",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "yarn_original_max_pos",
            "dtype",
            "tp_degree",
            "tp_rank",
            "num_heads",
            "num_kv_heads",
            "local_vocab_size",
            "lm_head_vocab_offset",
            "local_intermediate_size",
            "ep_degree",
            "ep_rank",
            "local_num_experts",
        )
        for field_name in fields:
            if getattr(current, field_name) != getattr(new, field_name):
                raise RuntimeError(
                    "Reloaded GPT-OSS weights are incompatible with the running "
                    f"executor: field {field_name} changed from "
                    f"{getattr(current, field_name)!r} to {getattr(new, field_name)!r}"
                )

    def reload_weights_from_disk(self, model_path: str) -> None:
        reload_config = replace(self._model_config, hf_model_id=str(model_path))
        snapshot_path, new_weights = _load_gpt_oss_weights(reload_config)
        self._validate_reload_compatibility(self._weights, new_weights)
        self._shared_tensors, self._layer_tensors = self._upload_all_weights(
            reload_config,
            weights=new_weights,
            snapshot_path=snapshot_path,
            existing_shared=self._shared_tensors,
            existing_layers=self._layer_tensors,
        )
        self._weights = new_weights
        self._model_config = reload_config

    def flush_cache(self) -> None:
        _flush_device_kv_cache(self._kv_cache_dev, self._kv_cache_zeros, self._kv_pool)

    # -- Kernel compilation ---------------------------------------------------

    @dataclass
    class _BucketScratch:
        input_ids: object
        cos: object
        sin: object
        hidden_shard: object
        hidden_attn_shard: object
        q: object
        k: object
        v: object
        context: object
        context_attn: object | None
        context_host: np.ndarray
        topk: object
        aff: object
        normed_full: object
        moe_out: object
        token_pos: object
        block_to_expert: object
        q_attn: object | None
        k_attn: object | None
        v_attn: object | None
        q_attn_host: np.ndarray | None
        k_attn_host: np.ndarray | None
        v_attn_host: np.ndarray | None
        nki_step_inputs: PreparedNkiStepInputs
        full_last_token_indices: object
        full_last_token_indices_host: np.ndarray
        full_last_token_indices_by_bs: dict[int, object]
        full_last_token_indices_host_by_bs: dict[int, np.ndarray]
        full_hidden_aux: object | None
        full_top1_vals: object | None
        full_top1_idx: object | None
        full_top1_vals_by_bs: dict[int, object]
        full_top1_idx_by_bs: dict[int, object]
        full_topk_vals: object | None
        full_topk_idx: object | None
        full_topk_vals_by_bs: dict[int, object]
        full_topk_idx_by_bs: dict[int, object]

    @dataclass
    class _CompiledKernels:
        token_bucket: int
        token_shard: int
        embed_kernel: object | None
        embed_all_reduce_kernel: object | None
        embed_reduce_scatter_kernel: object | None
        prefill_pre_moe_kernel: object | None
        prefill_post_moe_kernel: object | None
        pre_attn_kernel: object | None
        kv_update_kernel: object | None
        attention_kernel: object | None
        post_attn_kernel: object | None
        router_kernel: object | None
        moe_output_init: object | None
        moe_kernel: object | None
        moe_decode_kernel: object | None  # None when token_bucket > TILE_SIZE (128)
        full_decode_kernel: object | None
        # LM-head sampling kernels owned by LogitsProcessor (self._logits_processor_sp / _no_sp).
        full_static_inputs: dict[str, object] | None
        scratch: object

    def _use_top1_fast_path(self) -> bool:
        return self._sampled_local_topk == 1

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        value = int(value)
        multiple = int(multiple)
        if value <= 0:
            raise RuntimeError(f"value must be > 0, got {value}")
        if multiple <= 0:
            raise RuntimeError(f"multiple must be > 0, got {multiple}")
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _normalize_full_decode_token_buckets(
        bs_paddings: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(sorted({max(int(bucket), 1) for bucket in bs_paddings}))

    @classmethod
    def _normalize_prefill_token_buckets(
        cls,
        token_paddings: tuple[int, ...],
        tp_degree: int,
    ) -> tuple[int, ...]:
        # The LM-head now supports token_shard==1, but decode still performs
        # best on current kernels when each TP rank handles at least 2 tokens.
        # Apply the same floor to prefill buckets so TP8 does not execute
        # token_shard==1 prefill/extend kernels for small scheduler buckets.
        min_bucket = max(2 * int(tp_degree), 2)
        return tuple(
            sorted(
                {
                    cls._round_up_to_multiple(
                        max(int(bucket), min_bucket), int(tp_degree)
                    )
                    for bucket in token_paddings
                }
            )
        )

    @staticmethod
    def _lm_head_hidden_input_sample(hidden_shard: np.ndarray) -> np.ndarray:
        """Match NKIPy input canonicalization for token_shard==1 LM-head trace."""
        if hidden_shard.ndim == 2 and int(hidden_shard.shape[0]) == 1:
            return np.empty((int(hidden_shard.shape[1]),), dtype=hidden_shard.dtype)
        return hidden_shard

    @staticmethod
    def _make_lm_head_hidden_input_view(hidden_shard_dev, device_tensor_cls=None):
        """Alias a (1, H) shard as (H,) for kernels compiled with rank-1 input."""
        shape = tuple(int(dim) for dim in hidden_shard_dev.shape)
        if len(shape) != 2 or shape[0] != 1:
            return hidden_shard_dev
        cls = device_tensor_cls or _get_device_tensor_cls()
        if cls is None:
            raise RuntimeError("NKIPy DeviceTensor class is not initialized")
        return cls(
            tensor_ref=hidden_shard_dev.tensor_ref,
            shape=(shape[1],),
            dtype=hidden_shard_dev.dtype,
            name=f"{getattr(hidden_shard_dev, 'name', 'hidden_shard')}_rank1",
        )

    @staticmethod
    def _kernel_artifact_dir(kernel: object | None) -> str | None:
        if kernel is None:
            return None
        artifact_dir = getattr(kernel, "compiled_artifact_dir", None)
        if artifact_dir is None:
            return None
        return str(artifact_dir)

    def _startup_artifact_summary(self, **kernels: object | None) -> str:
        parts: list[str] = []
        for label, kernel in kernels.items():
            artifact_dir = self._kernel_artifact_dir(kernel)
            if artifact_dir:
                parts.append(f"{label}={artifact_dir}")
        return " ".join(parts)

    def _log_startup(self, message: str) -> None:
        if not getattr(self, "_log_startup_progress", False):
            return
        rank = int(self._weights.ep_rank) * int(self._weights.tp_degree) + int(
            self._weights.tp_rank
        )
        if rank != 0:
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{timestamp}] [gpt_oss][rank={rank}] {message}"
        logger.info("%s", line)
        log_path = getattr(self, "_startup_log_path", None)
        if log_path is None:
            return
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except OSError:
            # Startup logging is best-effort and must never block model init.
            return

    def _prepare_nki_step_inputs(
        self,
        *,
        scratch: _BucketScratch,
        token_bucket: int,
        attn_bucket: int,
        real_total_tokens: int,
        attn_metadata: AttentionMetadata,
    ) -> dict[str, object]:
        """Build per-step NKI inputs shared across grouped decode kernels."""
        return prepare_prepared_nki_step_inputs(
            scratch.nki_step_inputs,
            _overwrite_device_tensor,
            attn_metadata=attn_metadata,
            real_total_tokens=int(real_total_tokens),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
        )

    @staticmethod
    def _write_common_step_inputs(
        scratch: _BucketScratch,
        *,
        input_ids_shard: np.ndarray,
        cos_np: np.ndarray,
        sin_np: np.ndarray,
    ) -> dict[str, object]:
        _overwrite_device_tensor(scratch.input_ids, input_ids_shard)
        _overwrite_device_tensor(scratch.cos, cos_np)
        _overwrite_device_tensor(scratch.sin, sin_np)
        return {
            "input_ids": scratch.input_ids,
            "cos": scratch.cos,
            "sin": scratch.sin,
        }

    @staticmethod
    def _pad_host_step_vector(
        src: np.ndarray,
        *,
        target_size: int,
        dtype: np.dtype | None = None,
    ) -> np.ndarray:
        arr = np.asarray(src, dtype=dtype)
        if arr.ndim != 1:
            raise RuntimeError(
                f"expected rank-1 host step buffer, got shape={arr.shape}"
            )
        if int(arr.shape[0]) == int(target_size):
            return np.ascontiguousarray(arr)
        if int(arr.shape[0]) > int(target_size):
            return np.ascontiguousarray(arr[: int(target_size)])
        out = np.zeros((int(target_size),), dtype=arr.dtype)
        out[: int(arr.shape[0])] = arr
        return out

    def _prepare_full_decode_last_token_indices(
        self,
        *,
        scratch: _BucketScratch,
        attn_metadata: AttentionMetadata,
        bs_bucket: int | None = None,
    ) -> tuple[int, object]:
        bs = int(attn_metadata.batch_size)
        if bs_bucket is None:
            padded_last = scratch.full_last_token_indices_host
            target = scratch.full_last_token_indices
        else:
            padded_last = scratch.full_last_token_indices_host_by_bs[int(bs_bucket)]
            target = scratch.full_last_token_indices_by_bs[int(bs_bucket)]
        padded_last.fill(0)
        if bs > 0:
            last_indices = (attn_metadata.query_start_loc[1 : bs + 1] - 1).astype(
                np.int32
            )
            padded_last[:bs] = last_indices
        _overwrite_device_tensor(target, padded_last)
        return bs, target

    def _get_full_decode_kernel_fn(
        self,
        *,
        token_bucket: int,
        attn_bucket: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        ep_replica_groups: tuple[tuple[int, ...], ...],
    ):
        w = self._weights
        vocab_start, vocab_end = _lm_head_vocab_range(w)
        mod_name = (
            f"gpt_oss_full_decode_tp{w.tp_degree}_ep{w.ep_degree}"
            f"_embed_layers"
            f"_t{int(token_bucket)}_a{int(attn_bucket)}"
        )
        fn_name = (
            f"gpt_oss_full_decode_embed_layers_local_forward_t{int(token_bucket)}"
            f"_tp{w.tp_degree}_ep{w.ep_degree}"
        )
        return _load_generated_kernel_fn(
            build_dir=self._build_dir,
            mod_name=mod_name,
            fn_name=fn_name,
            source=_generate_full_decode_kernel_source(
                token_bucket=token_bucket,
                attn_bucket=attn_bucket,
                num_hidden_layers=int(w.num_hidden_layers),
                num_heads=int(w.num_heads),
                num_kv_heads=int(w.num_kv_heads),
                head_dim=int(w.head_dim),
                rms_norm_eps=float(w.rms_norm_eps),
                experts_per_token=int(w.experts_per_token),
                tp_degree=int(w.tp_degree),
                ep_degree=int(w.ep_degree),
                ep_rank=int(w.ep_rank),
                local_num_experts=int(w.local_num_experts),
                vocab_start_index=int(vocab_start),
                vocab_end_index=int(vocab_end),
                tp_replica_groups=tp_replica_groups,
                ep_replica_groups=ep_replica_groups,
            ),
        )

    def _resolve_token_buckets(
        self,
        *,
        forward_mode: int,
        input_token_bucket: int,
        real_total_tokens: int,
    ) -> tuple[int, int]:
        compute_token_bucket = int(input_token_bucket)
        if int(forward_mode) == int(FORWARD_MODE_DECODE):
            from nkipy_serving.runtime.shape_guard import select_bucket

            required = max(int(real_total_tokens), 1)
            compute_token_bucket = select_bucket(
                required,
                self._decode_token_buckets,
                "decode token",
            )
        elif int(forward_mode) == int(FORWARD_MODE_EXTEND):
            from nkipy_serving.runtime.shape_guard import select_bucket

            required = max(int(input_token_bucket), int(real_total_tokens))
            compute_token_bucket = select_bucket(
                required,
                self._prefill_token_buckets,
                "extend token",
            )
        attn_token_bucket = max(compute_token_bucket, int(_NKI_MIN_Q_SEQLEN))
        return int(compute_token_bucket), int(attn_token_bucket)

    def _startup_token_buckets(self, paddings=None) -> tuple[int, ...]:
        del paddings
        return tuple(
            sorted(set(self._prefill_token_buckets) | set(self._decode_token_buckets))
        )

    @contextmanager
    def _suspend_model_profiling(self):
        self._profile_suppression_depth += 1
        try:
            yield
        finally:
            self._profile_suppression_depth = max(
                0, self._profile_suppression_depth - 1
            )

    def _startup_warmup_steps(self, paddings=None) -> tuple[SyntheticWarmupStep, ...]:
        if paddings is None:
            paddings = self._precompile_paddings
        return build_standard_warmup_steps(paddings)

    def _ensure_kernels(self, token_bucket: int) -> _CompiledKernels:
        lock = getattr(self, "_kernel_init_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._kernel_init_lock = lock

        cached = self._kernels_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached
        with lock:
            cached = self._kernels_by_bucket.get(int(token_bucket))
            if cached is not None:
                return cached
            return self._compile_kernels_locked(token_bucket)

    def _compile_kernels_locked(self, token_bucket: int) -> _CompiledKernels:
        compile_started = time.perf_counter()
        w = self._weights
        tp_degree = int(w.tp_degree)
        tp_rank = int(w.tp_rank)
        ep_degree = int(w.ep_degree)
        ep_rank = int(w.ep_rank)
        total_workers = tp_degree * ep_degree
        global_rank = ep_rank * tp_degree + tp_rank

        # Build TP/EP replica groups for collective ops.
        from nkipy_serving.runtime.parallel_groups import (
            build_ep_replica_groups,
            build_tp_replica_groups,
        )

        tp_groups = build_tp_replica_groups(tp_degree, ep_degree)
        ep_groups = build_ep_replica_groups(tp_degree, ep_degree)
        tp_groups_tuple = tuple(tuple(g) for g in tp_groups)
        ep_groups_tuple = tuple(tuple(g) for g in ep_groups)

        # Sample tensors (shapes/dtypes only) for compilation.
        dtype = w.dtype
        hidden = w.hidden_size
        head_dim = w.head_dim
        num_heads = w.num_heads
        num_kv = w.num_kv_heads
        q_out = num_heads * head_dim
        kv_out = num_kv * head_dim
        local_E = w.local_num_experts
        vocab_start, vocab_end = _lm_head_vocab_range(w)
        attn_bucket = max(int(token_bucket), int(_NKI_MIN_Q_SEQLEN))
        decode_bucket_set = set(self._decode_token_buckets)
        prefill_bucket_set = set(getattr(self, "_prefill_token_buckets", ()))
        is_full_decode_bucket = (
            int(token_bucket) in decode_bucket_set and token_bucket <= MOE_BLOCK_SIZE
        )
        needs_prefill_kernels = (
            not is_full_decode_bucket or int(token_bucket) in prefill_bucket_set
        )
        if needs_prefill_kernels and token_bucket % tp_degree != 0:
            raise RuntimeError(
                "token_bucket must be divisible by tp_degree for seq-parallel. "
                f"{token_bucket=} {tp_degree=}"
            )
        token_shard = (
            token_bucket // tp_degree
            if token_bucket % tp_degree == 0
            else max(1, token_bucket // tp_degree)
        )

        input_ids = np.zeros((token_bucket,), dtype=np.int32)
        hidden_shard = np.empty((token_shard, hidden), dtype=dtype)
        cos = np.empty((token_bucket, head_dim // 2), dtype=dtype)
        sin = np.empty((token_bucket, head_dim // 2), dtype=dtype)

        # Layer weight samples (shapes must match; values are irrelevant for tracing).
        in_norm = np.empty((hidden,), dtype=dtype)
        post_norm = np.empty((hidden,), dtype=dtype)
        w_q = np.empty((hidden, q_out), dtype=dtype)
        w_k = np.empty((hidden, kv_out), dtype=dtype)
        w_v = np.empty((hidden, kv_out), dtype=dtype)
        b_q = np.empty((q_out,), dtype=dtype)
        b_k = np.empty((kv_out,), dtype=dtype)
        b_v = np.empty((kv_out,), dtype=dtype)
        w_o = np.empty((q_out, hidden), dtype=dtype)
        b_o = np.empty((hidden,), dtype=dtype)
        router_w = np.empty((hidden, w.num_experts), dtype=dtype)
        router_b = np.empty((w.num_experts,), dtype=dtype)

        # MoE sample tensors (use local_E for EP).
        num_blocks, num_static_blocks = moe_get_n_blocks(
            token_bucket, w.experts_per_token, local_E
        )
        moe_output = np.empty((token_bucket, hidden), dtype=dtype)
        expert_aff = np.empty((token_bucket, local_E), dtype=dtype)
        gup_w = np.empty(
            (local_E, hidden, 2, w.local_intermediate_size),
            dtype=ml_dtypes.float8_e5m2,
        )
        gup_b = np.empty((local_E, w.local_intermediate_size, 2), dtype=np.float32)
        down_w = np.empty(
            (local_E, w.local_intermediate_size, hidden),
            dtype=ml_dtypes.float8_e5m2,
        )
        down_bias_bc = np.empty((local_E, MOE_BLOCK_SIZE, hidden), dtype=dtype)
        token_pos = np.zeros((num_blocks, MOE_BLOCK_SIZE), dtype=np.int32)
        block_to_expert = np.zeros((num_blocks,), dtype=np.int8)

        cc_enabled = total_workers > 1
        needs_lm_head_no_sp_sampling_kernels = bool(is_full_decode_bucket)
        self._log_startup(
            "compile bucket start "
            f"token_bucket={int(token_bucket)} attn_bucket={int(attn_bucket)} "
            f"prefill={bool(needs_prefill_kernels)} "
            f"full_decode={bool(is_full_decode_bucket)} "
            f"build_dir={self._build_dir}"
        )

        # Compile.
        embed_kernel = None
        embed_all_reduce_kernel = None
        embed_reduce_scatter_kernel = None
        if needs_prefill_kernels:
            embed_kernel = _get_device_kernel_cls().compile_and_load(
                embedding_fn,
                input_ids,
                np.empty((w.local_vocab_size, hidden), dtype=dtype),
                vocab_start_index=int(vocab_start),
                vocab_end_index=int(vocab_end),
                tp_degree=tp_degree,
                tp_replica_groups=tp_groups_tuple,
                name=f"gpt_oss_embed_t{token_bucket}",
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )
            embed_reduce_scatter_kernel = _get_device_kernel_cls().compile_and_load(
                tp_reduce_scatter_hidden_fn,
                np.empty((token_bucket, hidden), dtype=dtype),
                tp_degree=tp_degree,
                tp_replica_groups=tp_groups_tuple,
                name=f"gpt_oss_embed_rs_t{token_bucket}",
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )
        pre_attn_kernel = None
        prefill_pre_moe_kernel = None
        prefill_post_moe_kernel = None
        kv_update_kernel = None
        attention_kernel = None
        post_attn_kernel = None
        router_kernel = None
        moe_output_init = None
        moe_kernel = None
        moe_decode_kernel = None
        if needs_prefill_kernels:
            from nkipy_serving.attention.nki_blocksparse_flash_attention import (
                NKI_COMPILER_ARGS,
                compute_max_tile_counts,
            )

            max_p, max_d = compute_max_tile_counts(
                token_bucket=attn_bucket,
                max_context_len=self._runtime_config.max_context_len,
                max_requests=self._max_requests_per_step,
                block_size=self._kv_pool.block_size,
            )
            kv_cache_sample = np.empty(
                (
                    2,
                    self._nki_num_blocks,
                    num_kv,
                    self._kv_pool.block_size,
                    head_dim,
                ),
                dtype=dtype,
            )
            prefill_pre_moe_kernel = _get_device_kernel_cls().compile_and_load(
                prefill_layer_pre_moe_nki_fn,
                hidden_shard,
                cos,
                sin,
                np.zeros((token_bucket,), dtype=np.int32),
                *_nki_tile_plan_sample_arrays(
                    max_num_prefill_tiles=max_p,
                    max_num_decode_tiles=max_d,
                    block_size=self._kv_pool.block_size,
                ),
                kv_cache_sample,
                in_norm,
                w_q,
                w_k,
                w_v,
                b_q,
                b_k,
                b_v,
                np.empty((num_heads, 1), dtype=dtype),
                w_o,
                b_o,
                post_norm,
                router_w,
                router_b,
                token_bucket=int(token_bucket),
                attn_bucket=int(attn_bucket),
                num_heads=num_heads,
                num_kv_heads=num_kv,
                head_dim=head_dim,
                rms_norm_eps=w.rms_norm_eps,
                softmax_scale=float(self._attn_softmax_scale),
                top_k=w.experts_per_token,
                tp_degree=tp_degree,
                ep_rank=ep_rank,
                local_num_experts=local_E,
                tp_replica_groups=tp_groups_tuple,
                name=f"gpt_oss_prefill_pre_moe_t{token_bucket}_a{attn_bucket}",
                additional_compiler_args=_join_compiler_args(
                    self._compiler_args,
                    NKI_COMPILER_ARGS,
                ),
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )
            prefill_post_moe_kernel = _get_device_kernel_cls().compile_and_load(
                prefill_layer_post_moe_fn,
                np.empty((token_bucket, hidden), dtype=dtype),
                hidden_shard,
                moe_output,
                expert_aff,
                gup_w,
                gup_b,
                down_w,
                down_bias_bc,
                token_pos,
                block_to_expert,
                num_static_blocks=int(num_static_blocks),
                tp_degree=tp_degree,
                ep_degree=ep_degree,
                ep_replica_groups=ep_groups_tuple,
                tp_replica_groups=tp_groups_tuple,
                name=f"gpt_oss_prefill_post_moe_t{token_bucket}_b{num_blocks}",
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )

        full_decode_kernel = None
        full_static_inputs = None
        if is_full_decode_bucket:
            from nkipy_serving.attention.nki_blocksparse_flash_attention import (
                NKI_COMPILER_ARGS,
                compute_max_tile_counts,
            )

            max_p, max_d = compute_max_tile_counts(
                token_bucket=attn_bucket,
                max_context_len=self._runtime_config.max_context_len,
                max_requests=self._max_requests_per_step,
                block_size=self._kv_pool.block_size,
            )
            (
                p_tqi,
                p_tbt,
                p_tm,
                p_ndls,
                p_qup,
                p_lti,
                d_tqi,
                d_tbt,
                d_tm,
                d_ndls,
                d_qup,
                d_lti,
            ) = _nki_tile_plan_sample_arrays(
                max_num_prefill_tiles=max_p,
                max_num_decode_tiles=max_d,
                block_size=self._kv_pool.block_size,
            )
            full_fn = self._get_full_decode_kernel_fn(
                token_bucket=token_bucket,
                attn_bucket=attn_bucket,
                tp_replica_groups=tp_groups_tuple,
                ep_replica_groups=ep_groups_tuple,
            )
            sample_ids = np.zeros((token_bucket,), dtype=np.int32)
            sample_embeddings = np.empty((w.local_vocab_size, hidden), dtype=dtype)
            kv_cache_sample = np.empty(
                (
                    2,
                    self._nki_num_blocks,
                    num_kv,
                    self._kv_pool.block_size,
                    head_dim,
                ),
                dtype=dtype,
            )
            sink = np.empty((num_heads, 1), dtype=dtype)

            sample_args: list[np.ndarray] = [
                sample_ids,
                sample_embeddings,
                cos,
                sin,
                np.zeros((token_bucket,), dtype=np.int32),
                p_tqi,
                p_tbt,
                p_tm,
                p_ndls,
                p_qup,
                p_lti,
                d_tqi,
                d_tbt,
                d_tm,
                d_ndls,
                d_qup,
                d_lti,
            ]
            for _layer_idx in range(w.num_hidden_layers):
                sample_args.extend(
                    [
                        kv_cache_sample,
                        in_norm,
                        w_q,
                        w_k,
                        w_v,
                        b_q,
                        b_k,
                        b_v,
                        sink,
                        w_o,
                        b_o,
                        post_norm,
                        router_w,
                        router_b,
                        gup_w,
                        gup_b,
                        down_w,
                        down_bias_bc,
                    ]
                )
            full_decode_started = time.perf_counter()
            full_decode_kernel = _get_device_kernel_cls().compile_and_load(
                full_fn,
                *sample_args,
                name=f"gpt_oss_full_decode_embed_layers_t{token_bucket}_a{attn_bucket}",
                additional_compiler_args=_join_compiler_args(
                    self._compiler_args,
                    NKI_COMPILER_ARGS,
                ),
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )
            self._log_startup(
                "full decode kernel ready "
                f"token_bucket={int(token_bucket)} attn_bucket={int(attn_bucket)} "
                f"elapsed_s={time.perf_counter() - full_decode_started:.3f} "
                f"artifact={getattr(full_decode_kernel, 'neff_path', '<unknown>')}"
            )
            full_static_inputs = {
                "embeddings": self._shared_tensors["embeddings"],
            }
            for layer_idx in range(w.num_hidden_layers):
                lt = self._layer_tensors[layer_idx]
                full_static_inputs.update(
                    {
                        f"kv_cache_L{layer_idx}.must_alias_input": self._kv_cache_dev[
                            layer_idx
                        ],
                        f"input_norm_L{layer_idx}": lt["input_norm"],
                        f"w_q_L{layer_idx}": lt["w_q"],
                        f"w_k_L{layer_idx}": lt["w_k"],
                        f"w_v_L{layer_idx}": lt["w_v"],
                        f"b_q_L{layer_idx}": lt["b_q"],
                        f"b_k_L{layer_idx}": lt["b_k"],
                        f"b_v_L{layer_idx}": lt["b_v"],
                        f"sink_L{layer_idx}": lt["sink"],
                        f"w_o_L{layer_idx}": lt["w_o"],
                        f"b_o_L{layer_idx}": lt["b_o"],
                        f"post_attn_norm_L{layer_idx}": lt["post_attn_norm"],
                        f"router_w_L{layer_idx}": lt["router_w"],
                        f"router_b_L{layer_idx}": lt["router_b"],
                        f"gup_w_L{layer_idx}": lt["gup_w"],
                        f"gup_bias_L{layer_idx}": lt["gup_bias"],
                        f"down_w_L{layer_idx}": lt["down_w"],
                        f"down_bias_bc_L{layer_idx}": lt["down_bias_bc"],
                    }
                )

        # LM head sampling kernels owned by LogitsProcessors.
        # The SP variant (gather_hidden=True) receives the TP-sharded hidden
        # tensor, so compile with token_shard = token_bucket / tp_degree.
        # The no-SP variant (gather_hidden=False, full-decode path) receives
        # the already-gathered hidden, so compile with the raw token_bucket.
        if needs_prefill_kernels:
            token_shard = int(token_bucket) // tp_degree
            self._logits_processor_sp._ensure_kernels(max(token_shard, 1))
        if needs_lm_head_no_sp_sampling_kernels:
            self._logits_processor_no_sp._ensure_kernels(int(token_bucket))

        q_attn = None
        k_attn = None
        v_attn = None
        context_attn = None
        q_attn_host = None
        k_attn_host = None
        v_attn_host = None
        if attn_bucket > int(token_bucket):
            q_attn = _alloc_device_scratch(
                (attn_bucket, num_heads, head_dim),
                dtype,
                name=f"gpt_oss_q_attn_t{token_bucket}",
            )
            k_attn = _alloc_device_scratch(
                (attn_bucket, num_kv, head_dim),
                dtype,
                name=f"gpt_oss_k_attn_t{token_bucket}",
            )
            v_attn = _alloc_device_scratch(
                (attn_bucket, num_kv, head_dim),
                dtype,
                name=f"gpt_oss_v_attn_t{token_bucket}",
            )
            q_attn_host = np.zeros((attn_bucket, num_heads, head_dim), dtype=dtype)
            k_attn_host = np.zeros((attn_bucket, num_kv, head_dim), dtype=dtype)
            v_attn_host = np.zeros((attn_bucket, num_kv, head_dim), dtype=dtype)
            context_attn = _alloc_device_scratch(
                (attn_bucket, num_heads, head_dim),
                dtype,
                name=f"gpt_oss_context_attn_t{token_bucket}",
            )

        nki_step_inputs = allocate_prepared_nki_step_inputs(
            _alloc_device_scratch,
            token_bucket=int(token_bucket),
            attn_bucket=int(attn_bucket),
            max_context_len=int(self._runtime_config.max_context_len),
            max_requests=int(self._max_requests_per_step),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
            prefix="gpt_oss",
        )

        scratch = GptOssExecutor._BucketScratch(
            input_ids=_alloc_device_scratch(
                (token_bucket,), np.int32, name=f"gpt_oss_input_ids_t{token_bucket}"
            ),
            cos=_alloc_device_scratch(
                (token_bucket, head_dim // 2),
                dtype,
                name=f"gpt_oss_cos_t{token_bucket}",
            ),
            sin=_alloc_device_scratch(
                (token_bucket, head_dim // 2),
                dtype,
                name=f"gpt_oss_sin_t{token_bucket}",
            ),
            hidden_shard=_alloc_device_scratch(
                (token_shard, hidden), dtype, name=f"gpt_oss_hidden_t{token_bucket}"
            ),
            hidden_attn_shard=_alloc_device_scratch(
                (token_shard, hidden),
                dtype,
                name=f"gpt_oss_hidden_attn_t{token_bucket}",
            ),
            q=_alloc_device_scratch(
                (token_bucket, num_heads, head_dim),
                dtype,
                name=f"gpt_oss_q_t{token_bucket}",
            ),
            k=_alloc_device_scratch(
                (token_bucket, num_kv, head_dim),
                dtype,
                name=f"gpt_oss_k_t{token_bucket}",
            ),
            v=_alloc_device_scratch(
                (token_bucket, num_kv, head_dim),
                dtype,
                name=f"gpt_oss_v_t{token_bucket}",
            ),
            context=_alloc_device_scratch(
                (token_bucket, num_heads, head_dim),
                dtype,
                name=f"gpt_oss_context_t{token_bucket}",
            ),
            context_attn=context_attn,
            context_host=np.zeros((token_bucket, num_heads, head_dim), dtype=dtype),
            topk=_alloc_device_scratch(
                (token_bucket, w.experts_per_token),
                np.int8,
                name=f"gpt_oss_topk_t{token_bucket}",
            ),
            aff=_alloc_device_scratch(
                (token_bucket, local_E), dtype, name=f"gpt_oss_aff_t{token_bucket}"
            ),
            normed_full=_alloc_device_scratch(
                (token_bucket, hidden), dtype, name=f"gpt_oss_normed_t{token_bucket}"
            ),
            moe_out=_alloc_device_scratch(
                (token_bucket, hidden), dtype, name=f"gpt_oss_moe_out_t{token_bucket}"
            ),
            token_pos=_alloc_device_scratch(
                (num_blocks, MOE_BLOCK_SIZE),
                np.int32,
                name=f"gpt_oss_token_pos_t{token_bucket}",
            ),
            block_to_expert=_alloc_device_scratch(
                (num_blocks,), np.int8, name=f"gpt_oss_block_to_expert_t{token_bucket}"
            ),
            q_attn=q_attn,
            k_attn=k_attn,
            v_attn=v_attn,
            q_attn_host=q_attn_host,
            k_attn_host=k_attn_host,
            v_attn_host=v_attn_host,
            nki_step_inputs=nki_step_inputs,
            full_last_token_indices=_alloc_device_scratch(
                (int(self._max_requests_per_step),),
                np.int32,
                name=f"gpt_oss_last_idx_t{token_bucket}",
            ),
            full_last_token_indices_host=np.zeros(
                (int(self._max_requests_per_step),),
                dtype=np.int32,
            ),
            full_last_token_indices_by_bs={
                int(bs_bucket): _alloc_device_scratch(
                    (int(bs_bucket),),
                    np.int32,
                    name=f"gpt_oss_last_idx_t{token_bucket}_bs{int(bs_bucket)}",
                )
                for bs_bucket in self._bs_buckets
            },
            full_last_token_indices_host_by_bs={
                int(bs_bucket): np.zeros((int(bs_bucket),), dtype=np.int32)
                for bs_bucket in self._bs_buckets
            },
            full_hidden_aux=(
                _alloc_device_scratch(
                    (token_bucket, hidden),
                    dtype,
                    name=f"gpt_oss_full_hidden_aux_t{token_bucket}",
                )
                if not False
                else None
            ),
            full_top1_vals=(
                _alloc_device_scratch(
                    (int(self._max_requests_per_step),),
                    np.float32,
                    name=f"gpt_oss_full_top1_vals_t{token_bucket}",
                )
                if self._use_top1_fast_path()
                else None
            ),
            full_top1_idx=(
                _alloc_device_scratch(
                    (int(self._max_requests_per_step),),
                    np.int32,
                    name=f"gpt_oss_full_top1_idx_t{token_bucket}",
                )
                if self._use_top1_fast_path()
                else None
            ),
            full_top1_vals_by_bs={
                int(bs_bucket): _alloc_device_scratch(
                    (int(bs_bucket),),
                    np.float32,
                    name=f"gpt_oss_full_top1_vals_t{token_bucket}_bs{int(bs_bucket)}",
                )
                for bs_bucket in self._bs_buckets
            }
            if self._use_top1_fast_path()
            else {},
            full_top1_idx_by_bs={
                int(bs_bucket): _alloc_device_scratch(
                    (int(bs_bucket),),
                    np.int32,
                    name=f"gpt_oss_full_top1_idx_t{token_bucket}_bs{int(bs_bucket)}",
                )
                for bs_bucket in self._bs_buckets
            }
            if self._use_top1_fast_path()
            else {},
            full_topk_vals=(
                _alloc_device_scratch(
                    (int(self._max_requests_per_step), int(self._sampled_local_topk)),
                    np.float32,
                    name=f"gpt_oss_full_topk_vals_t{token_bucket}",
                )
                if not self._use_top1_fast_path()
                else None
            ),
            full_topk_idx=(
                _alloc_device_scratch(
                    (int(self._max_requests_per_step), int(self._sampled_local_topk)),
                    np.int32,
                    name=f"gpt_oss_full_topk_idx_t{token_bucket}",
                )
                if not self._use_top1_fast_path()
                else None
            ),
            full_topk_vals_by_bs={
                int(bs_bucket): _alloc_device_scratch(
                    (int(bs_bucket), int(self._sampled_local_topk)),
                    np.float32,
                    name=f"gpt_oss_full_topk_vals_t{token_bucket}_bs{int(bs_bucket)}",
                )
                for bs_bucket in self._bs_buckets
            }
            if not self._use_top1_fast_path()
            else {},
            full_topk_idx_by_bs={
                int(bs_bucket): _alloc_device_scratch(
                    (int(bs_bucket), int(self._sampled_local_topk)),
                    np.int32,
                    name=f"gpt_oss_full_topk_idx_t{token_bucket}_bs{int(bs_bucket)}",
                )
                for bs_bucket in self._bs_buckets
            }
            if not self._use_top1_fast_path()
            else {},
        )
        initialize_prepared_nki_step_inputs(
            scratch.nki_step_inputs,
            _overwrite_device_tensor,
        )

        kernels = GptOssExecutor._CompiledKernels(
            token_bucket=int(token_bucket),
            token_shard=int(token_shard),
            embed_kernel=embed_kernel,
            embed_all_reduce_kernel=embed_all_reduce_kernel,
            embed_reduce_scatter_kernel=embed_reduce_scatter_kernel,
            prefill_pre_moe_kernel=prefill_pre_moe_kernel,
            prefill_post_moe_kernel=prefill_post_moe_kernel,
            pre_attn_kernel=pre_attn_kernel,
            kv_update_kernel=kv_update_kernel,
            attention_kernel=attention_kernel,
            post_attn_kernel=post_attn_kernel,
            router_kernel=router_kernel,
            moe_output_init=moe_output_init,
            moe_kernel=moe_kernel,
            moe_decode_kernel=moe_decode_kernel,
            full_decode_kernel=full_decode_kernel,
            full_static_inputs=full_static_inputs,
            scratch=scratch,
        )
        self._kernels_by_bucket[int(token_bucket)] = kernels
        artifact_summary = self._startup_artifact_summary(
            embed=embed_kernel,
            prefill_pre_moe=prefill_pre_moe_kernel,
            prefill_post_moe=prefill_post_moe_kernel,
            full_decode=full_decode_kernel,
        )
        self._log_startup(
            "compile bucket done "
            f"token_bucket={int(token_bucket)} elapsed_s={time.perf_counter() - compile_started:.3f}"
            + (f" {artifact_summary}" if artifact_summary else "")
        )
        return kernels

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_caches: list[np.ndarray],  # unused (device KV cache is persistent)
        attn_metadata: AttentionMetadata,
        token_bucket: int = 0,
        real_total_tokens: int = 0,
        sampling_batch: DeviceSamplingBatch | None = None,
        attention_lane: int = -1,
    ) -> dict[str, np.ndarray]:
        w = self._weights
        if token_bucket <= 0:
            token_bucket = int(input_ids.shape[0])
        if real_total_tokens <= 0:
            real_total_tokens = int(input_ids.shape[0])
        input_token_bucket = int(token_bucket)
        real_total_tokens = int(real_total_tokens)
        token_bucket, attn_token_bucket = self._resolve_token_buckets(
            forward_mode=int(attn_metadata.forward_mode),
            input_token_bucket=input_token_bucket,
            real_total_tokens=real_total_tokens,
        )
        if token_bucket < real_total_tokens:
            raise RuntimeError(
                f"token_bucket must cover real_total_tokens, got {token_bucket=} {real_total_tokens=}"
            )
        input_ids = self._pad_host_step_vector(
            input_ids,
            target_size=int(token_bucket),
            dtype=np.int32,
        )
        positions = self._pad_host_step_vector(
            positions,
            target_size=int(token_bucket),
            dtype=np.int32,
        )

        kernels = self._ensure_kernels(token_bucket)
        scratch = kernels.scratch
        profiler = self._model_profiler
        profile_enabled = (
            profiler.enabled and getattr(self, "_profile_suppression_depth", 0) == 0
        )
        profile_step = profiler.next_step() if profile_enabled else 0
        forward_mode_name = (
            "decode" if attn_metadata.forward_mode == FORWARD_MODE_DECODE else "extend"
        )
        profile_totals = {
            "rope_build": 0.0,
            "rope_upload": 0.0,
            "embed_alloc": 0.0,
            "embed": 0.0,
            "layer_alloc": 0.0,
            "attn_prep": 0.0,
            "prefill_pre_moe_graph": 0.0,
            "prefill_post_moe_graph": 0.0,
            "decode_full": 0.0,
            "pre_attn": 0.0,
            "kv_update": 0.0,
            "attention": 0.0,
            "post_attn": 0.0,
            "router": 0.0,
            "moe_cpu_schedule": 0.0,
            "moe_output_init": 0.0,
            "moe": 0.0,
            "lm_head_alloc": 0.0,
            "lm_head": 0.0,
        }
        forward_t0 = time.perf_counter() if profile_enabled else 0.0
        decode_moe_layers = 0
        prefill_moe_layers = 0

        from nkipy_serving.runtime.shape_guard import select_bucket

        bs = int(attn_metadata.batch_size)
        bs_bucket = select_bucket(max(bs, 2), tuple(self._bs_buckets), "batch")

        rope_t0 = time.perf_counter() if profile_enabled else 0.0
        cos_np, sin_np = _build_rope_cache_for_positions_yarn(
            positions.astype(np.int32),
            head_dim=w.head_dim,
            theta=w.rope_theta,
            initial_context_length=w.yarn_original_max_pos,
            scaling_factor=w.yarn_factor,
            ntk_alpha=w.yarn_beta_slow,
            ntk_beta=w.yarn_beta_fast,
            dtype=w.dtype,
        )
        if profile_enabled:
            profile_totals["rope_build"] += time.perf_counter() - rope_t0
        embed_alloc_t0 = time.perf_counter() if profile_enabled else 0.0
        common_step_inputs = self._write_common_step_inputs(
            scratch,
            input_ids_shard=input_ids.astype(np.int32, copy=False),
            cos_np=cos_np,
            sin_np=sin_np,
        )
        if profile_enabled:
            profile_totals["rope_upload"] += time.perf_counter() - embed_alloc_t0

        use_full_decode = (
            attn_metadata.forward_mode == FORWARD_MODE_DECODE
            and kernels.full_decode_kernel is not None
        )

        if use_full_decode:
            attn_prep_t0 = time.perf_counter() if profile_enabled else 0.0
            step_inputs = self._prepare_nki_step_inputs(
                scratch=scratch,
                token_bucket=token_bucket,
                attn_bucket=attn_token_bucket,
                real_total_tokens=real_total_tokens,
                attn_metadata=attn_metadata,
            )
            if profile_enabled:
                profile_totals["attn_prep"] += time.perf_counter() - attn_prep_t0
            if profile_enabled:
                profile_totals["embed_alloc"] += time.perf_counter() - embed_alloc_t0
                full_t0 = time.perf_counter()
            full_inputs = dict(kernels.full_static_inputs or {})
            full_inputs.update(common_step_inputs)
            full_inputs.update(step_inputs)
            # Full-decode graph outputs hidden states; LM head runs via LogitsProcessor.
            hidden_full_dev = scratch.normed_full
            hidden_aux_dev = scratch.full_hidden_aux
            if hidden_aux_dev is None:
                raise RuntimeError(
                    "GPT-OSS full-decode hidden auxiliary buffer is unavailable. "
                    f"token_bucket={token_bucket}"
                )
            full_outputs: dict[str, object] = {
                "output0": hidden_full_dev,
                "output1": hidden_aux_dev,
            }
            for _li in range(int(w.num_hidden_layers)):
                full_outputs[f"kv_cache_L{_li}"] = self._kv_cache_dev[_li]
            kernels.full_decode_kernel(
                inputs=full_inputs,
                outputs=full_outputs,
            )
            lm_alloc_t0 = time.perf_counter() if profile_enabled else 0.0
            _, last_dev = self._prepare_full_decode_last_token_indices(
                scratch=scratch,
                attn_metadata=attn_metadata,
                bs_bucket=int(bs_bucket),
            )
            if profile_enabled:
                profile_totals["lm_head_alloc"] += time.perf_counter() - lm_alloc_t0
                lm_head_t0 = time.perf_counter()
            lp_output = self._logits_processor_no_sp.forward(
                hidden_full_dev,
                self._shared_tensors["final_norm"],
                self._shared_tensors["lm_head"],
                last_dev,
                batch_size=bs,
                token_bucket=int(bs_bucket),
                sampling_batch=sampling_batch,
                needs_logprobs=bool(sampling_batch.needs_logprobs)
                if sampling_batch
                else False,
                logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
            )
            result = lp_output.to_shm_dict(vocab_offset=w.lm_head_vocab_offset)
            if profile_enabled:
                profile_totals["lm_head"] += time.perf_counter() - lm_head_t0
            if profile_enabled:
                profile_totals["decode_full"] += time.perf_counter() - full_t0
                profiler.write_step(
                    {
                        "step": profile_step,
                        "rank": profiler.rank,
                        "ts": time.time(),
                        "forward_mode": forward_mode_name,
                        "batch_size": bs,
                        "token_bucket": token_bucket,
                        "input_token_bucket": input_token_bucket,
                        "attn_token_bucket": attn_token_bucket,
                        "real_tokens": real_total_tokens,
                        "bs_bucket": int(bs_bucket),
                        "num_layers": int(w.num_hidden_layers),
                        "decode_moe_layers": int(w.num_hidden_layers),
                        "prefill_moe_layers": 0,
                        "execution_path": "full_decode_embed_layers",
                        "sampled_local_topk": int(self._sampled_local_topk),
                        **{
                            f"t_{name}": round(value, 6)
                            for name, value in profile_totals.items()
                        },
                        "t_total": round(time.perf_counter() - forward_t0, 6),
                    }
                )
            return result

        hidden_shard_dev = scratch.hidden_shard
        hidden_full_dev = scratch.normed_full
        if kernels.embed_kernel is None or kernels.embed_reduce_scatter_kernel is None:
            raise RuntimeError(
                "GPT-OSS embed kernel is unavailable for the selected bucket/path. "
                f"token_bucket={token_bucket}"
            )
        if profile_enabled:
            profile_totals["embed_alloc"] += time.perf_counter() - embed_alloc_t0
            embed_t0 = time.perf_counter()
        kernels.embed_kernel(
            inputs={
                "input_ids": common_step_inputs["input_ids"],
                "embeddings": self._shared_tensors["embeddings"],
            },
            outputs={"output0": hidden_full_dev},
        )
        kernels.embed_reduce_scatter_kernel(
            inputs={"hidden": hidden_full_dev},
            outputs={"output0": hidden_shard_dev},
        )
        if profile_enabled:
            profile_totals["embed"] += time.perf_counter() - embed_t0

        use_fused_prefill_layer = (
            attn_metadata.forward_mode == FORWARD_MODE_EXTEND
            and kernels.prefill_pre_moe_kernel is not None
            and kernels.prefill_post_moe_kernel is not None
        )

        if (
            attn_metadata.forward_mode == FORWARD_MODE_EXTEND
            and not use_fused_prefill_layer
        ):
            raise RuntimeError(
                "GPT-OSS fused prefill kernels are unavailable for the selected bucket/path. "
                f"token_bucket={token_bucket}"
            )
        if attn_metadata.forward_mode == FORWARD_MODE_DECODE:
            raise RuntimeError(
                "GPT-OSS full decode kernel is unavailable for the selected bucket/path. "
                f"token_bucket={token_bucket}"
            )
        attn_prep_t0 = time.perf_counter() if profile_enabled else 0.0
        step_inputs = self._prepare_nki_step_inputs(
            scratch=scratch,
            token_bucket=token_bucket,
            attn_bucket=attn_token_bucket,
            real_total_tokens=real_total_tokens,
            attn_metadata=attn_metadata,
        )
        if profile_enabled:
            profile_totals["attn_prep"] += time.perf_counter() - attn_prep_t0

        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_tensors[layer_idx]
            layer_alloc = 0.0
            layer_pre_attn = 0.0
            layer_kv_update = 0.0
            layer_attention = 0.0
            layer_post_attn = 0.0
            layer_router = 0.0
            layer_moe_cpu_schedule = 0.0
            layer_moe_output_init = 0.0
            layer_moe = 0.0
            layer_prefill_pre_moe_graph = 0.0
            layer_prefill_post_moe_graph = 0.0
            layer_t0 = time.perf_counter() if profile_enabled else 0.0

            kv_cache_dev = self._kv_cache_dev[layer_idx]
            hidden_out_shard_dev = scratch.hidden_attn_shard
            topk_dev = scratch.topk
            aff_dev = scratch.aff
            normed_full_dev = scratch.normed_full
            if use_fused_prefill_layer:
                prefill_pre_t0 = time.perf_counter() if profile_enabled else 0.0
                kernels.prefill_pre_moe_kernel(
                    inputs={
                        "hidden_shard": hidden_shard_dev,
                        "cos": common_step_inputs["cos"],
                        "sin": common_step_inputs["sin"],
                        "slot_mapping": step_inputs["slot_mapping"],
                        "p_tqi": step_inputs["p_tqi"],
                        "p_tbt": step_inputs["p_tbt"],
                        "p_tm": step_inputs["p_tm"],
                        "p_ndls": step_inputs["p_ndls"],
                        "p_qup": step_inputs["p_qup"],
                        "p_lti": step_inputs["p_lti"],
                        "d_tqi": step_inputs["d_tqi"],
                        "d_tbt": step_inputs["d_tbt"],
                        "d_tm": step_inputs["d_tm"],
                        "d_ndls": step_inputs["d_ndls"],
                        "d_qup": step_inputs["d_qup"],
                        "d_lti": step_inputs["d_lti"],
                        "kv_cache.must_alias_input": kv_cache_dev,
                        "input_norm": lt["input_norm"],
                        "w_q": lt["w_q"],
                        "w_k": lt["w_k"],
                        "w_v": lt["w_v"],
                        "b_q": lt["b_q"],
                        "b_k": lt["b_k"],
                        "b_v": lt["b_v"],
                        "sink": lt["sink"],
                        "w_o": lt["w_o"],
                        "b_o": lt["b_o"],
                        "post_attn_norm": lt["post_attn_norm"],
                        "router_w": lt["router_w"],
                        "router_b": lt["router_b"],
                    },
                    outputs={
                        "kv_cache": kv_cache_dev,
                        "output1": hidden_out_shard_dev,
                        "output2": topk_dev,
                        "output3": aff_dev,
                        "output4": normed_full_dev,
                    },
                )
                hidden_attn_shard_dev = hidden_out_shard_dev
                if profile_enabled:
                    dur = time.perf_counter() - prefill_pre_t0
                    layer_prefill_pre_moe_graph += dur
                    profile_totals["prefill_pre_moe_graph"] += dur
            else:
                q_dev = scratch.q
                k_dev = scratch.k
                v_dev = scratch.v
                if profile_enabled:
                    pre_attn_t0 = time.perf_counter()
                kernels.pre_attn_kernel(
                    inputs={
                        "hidden_shard": hidden_shard_dev,
                        "input_norm": lt["input_norm"],
                        "w_q": lt["w_q"],
                        "w_k": lt["w_k"],
                        "w_v": lt["w_v"],
                        "b_q": lt["b_q"],
                        "b_k": lt["b_k"],
                        "b_v": lt["b_v"],
                        "cos": common_step_inputs["cos"],
                        "sin": common_step_inputs["sin"],
                    },
                    outputs={"output0": q_dev, "output1": k_dev, "output2": v_dev},
                )
                if profile_enabled:
                    dur = time.perf_counter() - pre_attn_t0
                    layer_pre_attn += dur
                    profile_totals["pre_attn"] += dur

                kv_update_t0 = time.perf_counter() if profile_enabled else 0.0
                kernels.kv_update_kernel(
                    inputs={
                        "key": k_dev,
                        "value": v_dev,
                        "kv_cache.must_alias_input": kv_cache_dev,
                        "slot_mapping": step_inputs["slot_mapping"],
                    },
                    outputs={"kv_cache": kv_cache_dev},
                )
                if profile_enabled:
                    dur = time.perf_counter() - kv_update_t0
                    layer_kv_update += dur
                    profile_totals["kv_update"] += dur
                    attention_t0 = time.perf_counter()
                q_attn_dev = q_dev
                k_attn_dev = k_dev
                v_attn_dev = v_dev
                if attn_token_bucket > token_bucket:
                    scratch.q_attn_host.fill(0)
                    scratch.k_attn_host.fill(0)
                    scratch.v_attn_host.fill(0)
                    scratch.q_attn_host[:token_bucket] = q_dev.numpy()
                    scratch.k_attn_host[:token_bucket] = k_dev.numpy()
                    scratch.v_attn_host[:token_bucket] = v_dev.numpy()
                    _overwrite_device_tensor(scratch.q_attn, scratch.q_attn_host)
                    _overwrite_device_tensor(scratch.k_attn, scratch.k_attn_host)
                    _overwrite_device_tensor(scratch.v_attn, scratch.v_attn_host)
                    q_attn_dev = scratch.q_attn
                    k_attn_dev = scratch.k_attn
                    v_attn_dev = scratch.v_attn
                context_attn_dev = (
                    scratch.context_attn
                    if attn_token_bucket > token_bucket
                    and scratch.context_attn is not None
                    else scratch.context
                )
                kernels.attention_kernel(
                    inputs={
                        "q": q_attn_dev,
                        "k": k_attn_dev,
                        "v": v_attn_dev,
                        "kv_cache": kv_cache_dev,
                        "sink": lt["sink"],
                        "p_tqi": step_inputs["p_tqi"],
                        "p_tbt": step_inputs["p_tbt"],
                        "p_tm": step_inputs["p_tm"],
                        "p_ndls": step_inputs["p_ndls"],
                        "p_qup": step_inputs["p_qup"],
                        "p_lti": step_inputs["p_lti"],
                        "d_tqi": step_inputs["d_tqi"],
                        "d_tbt": step_inputs["d_tbt"],
                        "d_tm": step_inputs["d_tm"],
                        "d_ndls": step_inputs["d_ndls"],
                        "d_qup": step_inputs["d_qup"],
                        "d_lti": step_inputs["d_lti"],
                    },
                    outputs={"output0": context_attn_dev},
                )
                if attn_token_bucket > token_bucket:
                    scratch.context_host[:] = context_attn_dev.numpy()[:token_bucket]
                    _overwrite_device_tensor(scratch.context, scratch.context_host)
                    context_dev = scratch.context
                else:
                    context_dev = context_attn_dev
                if profile_enabled:
                    dur = time.perf_counter() - attention_t0
                    layer_attention += dur
                    profile_totals["attention"] += dur

                if profile_enabled:
                    post_attn_t0 = time.perf_counter()
                kernels.post_attn_kernel(
                    inputs={
                        "residual_shard": hidden_shard_dev,
                        "context": context_dev,
                        "w_o": lt["w_o"],
                        "b_o_sharded": lt["b_o"],
                    },
                    outputs={"output0": hidden_out_shard_dev},
                )
                if profile_enabled:
                    dur = time.perf_counter() - post_attn_t0
                    layer_post_attn += dur
                    profile_totals["post_attn"] += dur
                hidden_attn_shard_dev = hidden_out_shard_dev

                if profile_enabled:
                    router_t0 = time.perf_counter()
                kernels.router_kernel(
                    inputs={
                        "hidden_shard": hidden_attn_shard_dev,
                        "post_attn_norm": lt["post_attn_norm"],
                        "router_weight": lt["router_w"],
                        "router_bias": lt["router_b"],
                    },
                    outputs={
                        "output0": topk_dev,
                        "output1": aff_dev,
                        "output2": normed_full_dev,
                    },
                )
                if profile_enabled:
                    dur = time.perf_counter() - router_t0
                    layer_router += dur
                    profile_totals["router"] += dur

            use_decode_moe = (
                attn_metadata.forward_mode == FORWARD_MODE_DECODE
                and kernels.moe_decode_kernel is not None
            )
            if use_decode_moe:
                decode_moe_layers += 1
            else:
                prefill_moe_layers += 1

            hidden_next_shard_dev = scratch.hidden_shard
            if use_decode_moe:
                moe_t0 = time.perf_counter() if profile_enabled else 0.0
                kernels.moe_decode_kernel(
                    inputs={
                        "hidden_states": normed_full_dev,
                        "residual_2d_shard": hidden_attn_shard_dev,
                        "expert_affinities_masked_hbm": aff_dev,
                        "gate_up_proj_weight": lt["gup_w"],
                        "gate_up_bias_plus1_T_hbm": lt["gup_bias"],
                        "down_proj_weight": lt["down_w"],
                        "down_bias_broadcasted_hbm": lt["down_bias_bc"],
                    },
                    outputs={"output0": hidden_next_shard_dev},
                )
                if profile_enabled:
                    dur = time.perf_counter() - moe_t0
                    layer_moe += dur
                    profile_totals["moe"] += dur
            else:
                moe_cpu_t0 = time.perf_counter() if profile_enabled else 0.0
                topk_np = topk_dev.numpy()
                token_pos_to_id, block_to_expert, _num_blocks, num_static_blocks = (
                    build_prefill_moe_schedule(
                        topk_np,
                        token_bucket=int(token_bucket),
                        real_total_tokens=int(real_total_tokens),
                        experts_per_token=int(w.experts_per_token),
                        local_num_experts=int(w.local_num_experts),
                        ep_degree=int(w.ep_degree),
                        ep_rank=int(w.ep_rank),
                    )
                )
                _overwrite_device_tensor(scratch.token_pos, token_pos_to_id)
                _overwrite_device_tensor(scratch.block_to_expert, block_to_expert)
                if profile_enabled:
                    dur = time.perf_counter() - moe_cpu_t0
                    layer_moe_cpu_schedule += dur
                    profile_totals["moe_cpu_schedule"] += dur
                    moe_init_t0 = time.perf_counter()

                moe_out_dev = scratch.moe_out
                if use_fused_prefill_layer:
                    kernels.prefill_post_moe_kernel(
                        inputs={
                            "hidden_states": normed_full_dev,
                            "residual_2d_shard": hidden_attn_shard_dev,
                            "output.must_alias_input": moe_out_dev,
                            "expert_affinities_masked_hbm": aff_dev,
                            "gate_up_proj_weight": lt["gup_w"],
                            "gate_up_bias_plus1_T_hbm": lt["gup_bias"],
                            "down_proj_weight": lt["down_w"],
                            "down_bias_broadcasted_hbm": lt["down_bias_bc"],
                            "token_position_to_id": scratch.token_pos,
                            "block_to_expert": scratch.block_to_expert,
                        },
                        outputs=(
                            {
                                "output0": hidden_next_shard_dev,
                                "output1": moe_out_dev,
                                "output": moe_out_dev,
                            }
                            if self._weights.ep_degree > 1
                            else {
                                "output0": hidden_next_shard_dev,
                                "output": moe_out_dev,
                            }
                        ),
                    )
                    if profile_enabled:
                        dur = time.perf_counter() - moe_init_t0
                        layer_prefill_post_moe_graph += dur
                        profile_totals["prefill_post_moe_graph"] += dur
                else:
                    kernels.moe_output_init(
                        inputs={"output.must_alias_input": moe_out_dev},
                        outputs={"output": moe_out_dev},
                    )
                    if profile_enabled:
                        dur = time.perf_counter() - moe_init_t0
                        layer_moe_output_init += dur
                        profile_totals["moe_output_init"] += dur
                        moe_t0 = time.perf_counter()

                    kernels.moe_kernel(
                        inputs={
                            "hidden_states": normed_full_dev,
                            "residual_2d_shard": hidden_attn_shard_dev,
                            "output.must_alias_input": moe_out_dev,
                            "expert_affinities_masked_hbm": aff_dev,
                            "gate_up_proj_weight": lt["gup_w"],
                            "gate_up_bias_plus1_T_hbm": lt["gup_bias"],
                            "down_proj_weight": lt["down_w"],
                            "down_bias_broadcasted_hbm": lt["down_bias_bc"],
                            "token_position_to_id": scratch.token_pos,
                            "block_to_expert": scratch.block_to_expert,
                        },
                        outputs=(
                            {
                                "output0": hidden_next_shard_dev,
                                "output1": moe_out_dev,
                                "output": moe_out_dev,
                            }
                            if self._weights.ep_degree > 1
                            else {
                                "output0": hidden_next_shard_dev,
                                "output": moe_out_dev,
                            }
                        ),
                    )
                    if profile_enabled:
                        dur = time.perf_counter() - moe_t0
                        layer_moe += dur
                        profile_totals["moe"] += dur

            hidden_shard_dev = hidden_next_shard_dev

            if profile_enabled and profiler.layer_enabled:
                profiler.write_layer(
                    {
                        "step": profile_step,
                        "rank": profiler.rank,
                        "ts": time.time(),
                        "forward_mode": forward_mode_name,
                        "layer_idx": layer_idx,
                        "batch_size": bs,
                        "token_bucket": token_bucket,
                        "input_token_bucket": input_token_bucket,
                        "attn_token_bucket": attn_token_bucket,
                        "real_tokens": real_total_tokens,
                        "moe_mode": "decode" if use_decode_moe else "prefill",
                        "t_alloc": round(layer_alloc, 6),
                        "t_pre_attn": round(layer_pre_attn, 6),
                        "t_kv_update": round(layer_kv_update, 6),
                        "t_attention": round(layer_attention, 6),
                        "t_post_attn": round(layer_post_attn, 6),
                        "t_router": round(layer_router, 6),
                        "t_prefill_pre_moe_graph": round(
                            layer_prefill_pre_moe_graph, 6
                        ),
                        "t_moe_cpu_schedule": round(layer_moe_cpu_schedule, 6),
                        "t_moe_output_init": round(layer_moe_output_init, 6),
                        "t_prefill_post_moe_graph": round(
                            layer_prefill_post_moe_graph, 6
                        ),
                        "t_moe": round(layer_moe, 6),
                        "t_total": round(time.perf_counter() - layer_t0, 6),
                    }
                )

        last_indices = (attn_metadata.query_start_loc[1:] - 1).astype(np.int32)
        padded_last = np.zeros((int(bs_bucket),), dtype=np.int32)
        padded_last[:bs] = last_indices[:bs]
        lm_alloc_t0 = time.perf_counter() if profile_enabled else 0.0
        last_dev = _get_device_tensor_cls().from_numpy(
            padded_last, name="last_token_indices"
        )
        lm_hidden_input = self._make_lm_head_hidden_input_view(hidden_shard_dev)

        if profile_enabled:
            profile_totals["lm_head_alloc"] += time.perf_counter() - lm_alloc_t0
            lm_t0 = time.perf_counter()
        # For the SP variant (gather_hidden=True), the hidden input is the TP shard
        # with first dim = token_bucket / tp_degree.
        lp_token_bucket = int(token_bucket) // int(w.tp_degree)
        lp_output = self._logits_processor_sp.forward(
            lm_hidden_input,
            self._shared_tensors["final_norm"],
            self._shared_tensors["lm_head"],
            last_dev,
            batch_size=bs,
            token_bucket=lp_token_bucket,
            sampling_batch=sampling_batch,
            needs_logprobs=bool(sampling_batch.needs_logprobs)
            if sampling_batch
            else False,
            logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
        )
        result = lp_output.to_shm_dict(vocab_offset=w.lm_head_vocab_offset)

        if profile_enabled:
            profile_totals["lm_head"] += time.perf_counter() - lm_t0
            profiler.write_step(
                {
                    "step": profile_step,
                    "rank": profiler.rank,
                    "ts": time.time(),
                    "forward_mode": forward_mode_name,
                    "batch_size": bs,
                    "token_bucket": token_bucket,
                    "input_token_bucket": input_token_bucket,
                    "attn_token_bucket": attn_token_bucket,
                    "real_tokens": real_total_tokens,
                    "bs_bucket": int(bs_bucket),
                    "num_layers": int(w.num_hidden_layers),
                    "decode_moe_layers": decode_moe_layers,
                    "prefill_moe_layers": prefill_moe_layers,
                    "execution_path": (
                        "prefill_layer_fused"
                        if use_fused_prefill_layer
                        else "prefill_per_op"
                    ),
                    "sampled_local_topk": int(self._sampled_local_topk),
                    **{
                        f"t_{name}": round(value, 6)
                        for name, value in profile_totals.items()
                    },
                    "t_total": round(time.perf_counter() - forward_t0, 6),
                }
            )
        return result

    # -- Warmup ---------------------------------------------------------------

    def warmup(self, paddings=None) -> None:
        if paddings is None:
            return
        preload_blockwise_index_ext()
        for token_bucket in self._startup_token_buckets(paddings):
            bucket_started = time.perf_counter()
            self._log_startup(f"ensure bucket start token_bucket={int(token_bucket)}")
            self._ensure_kernels(int(token_bucket))
            self._log_startup(
                f"ensure bucket done token_bucket={int(token_bucket)} "
                f"elapsed_s={time.perf_counter() - bucket_started:.3f}"
            )
        steps = self._startup_warmup_steps(paddings)
        token_paddings = tuple(int(bucket) for bucket in paddings.token_paddings)
        bs_paddings = tuple(int(bucket) for bucket in paddings.bs_paddings)
        if not getattr(self, "_log_startup_progress", False):
            run_synthetic_warmup_steps(
                steps,
                token_paddings=token_paddings,
                bs_paddings=bs_paddings,
                num_blocks=int(self._nki_num_blocks),
                block_size=int(self._kv_pool.block_size),
                num_kv_heads=int(self._weights.num_kv_heads),
                head_dim=int(self._weights.head_dim),
                forward=self.forward,
                profiling_context=self._suspend_model_profiling,
            )
            return
        with self._suspend_model_profiling():
            for step in steps:
                step_started = time.perf_counter()
                input_ids, positions, attn_metadata = build_synthetic_warmup_inputs(
                    step,
                    token_paddings=token_paddings,
                    bs_paddings=bs_paddings,
                    num_blocks=int(self._nki_num_blocks),
                    block_size=int(self._kv_pool.block_size),
                    num_kv_heads=int(self._weights.num_kv_heads),
                    head_dim=int(self._weights.head_dim),
                )
                compute_bucket, attn_bucket = self._resolve_token_buckets(
                    forward_mode=int(attn_metadata.forward_mode),
                    input_token_bucket=int(step.input_token_bucket),
                    real_total_tokens=int(attn_metadata.total_tokens),
                )
                self._log_startup(
                    "warmup step start "
                    f"name={step.name} forward_mode={int(step.forward_mode)} "
                    f"input_token_bucket={int(step.input_token_bucket)} "
                    f"compute_token_bucket={int(compute_bucket)} "
                    f"attn_token_bucket={int(attn_bucket)} batch_size={int(step.batch_size)}"
                )
                self.forward(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=[],
                    attn_metadata=attn_metadata,
                    token_bucket=int(step.input_token_bucket),
                    real_total_tokens=int(attn_metadata.total_tokens),
                )
                self._log_startup(
                    f"warmup step done name={step.name} "
                    f"elapsed_s={time.perf_counter() - step_started:.3f}"
                )
