"""Qwen3MoeExecutor: owns weights, device tensors, compiled kernels, and forward pass."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import FORWARD_MODE_DECODE, AttentionMetadata
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
from nkipy_serving.models.qwen3_moe.codegen import (
    generate_full_decode_kernel_source as _generate_full_decode_kernel_source,
)
from nkipy_serving.models.qwen3_moe.config import (
    Qwen3MoeModelConfig,
    Qwen3MoeWeights,
)
from nkipy_serving.models.qwen3_moe.graph_fns import (
    embedding_fn,
    post_attn_fn,
    pre_attn_fn,
    router_fn,
)
from nkipy_serving.models.qwen3_moe.weights import (
    _kv_head_indices_for_rank,
    _load_qwen3_moe_weights,
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
from nkipy_serving.ops.moe.blockwise_nki import (
    blockwise_add_residual,
    blockwise_decode_add_residual,
    output_init,
)
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule
from nkipy_serving.ops.nn import (
    build_rope_cache_for_positions as _build_rope_cache_for_positions,
)
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge
from nkipy_serving.runtime.warmup import (
    build_standard_warmup_steps,
    run_synthetic_warmup_steps,
)
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessor

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Qwen3MoeExecutor:
    """Qwen3 MoE executor (device-only, per-layer prefill graphs, CPU-scheduled MoE)."""

    def __init__(self, model_config: Qwen3MoeModelConfig, kv_pool, runtime_config):
        from nkipy_serving.attention._kernel_cache import AttentionKernelCache

        self._model_config = model_config
        if runtime_config.execution_backend != "nkipy":
            raise RuntimeError("qwen3-moe executor requires execution_backend='nkipy'")
        if runtime_config.attention_backend != "NKIBlockSparseFlashAttention":
            raise RuntimeError(
                "qwen3-moe executor requires attention_backend='NKIBlockSparseFlashAttention'"
            )
        try:
            _get_device_tensor_cls()
        except ImportError:
            raise RuntimeError("nkipy runtime not available")

        snapshot_path, self._weights = _load_qwen3_moe_weights(model_config)
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config
        self._attention_kernel_cache = AttentionKernelCache()
        self._nki_step_inputs_by_bucket: dict[int, PreparedNkiStepInputs] = {}
        self._compiler_args = runtime_config.nkipy_compiler_args
        self._sampled_local_topk = int(runtime_config.dense_local_topk)
        _global_rank = (
            self._weights.ep_rank * self._weights.tp_degree + self._weights.tp_rank
        )
        self._build_dir = f"{runtime_config.config_build_dir()}/rank{_global_rank}"

        # Qwen3 MoE head_dim=128 matches NKI attention natively (no padding needed).
        self._attn_head_dim = int(self._weights.head_dim)
        self._attn_softmax_scale = 1.0 / (float(self._weights.head_dim) ** 0.5)
        if int(self._weights.head_dim) != 128:
            raise RuntimeError(
                "Qwen3 MoE expected head_dim=128 for NKI attention. "
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
        # NeuronX compiler requires >= 8 elements per partition for certain ops
        # (e.g. topk output reshape in lm_head). Normalize bs buckets to >= 8.
        self._bs_buckets = tuple(
            sorted({max(int(b), 8) for b in runtime_config.request_buckets})
        )

        # LogitsProcessor owns all sampling-related kernel compilation,
        # warmup, and forward dispatch.  Qwen3 MoE uses gather_hidden=True
        # (seq-parallel all-gather before LM head) and compiles per bs_bucket.
        from nkipy_serving.runtime.parallel_groups import build_tp_replica_groups

        w = self._weights
        _global_rank = w.ep_rank * w.tp_degree + w.tp_rank
        tp_groups = build_tp_replica_groups(int(w.tp_degree), int(w.ep_degree))
        self._logits_processor = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=tuple(tuple(g) for g in tp_groups),
            collective_rank=int(_global_rank),
            collective_world_size=int(w.tp_degree) * int(w.ep_degree),
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._sampled_local_topk,
            gather_hidden=True,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=max(self._bs_buckets),
        )
        # No-SP LogitsProcessor for full-decode path (hidden is already full,
        # no seq-parallel gather needed).
        self._logits_processor_no_sp = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=tuple(tuple(g) for g in tp_groups),
            collective_rank=int(_global_rank),
            collective_world_size=int(w.tp_degree) * int(w.ep_degree),
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._sampled_local_topk,
            gather_hidden=False,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=max(self._bs_buckets),
        )

    @property
    def weights(self) -> Qwen3MoeWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- Weight upload --------------------------------------------------------

    def _upload_all_weights(
        self,
        model_config: Qwen3MoeModelConfig,
        *,
        weights: Qwen3MoeWeights | None = None,
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
            emb = np.asarray(
                reader.get_tensor("model.embed_tokens.weight"), dtype=w.dtype
            )
            shared["embeddings"] = upsert_device_tensor(
                _get_device_tensor_cls(),
                emb,
                name="embeddings",
                existing=None
                if existing_shared is None
                else existing_shared["embeddings"],
            )
            del emb

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
            v0 = int(w.lm_head_vocab_offset)
            v1 = v0 + int(w.local_vocab_size)
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

            # KV head selection via _kv_head_indices_for_rank.
            kv_indices = _kv_head_indices_for_rank(
                w.num_key_value_heads, tp_degree, tp_rank
            )
            kv_row0 = kv_indices[0] * head_dim
            kv_row1 = kv_row0 + kv_out

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

                # Attention weights: no bias. Row-slice Q, KV head select, transpose.
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

                # Output projection: slice input columns for TP, transpose.
                o_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.o_proj.weight")[
                        :, q_row0:q_row1
                    ],
                    dtype=w.dtype,
                ).T

                # Per-head QK norms (applied before RoPE).
                q_norm_w = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.q_norm.weight"),
                    dtype=w.dtype,
                )
                k_norm_w = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.k_norm.weight"),
                    dtype=w.dtype,
                )

                # Qwen3 has no attention sinks; keep the shared sink tensor zeroed.
                sinks = np.zeros((local_num_heads, 1), dtype=w.dtype)

                # Router (no bias).
                router_w = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.gate.weight"), dtype=w.dtype
                ).T

                # Experts: load per-expert gate/up/down, shard intermediate dim,
                # reformat for NKI blockwise kernel, cast to fp8.
                # Only load local experts (EP shard).
                gup_list = []
                down_list = []
                for expert_idx in range(e0, e1):
                    ep = f"{prefix}.mlp.experts.{expert_idx}"
                    # gate_proj.weight: [moe_intermediate_size, hidden_size]
                    gate_w = np.asarray(
                        reader.get_slice(f"{ep}.gate_proj.weight")[i0:i1, :],
                        dtype=w.dtype,
                    )  # [I_local, H]
                    up_w = np.asarray(
                        reader.get_slice(f"{ep}.up_proj.weight")[i0:i1, :],
                        dtype=w.dtype,
                    )  # [I_local, H]
                    # Stack gate+up: [H, 2, I_local]
                    gate_t = gate_w.T  # [H, I_local]
                    up_t = up_w.T  # [H, I_local]
                    gup = np.stack([gate_t, up_t], axis=1)  # [H, 2, I_local]
                    gup_list.append(gup)
                    del gate_w, up_w, gate_t, up_t, gup

                    # down_proj.weight: [hidden_size, moe_intermediate_size]
                    down_w_expert = np.asarray(
                        reader.get_slice(f"{ep}.down_proj.weight")[:, i0:i1],
                        dtype=w.dtype,
                    )  # [H, I_local]
                    down_list.append(down_w_expert.T)  # [I_local, H]
                    del down_w_expert

                gup_w = np.stack(gup_list, axis=0).astype(
                    ml_dtypes.float8_e5m2
                )  # [E_local, H, 2, I_local]
                down_w = np.stack(down_list, axis=0).astype(
                    ml_dtypes.float8_e5m2
                )  # [E_local, I_local, H]
                del gup_list, down_list

                # Zero biases (NKI blockwise kernel requires these tensors).
                # Qwen3 MoE has no expert biases -- all zeros. Unlike GPT-OSS, we do
                # NOT pre-apply +1 to the up bias because Qwen3 uses standard SiLU
                # (the +1 shift in GPT-OSS is model-specific, not a kernel convention).
                gup_bias = np.zeros((E_local, I_local, 2), dtype=np.float32)
                down_bias_bc = np.zeros(
                    (E_local, MOE_BLOCK_SIZE, hidden), dtype=w.dtype
                )

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
                    "w_o": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        o_w,
                        name=f"wo_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["w_o"],
                    ),
                    "q_norm": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        q_norm_w,
                        name=f"q_norm_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["q_norm"],
                    ),
                    "k_norm": upsert_device_tensor(
                        _get_device_tensor_cls(),
                        k_norm_w,
                        name=f"k_norm_L{layer_idx}",
                        existing=None
                        if existing_layer is None
                        else existing_layer["k_norm"],
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
                del q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, sinks, router_w
                del gup_w, gup_bias, down_w, down_bias_bc

            return shared, layers
        finally:
            reader.close()

    @staticmethod
    def _validate_reload_compatibility(
        current: Qwen3MoeWeights,
        new: Qwen3MoeWeights,
    ) -> None:
        fields = (
            "vocab_size",
            "hidden_size",
            "head_dim",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "moe_intermediate_size",
            "num_experts",
            "experts_per_token",
            "rms_norm_eps",
            "rope_theta",
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
                    "Reloaded Qwen3 MoE weights are incompatible with the running "
                    f"executor: field {field_name} changed from "
                    f"{getattr(current, field_name)!r} to {getattr(new, field_name)!r}"
                )

    def reload_weights_from_disk(self, model_path: str) -> None:
        reload_config = replace(self._model_config, hf_model_id=str(model_path))
        snapshot_path, new_weights = _load_qwen3_moe_weights(reload_config)
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
        hidden_shard: object
        hidden_attn_shard: object
        q: object
        k: object
        v: object
        context: object
        topk: object
        aff: object
        normed_full: object
        moe_out: object
        token_pos: object
        block_to_expert: object
        hidden_full: object | None = None  # (token_bucket, hidden) for full decode

    @dataclass
    class _CompiledKernels:
        token_bucket: int
        token_shard: int
        embed_kernel: object
        pre_attn_kernel: object
        post_attn_kernel: object
        router_kernel: object
        moe_output_init: object
        moe_kernel: object
        moe_decode_kernel: object | None  # None when token_bucket > TILE_SIZE (128)
        full_embed_kernel: (
            object | None
        )  # Embed kernel for full token_bucket (no shard)
        full_decode_kernel: object | None  # All-layers-in-one-graph decode NEFF
        full_static_inputs: dict | None  # Per-layer weights for full decode kernel
        scratch: object

    def _use_top1_fast_path(self) -> bool:
        return self._sampled_local_topk == 1

    def _ensure_nki_step_inputs(self, token_bucket: int) -> PreparedNkiStepInputs:
        cached = self._nki_step_inputs_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached
        step_inputs = allocate_prepared_nki_step_inputs(
            _alloc_device_scratch,
            token_bucket=int(token_bucket),
            attn_bucket=int(token_bucket),
            max_context_len=int(self._runtime_config.max_context_len),
            max_requests=int(self._runtime_config.max_requests),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
            prefix="qwen3moe",
        )
        initialize_prepared_nki_step_inputs(step_inputs, _overwrite_device_tensor)
        self._nki_step_inputs_by_bucket[int(token_bucket)] = step_inputs
        return step_inputs

    # -- Full decode codegen ---------------------------------------------------

    def _get_full_decode_kernel_fn(
        self,
        *,
        token_bucket: int,
        attn_bucket: int,
        tp_replica_groups: tuple[tuple[int, ...], ...],
        ep_replica_groups: tuple[tuple[int, ...], ...],
    ):
        """Write generated source to disk and load the decode function."""
        w = self._weights
        mod_name = (
            f"qwen3_moe_full_decode_tp{w.tp_degree}_ep{w.ep_degree}"
            f"_layers"
            f"_t{int(token_bucket)}_a{int(attn_bucket)}"
        )
        fn_name = (
            f"qwen3_moe_full_decode_layers_forward_t{int(token_bucket)}"
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
                softmax_scale=float(self._attn_softmax_scale),
                experts_per_token=int(w.experts_per_token),
                tp_degree=int(w.tp_degree),
                ep_degree=int(w.ep_degree),
                ep_rank=int(w.ep_rank),
                local_num_experts=int(w.local_num_experts),
                tp_replica_groups=tp_replica_groups,
                ep_replica_groups=ep_replica_groups,
            ),
        )

    def _ensure_kernels(self, token_bucket: int) -> _CompiledKernels:
        cached = self._kernels_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached

        w = self._weights
        tp_degree = int(w.tp_degree)
        tp_rank = int(w.tp_rank)
        ep_degree = int(w.ep_degree)
        ep_rank = int(w.ep_rank)
        total_workers = tp_degree * ep_degree
        global_rank = ep_rank * tp_degree + tp_rank

        if token_bucket % tp_degree != 0:
            raise RuntimeError(
                "token_bucket must be divisible by tp_degree for seq-parallel. "
                f"{token_bucket=} {tp_degree=}"
            )
        token_shard = token_bucket // tp_degree

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

        input_ids = np.zeros((token_shard,), dtype=np.int32)
        hidden_shard = np.empty((token_shard, hidden), dtype=dtype)
        cos = np.empty((token_bucket, head_dim // 2), dtype=dtype)
        sin = np.empty((token_bucket, head_dim // 2), dtype=dtype)

        # Layer weight samples (shapes must match; values are irrelevant for tracing).
        in_norm = np.empty((hidden,), dtype=dtype)
        post_norm = np.empty((hidden,), dtype=dtype)
        w_q = np.empty((hidden, q_out), dtype=dtype)
        w_k = np.empty((hidden, kv_out), dtype=dtype)
        w_v = np.empty((hidden, kv_out), dtype=dtype)
        w_o = np.empty((q_out, hidden), dtype=dtype)
        q_norm_w = np.empty((head_dim,), dtype=dtype)
        k_norm_w = np.empty((head_dim,), dtype=dtype)
        router_w = np.empty((hidden, w.num_experts), dtype=dtype)

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

        # Compile.
        embed_kernel = _get_device_kernel_cls().compile_and_load(
            embedding_fn,
            input_ids,
            np.empty((w.vocab_size, hidden), dtype=dtype),
            name=f"qwen3moe_embed_s{token_shard}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
        )
        pre_attn_kernel = _get_device_kernel_cls().compile_and_load(
            pre_attn_fn,
            hidden_shard,
            in_norm,
            w_q,
            w_k,
            w_v,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv,
            head_dim=head_dim,
            rms_norm_eps=w.rms_norm_eps,
            tp_degree=tp_degree,
            tp_replica_groups=tp_groups_tuple,
            name=f"qwen3moe_pre_attn_t{token_bucket}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
            cc_enabled=cc_enabled,
            rank_id=global_rank,
            world_size=total_workers,
        )
        post_attn_kernel = _get_device_kernel_cls().compile_and_load(
            post_attn_fn,
            hidden_shard,
            np.empty((token_bucket, num_heads, head_dim), dtype=dtype),
            w_o,
            num_heads=num_heads,
            head_dim=head_dim,
            tp_degree=tp_degree,
            tp_replica_groups=tp_groups_tuple,
            name=f"qwen3moe_post_attn_t{token_bucket}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
            cc_enabled=cc_enabled,
            rank_id=global_rank,
            world_size=total_workers,
        )
        router_kernel = _get_device_kernel_cls().compile_and_load(
            router_fn,
            hidden_shard,
            post_norm,
            router_w,
            rms_norm_eps=w.rms_norm_eps,
            top_k=w.experts_per_token,
            tp_degree=tp_degree,
            ep_rank=ep_rank,
            local_num_experts=local_E,
            tp_replica_groups=tp_groups_tuple,
            name=f"qwen3moe_router_t{token_bucket}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
            cc_enabled=cc_enabled,
            rank_id=global_rank,
            world_size=total_workers,
        )

        moe_output_init = _get_device_kernel_cls().compile_and_load(
            output_init,
            moe_output,
            name=f"qwen3moe_moe_out_init_t{token_bucket}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
        )
        moe_kernel = _get_device_kernel_cls().compile_and_load(
            blockwise_add_residual,
            hidden_states=np.empty((token_bucket, hidden), dtype=dtype),
            residual_2d_shard=hidden_shard,
            output=moe_output,
            expert_affinities_masked_hbm=expert_aff,
            gate_up_proj_weight=gup_w,
            gate_up_bias_plus1_T_hbm=gup_b,
            down_proj_weight=down_w,
            down_bias_broadcasted_hbm=down_bias_bc,
            token_position_to_id=token_pos,
            block_to_expert=block_to_expert,
            num_static_blocks=int(num_static_blocks),
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            ep_replica_groups=ep_groups_tuple,
            tp_replica_groups=tp_groups_tuple,
            name=f"qwen3moe_moe_t{token_bucket}_b{num_blocks}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
            cc_enabled=cc_enabled,
            rank_id=global_rank,
            world_size=total_workers,
        )

        # Decode MoE kernel: only when token_bucket fits in one TILE_SIZE (128).
        moe_decode_kernel = None
        if token_bucket <= MOE_BLOCK_SIZE:
            moe_decode_kernel = _get_device_kernel_cls().compile_and_load(
                blockwise_decode_add_residual,
                hidden_states=np.empty((token_bucket, hidden), dtype=dtype),
                residual_2d_shard=hidden_shard,
                expert_affinities_masked_hbm=expert_aff,
                gate_up_proj_weight=gup_w,
                gate_up_bias_plus1_T_hbm=gup_b,
                down_proj_weight=down_w,
                down_bias_broadcasted_hbm=down_bias_bc,
                tp_degree=tp_degree,
                num_experts=local_E,
                ep_degree=ep_degree,
                ep_replica_groups=ep_groups_tuple,
                tp_replica_groups=tp_groups_tuple,
                name=f"qwen3moe_moe_decode_t{token_bucket}",
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
                cc_enabled=cc_enabled,
                rank_id=global_rank,
                world_size=total_workers,
            )

        # Full decode kernel: all layers fused into one graph (decode only).
        _NKI_MIN_Q_SEQLEN = 128
        is_full_decode_bucket = token_bucket <= MOE_BLOCK_SIZE
        full_embed_kernel = None
        full_decode_kernel = None
        full_static_inputs = None
        if is_full_decode_bucket:
            from nkipy_serving.attention.nki_blocksparse_flash_attention import (
                NKI_COMPILER_ARGS,
                compute_max_tile_counts,
            )

            attn_bucket = max(int(token_bucket), _NKI_MIN_Q_SEQLEN)
            max_p, max_d = compute_max_tile_counts(
                token_bucket=attn_bucket,
                max_context_len=self._runtime_config.max_context_len,
                max_requests=int(self._runtime_config.max_requests),
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

            # Full embed kernel: operates on full token_bucket (not sharded).
            full_embed_kernel = _get_device_kernel_cls().compile_and_load(
                embedding_fn,
                np.zeros((token_bucket,), dtype=np.int32),
                np.empty((w.vocab_size, hidden), dtype=dtype),
                name=f"qwen3moe_full_embed_t{token_bucket}",
                additional_compiler_args=self._compiler_args,
                use_cached_if_exists=True,
                build_dir=self._build_dir,
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

            sample_args: list[np.ndarray] = [
                np.empty((token_bucket, hidden), dtype=dtype),  # hidden
                cos,
                sin,
                np.zeros((token_bucket,), dtype=np.int32),  # slot_mapping
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
                        q_norm_w,
                        k_norm_w,
                        w_o,
                        post_norm,
                        router_w,
                        gup_w,
                        gup_b,
                        down_w,
                        down_bias_bc,
                    ]
                )
            full_decode_kernel = _get_device_kernel_cls().compile_and_load(
                full_fn,
                *sample_args,
                name=f"qwen3moe_full_decode_layers_t{token_bucket}_a{attn_bucket}",
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
            full_static_inputs = {}
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
                        f"q_norm_L{layer_idx}": lt["q_norm"],
                        f"k_norm_L{layer_idx}": lt["k_norm"],
                        f"w_o_L{layer_idx}": lt["w_o"],
                        f"post_attn_norm_L{layer_idx}": lt["post_attn_norm"],
                        f"router_w_L{layer_idx}": lt["router_w"],
                        f"gup_w_L{layer_idx}": lt["gup_w"],
                        f"gup_bias_L{layer_idx}": lt["gup_bias"],
                        f"down_w_L{layer_idx}": lt["down_w"],
                        f"down_bias_bc_L{layer_idx}": lt["down_bias_bc"],
                    }
                )

        # LM head sampling kernels owned by LogitsProcessor.
        # gather_hidden=True: the hidden input is the TP shard (token_shard, H).
        self._logits_processor._ensure_kernels(max(int(token_shard), 1))
        # No-SP variant for full-decode path: hidden is full token_bucket.
        if is_full_decode_bucket:
            self._logits_processor_no_sp._ensure_kernels(int(token_bucket))

        scratch = Qwen3MoeExecutor._BucketScratch(
            hidden_shard=_alloc_device_scratch(
                (token_shard, hidden), dtype, name=f"qwen3moe_hidden_t{token_bucket}"
            ),
            hidden_attn_shard=_alloc_device_scratch(
                (token_shard, hidden),
                dtype,
                name=f"qwen3moe_hidden_attn_t{token_bucket}",
            ),
            q=_alloc_device_scratch(
                (token_bucket, num_heads, head_dim),
                dtype,
                name=f"qwen3moe_q_t{token_bucket}",
            ),
            k=_alloc_device_scratch(
                (token_bucket, num_kv, head_dim),
                dtype,
                name=f"qwen3moe_k_t{token_bucket}",
            ),
            v=_alloc_device_scratch(
                (token_bucket, num_kv, head_dim),
                dtype,
                name=f"qwen3moe_v_t{token_bucket}",
            ),
            context=_alloc_device_scratch(
                (token_bucket, num_heads, head_dim),
                dtype,
                name=f"qwen3moe_context_t{token_bucket}",
            ),
            topk=_alloc_device_scratch(
                (token_bucket, w.experts_per_token),
                np.int8,
                name=f"qwen3moe_topk_t{token_bucket}",
            ),
            aff=_alloc_device_scratch(
                (token_bucket, local_E), dtype, name=f"qwen3moe_aff_t{token_bucket}"
            ),
            normed_full=_alloc_device_scratch(
                (token_bucket, hidden),
                dtype,
                name=f"qwen3moe_normed_t{token_bucket}",
            ),
            moe_out=_alloc_device_scratch(
                (token_bucket, hidden),
                dtype,
                name=f"qwen3moe_moe_out_t{token_bucket}",
            ),
            token_pos=_alloc_device_scratch(
                (num_blocks, MOE_BLOCK_SIZE),
                np.int32,
                name=f"qwen3moe_token_pos_t{token_bucket}",
            ),
            block_to_expert=_alloc_device_scratch(
                (num_blocks,),
                np.int8,
                name=f"qwen3moe_block_to_expert_t{token_bucket}",
            ),
            hidden_full=(
                _alloc_device_scratch(
                    (token_bucket, hidden),
                    dtype,
                    name=f"qwen3moe_hidden_full_t{token_bucket}",
                )
                if is_full_decode_bucket
                else None
            ),
        )

        kernels = Qwen3MoeExecutor._CompiledKernels(
            token_bucket=int(token_bucket),
            token_shard=int(token_shard),
            embed_kernel=embed_kernel,
            pre_attn_kernel=pre_attn_kernel,
            post_attn_kernel=post_attn_kernel,
            router_kernel=router_kernel,
            moe_output_init=moe_output_init,
            moe_kernel=moe_kernel,
            moe_decode_kernel=moe_decode_kernel,
            full_embed_kernel=full_embed_kernel,
            full_decode_kernel=full_decode_kernel,
            full_static_inputs=full_static_inputs,
            scratch=scratch,
        )
        self._kernels_by_bucket[int(token_bucket)] = kernels
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
        token_bucket = int(token_bucket)
        real_total_tokens = int(real_total_tokens)

        kernels = self._ensure_kernels(token_bucket)
        token_shard = kernels.token_shard
        scratch = kernels.scratch
        tp_rank = int(w.tp_rank)

        # Select batch-size bucket for LM head sampling.
        bs = int(attn_metadata.batch_size)
        from nkipy_serving.runtime.shape_guard import select_bucket

        bs_bucket = select_bucket(max(bs, 8), tuple(self._bs_buckets), "batch")

        # --- Build RoPE cos/sin on host, upload ---
        cos_np, sin_np = _build_rope_cache_for_positions(
            positions.astype(np.int32),
            head_dim=w.head_dim,
            theta=w.rope_theta,
            dtype=w.dtype,
        )
        cos_dev = _get_device_tensor_cls().from_numpy(cos_np, name="rope_cos")
        sin_dev = _get_device_tensor_cls().from_numpy(sin_np, name="rope_sin")

        # --- Check for full decode path ---
        use_full_decode = (
            attn_metadata.forward_mode == FORWARD_MODE_DECODE
            and kernels.full_decode_kernel is not None
        )

        if use_full_decode:
            # Full decode: all layers in one NEFF. Embedding runs separately
            # with full token_bucket (not seq-parallel sharded).
            input_ids_full = input_ids.astype(np.int32, copy=False)
            input_ids_full_dev = _get_device_tensor_cls().from_numpy(
                input_ids_full, name="input_ids_full"
            )
            hidden_full_dev = scratch.hidden_full
            kernels.full_embed_kernel(
                inputs={
                    "input_ids": input_ids_full_dev,
                    "embeddings": self._shared_tensors["embeddings"],
                },
                outputs={"output0": hidden_full_dev},
            )

            nki_step_inputs = self._ensure_nki_step_inputs(int(token_bucket))
            nki_step_input_map = prepare_prepared_nki_step_inputs(
                nki_step_inputs,
                _overwrite_device_tensor,
                attn_metadata=attn_metadata,
                real_total_tokens=int(real_total_tokens),
                num_blocks=int(self._nki_num_blocks),
                block_size=int(self._kv_pool.block_size),
            )

            full_inputs = dict(kernels.full_static_inputs or {})
            full_inputs["hidden"] = hidden_full_dev
            full_inputs["cos"] = cos_dev
            full_inputs["sin"] = sin_dev
            full_inputs.update(nki_step_input_map)

            # Full decode outputs: hidden_out + hidden_aux + mutated KV caches.
            hidden_full_dev = scratch.normed_full
            hidden_aux_dev = scratch.moe_out
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

            # LM head via no-SP LogitsProcessor (hidden is full, not sharded).
            last_indices = (attn_metadata.query_start_loc[1:] - 1).astype(np.int32)
            padded_last = np.zeros((int(bs_bucket),), dtype=np.int32)
            padded_last[:bs] = last_indices[:bs]
            last_dev = _get_device_tensor_cls().from_numpy(
                padded_last, name="last_token_indices"
            )

            lp_output = self._logits_processor_no_sp.forward(
                hidden_full_dev,
                self._shared_tensors["final_norm"],
                self._shared_tensors["lm_head"],
                last_dev,
                batch_size=bs,
                token_bucket=int(token_bucket),
                sampling_batch=sampling_batch,
                needs_logprobs=bool(sampling_batch.needs_logprobs)
                if sampling_batch
                else False,
                logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
            )
            return lp_output.to_shm_dict(vocab_offset=w.lm_head_vocab_offset)

        # --- Per-layer (prefill or decode fallback) ---
        # Embedding (seq-sharded).
        s0 = tp_rank * token_shard
        s1 = s0 + token_shard
        input_ids_shard = input_ids[s0:s1].astype(np.int32, copy=False)

        input_ids_dev = _get_device_tensor_cls().from_numpy(
            input_ids_shard, name="input_ids_shard"
        )
        hidden_shard_dev = scratch.hidden_shard
        kernels.embed_kernel(
            inputs={
                "input_ids": input_ids_dev,
                "embeddings": self._shared_tensors["embeddings"],
            },
            outputs={"output0": hidden_shard_dev},
        )

        from nkipy_serving.attention.nki_blocksparse_flash_attention import (
            run_prepared_nki_blocksparse_attention,
            run_prepared_nki_kv_update,
        )

        nki_step_inputs = self._ensure_nki_step_inputs(int(token_bucket))
        nki_step_input_map = prepare_prepared_nki_step_inputs(
            nki_step_inputs,
            _overwrite_device_tensor,
            attn_metadata=attn_metadata,
            real_total_tokens=int(real_total_tokens),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
        )

        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_tensors[layer_idx]

            # Pre-attn (produces full QKV with head norms + RoPE).
            q_dev = scratch.q
            k_dev = scratch.k
            v_dev = scratch.v
            kernels.pre_attn_kernel(
                inputs={
                    "hidden_shard": hidden_shard_dev,
                    "input_norm": lt["input_norm"],
                    "w_q": lt["w_q"],
                    "w_k": lt["w_k"],
                    "w_v": lt["w_v"],
                    "q_norm": lt["q_norm"],
                    "k_norm": lt["k_norm"],
                    "cos": cos_dev,
                    "sin": sin_dev,
                },
                outputs={"output0": q_dev, "output1": k_dev, "output2": v_dev},
            )

            # NKI KV cache update + attention (head_dim=128, no padding needed).
            kv_cache_dev = self._kv_cache_dev[layer_idx]
            run_prepared_nki_kv_update(
                k_dev,
                v_dev,
                kv_cache_dev,
                slot_mapping_dev=nki_step_input_map["slot_mapping"],
                token_bucket=token_bucket,
                num_kv_heads=w.num_kv_heads,
                head_dim=int(self._attn_head_dim),
                num_blocks=self._nki_num_blocks,
                block_size=self._kv_pool.block_size,
                kernel_cache=self._attention_kernel_cache,
                build_dir=self._build_dir,
            )
            context_dev = run_prepared_nki_blocksparse_attention(
                q_dev,
                k_dev,
                v_dev,
                kv_cache_dev,
                lt["sink"],
                prepared_inputs=nki_step_input_map,
                out_dev=scratch.context,
                num_heads=w.num_heads,
                num_kv_heads=w.num_kv_heads,
                head_dim=int(self._attn_head_dim),
                token_bucket=token_bucket,
                num_blocks=self._nki_num_blocks,
                block_size=self._kv_pool.block_size,
                kernel_cache=self._attention_kernel_cache,
                max_num_prefill_tiles=int(nki_step_inputs.max_num_prefill_tiles),
                max_num_decode_tiles=int(nki_step_inputs.max_num_decode_tiles),
                build_dir=self._build_dir,
                softmax_scale=float(self._attn_softmax_scale),
            )

            # Post-attn (reduce-scatter + residual, no bias).
            hidden_out_shard_dev = scratch.hidden_attn_shard
            kernels.post_attn_kernel(
                inputs={
                    "residual_shard": hidden_shard_dev,
                    "context": context_dev,
                    "w_o": lt["w_o"],
                },
                outputs={"output0": hidden_out_shard_dev},
            )
            hidden_attn_shard_dev = hidden_out_shard_dev

            # Router (produces full topk/affinities/normed hidden, no bias).
            # With EP, affinities are sliced to local_num_experts inside the kernel.
            topk_dev = scratch.topk
            aff_dev = scratch.aff
            normed_full_dev = scratch.normed_full
            kernels.router_kernel(
                inputs={
                    "hidden_shard": hidden_attn_shard_dev,
                    "post_attn_norm": lt["post_attn_norm"],
                    "router_weight": lt["router_w"],
                },
                outputs={
                    "output0": topk_dev,
                    "output1": aff_dev,
                    "output2": normed_full_dev,
                },
            )

            # MoE kernel: decode path (no CPU scheduling) or prefill path.
            use_decode_moe = (
                attn_metadata.forward_mode == FORWARD_MODE_DECODE
                and kernels.moe_decode_kernel is not None
            )

            # The old layer input is dead after post-attn, so reuse that scratch
            # buffer for the next layer's hidden shard instead of reallocating it.
            hidden_next_shard_dev = scratch.hidden_shard

            if use_decode_moe:
                # Decode: static block mappings baked into the compiled kernel.
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
            else:
                # Prefill: CPU scheduling builds dynamic block mappings.
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

                # Zero output buffer.
                moe_out_dev = scratch.moe_out
                kernels.moe_output_init(
                    inputs={"output.must_alias_input": moe_out_dev},
                    outputs={"output": moe_out_dev},
                )

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
                    outputs={
                        "output0": hidden_next_shard_dev,
                        "output": moe_out_dev,
                    },
                )

            hidden_shard_dev = hidden_next_shard_dev

        # --- LM-head + sampling via LogitsProcessor ---
        last_indices = (attn_metadata.query_start_loc[1:] - 1).astype(np.int32)
        padded_last = np.zeros((int(bs_bucket),), dtype=np.int32)
        padded_last[:bs] = last_indices[:bs]
        last_dev = _get_device_tensor_cls().from_numpy(
            padded_last, name="last_token_indices"
        )

        # gather_hidden=True: the hidden input is the TP shard.
        lp_token_shard = int(token_bucket) // max(int(w.tp_degree), 1)
        lp_output = self._logits_processor.forward(
            hidden_shard_dev,
            self._shared_tensors["final_norm"],
            self._shared_tensors["lm_head"],
            last_dev,
            batch_size=bs,
            token_bucket=max(lp_token_shard, 1),
            sampling_batch=sampling_batch,
            needs_logprobs=bool(sampling_batch.needs_logprobs)
            if sampling_batch
            else False,
            logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
        )
        return lp_output.to_shm_dict(vocab_offset=w.lm_head_vocab_offset)

    # -- Warmup ---------------------------------------------------------------

    def warmup(self, paddings=None) -> None:
        if paddings is None:
            return
        preload_blockwise_index_ext()
        for token_bucket in sorted(set(paddings.token_paddings)):
            self._ensure_kernels(int(token_bucket))
        run_synthetic_warmup_steps(
            build_standard_warmup_steps(paddings),
            token_paddings=tuple(int(bucket) for bucket in paddings.token_paddings),
            bs_paddings=tuple(int(bucket) for bucket in paddings.bs_paddings),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
            num_kv_heads=int(self._weights.num_kv_heads),
            head_dim=int(self._weights.head_dim),
            forward=self.forward,
        )
