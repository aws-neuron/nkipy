"""Qwen3MoeEagerExecutor: stage-level composable execution via Fragment.

Eager executor for Qwen3 MoE supporting both prefill and decode. Uses no-SP
(full-token) graph functions on every rank. Each transformer layer is
decomposed into:
  decode:  pre_attn → attn → post_attn → router_moe   (fused router+MoE)
  prefill: pre_attn → attn → post_attn → router_prefill
                       → [CPU block scheduling] → moe_dispatch_prefill

``self.attn`` is the swappable debug point (device ↔ CPU).
All other stages always run on device.

Prefill performs a host sync inside the layer loop (topk → numpy copy →
schedule → upload) — matches production; slower than decode per-layer.
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import FORWARD_MODE_DECODE, AttentionMetadata
from nkipy_serving.attention.nki_blocksparse_flash_attention import NKI_COMPILER_ARGS
from nkipy_serving.fragment_jit import jit
from nkipy_serving.models._device_utils import (
    _get_device_tensor_cls,
)
from nkipy_serving.models._device_utils import (
    allocate_device_kv_cache as _allocate_device_kv_cache,
)
from nkipy_serving.models._device_utils import (
    join_compiler_args as _join_compiler_args,
)
from nkipy_serving.models._device_utils import (
    pre_allocate_kv_cache_zeros as _pre_allocate_kv_cache_zeros,
)
from nkipy_serving.models.common.attn_fns import cpu_attn_fn, nki_attn_fn
from nkipy_serving.models.common.eager_executor_base import EagerExecutorBase
from nkipy_serving.models.common.moe_cpu_ops import (
    cpu_moe_dispatch_swish,
    softmax_topk_masked,
)
from nkipy_serving.models.qwen3_moe.config import (
    Qwen3MoeModelConfig,
    Qwen3MoeWeights,
)
from nkipy_serving.models.qwen3_moe.graph_fns import (
    embedding_fn,
    moe_dispatch_prefill_no_sp_fn,
    post_attn_decode_no_sp_fn,
    pre_attn_decode_no_sp_fn,
    router_moe_decode_no_sp_fn,
    router_prefill_no_sp_fn,
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
)
from nkipy_serving.ops.moe.blockwise_index import BLOCK_SIZE as MOE_BLOCK_SIZE
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule
from nkipy_serving.ops.nn import (
    apply_rms_norm as _apply_rms_norm,
)
from nkipy_serving.ops.nn import (
    build_rope_cache_for_positions as _build_rope_cache_for_positions,
)
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessor


class Qwen3MoeEagerExecutor(EagerExecutorBase):
    """Stage-level composable executor for Qwen3 MoE; supports prefill and decode.

    ``self.attn`` is the swappable debug point.
    All other stages always run on device. Decode uses fused ``router_moe``;
    prefill splits into ``router_prefill`` → CPU block scheduling → ``moe_dispatch_prefill``.
    """

    def __init__(self, model_config: Qwen3MoeModelConfig, kv_pool, runtime_config):
        self._model_config = model_config
        snapshot_path, self._weights = _load_qwen3_moe_weights(model_config)
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config

        from nkipy_serving.runtime.parallel_groups import (
            build_ep_replica_groups,
            build_tp_replica_groups,
        )
        from nkipy_serving.runtime.precompile_paddings import (
            build_precompile_paddings,
        )

        w = self._weights
        self._precompile_paddings = build_precompile_paddings(runtime_config)
        self._max_requests_per_step = int(
            self._precompile_paddings.max_padded_batch_size
        )
        self._dense_local_topk = int(runtime_config.dense_local_topk)
        self._compiler_args = runtime_config.nkipy_compiler_args
        _global_rank = w.ep_rank * w.tp_degree + w.tp_rank
        self._build_dir = f"{runtime_config.config_build_dir()}/rank{_global_rank}"

        # TP/EP replica groups
        tp_groups = build_tp_replica_groups(int(w.tp_degree), int(w.ep_degree))
        ep_groups = build_ep_replica_groups(int(w.tp_degree), int(w.ep_degree))
        self._tp_replica_groups = tuple(tuple(g) for g in tp_groups)
        self._ep_replica_groups = tuple(tuple(g) for g in ep_groups)
        self._total_workers = int(w.tp_degree) * int(w.ep_degree)
        self._global_rank = _global_rank

        # KV caches
        ensure_nki_bridge()

        self._nki_num_blocks = kv_pool.num_blocks + 1
        self._kv_cache_dev = _allocate_device_kv_cache(
            num_hidden_layers=w.num_hidden_layers,
            num_kv_heads=w.num_kv_heads,
            head_dim=int(w.head_dim),
            block_size=kv_pool.block_size,
            num_blocks=self._nki_num_blocks,
            dtype=w.dtype,
        )
        self._kv_cache_zeros = _pre_allocate_kv_cache_zeros(
            num_blocks=self._nki_num_blocks,
            num_kv_heads=w.num_kv_heads,
            block_size=kv_pool.block_size,
            head_dim=int(w.head_dim),
            dtype=w.dtype,
        )

        # Upload weights (also stash numpy copies for forward_cpu)
        (
            self._shared_tensors,
            self._layer_tensors,
            self._shared_np,
            self._layer_np,
        ) = self._upload_all_weights(
            model_config,
            snapshot_path=snapshot_path,
        )

        # Base class init
        self._init_nki_step_inputs_cache()
        self._init_lm_head_scratch(self._max_requests_per_step, prefix="eager_moe")

        # LogitsProcessor (no-SP: hidden is full, no gather needed)
        self._logits_processor = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=self._tp_replica_groups,
            collective_rank=_global_rank,
            collective_world_size=self._total_workers,
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._dense_local_topk,
            gather_hidden=False,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=self._max_requests_per_step,
        )

        # Fragments
        _cc_kw = dict(
            cc_enabled=True if self._total_workers > 1 else None,
            rank_id=_global_rank if self._total_workers > 1 else None,
            world_size=self._total_workers if self._total_workers > 1 else None,
        )
        _nki_kw = dict(
            additional_compiler_args=_join_compiler_args(
                self._compiler_args,
                NKI_COMPILER_ARGS,
            ),
            **_cc_kw,
        )

        self._embed = jit(
            embedding_fn,
            name="eager_moe_embed",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self._pre_attn = jit(
            pre_attn_decode_no_sp_fn,
            name="eager_moe_pre_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self.attn = jit(
            nki_attn_fn,
            name="eager_moe_attn",
            build_dir=self._build_dir,
            **_nki_kw,
        )
        self._post_attn = jit(
            post_attn_decode_no_sp_fn,
            name="eager_moe_post_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
            **_cc_kw,
        )
        self._router_moe = jit(
            router_moe_decode_no_sp_fn,
            name="eager_moe_router_moe",
            build_dir=self._build_dir,
            **_nki_kw,
        )
        self._router_prefill = jit(
            router_prefill_no_sp_fn,
            name="eager_moe_router_prefill",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self._moe_dispatch_prefill = jit(
            moe_dispatch_prefill_no_sp_fn,
            name="eager_moe_moe_dispatch_prefill",
            build_dir=self._build_dir,
            **_nki_kw,
        )

    # -- Properties ---------------------------------------------------------------

    @property
    def weights(self) -> Qwen3MoeWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- Weight upload (mirrors production executor) ---------------------------------

    def _upload_all_weights(
        self,
        model_config: Qwen3MoeModelConfig,
        *,
        snapshot_path: Path | None = None,
    ) -> tuple[
        dict[str, object],
        list[dict[str, object]],
        dict[str, np.ndarray],
        list[dict[str, np.ndarray]],
    ]:
        w = self._weights
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
            shared_np: dict[str, np.ndarray] = {}
            layers_np: list[dict[str, np.ndarray]] = []
            DeviceTensor = _get_device_tensor_cls()

            # Shared weights
            emb = np.asarray(
                reader.get_tensor("model.embed_tokens.weight"), dtype=w.dtype
            )
            shared["embeddings"] = DeviceTensor.from_numpy(emb, name="embeddings")
            shared_np["embeddings"] = emb

            fn = np.asarray(reader.get_tensor("model.norm.weight"), dtype=w.dtype)
            shared["final_norm"] = DeviceTensor.from_numpy(fn, name="final_norm")
            shared_np["final_norm"] = fn

            v0 = int(w.lm_head_vocab_offset)
            v1 = v0 + int(w.local_vocab_size)
            lm = np.asarray(reader.get_slice("lm_head.weight")[v0:v1, :], dtype=w.dtype)
            shared["lm_head"] = DeviceTensor.from_numpy(lm, name="lm_head")
            shared_np["lm_head"] = lm

            # Per-layer weights
            local_num_heads = w.num_heads
            head_dim = w.head_dim
            tp_rank = w.tp_rank
            q_out = local_num_heads * head_dim
            q_row0 = (tp_rank * local_num_heads) * head_dim
            q_row1 = q_row0 + q_out

            kv_indices = _kv_head_indices_for_rank(
                w.num_key_value_heads,
                w.tp_degree,
                tp_rank,
            )
            kv_out = w.num_kv_heads * head_dim
            kv_row0 = kv_indices[0] * head_dim
            kv_row1 = kv_row0 + kv_out

            I_local = w.local_intermediate_size
            i0 = tp_rank * I_local
            i1 = i0 + I_local
            E_local = w.local_num_experts
            e0 = w.ep_rank * E_local
            e1 = e0 + E_local

            for layer_idx in range(w.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"

                in_norm = np.asarray(
                    reader.get_tensor(f"{prefix}.input_layernorm.weight"),
                    dtype=w.dtype,
                )
                post_norm = np.asarray(
                    reader.get_tensor(f"{prefix}.post_attention_layernorm.weight"),
                    dtype=w.dtype,
                )

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
                o_w = np.asarray(
                    reader.get_slice(f"{prefix}.self_attn.o_proj.weight")[
                        :, q_row0:q_row1
                    ],
                    dtype=w.dtype,
                ).T

                q_norm_w = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.q_norm.weight"),
                    dtype=w.dtype,
                )
                k_norm_w = np.asarray(
                    reader.get_tensor(f"{prefix}.self_attn.k_norm.weight"),
                    dtype=w.dtype,
                )

                router_w = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.gate.weight"),
                    dtype=w.dtype,
                ).T

                # Experts (EP-sharded, TP-sharded intermediate)
                gup_list, down_list = [], []
                for expert_idx in range(e0, e1):
                    ep = f"{prefix}.mlp.experts.{expert_idx}"
                    gate_w = np.asarray(
                        reader.get_slice(f"{ep}.gate_proj.weight")[i0:i1, :],
                        dtype=w.dtype,
                    )
                    up_w = np.asarray(
                        reader.get_slice(f"{ep}.up_proj.weight")[i0:i1, :],
                        dtype=w.dtype,
                    )
                    gup_list.append(np.stack([gate_w.T, up_w.T], axis=1))
                    down_w_e = np.asarray(
                        reader.get_slice(f"{ep}.down_proj.weight")[:, i0:i1],
                        dtype=w.dtype,
                    )
                    down_list.append(down_w_e.T)

                gup_w = np.stack(gup_list, axis=0).astype(ml_dtypes.float8_e5m2)
                down_w = np.stack(down_list, axis=0).astype(ml_dtypes.float8_e5m2)
                gup_bias = np.zeros((E_local, I_local, 2), dtype=np.float32)
                down_bias_bc = np.zeros(
                    (E_local, MOE_BLOCK_SIZE, w.hidden_size), dtype=w.dtype
                )

                lt: dict[str, object] = {
                    "input_norm": DeviceTensor.from_numpy(
                        in_norm, name=f"in_norm_L{layer_idx}"
                    ),
                    "post_attn_norm": DeviceTensor.from_numpy(
                        post_norm, name=f"post_norm_L{layer_idx}"
                    ),
                    "w_q": DeviceTensor.from_numpy(q_w, name=f"wq_L{layer_idx}"),
                    "w_k": DeviceTensor.from_numpy(k_w, name=f"wk_L{layer_idx}"),
                    "w_v": DeviceTensor.from_numpy(v_w, name=f"wv_L{layer_idx}"),
                    "w_o": DeviceTensor.from_numpy(o_w, name=f"wo_L{layer_idx}"),
                    "q_norm": DeviceTensor.from_numpy(
                        q_norm_w, name=f"q_norm_L{layer_idx}"
                    ),
                    "k_norm": DeviceTensor.from_numpy(
                        k_norm_w, name=f"k_norm_L{layer_idx}"
                    ),
                    "router_w": DeviceTensor.from_numpy(
                        router_w, name=f"router_w_L{layer_idx}"
                    ),
                    "gup_w": DeviceTensor.from_numpy(gup_w, name=f"gup_w_L{layer_idx}"),
                    "gup_bias": DeviceTensor.from_numpy(
                        gup_bias, name=f"gup_bias_L{layer_idx}"
                    ),
                    "down_w": DeviceTensor.from_numpy(
                        down_w, name=f"down_w_L{layer_idx}"
                    ),
                    "down_bias_bc": DeviceTensor.from_numpy(
                        down_bias_bc, name=f"down_bias_L{layer_idx}"
                    ),
                }
                layers.append(lt)
                layers_np.append(
                    {
                        "input_norm": in_norm,
                        "post_attn_norm": post_norm,
                        "w_q": q_w,
                        "w_k": k_w,
                        "w_v": v_w,
                        "w_o": o_w,
                        "q_norm": q_norm_w,
                        "k_norm": k_norm_w,
                        "router_w": router_w,
                        "gup_w": gup_w,
                        "gup_bias": gup_bias,
                        "down_w": down_w,
                        "down_bias_bc": down_bias_bc,
                    }
                )

            return shared, layers, shared_np, layers_np
        finally:
            reader.close()

    # -- CPU reference forward ---------------------------------------------------

    def forward_cpu(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_caches: list[np.ndarray],
        attn_metadata: AttentionMetadata,
    ) -> np.ndarray:
        """TP=1/EP=1 CPU reference forward. Returns logits [total_tokens, local_vocab_size].

        Uses numpy graph-fn stages + vanilla attention + a numpy MoE
        dispatch. Hardcodes tp_degree=1/ep_degree=1: rejects sharded
        weight containers because it would silently return rank-local
        partial output (vocab shard, expert shard, KV head shard).
        """
        w = self._weights
        self._require_single_rank_for_forward_cpu(w.tp_degree, w.ep_degree)
        hidden = self._shared_np["embeddings"][input_ids]
        cos, sin = _build_rope_cache_for_positions(
            positions.astype(np.float32),
            w.head_dim,
            w.rope_theta,
            dtype=w.dtype,
        )
        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_np[layer_idx]
            q, k, v = pre_attn_decode_no_sp_fn(
                hidden,
                lt["input_norm"],
                lt["w_q"],
                lt["w_k"],
                lt["w_v"],
                lt["q_norm"],
                lt["k_norm"],
                cos,
                sin,
                num_heads=w.num_heads,
                num_kv_heads=w.num_kv_heads,
                head_dim=w.head_dim,
                rms_norm_eps=float(w.rms_norm_eps),
            )
            context = cpu_attn_fn(q, k, v, kv_caches[layer_idx], attn_metadata)
            attn_out = (
                context.reshape(context.shape[0], w.num_heads * w.head_dim).astype(
                    w.dtype
                )
                @ lt["w_o"]
            ).astype(w.dtype)
            hidden = (hidden + attn_out).astype(w.dtype)

            # MoE: norm -> router logits -> softmax(topk) -> dispatch -> residual add
            normed = _apply_rms_norm(
                hidden, lt["post_attn_norm"], eps=float(w.rms_norm_eps)
            )
            logits = (normed.astype(w.dtype) @ lt["router_w"]).astype(np.float32)
            affinities = softmax_topk_masked(logits, int(w.experts_per_token))
            # EP slice (TP=1/EP=1 → identity).
            if int(w.local_num_experts) < int(w.num_experts):
                e0 = int(w.ep_rank) * int(w.local_num_experts)
                affinities = affinities[:, e0 : e0 + int(w.local_num_experts)]
            moe_out = cpu_moe_dispatch_swish(
                normed,
                affinities,
                lt["gup_w"],
                lt["down_w"],
            )
            hidden = (hidden + moe_out).astype(w.dtype)

        hidden = _apply_rms_norm(
            hidden, self._shared_np["final_norm"], eps=float(w.rms_norm_eps)
        )
        lm_head = self._shared_np["lm_head"]
        return (hidden.astype(lm_head.dtype) @ lm_head.T).astype(np.float32)

    # -- Forward -----------------------------------------------------------------

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_caches: list[np.ndarray],
        attn_metadata: AttentionMetadata,
        token_bucket: int = 0,
        real_total_tokens: int = 0,
        sampling_batch: DeviceSamplingBatch | None = None,
        attention_lane: int = -1,
    ) -> dict[str, np.ndarray]:
        """Device forward using stage-level fragments; supports prefill and decode."""
        effective_real_tt = (
            real_total_tokens if real_total_tokens > 0 else input_ids.shape[0]
        )
        effective_bucket = token_bucket if token_bucket > 0 else input_ids.shape[0]
        is_decode = int(attn_metadata.forward_mode) == FORWARD_MODE_DECODE
        if is_decode and effective_bucket > MOE_BLOCK_SIZE:
            raise ValueError(
                f"token_bucket={effective_bucket} exceeds decode MoE BLOCK_SIZE="
                f"{MOE_BLOCK_SIZE}; use prefill mode for larger buckets"
            )
        w = self._weights

        # RoPE on host
        cos_np, sin_np = _build_rope_cache_for_positions(
            positions.astype(np.float32),
            w.head_dim,
            w.rope_theta,
            dtype=w.dtype,
        )

        # Embedding
        hidden = self._embed(
            input_ids.astype(np.int32, copy=False),
            self._shared_tensors["embeddings"],
        )

        # NKI step inputs
        step_inputs = None
        if self.attn.device:
            step_inputs = self._prepare_nki_step_inputs(
                token_bucket=effective_bucket,
                real_total_tokens=effective_real_tt,
                attn_metadata=attn_metadata,
            )

        # Layer loop: pre_attn → attn → post_attn → router_moe
        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_tensors[layer_idx]
            kv_dev = self._kv_cache_dev[layer_idx]

            q, k, v = self._pre_attn(
                hidden,
                lt["input_norm"],
                lt["w_q"],
                lt["w_k"],
                lt["w_v"],
                lt["q_norm"],
                lt["k_norm"],
                cos_np,
                sin_np,
                num_heads=int(w.num_heads),
                num_kv_heads=int(w.num_kv_heads),
                head_dim=int(w.head_dim),
                rms_norm_eps=float(w.rms_norm_eps),
            )

            if self.attn.device:
                context = self.attn(
                    q,
                    k,
                    v,
                    kv_dev,
                    step_inputs["slot_mapping"],
                    step_inputs["p_tqi"],
                    step_inputs["p_tbt"],
                    step_inputs["p_tm"],
                    step_inputs["p_ndls"],
                    step_inputs["p_qup"],
                    step_inputs["p_lti"],
                    step_inputs["d_tqi"],
                    step_inputs["d_tbt"],
                    step_inputs["d_tm"],
                    step_inputs["d_ndls"],
                    step_inputs["d_qup"],
                    step_inputs["d_lti"],
                    num_heads=int(w.num_heads),
                    num_kv_heads=int(w.num_kv_heads),
                    head_dim=int(w.head_dim),
                )
            else:
                context = self._run_cpu_attn(
                    self.attn,
                    q,
                    k,
                    v,
                    kv_dev,
                    attn_metadata,
                    effective_real_tt,
                    effective_bucket,
                )

            hidden = self._post_attn(
                hidden,
                context,
                lt["w_o"],
                num_heads=int(w.num_heads),
                head_dim=int(w.head_dim),
                tp_degree=int(w.tp_degree),
                tp_replica_groups=self._tp_replica_groups,
            )

            if is_decode:
                hidden = self._router_moe(
                    hidden,
                    lt["post_attn_norm"],
                    lt["router_w"],
                    lt["gup_w"],
                    lt["gup_bias"],
                    lt["down_w"],
                    lt["down_bias_bc"],
                    hidden,  # residual_2d
                    rms_norm_eps=float(w.rms_norm_eps),
                    top_k=int(w.experts_per_token),
                    tp_degree=int(w.tp_degree),
                    ep_degree=int(w.ep_degree),
                    ep_rank=int(w.ep_rank),
                    local_num_experts=int(w.local_num_experts),
                    ep_replica_groups=self._ep_replica_groups,
                    tp_replica_groups=self._tp_replica_groups,
                )
            else:
                topk_dev, aff_dev, normed_dev = self._router_prefill(
                    hidden,
                    lt["post_attn_norm"],
                    lt["router_w"],
                    rms_norm_eps=float(w.rms_norm_eps),
                    top_k=int(w.experts_per_token),
                    ep_rank=int(w.ep_rank),
                    local_num_experts=int(w.local_num_experts),
                )
                topk_np = np.asarray(topk_dev.numpy())
                token_pos_np, b2e_np, _nb, n_static = build_prefill_moe_schedule(
                    topk_np,
                    token_bucket=int(effective_bucket),
                    real_total_tokens=int(effective_real_tt),
                    experts_per_token=int(w.experts_per_token),
                    local_num_experts=int(w.local_num_experts),
                    ep_degree=int(w.ep_degree),
                    ep_rank=int(w.ep_rank),
                )
                scr = self._ensure_prefill_scratch(
                    int(effective_bucket),
                    prefix="eager_moe",
                )
                _overwrite_device_tensor(scr["token_pos"], token_pos_np)
                _overwrite_device_tensor(scr["block_to_expert"], b2e_np)
                _overwrite_device_tensor(scr["moe_out"], scr["moe_out_zero"])
                hidden = self._moe_dispatch_prefill(
                    normed_dev,
                    hidden,  # residual_2d
                    aff_dev,
                    scr["token_pos"],
                    scr["block_to_expert"],
                    lt["gup_w"],
                    lt["gup_bias"],
                    lt["down_w"],
                    lt["down_bias_bc"],
                    scr["moe_out"],
                    num_static_blocks=int(n_static),
                    tp_degree=int(w.tp_degree),
                    ep_degree=int(w.ep_degree),
                    ep_replica_groups=self._ep_replica_groups,
                    tp_replica_groups=self._tp_replica_groups,
                )

        # LM head
        return self._run_lm_head(
            hidden,
            attn_metadata,
            token_bucket=effective_bucket,
            sampling_batch=sampling_batch,
        )
