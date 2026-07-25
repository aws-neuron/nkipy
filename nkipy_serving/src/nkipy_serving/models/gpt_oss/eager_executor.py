"""GptOssEagerExecutor: stage-level composable execution via Fragment.

Eager executor for GPT-OSS supporting both prefill and decode. Uses no-SP
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
from nkipy_serving.models.common.eager_executor_base import EagerExecutorBase
from nkipy_serving.models.common.moe_cpu_ops import (
    cpu_moe_dispatch_swiglu_oai,
    softmax_topk_masked,
)
from nkipy_serving.models.gpt_oss.config import GptOssModelConfig, GptOssWeights
from nkipy_serving.models.gpt_oss.graph_fns import (
    _build_rope_cache_for_positions_yarn,
    cpu_attn_with_sink_fn,
    embedding_fn,
    moe_dispatch_prefill_no_sp_fn,
    nki_attn_with_sink_fn,
    post_attn_decode_no_sp_fn,
    pre_attn_decode_no_sp_fn,
    router_moe_decode_no_sp_fn,
    router_prefill_no_sp_fn,
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
)
from nkipy_serving.ops.moe.blockwise_index import BLOCK_SIZE as MOE_BLOCK_SIZE
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule
from nkipy_serving.ops.nn import apply_rms_norm as _apply_rms_norm
from nkipy_serving.ops.vocab_parallel_embedding import (
    get_vocab_parallel_shard,
    vocab_parallel_embedding_local_fn,
)
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessor


class GptOssEagerExecutor(EagerExecutorBase):
    """Stage-level composable executor for GPT-OSS; supports prefill and decode.

    ``self.attn`` is the swappable debug point.
    All other stages always run on device. Decode uses fused ``router_moe``;
    prefill splits into ``router_prefill`` → CPU block scheduling → ``moe_dispatch_prefill``.
    """

    def __init__(self, model_config: GptOssModelConfig, kv_pool, runtime_config):
        self._model_config = model_config
        snapshot_path, self._weights = _load_gpt_oss_weights(model_config)
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

        # Vocab-parallel shard info for embedding
        self._vocab_shard = get_vocab_parallel_shard(
            vocab_size=int(w.vocab_size),
            rank=int(w.tp_rank),
            world_size=int(w.tp_degree),
        )

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
        self._init_lm_head_scratch(self._max_requests_per_step, prefix="eager_gptoss")

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
            name="eager_gptoss_embed",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
            **_cc_kw,
        )
        self._pre_attn = jit(
            pre_attn_decode_no_sp_fn,
            name="eager_gptoss_pre_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self.attn = jit(
            nki_attn_with_sink_fn,
            name="eager_gptoss_attn",
            build_dir=self._build_dir,
            **_nki_kw,
        )
        self._post_attn = jit(
            post_attn_decode_no_sp_fn,
            name="eager_gptoss_post_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
            **_cc_kw,
        )
        self._router_moe = jit(
            router_moe_decode_no_sp_fn,
            name="eager_gptoss_router_moe",
            build_dir=self._build_dir,
            **_nki_kw,
        )
        self._router_prefill = jit(
            router_prefill_no_sp_fn,
            name="eager_gptoss_router_prefill",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self._moe_dispatch_prefill = jit(
            moe_dispatch_prefill_no_sp_fn,
            name="eager_gptoss_moe_dispatch_prefill",
            build_dir=self._build_dir,
            **_nki_kw,
        )

    # -- Properties ---------------------------------------------------------------

    @property
    def weights(self) -> GptOssWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- CPU attention fallback (with sink) -------------------------------------

    @staticmethod
    def _run_cpu_attn_with_sink(
        attn_frag,
        q,
        k,
        v,
        kv_dev,
        sink_dev,
        attn_metadata: AttentionMetadata,
        effective_real_tt: int,
        effective_bucket: int,
    ):
        rt = effective_real_tt
        q_cpu = q[:rt] if hasattr(q, "__getitem__") else q.numpy()[:rt]
        k_cpu = k[:rt] if hasattr(k, "__getitem__") else k.numpy()[:rt]
        v_cpu = v[:rt] if hasattr(v, "__getitem__") else v.numpy()[:rt]
        kv_np = kv_dev.numpy() if hasattr(kv_dev, "numpy") else kv_dev
        sink_np = sink_dev.numpy() if hasattr(sink_dev, "numpy") else sink_dev

        ctx_cpu = attn_frag(q_cpu, k_cpu, v_cpu, kv_np, sink_np, attn_metadata)

        _overwrite_device_tensor(kv_dev, kv_np)

        if rt < effective_bucket:
            pad_shape = list(ctx_cpu.shape)
            pad_shape[0] = effective_bucket - rt
            return np.concatenate(
                [ctx_cpu, np.zeros(pad_shape, dtype=ctx_cpu.dtype)],
                axis=0,
            )
        return ctx_cpu

    # -- Weight upload (mirrors production executor) ---------------------------------

    def _upload_all_weights(
        self,
        model_config: GptOssModelConfig,
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
            vs = self._vocab_shard
            emb = np.asarray(
                reader.get_slice("model.embed_tokens.weight")[
                    int(vs.vocab_start_index) : int(vs.vocab_end_index), :
                ],
                dtype=w.dtype,
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
            hidden = w.hidden_size
            tp_rank = w.tp_rank
            tp_degree = w.tp_degree
            q_out = local_num_heads * head_dim
            q_row0 = (tp_rank * local_num_heads) * head_dim
            q_row1 = q_row0 + q_out
            kv_out = w.num_kv_heads * head_dim
            kv_row0 = tp_rank * head_dim
            kv_row1 = kv_row0 + kv_out

            hidden_shard = hidden // tp_degree
            h0 = tp_rank * hidden_shard
            h1 = h0 + hidden_shard

            I_local = w.local_intermediate_size
            i0 = tp_rank * I_local
            i1 = i0 + I_local
            E_local = w.local_num_experts
            e0 = w.ep_rank * E_local
            e1 = e0 + E_local

            for layer_idx in range(w.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"

                in_norm = np.asarray(
                    reader.get_tensor(f"{prefix}.input_layernorm.weight"), dtype=w.dtype
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

                router_w = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.router.weight"), dtype=w.dtype
                ).T
                router_b = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.router.bias"), dtype=w.dtype
                )

                # Experts (gate-up interleaved layout)
                gup_sl = reader.get_slice(f"{prefix}.mlp.experts.gate_up_proj")
                gup_chunk = np.asarray(gup_sl[:, :, 2 * i0 : 2 * i1], dtype=w.dtype)
                gup_chunk = gup_chunk.reshape(
                    (gup_chunk.shape[0], gup_chunk.shape[1], I_local, 2)
                )
                gup_chunk = np.transpose(gup_chunk, (0, 1, 3, 2))
                gup_w = gup_chunk[e0:e1].astype(ml_dtypes.float8_e5m2)

                gupb_sl = reader.get_slice(f"{prefix}.mlp.experts.gate_up_proj_bias")
                gupb_chunk = np.asarray(gupb_sl[:, 2 * i0 : 2 * i1], dtype=np.float32)
                gup_bias = gupb_chunk.reshape((gupb_chunk.shape[0], I_local, 2))
                gup_bias[:, :, 1] = gup_bias[:, :, 1] + np.float32(1.0)
                gup_bias = gup_bias[e0:e1]

                down_sl = reader.get_slice(f"{prefix}.mlp.experts.down_proj")
                down_w = np.asarray(down_sl[:, i0:i1, :], dtype=w.dtype)
                down_w = down_w[e0:e1].astype(ml_dtypes.float8_e5m2)

                down_b = np.asarray(
                    reader.get_tensor(f"{prefix}.mlp.experts.down_proj_bias"),
                    dtype=w.dtype,
                )
                down_b[:, :h0] = 0
                down_b[:, h1:] = 0
                down_b = down_b[e0:e1]
                down_bias_bc = np.broadcast_to(
                    down_b[:, None, :],
                    (down_b.shape[0], MOE_BLOCK_SIZE, down_b.shape[1]),
                ).copy()

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
                    "b_q": DeviceTensor.from_numpy(q_b, name=f"bq_L{layer_idx}"),
                    "b_k": DeviceTensor.from_numpy(k_b, name=f"bk_L{layer_idx}"),
                    "b_v": DeviceTensor.from_numpy(v_b, name=f"bv_L{layer_idx}"),
                    "w_o": DeviceTensor.from_numpy(o_w, name=f"wo_L{layer_idx}"),
                    "b_o": DeviceTensor.from_numpy(o_b_full, name=f"bo_L{layer_idx}"),
                    "sink": DeviceTensor.from_numpy(sinks, name=f"sink_L{layer_idx}"),
                    "router_w": DeviceTensor.from_numpy(
                        router_w, name=f"router_w_L{layer_idx}"
                    ),
                    "router_b": DeviceTensor.from_numpy(
                        router_b, name=f"router_b_L{layer_idx}"
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
                        "b_q": q_b,
                        "b_k": k_b,
                        "b_v": v_b,
                        "w_o": o_w,
                        "b_o": o_b_full,
                        "sink": sinks,
                        "router_w": router_w,
                        "router_b": router_b,
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

        Uses numpy graph-fn stages + sink-aware vanilla attention + swiglu_oai
        MoE dispatch. Hardcodes tp_degree=1/ep_degree=1: rejects sharded
        weight containers because the vocab-parallel embed, EP expert shards,
        and TP KV shards would silently yield rank-local partial output.
        """
        w = self._weights
        self._require_single_rank_for_forward_cpu(w.tp_degree, w.ep_degree)
        vs = self._vocab_shard
        hidden = vocab_parallel_embedding_local_fn(
            input_ids.astype(np.int32, copy=False),
            self._shared_np["embeddings"],
            vocab_start_index=int(vs.vocab_start_index),
            vocab_end_index=int(vs.vocab_end_index),
        )
        cos, sin = _build_rope_cache_for_positions_yarn(
            positions.astype(np.int32),
            head_dim=w.head_dim,
            theta=w.rope_theta,
            initial_context_length=w.yarn_original_max_pos,
            scaling_factor=w.yarn_factor,
            ntk_alpha=w.yarn_beta_slow,
            ntk_beta=w.yarn_beta_fast,
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
                lt["b_q"],
                lt["b_k"],
                lt["b_v"],
                cos,
                sin,
                num_heads=w.num_heads,
                num_kv_heads=w.num_kv_heads,
                head_dim=w.head_dim,
                rms_norm_eps=float(w.rms_norm_eps),
            )
            context = cpu_attn_with_sink_fn(
                q,
                k,
                v,
                kv_caches[layer_idx],
                lt["sink"],
                attn_metadata,
            )
            attn_out = (
                context.reshape(context.shape[0], w.num_heads * w.head_dim).astype(
                    w.dtype
                )
                @ lt["w_o"]
                + lt["b_o"]
            ).astype(w.dtype)
            hidden = (hidden + attn_out).astype(w.dtype)

            # MoE: norm -> router (bias) -> softmax(topk) -> swigluoai dispatch -> residual add
            normed = _apply_rms_norm(
                hidden, lt["post_attn_norm"], eps=float(w.rms_norm_eps)
            )
            logits = (normed.astype(w.dtype) @ lt["router_w"] + lt["router_b"]).astype(
                np.float32
            )
            affinities = softmax_topk_masked(logits, int(w.experts_per_token))
            if int(w.local_num_experts) < int(w.num_experts):
                e0 = int(w.ep_rank) * int(w.local_num_experts)
                affinities = affinities[:, e0 : e0 + int(w.local_num_experts)]
            moe_out = cpu_moe_dispatch_swiglu_oai(
                normed,
                affinities,
                lt["gup_w"],
                lt["gup_bias"],
                lt["down_w"],
                lt["down_bias_bc"],
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

        # YaRN RoPE on host (arg order matches production: slow then fast).
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

        # Vocab-parallel embedding
        vs = self._vocab_shard
        hidden = self._embed(
            input_ids.astype(np.int32, copy=False),
            self._shared_tensors["embeddings"],
            vocab_start_index=int(vs.vocab_start_index),
            vocab_end_index=int(vs.vocab_end_index),
            tp_degree=int(w.tp_degree),
            tp_replica_groups=self._tp_replica_groups,
        )

        # NKI step inputs
        step_inputs = None
        if self.attn.device:
            step_inputs = self._prepare_nki_step_inputs(
                token_bucket=effective_bucket,
                real_total_tokens=effective_real_tt,
                attn_metadata=attn_metadata,
            )

        # Layer loop
        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_tensors[layer_idx]
            kv_dev = self._kv_cache_dev[layer_idx]

            q, k, v = self._pre_attn(
                hidden,
                lt["input_norm"],
                lt["w_q"],
                lt["w_k"],
                lt["w_v"],
                lt["b_q"],
                lt["b_k"],
                lt["b_v"],
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
                    lt["sink"],
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
                context = self._run_cpu_attn_with_sink(
                    self.attn,
                    q,
                    k,
                    v,
                    kv_dev,
                    lt["sink"],
                    attn_metadata,
                    effective_real_tt,
                    effective_bucket,
                )

            hidden = self._post_attn(
                hidden,
                context,
                lt["w_o"],
                lt["b_o"],
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
                    lt["router_b"],
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
                    lt["router_b"],
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
                    prefix="eager_gptoss",
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
