"""Qwen3DenseEagerExecutor: stage-level composable execution via Fragment.

Each transformer layer is decomposed into three Fragment calls:
  pre_attn → attn → post_attn

Attention is the swappable stage: set ``self.attn`` to a CPU fragment
(e.g. ``jit(cpu_attn_fn, device=False)``) to run vanilla attention while
keeping pre_attn/post_attn/lm_head on device. The forward() method
dispatches based on ``self.attn.device`` and handles the device↔CPU
boundary (download, slice, re-pad, writeback).

Embed, pre_attn, post_attn, and lm_head always run on device.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import AttentionMetadata
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
from nkipy_serving.models.qwen3_dense.config import Qwen3DenseModelConfig
from nkipy_serving.models.qwen3_dense.graph_fns import (
    cpu_attn_fn,
    embedding_fn,
    nki_attn_fn,
    post_attn_fn,
    pre_attn_fn,
)
from nkipy_serving.models.qwen3_dense.weights import (
    Qwen3DenseWeights,
    init_qwen3_dense_weights,
)
from nkipy_serving.ops.nn import (
    apply_rms_norm as _apply_rms_norm,
)
from nkipy_serving.ops.nn import (
    build_rope_cache_for_positions as _build_rope_cache_for_positions,
)
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessor


class Qwen3DenseEagerExecutor(EagerExecutorBase):
    """Stage-level composable executor for Qwen3 Dense.

    ``self.attn`` is the swappable debug point — set it to
    ``jit(cpu_attn_fn, device=False)`` to run vanilla attention on CPU
    while all other stages stay on device.

    Embed, pre_attn, post_attn, and lm_head always run on device.
    """

    def __init__(self, model_config: Qwen3DenseModelConfig, kv_pool, runtime_config):
        self._model_config = model_config
        self._weights = init_qwen3_dense_weights(model_config)
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config

        from nkipy_serving.runtime.precompile_paddings import (
            build_precompile_paddings,
        )

        self._precompile_paddings = build_precompile_paddings(runtime_config)
        self._max_requests_per_step = int(
            self._precompile_paddings.max_padded_batch_size
        )
        self._dense_local_topk = int(runtime_config.dense_local_topk)
        self._compiler_args = runtime_config.nkipy_compiler_args
        self._build_dir = (
            f"{runtime_config.config_build_dir()}/rank{self._weights.tp_rank}"
        )

        # Upload weights to device
        DeviceTensor = _get_device_tensor_cls()
        w = self._weights
        self._shared_tensors = {
            "embeddings": DeviceTensor.from_numpy(w.embeddings, name="embeddings"),
            "final_norm": DeviceTensor.from_numpy(w.final_norm, name="final_norm"),
            "lm_head": DeviceTensor.from_numpy(w.lm_head, name="lm_head"),
        }
        self._layer_tensors = [
            self._upload_layer_weights(i) for i in range(w.num_hidden_layers)
        ]

        # KV caches
        ensure_nki_bridge()

        self._nki_num_blocks = kv_pool.num_blocks + 1
        self._kv_cache_dev = _allocate_device_kv_cache(
            num_hidden_layers=w.num_hidden_layers,
            num_kv_heads=w.num_kv_heads,
            head_dim=w.head_dim,
            block_size=kv_pool.block_size,
            num_blocks=self._nki_num_blocks,
            dtype=ml_dtypes.bfloat16,
        )
        self._kv_cache_zeros = _pre_allocate_kv_cache_zeros(
            num_blocks=self._nki_num_blocks,
            num_kv_heads=w.num_kv_heads,
            block_size=kv_pool.block_size,
            head_dim=w.head_dim,
            dtype=ml_dtypes.bfloat16,
        )

        # Base class init
        self._init_nki_step_inputs_cache()
        self._init_lm_head_scratch(self._max_requests_per_step, prefix="eager_dense")

        # LogitsProcessor
        self._logits_processor = LogitsProcessor(
            vocab_size=int(w.vocab_size),
            local_vocab_size=int(w.local_vocab_size),
            vocab_offset=int(w.lm_head_vocab_offset),
            hidden_size=int(w.hidden_size),
            dtype=w.embeddings.dtype,
            tp_degree=int(w.tp_degree),
            tp_rank=int(w.tp_rank),
            tp_replica_groups=(),
            rms_norm_eps=float(w.rms_norm_eps),
            dense_local_topk=self._dense_local_topk,
            gather_hidden=False,
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=self._max_requests_per_step,
        )

        # Fragments
        tp_cc = w.tp_degree > 1
        _cc_kw = dict(
            cc_enabled=True if tp_cc else None,
            rank_id=w.tp_rank if tp_cc else None,
            world_size=w.tp_degree if tp_cc else None,
        )

        self._embed = jit(
            embedding_fn,
            name="eager_embed",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
        )
        self._pre_attn = jit(
            pre_attn_fn,
            name="eager_pre_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
            **_cc_kw,
        )
        self.attn = jit(
            nki_attn_fn,
            name="eager_attn",
            build_dir=self._build_dir,
            additional_compiler_args=_join_compiler_args(
                self._compiler_args,
                NKI_COMPILER_ARGS,
            ),
            **_cc_kw,
        )
        self._post_attn = jit(
            post_attn_fn,
            name="eager_post_attn",
            build_dir=self._build_dir,
            additional_compiler_args=self._compiler_args,
            **_cc_kw,
        )

    # -- Properties ---------------------------------------------------------------

    @property
    def weights(self) -> Qwen3DenseWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- Weight upload ---------------------------------------------------------------

    def _upload_layer_weights(self, layer_idx: int) -> dict[str, object]:
        DeviceTensor = _get_device_tensor_cls()
        layer = self._weights.layers[layer_idx]
        return {
            "input_norm": DeviceTensor.from_numpy(
                layer.input_norm, name=f"input_norm_L{layer_idx}"
            ),
            "w_q": DeviceTensor.from_numpy(layer.w_q, name=f"w_q_L{layer_idx}"),
            "w_k": DeviceTensor.from_numpy(layer.w_k, name=f"w_k_L{layer_idx}"),
            "w_v": DeviceTensor.from_numpy(layer.w_v, name=f"w_v_L{layer_idx}"),
            "q_norm": DeviceTensor.from_numpy(
                layer.q_norm, name=f"q_norm_L{layer_idx}"
            ),
            "k_norm": DeviceTensor.from_numpy(
                layer.k_norm, name=f"k_norm_L{layer_idx}"
            ),
            "w_o": DeviceTensor.from_numpy(layer.w_o, name=f"w_o_L{layer_idx}"),
            "post_attn_norm": DeviceTensor.from_numpy(
                layer.post_attn_norm, name=f"post_attn_norm_L{layer_idx}"
            ),
            "w_gate": DeviceTensor.from_numpy(
                layer.w_gate, name=f"w_gate_L{layer_idx}"
            ),
            "w_up": DeviceTensor.from_numpy(layer.w_up, name=f"w_up_L{layer_idx}"),
            "w_down": DeviceTensor.from_numpy(
                layer.w_down, name=f"w_down_L{layer_idx}"
            ),
        }

    # -- CPU forward ---------------------------------------------------------------

    def forward_cpu(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_caches: list[np.ndarray],
        attn_metadata: AttentionMetadata,
    ) -> np.ndarray:
        """TP=1 CPU reference forward. Returns logits [total_tokens, local_vocab_size].

        Uses graph_fns stages with vanilla attention. Hardcodes tp_degree=1:
        rejects sharded weight containers because it would silently return
        rank-local partial output. On TP=1, local_vocab_size == vocab_size.
        """
        w = self._weights
        self._require_single_rank_for_forward_cpu(w.tp_degree)
        hidden = embedding_fn(input_ids, w.embeddings)
        cos, sin = _build_rope_cache_for_positions(
            positions.astype(np.float32),
            w.head_dim,
            w.rope_theta,
            dtype=w.embeddings.dtype,
        )
        for layer_idx, layer in enumerate(w.layers):
            q, k, v = pre_attn_fn(
                hidden,
                layer.input_norm,
                layer.w_q,
                layer.w_k,
                layer.w_v,
                layer.q_norm,
                layer.k_norm,
                cos,
                sin,
                num_heads=w.num_heads,
                num_kv_heads=w.num_kv_heads,
                head_dim=w.head_dim,
                rms_norm_eps=w.rms_norm_eps,
            )
            context = cpu_attn_fn(q, k, v, kv_caches[layer_idx], attn_metadata)
            hidden = post_attn_fn(
                hidden,
                context,
                layer.w_o,
                layer.post_attn_norm,
                layer.w_gate,
                layer.w_up,
                layer.w_down,
                num_heads=w.num_heads,
                head_dim=w.head_dim,
                rms_norm_eps=w.rms_norm_eps,
                tp_degree=1,
            )
        hidden = _apply_rms_norm(hidden, w.final_norm, eps=w.rms_norm_eps)
        return (hidden.astype(w.lm_head.dtype) @ w.lm_head.T).astype(np.float32)

    # -- Device forward -------------------------------------------------------------

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
        effective_real_tt = (
            real_total_tokens if real_total_tokens > 0 else input_ids.shape[0]
        )
        effective_bucket = token_bucket if token_bucket > 0 else input_ids.shape[0]
        w = self._weights
        dtype = w.embeddings.dtype

        cos_np, sin_np = _build_rope_cache_for_positions(
            positions.astype(np.float32),
            w.head_dim,
            w.rope_theta,
            dtype=dtype,
        )

        hidden = self._embed(
            input_ids.astype(np.int32, copy=False),
            self._shared_tensors["embeddings"],
        )

        step_inputs = None
        if self.attn.device:
            step_inputs = self._prepare_nki_step_inputs(
                token_bucket=effective_bucket,
                real_total_tokens=effective_real_tt,
                attn_metadata=attn_metadata,
            )

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
                lt["post_attn_norm"],
                lt["w_gate"],
                lt["w_up"],
                lt["w_down"],
                num_heads=int(w.num_heads),
                head_dim=int(w.head_dim),
                rms_norm_eps=float(w.rms_norm_eps),
                tp_degree=int(w.tp_degree),
            )

        return self._run_lm_head(
            hidden,
            attn_metadata,
            token_bucket=effective_bucket,
            sampling_batch=sampling_batch,
        )
