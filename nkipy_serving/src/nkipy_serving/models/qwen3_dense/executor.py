"""Qwen3DenseExecutor: owns weights, device tensors, compiled kernels, and forward pass."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import (
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
from nkipy_serving.models.qwen3_dense.codegen import (
    generate_full_kernel_source as _generate_full_kernel_source,
)
from nkipy_serving.models.qwen3_dense.graph_fns import (
    embedding_fn,
)
from nkipy_serving.models.qwen3_dense.weights import (
    Qwen3DenseLayerWeights,
    Qwen3DenseWeights,
    init_qwen3_dense_weights,
)
from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor as _overwrite_device_tensor,
)
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
# Compiled kernel bookkeeping
# ---------------------------------------------------------------------------


class Qwen3DenseExecutor:
    """Owns weights, device tensors, compiled kernels, and forward pass.

    Device execution compiles an embedding kernel + all-layers-one-graph kernel
    + LM head via LogitsProcessor.  Requires NKIBlockSparseFlashAttention.
    Also supports pure-numpy CPU forward for testing.
    """

    def __init__(self, model_config, kv_pool, runtime_config):
        self._model_config = model_config
        self._weights = init_qwen3_dense_weights(model_config)
        self._kv_pool = kv_pool
        self._runtime_config = runtime_config
        self._attention_backend = model_config.attention_backend
        self._full_kernels: dict[int, object] = {}
        self._dense_local_topk = int(runtime_config.dense_local_topk)
        if self._dense_local_topk > int(self._weights.lm_head.shape[0]):
            raise RuntimeError(
                "dense_local_topk exceeds local LM-head vocab shard size: "
                f"{self._dense_local_topk} > {int(self._weights.lm_head.shape[0])}"
            )

        from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

        try:
            _get_device_tensor_cls()
        except ImportError:
            raise RuntimeError(
                "nkipy runtime not available. "
                "Install nkipy to use execution_backend='nkipy'."
            )
        self._precompile_paddings = build_precompile_paddings(runtime_config)
        self._max_requests_per_step = int(
            self._precompile_paddings.max_padded_batch_size
        )
        self._compiler_args = runtime_config.nkipy_compiler_args
        self._build_dir = (
            f"{runtime_config.config_build_dir()}/rank{self._weights.tp_rank}"
        )
        self._step_scratch_by_bucket: dict[int, object] = {}
        self._nki_step_inputs_by_bucket: dict[int, PreparedNkiStepInputs] = {}
        self._shared_tensors = self._upload_shared_weights()
        self._layer_tensors = [
            self._upload_layer_weights(i)
            for i in range(self._weights.num_hidden_layers)
        ]

        ensure_nki_bridge()
        self._nki_num_blocks = self._kv_pool.num_blocks + 1
        self._kv_cache_dev = _allocate_device_kv_cache_shared(
            num_hidden_layers=self._weights.num_hidden_layers,
            num_kv_heads=self._weights.num_kv_heads,
            head_dim=self._weights.head_dim,
            block_size=self._kv_pool.block_size,
            num_blocks=self._nki_num_blocks,
            dtype=ml_dtypes.bfloat16,
        )
        self._kv_cache_zeros = _pre_allocate_kv_cache_zeros(
            num_blocks=self._nki_num_blocks,
            num_kv_heads=self._weights.num_kv_heads,
            block_size=self._kv_pool.block_size,
            head_dim=self._weights.head_dim,
            dtype=ml_dtypes.bfloat16,
        )

        w = self._weights
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
            gather_hidden=False,  # qwen3 dense: no seq-parallel gather
            nkipy_compiler_args=self._compiler_args,
            build_dir=self._build_dir,
            max_requests_per_step=self._max_requests_per_step,
        )

    @property
    def weights(self) -> Qwen3DenseWeights:
        return self._weights

    @property
    def kv_pool(self):
        return self._kv_pool

    # -- Device weight upload ---------------------------------------------------

    def _upload_shared_weights(self) -> dict[str, object]:
        w = self._weights
        return {
            "embeddings": _get_device_tensor_cls().from_numpy(
                w.embeddings, name="embeddings"
            ),
            "final_norm": _get_device_tensor_cls().from_numpy(
                w.final_norm, name="final_norm"
            ),
            "lm_head": _get_device_tensor_cls().from_numpy(w.lm_head, name="lm_head"),
        }

    def _upload_layer_weights(self, layer_idx: int) -> dict[str, object]:
        layer = self._weights.layers[layer_idx]
        return {
            "input_norm": _get_device_tensor_cls().from_numpy(
                layer.input_norm, name=f"input_norm_L{layer_idx}"
            ),
            "w_q": _get_device_tensor_cls().from_numpy(
                layer.w_q, name=f"w_q_L{layer_idx}"
            ),
            "w_k": _get_device_tensor_cls().from_numpy(
                layer.w_k, name=f"w_k_L{layer_idx}"
            ),
            "w_v": _get_device_tensor_cls().from_numpy(
                layer.w_v, name=f"w_v_L{layer_idx}"
            ),
            "q_norm": _get_device_tensor_cls().from_numpy(
                layer.q_norm, name=f"q_norm_L{layer_idx}"
            ),
            "k_norm": _get_device_tensor_cls().from_numpy(
                layer.k_norm, name=f"k_norm_L{layer_idx}"
            ),
            "w_o": _get_device_tensor_cls().from_numpy(
                layer.w_o, name=f"w_o_L{layer_idx}"
            ),
            "post_attn_norm": _get_device_tensor_cls().from_numpy(
                layer.post_attn_norm, name=f"post_attn_norm_L{layer_idx}"
            ),
            "w_gate": _get_device_tensor_cls().from_numpy(
                layer.w_gate, name=f"w_gate_L{layer_idx}"
            ),
            "w_up": _get_device_tensor_cls().from_numpy(
                layer.w_up, name=f"w_up_L{layer_idx}"
            ),
            "w_down": _get_device_tensor_cls().from_numpy(
                layer.w_down, name=f"w_down_L{layer_idx}"
            ),
        }

    def _overwrite_shared_weights(self, weights: Qwen3DenseWeights) -> None:
        _overwrite_device_tensor(self._shared_tensors["embeddings"], weights.embeddings)
        _overwrite_device_tensor(self._shared_tensors["final_norm"], weights.final_norm)
        _overwrite_device_tensor(self._shared_tensors["lm_head"], weights.lm_head)

    def _overwrite_layer_weights(
        self,
        layer_idx: int,
        layer: Qwen3DenseLayerWeights,
    ) -> None:
        layer_tensors = self._layer_tensors[layer_idx]
        _overwrite_device_tensor(layer_tensors["input_norm"], layer.input_norm)
        _overwrite_device_tensor(layer_tensors["w_q"], layer.w_q)
        _overwrite_device_tensor(layer_tensors["w_k"], layer.w_k)
        _overwrite_device_tensor(layer_tensors["w_v"], layer.w_v)
        _overwrite_device_tensor(layer_tensors["q_norm"], layer.q_norm)
        _overwrite_device_tensor(layer_tensors["k_norm"], layer.k_norm)
        _overwrite_device_tensor(layer_tensors["w_o"], layer.w_o)
        _overwrite_device_tensor(layer_tensors["post_attn_norm"], layer.post_attn_norm)
        _overwrite_device_tensor(layer_tensors["w_gate"], layer.w_gate)
        _overwrite_device_tensor(layer_tensors["w_up"], layer.w_up)
        _overwrite_device_tensor(layer_tensors["w_down"], layer.w_down)

    @staticmethod
    def _validate_reload_compatibility(
        current: Qwen3DenseWeights,
        new: Qwen3DenseWeights,
    ) -> None:
        scalar_fields = (
            "num_heads",
            "num_kv_heads",
            "global_num_heads",
            "global_num_kv_heads",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "global_intermediate_size",
            "num_hidden_layers",
            "vocab_size",
            "lm_head_vocab_offset",
            "local_vocab_size",
            "tp_degree",
            "tp_rank",
        )
        for field_name in scalar_fields:
            if getattr(current, field_name) != getattr(new, field_name):
                raise RuntimeError(
                    "Reloaded Qwen3 dense weights are incompatible with the running "
                    f"executor: field {field_name} changed from "
                    f"{getattr(current, field_name)!r} to {getattr(new, field_name)!r}"
                )
        if current.embeddings.shape != new.embeddings.shape:
            raise RuntimeError(
                "Reloaded Qwen3 dense embeddings shape is incompatible: "
                f"{current.embeddings.shape} != {new.embeddings.shape}"
            )
        if current.final_norm.shape != new.final_norm.shape:
            raise RuntimeError(
                "Reloaded Qwen3 dense final_norm shape is incompatible: "
                f"{current.final_norm.shape} != {new.final_norm.shape}"
            )
        if current.lm_head.shape != new.lm_head.shape:
            raise RuntimeError(
                "Reloaded Qwen3 dense lm_head shape is incompatible: "
                f"{current.lm_head.shape} != {new.lm_head.shape}"
            )
        if len(current.layers) != len(new.layers):
            raise RuntimeError(
                "Reloaded Qwen3 dense layer count is incompatible: "
                f"{len(current.layers)} != {len(new.layers)}"
            )
        layer_fields = (
            "input_norm",
            "post_attn_norm",
            "w_q",
            "w_k",
            "w_v",
            "w_o",
            "q_norm",
            "k_norm",
            "w_gate",
            "w_up",
            "w_down",
        )
        for layer_idx, (current_layer, new_layer) in enumerate(
            zip(current.layers, new.layers, strict=True)
        ):
            for field_name in layer_fields:
                current_value = getattr(current_layer, field_name)
                new_value = getattr(new_layer, field_name)
                if current_value.shape != new_value.shape:
                    raise RuntimeError(
                        "Reloaded Qwen3 dense layer tensor is incompatible: "
                        f"layer={layer_idx} field={field_name} "
                        f"{current_value.shape} != {new_value.shape}"
                    )

    def reload_weights_from_disk(self, model_path: str) -> None:
        reload_config = replace(self._model_config, hf_model_id=str(model_path))
        new_weights = init_qwen3_dense_weights(reload_config)
        self._validate_reload_compatibility(self._weights, new_weights)
        self._overwrite_shared_weights(new_weights)
        for layer_idx, layer in enumerate(new_weights.layers):
            self._overwrite_layer_weights(layer_idx, layer)
        self._weights = new_weights
        self._model_config = reload_config

    def flush_cache(self) -> None:
        _flush_device_kv_cache(self._kv_cache_dev, self._kv_cache_zeros, self._kv_pool)

    @dataclass
    class _CompiledKernels:
        """Compiled full-model graph kernels for a specific token_bucket shape."""

        token_bucket: int
        full_kernel: object
        embed_kernel: object = None

    def _format_local_topk_output(
        self,
        topk_values: np.ndarray,
        topk_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        topk_values = np.asarray(topk_values, dtype=np.float32)
        topk_indices = np.asarray(topk_indices, dtype=np.int32)
        if self._dense_local_topk == 1:
            return {
                "top1_values": topk_values.reshape((-1,)),
                "top1_indices": topk_indices.reshape((-1,)),
                "vocab_offset": np.asarray(
                    [self._weights.lm_head_vocab_offset],
                    dtype=np.int32,
                ),
            }
        return {
            "topk_values": topk_values,
            "topk_indices": topk_indices,
            "vocab_offset": np.asarray(
                [self._weights.lm_head_vocab_offset],
                dtype=np.int32,
            ),
        }

    def _use_top1_fast_path(self) -> bool:
        return self._dense_local_topk == 1

    @dataclass
    class _BucketScratch:
        input_ids: object
        cos: object
        sin: object
        hidden_a: object
        hidden_b: object
        last_token_indices: object
        last_token_indices_host: np.ndarray
        top1_values: object | None
        top1_indices: object | None
        topk_values: object | None
        topk_indices: object | None

    @staticmethod
    def _fill_padded_last_token_indices(
        host_buffer: np.ndarray,
        attn_metadata: AttentionMetadata,
    ) -> int:
        bs = int(attn_metadata.batch_size)
        host_buffer.fill(0)
        if bs > 0:
            last_indices = (attn_metadata.query_start_loc[1 : bs + 1] - 1).astype(
                np.int32
            )
            host_buffer[:bs] = last_indices
        return bs

    @staticmethod
    def _write_common_step_inputs(
        step_scratch: _BucketScratch,
        *,
        input_ids: np.ndarray,
        cos_np: np.ndarray,
        sin_np: np.ndarray,
    ) -> dict[str, object]:
        _overwrite_device_tensor(step_scratch.input_ids, input_ids)
        _overwrite_device_tensor(step_scratch.cos, cos_np)
        _overwrite_device_tensor(step_scratch.sin, sin_np)
        return {
            "input_ids": step_scratch.input_ids,
            "cos": step_scratch.cos,
            "sin": step_scratch.sin,
        }

    def _prepare_last_token_indices(
        self,
        *,
        step_scratch: _BucketScratch,
        attn_metadata: AttentionMetadata,
    ) -> tuple[int, object]:
        bs = self._fill_padded_last_token_indices(
            step_scratch.last_token_indices_host,
            attn_metadata,
        )
        _overwrite_device_tensor(
            step_scratch.last_token_indices,
            step_scratch.last_token_indices_host,
        )
        return bs, step_scratch.last_token_indices

    def _ensure_step_scratch(self, token_bucket: int) -> _BucketScratch:
        scratch_by_bucket = getattr(self, "_step_scratch_by_bucket", None)
        if scratch_by_bucket is None:
            scratch_by_bucket = {}
            self._step_scratch_by_bucket = scratch_by_bucket
        cached = scratch_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached

        w = self._weights
        dtype = w.embeddings.dtype
        half_dim = int(w.head_dim) // 2
        max_requests_per_step = int(self._max_requests_per_step)
        dense_local_topk = int(self._dense_local_topk)

        scratch = Qwen3DenseExecutor._BucketScratch(
            input_ids=_alloc_device_scratch(
                (token_bucket,),
                np.int32,
                name=f"qwen3dense_input_ids_t{token_bucket}",
            ),
            cos=_alloc_device_scratch(
                (token_bucket, half_dim),
                dtype,
                name=f"qwen3dense_cos_t{token_bucket}",
            ),
            sin=_alloc_device_scratch(
                (token_bucket, half_dim),
                dtype,
                name=f"qwen3dense_sin_t{token_bucket}",
            ),
            hidden_a=_alloc_device_scratch(
                (token_bucket, int(w.hidden_size)),
                dtype,
                name=f"qwen3dense_hidden_a_t{token_bucket}",
            ),
            hidden_b=_alloc_device_scratch(
                (token_bucket, int(w.hidden_size)),
                dtype,
                name=f"qwen3dense_hidden_b_t{token_bucket}",
            ),
            last_token_indices=_alloc_device_scratch(
                (max_requests_per_step,),
                np.int32,
                name=f"qwen3dense_last_idx_t{token_bucket}",
            ),
            last_token_indices_host=np.zeros((max_requests_per_step,), dtype=np.int32),
            top1_values=(
                _alloc_device_scratch(
                    (max_requests_per_step,),
                    np.float32,
                    name=f"qwen3dense_top1_vals_t{token_bucket}",
                )
                if self._use_top1_fast_path()
                else None
            ),
            top1_indices=(
                _alloc_device_scratch(
                    (max_requests_per_step,),
                    np.int32,
                    name=f"qwen3dense_top1_idx_t{token_bucket}",
                )
                if self._use_top1_fast_path()
                else None
            ),
            topk_values=(
                _alloc_device_scratch(
                    (max_requests_per_step, dense_local_topk),
                    np.float32,
                    name=f"qwen3dense_topk_vals_t{token_bucket}",
                )
                if not self._use_top1_fast_path()
                else None
            ),
            topk_indices=(
                _alloc_device_scratch(
                    (max_requests_per_step, dense_local_topk),
                    np.int32,
                    name=f"qwen3dense_topk_idx_t{token_bucket}",
                )
                if not self._use_top1_fast_path()
                else None
            ),
        )
        scratch_by_bucket[int(token_bucket)] = scratch
        return scratch

    def _run_lm_head_via_logits_processor(
        self,
        hidden_dev: object,
        attn_metadata: AttentionMetadata,
        step_scratch: _BucketScratch,
        *,
        sampling_batch: DeviceSamplingBatch | None = None,
    ) -> dict[str, np.ndarray]:
        """Delegate LM-head + sampling to LogitsProcessor."""
        bs, last_dev = self._prepare_last_token_indices(
            step_scratch=step_scratch,
            attn_metadata=attn_metadata,
        )
        hidden_shape = getattr(hidden_dev, "shape", None)
        if hidden_shape is not None and len(hidden_shape) > 0:
            token_bucket = int(hidden_shape[0])
        else:
            token_bucket = (
                int(attn_metadata.total_tokens)
                if attn_metadata.forward_mode == FORWARD_MODE_EXTEND
                else int(attn_metadata.batch_size)
            )
        lp_output = self._logits_processor.forward(
            hidden_dev,
            self._shared_tensors["final_norm"],
            self._shared_tensors["lm_head"],
            last_dev,
            batch_size=bs,
            token_bucket=token_bucket,
            sampling_batch=sampling_batch,
            needs_logprobs=bool(sampling_batch.needs_logprobs)
            if sampling_batch
            else False,
            logprobs_k=int(sampling_batch.logprobs_k) if sampling_batch else 0,
        )
        return lp_output.to_shm_dict(
            vocab_offset=self._weights.lm_head_vocab_offset,
        )

    # -- Kernel compilation -----------------------------------------------------

    def _ensure_nki_step_inputs(self, token_bucket: int) -> PreparedNkiStepInputs:
        cached = self._nki_step_inputs_by_bucket.get(int(token_bucket))
        if cached is not None:
            return cached
        if self._precompile_paddings is None:
            raise RuntimeError(
                "Expected precompile paddings to be initialized for nkipy"
            )
        from nkipy_serving.attention.nki_blocksparse_flash_attention import (
            NKI_MIN_Q_SEQLEN,
        )

        attn_bucket = max(int(token_bucket), int(NKI_MIN_Q_SEQLEN))
        step_inputs = allocate_prepared_nki_step_inputs(
            _alloc_device_scratch,
            token_bucket=int(token_bucket),
            attn_bucket=int(attn_bucket),
            max_context_len=int(self._runtime_config.max_context_len),
            max_requests=int(self._precompile_paddings.max_padded_batch_size),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
            prefix="qwen3dense",
        )
        initialize_prepared_nki_step_inputs(step_inputs, _overwrite_device_tensor)
        self._nki_step_inputs_by_bucket[int(token_bucket)] = step_inputs
        return step_inputs

    def _get_full_kernel_fn(self, *, token_bucket: int, attn_bucket: int):
        w = self._weights
        mod_name = (
            f"qwen3_dense_full_tp{w.tp_degree}_t{int(token_bucket)}_a{int(attn_bucket)}"
        )
        fn_name = f"qwen3_dense_full_forward_t{int(token_bucket)}_tp{w.tp_degree}"
        return _load_generated_kernel_fn(
            build_dir=self._build_dir,
            mod_name=mod_name,
            fn_name=fn_name,
            source=_generate_full_kernel_source(
                token_bucket=token_bucket,
                attn_bucket=attn_bucket,
                num_hidden_layers=int(w.num_hidden_layers),
                num_heads=int(w.num_heads),
                num_kv_heads=int(w.num_kv_heads),
                head_dim=int(w.head_dim),
                rms_norm_eps=float(w.rms_norm_eps),
                tp_degree=int(w.tp_degree),
            ),
        )

    def _ensure_kernels(self, token_bucket: int):
        """Compile full-model graph kernel for the given token_bucket shape if not cached."""
        cached = self._full_kernels.get(token_bucket)
        if cached is not None:
            return cached

        if self._attention_backend != "NKIBlockSparseFlashAttention":
            raise RuntimeError(
                "Full graph mode requires NKIBlockSparseFlashAttention. "
                f"Got attention_backend={self._attention_backend}"
            )
        if self._precompile_paddings is None:
            raise RuntimeError(
                "Expected precompile paddings to be initialized for nkipy"
            )

        from nkipy_serving.attention.nki_blocksparse_flash_attention import (
            NKI_COMPILER_ARGS,
            NKI_MIN_Q_SEQLEN,
            compute_max_tile_counts,
        )

        w = self._weights
        dtype = w.embeddings.dtype
        hidden_size = w.hidden_size
        num_kv_heads = w.num_kv_heads
        head_dim = w.head_dim
        half_dim = head_dim // 2

        attn_bucket = max(int(token_bucket), int(NKI_MIN_Q_SEQLEN))
        max_p, max_d = compute_max_tile_counts(
            token_bucket=attn_bucket,
            max_context_len=self._runtime_config.max_context_len,
            max_requests=int(self._precompile_paddings.max_padded_batch_size),
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

        full_fn = self._get_full_kernel_fn(
            token_bucket=token_bucket, attn_bucket=attn_bucket
        )

        # Sample tensors for shape specialization.
        sample_hidden = np.zeros((token_bucket, hidden_size), dtype=dtype)
        sample_ids = np.zeros((token_bucket,), dtype=np.int32)
        sample_cos = np.zeros((token_bucket, half_dim), dtype=dtype)
        sample_sin = np.zeros((token_bucket, half_dim), dtype=dtype)
        sample_slot = np.zeros((token_bucket,), dtype=np.int32)

        # Use a single KV-cache sample array and reuse it for every layer arg.
        kv_blocks = self._kv_pool.num_blocks + 1
        sample_kv_cache = np.zeros(
            (2, kv_blocks, num_kv_heads, self._kv_pool.block_size, head_dim),
            dtype=ml_dtypes.bfloat16,
        )

        # Compile embed_kernel (embedding outside the all-layers graph).
        embed_kernel = _get_device_kernel_cls().compile_and_load(
            embedding_fn,
            sample_ids,
            w.embeddings,
            name=f"embedding_full_t{token_bucket}",
            additional_compiler_args=self._compiler_args,
            use_cached_if_exists=True,
            build_dir=self._build_dir,
        )

        # Compile all-layers kernel (takes hidden, not input_ids+embeddings).
        full_cc = w.tp_degree > 1
        sample_args: list[np.ndarray] = [
            sample_hidden,
            sample_cos,
            sample_sin,
            sample_slot,
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
        for layer in w.layers:
            sample_args.extend(
                [
                    sample_kv_cache,
                    layer.input_norm,
                    layer.w_q,
                    layer.w_k,
                    layer.w_v,
                    layer.q_norm,
                    layer.k_norm,
                    layer.w_o,
                    layer.post_attn_norm,
                    layer.w_gate,
                    layer.w_up,
                    layer.w_down,
                ]
            )

        full_kernel = _get_device_kernel_cls().compile_and_load(
            full_fn,
            *sample_args,
            name=(
                f"full_qwen3_tp{w.tp_degree}_t{token_bucket}"
                f"_bsmax{self._max_requests_per_step}"
            ),
            additional_compiler_args=_join_compiler_args(
                self._compiler_args, NKI_COMPILER_ARGS
            ),
            use_cached_if_exists=True,
            build_dir=self._build_dir,
            cc_enabled=full_cc,
            rank_id=w.tp_rank,
            world_size=w.tp_degree,
        )

        # LogitsProcessor owns LM-head sampling kernels.
        self._logits_processor._ensure_kernels(token_bucket)

        compiled = self._CompiledKernels(
            token_bucket=token_bucket,
            full_kernel=full_kernel,
            embed_kernel=embed_kernel,
        )
        self._full_kernels[token_bucket] = compiled
        return compiled

    def _prepare_nki_step_inputs(
        self,
        *,
        token_bucket: int,
        real_total_tokens: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[object, dict[str, object]]:
        """Build and upload per-step NKI inputs (slot mapping + unified tile plans)."""
        if self._attention_backend != "NKIBlockSparseFlashAttention":
            raise RuntimeError(
                "Expected NKIBlockSparseFlashAttention for fused graph modes. "
                f"Got attention_backend={self._attention_backend}"
            )
        step_inputs = self._ensure_nki_step_inputs(int(token_bucket))
        return int(step_inputs.attn_bucket), prepare_prepared_nki_step_inputs(
            step_inputs,
            _overwrite_device_tensor,
            attn_metadata=attn_metadata,
            real_total_tokens=int(real_total_tokens),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
        )

    # -- Forward dispatch -------------------------------------------------------

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
    ) -> object:
        effective_real_tt = (
            real_total_tokens if real_total_tokens > 0 else input_ids.shape[0]
        )
        effective_bucket = token_bucket if token_bucket > 0 else input_ids.shape[0]

        return self._forward_device_full(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            token_bucket=effective_bucket,
            real_total_tokens=effective_real_tt,
            sampling_batch=sampling_batch,
        )

    def _forward_device_full(
        self,
        *,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: AttentionMetadata,
        token_bucket: int,
        real_total_tokens: int,
        sampling_batch: DeviceSamplingBatch | None = None,
    ) -> dict[str, np.ndarray]:
        """All-layers-one-graph device forward: embed_kernel -> full_kernel -> LM head.

        Embedding runs as a separate kernel. All transformer layers run in one
        NEFF (full_kernel). LM head runs via LogitsProcessor.
        """
        w = self._weights
        kernels = self._ensure_kernels(token_bucket)
        step_scratch = self._ensure_step_scratch(token_bucket)
        head_dim = w.head_dim
        dtype = w.embeddings.dtype

        # --- RoPE cache on host, upload to device ---
        cos_np, sin_np = _build_rope_cache_for_positions(
            positions.astype(np.float32),
            head_dim,
            w.rope_theta,
            dtype=dtype,
        )
        common_step_inputs = self._write_common_step_inputs(
            step_scratch,
            input_ids=input_ids.astype(np.int32, copy=False),
            cos_np=cos_np,
            sin_np=sin_np,
        )

        # --- Embedding on device (outside the all-layers graph) ---
        hidden_dev = step_scratch.hidden_a
        kernels.embed_kernel(
            inputs={
                "input_ids": common_step_inputs["input_ids"],
                "embeddings": self._shared_tensors["embeddings"],
            },
            outputs={"output0": hidden_dev},
        )

        # --- Per-step shared inputs for NKI attention (slot mapping + tile plans) ---
        _, step_inputs = self._prepare_nki_step_inputs(
            token_bucket=token_bucket,
            real_total_tokens=real_total_tokens,
            attn_metadata=attn_metadata,
        )

        inputs: dict[str, object] = {
            "hidden": hidden_dev,
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
        }
        for layer_idx in range(w.num_hidden_layers):
            lt = self._layer_tensors[layer_idx]
            inputs.update(
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
                    f"w_gate_L{layer_idx}": lt["w_gate"],
                    f"w_up_L{layer_idx}": lt["w_up"],
                    f"w_down_L{layer_idx}": lt["w_down"],
                }
            )

        # Run all transformer layers -- outputs hidden states.
        # KV caches are updated in-place by nki_update_kv_cache_core;
        # the compiler marks them as aliased outputs that must appear in
        # the output set for the runtime to bind them correctly.
        hidden_out_dev = step_scratch.hidden_b
        outputs: dict[str, object] = {"output0": hidden_out_dev}
        for layer_idx in range(w.num_hidden_layers):
            outputs[f"kv_cache_L{layer_idx}"] = self._kv_cache_dev[layer_idx]
        kernels.full_kernel(
            inputs=inputs,
            outputs=outputs,
        )

        # Run the separate LM-head kernel (topk / filtered / unfiltered).
        return self._run_lm_head_via_logits_processor(
            hidden_out_dev,
            attn_metadata,
            step_scratch=step_scratch,
            sampling_batch=sampling_batch,
        )

    # -- Warmup -----------------------------------------------------------------

    def warmup(self, paddings) -> None:
        """Precompile embed + all-layers kernel for every configured bucket size."""
        all_buckets = sorted(set(paddings.token_paddings) | set(paddings.bs_paddings))
        w = self._weights

        for bucket in all_buckets:
            self._ensure_kernels(bucket)

        run_synthetic_warmup_steps(
            build_standard_warmup_steps(paddings),
            token_paddings=tuple(int(bucket) for bucket in paddings.token_paddings),
            bs_paddings=tuple(int(bucket) for bucket in paddings.bs_paddings),
            num_blocks=int(self._nki_num_blocks),
            block_size=int(self._kv_pool.block_size),
            num_kv_heads=int(w.num_kv_heads),
            head_dim=int(w.head_dim),
            forward=self.forward,
        )
