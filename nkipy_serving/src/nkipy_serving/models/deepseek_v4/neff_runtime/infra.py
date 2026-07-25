"""Miscellaneous product-executor mixins for DeepSeek-V4.

Merged from the former alias / head / embedding / frequency / metadata /
stage_profile modules — each was a single small ``Dsv4Product*Mixin`` used only
by the product executor. Bodies are byte-identical to their pre-merge form.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.diagnostics import (
    rank_trace_allowed,
    stage_profile_enabled,
)
from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _compile_product_kernel,
    _run_product_kernel,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import common as graph_common
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _as_product_device_input,
    _host_array_signature,
    _is_device_value,
    _product_executor_coord,
    _require_product_device_value,
    _sample_array,
    _value_dtype,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _product_canonical_neff_cache_key,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
    _TensorSpec,
)
from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor_if_changed as _overwrite_device_tensor_if_changed,
)
from nkipy_serving.profiling import ProfileWriter
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)


# --- from alias.py ---
def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


class Dsv4ProductAliasMixin:
    def _product_alias_registry(self) -> dict[int, Any]:
        registry = getattr(self, "_product_active_alias_full_values", None)
        if registry is None:
            registry = {}
            self._product_active_alias_full_values = registry
        return registry

    def _product_alias_ref_registry(self) -> dict[tuple[int, tuple[int, ...]], Any]:
        registry = getattr(self, "_product_active_alias_full_values_by_ref", None)
        if registry is None:
            registry = {}
            self._product_active_alias_full_values_by_ref = registry
        return registry

    def _product_active_alias(
        self,
        value: Any,
        active_shape: tuple[int, ...],
    ) -> Any:
        alias = _alias_device_value_shape(value, active_shape)
        if alias is None:
            return value
        if alias is not value:
            self._product_alias_registry()[id(alias)] = value
            tensor_ref = getattr(alias, "tensor_ref", None)
            if tensor_ref is not None:
                full_shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
                self._product_alias_ref_registry()[(id(tensor_ref), full_shape)] = value
        return alias

    def _product_full_value_for(
        self,
        value: Any,
        full_shape: tuple[int, ...],
    ) -> Any | None:
        full_shape_t = tuple(int(dim) for dim in full_shape)
        if tuple(int(dim) for dim in getattr(value, "shape", ())) == full_shape_t:
            return value

        def _matching_full_or_alias(full: Any | None) -> Any | None:
            if full is None:
                return None
            current_shape = tuple(int(dim) for dim in getattr(full, "shape", ()))
            if current_shape == full_shape_t:
                return full
            if _shape_numel(current_shape) < _shape_numel(full_shape_t):
                return None
            alias = _alias_device_value_shape(full, full_shape_t)
            if alias is None:
                return None
            self._product_alias_registry()[id(alias)] = full
            tensor_ref = getattr(alias, "tensor_ref", None)
            if tensor_ref is not None:
                self._product_alias_ref_registry()[(id(tensor_ref), full_shape_t)] = (
                    alias
                )
            return alias

        full = self._product_alias_registry().get(id(value))
        matched = _matching_full_or_alias(full)
        if matched is not None:
            return matched
        tensor_ref = getattr(value, "tensor_ref", None)
        if tensor_ref is not None:
            full = self._product_alias_ref_registry().get(
                (id(tensor_ref), full_shape_t)
            )
            matched = _matching_full_or_alias(full)
            if matched is not None:
                return matched
            for (ref_id, _shape), candidate in list(
                self._product_alias_ref_registry().items()
            ):
                if int(ref_id) != id(tensor_ref):
                    continue
                matched = _matching_full_or_alias(candidate)
                if matched is not None:
                    return matched
        return None

    def _product_promote_mhc_state_shape(
        self,
        residual: Any,
        post: Any,
        comb: Any,
        *,
        compile_bsz: int,
        compile_seqlen: int,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        hc_mult: int,
    ) -> tuple[int, int, Any, Any, Any]:
        compile_bsz_i = int(compile_bsz)
        compile_seqlen_i = int(compile_seqlen)
        bsz_i = int(bsz)
        seqlen_i = int(seqlen)
        hidden_i = int(hidden_size)
        hc_i = int(hc_mult)
        if compile_bsz_i == bsz_i and compile_seqlen_i == seqlen_i:
            return compile_bsz_i, compile_seqlen_i, residual, post, comb

        residual_full = self._product_full_value_for(
            residual,
            (compile_bsz_i, compile_seqlen_i, hc_i, hidden_i),
        )
        post_full = self._product_full_value_for(
            post,
            (compile_bsz_i, compile_seqlen_i, hc_i),
        )
        comb_full = self._product_full_value_for(
            comb,
            (compile_bsz_i, compile_seqlen_i, hc_i, hc_i),
        )
        if residual_full is None or post_full is None or comb_full is None:
            return bsz_i, seqlen_i, residual, post, comb
        return compile_bsz_i, compile_seqlen_i, residual_full, post_full, comb_full

    def _product_promote_mhc_state_batch(
        self,
        residual: Any,
        post: Any,
        comb: Any,
        *,
        compile_bsz: int,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        hc_mult: int,
    ) -> tuple[int, Any, Any, Any]:
        compile_bsz_i, _compile_seqlen_i, residual_out, post_out, comb_out = (
            self._product_promote_mhc_state_shape(
                residual,
                post,
                comb,
                compile_bsz=int(compile_bsz),
                compile_seqlen=int(seqlen),
                bsz=int(bsz),
                seqlen=int(seqlen),
                hidden_size=int(hidden_size),
                hc_mult=int(hc_mult),
            )
        )
        return compile_bsz_i, residual_out, post_out, comb_out


# --- from head.py ---
class Dsv4ProductHeadMixin:
    def _can_use_fused_head_top1(self) -> bool:
        logits_processor = getattr(self, "logits_processor", None)
        use_top1 = getattr(logits_processor, "_use_top1_fast_path", None)
        if not callable(use_top1) or not bool(use_top1()):
            return False
        if bool(getattr(logits_processor, "_gather_hidden", False)):
            return False
        return (
            getattr(self, "final_norm_dev", None) is not None
            and getattr(self, "lm_head_dev", None) is not None
        )

    def precompile_logits_processor_bucket(self, token_bucket: int) -> None:
        """Compile logits/sampler/logprob NEFFs during warmup."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        logits_processor = getattr(self, "logits_processor", None)
        ensure_kernels = getattr(logits_processor, "_ensure_kernels", None)
        if callable(ensure_kernels):
            ensure_kernels(
                int(runtime_token_bucket),
                include_sampler=True,
                include_logprobs=True,
                deferred_sampler_load=True,
            )

    def seal_logits_processor_precompiled_kernels(self) -> None:
        logits_processor = getattr(self, "logits_processor", None)
        seal = getattr(logits_processor, "seal_precompiled_kernels", None)
        if callable(seal):
            seal()


# --- from embedding.py ---
class Dsv4ProductEmbeddingMixin:
    def _embedding_hc_mhc_pre_from_ids_kernel_for(
        self,
        bucket: Dsv4ProductBucket,
        input_ids: Any,
        embeddings: Any,
        vocab_range: Any,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        *,
        bsz: int,
        seqlen: int,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        tp_degree: int,
        tp_replica_groups: tuple,
    ) -> Any:
        ids_shape = tuple(int(dim) for dim in getattr(input_ids, "shape", ()))
        range_shape = tuple(int(dim) for dim in getattr(vocab_range, "shape", ()))
        groups = tuple(
            tuple(int(rank) for rank in group) for group in tp_replica_groups
        )
        if int(tp_degree) <= 1:
            raise RuntimeError(
                "DSV4 product first-layer embedding requires TP-sharded "
                "vocab-parallel weights"
            )
        key = (
            ids_shape,
            str(getattr(input_ids, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(embeddings, "shape", ())),
            str(getattr(embeddings, "dtype", "unknown")),
            range_shape,
            str(getattr(vocab_range, "dtype", "unknown")),
            tuple(int(dim) for dim in getattr(hc_fn, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_scale, "shape", ())),
            tuple(int(dim) for dim in getattr(hc_base, "shape", ())),
            tuple(int(dim) for dim in getattr(norm_weight, "shape", ())),
            int(bsz),
            int(seqlen),
            int(hc_mult),
            int(sinkhorn_iters),
            float(norm_eps),
            float(hc_eps),
            int(tp_degree),
            groups,
        )
        cached = bucket.kernel_caches["embedding_hc_mhc_pre_from_ids_kernels"].get(key)
        if cached is not None:
            return cached
        group_tag = hashlib.sha1(repr(groups).encode("utf-8")).hexdigest()[:8]
        rank_id, world_size = self._collective_graph_metadata(
            "vocab_parallel_embedding_hc",
            where="vocab embedding/mHC pre",
        )
        compile_kwargs: dict[str, Any] = {
            "cc_enabled": True,
            "rank_id": int(rank_id),
            "world_size": int(world_size),
            "is_spmd": False,
            "load_barrier_name": (
                "dsv4_product_vocab_embedding_mhc_pre_"
                f"t{int(bucket.token_bucket)}_"
                f"ids{'x'.join(str(v) for v in ids_shape)}_"
                f"b{int(bsz)}_s{int(seqlen)}_tp{int(tp_degree)}"
            ),
            "canonical_neff_cache_key": _product_canonical_neff_cache_key(
                "dsv4_product_vocab_embedding_mhc_pre",
                "v1",
                key,
            ),
        }
        name = (
            "dsv4_product_vocab_embedding_mhc_pre_"
            f"t{int(bucket.token_bucket)}_"
            f"ids{'x'.join(str(v) for v in ids_shape)}_"
            f"b{int(bsz)}_s{int(seqlen)}_"
            f"tp{int(tp_degree)}_hc{int(hc_mult)}_{group_tag}"
        )
        return self._cached_product_kernel(
            bucket=bucket,
            cache_name="embedding_hc_mhc_pre_from_ids_kernels",
            key=key,
            compile_kernel=lambda: _compile_product_kernel(
                graph_common.vocab_parallel_embedding_hc_mhc_pre_from_ids_dynamic_range_fn,
                _sample_array(input_ids, fallback_dtype=np.int32),
                _sample_array(embeddings, fallback_dtype=ml_dtypes.bfloat16),
                _sample_array(vocab_range, fallback_dtype=np.int32),
                _sample_array(hc_fn, fallback_dtype=np.float32),
                _sample_array(hc_scale, fallback_dtype=np.float32),
                _sample_array(hc_base, fallback_dtype=np.float32),
                _sample_array(norm_weight, fallback_dtype=np.float32),
                bsz=int(bsz),
                seqlen=int(seqlen),
                tp_degree=int(tp_degree),
                tp_replica_groups=groups,
                hc_mult=int(hc_mult),
                sinkhorn_iters=int(sinkhorn_iters),
                norm_eps=float(norm_eps),
                hc_eps=float(hc_eps),
                name=name,
                additional_compiler_args=getattr(self, "compiler_args", ""),
                build_dir=self.build_dir,
                load=False,
                **compile_kwargs,
            ),
        )

    def precompile_first_layer_embedding_mhc_shapes(
        self,
        token_bucket: int,
        *,
        batch_size: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> None:
        """Precompile first-layer active embedding unpad fused with mHC pre."""
        runtime_token_bucket = self._require_runtime_token_bucket(int(token_bucket))
        bucket = self._ensure_product_bucket(runtime_token_bucket)
        bsz = int(batch_size)
        seq = int(seqlen)
        if bsz <= 0 or seq <= 0:
            return
        full_spec = self._embedding_full_spec_for_bucket(bucket)
        full_shape = tuple(int(dim) for dim in full_spec.shape)
        if bsz > full_shape[0] or bsz * seq > int(bucket.token_bucket):
            raise RuntimeError(
                "DSV4 product first-layer embedding mHC precompile shape exceeds "
                f"bucket: batch={bsz}, seqlen={seq}, "
                f"token_bucket={int(bucket.token_bucket)}, full_shape={full_shape}"
            )
        compile_bsz, compile_seqlen = self._product_compile_embedding_shape(
            bucket,
            bsz=bsz,
            seqlen=seq,
            bucket_single_token=not bool(is_decode),
        )
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()) or ())
        if not blocks:
            return
        block = blocks[0]
        args = self.runtime_surface.args
        embeddings_dev = _as_product_device_input(
            self.runtime_surface.w.embed,
            name="dsv4_product_embeddings",
        )
        input_ids_spec = _TensorSpec(
            tuple(int(dim) for dim in getattr(bucket.input_ids_dev, "shape", ())),
            np.dtype(np.int32),
        )
        vocab_range_spec = _TensorSpec((2,), np.dtype(np.int32))
        kernel = self._embedding_hc_mhc_pre_from_ids_kernel_for(
            bucket,
            input_ids_spec,
            embeddings_dev,
            vocab_range_spec,
            block.hc_attn_fn,
            block.hc_attn_scale,
            block.hc_attn_base,
            block.attn_norm,
            bsz=compile_bsz,
            seqlen=compile_seqlen,
            hc_mult=int(args.hc_mult),
            sinkhorn_iters=int(args.hc_sinkhorn_iters),
            norm_eps=float(args.norm_eps),
            hc_eps=float(args.hc_eps),
            tp_degree=self.embed_tp_degree,
            tp_replica_groups=self.embed_tp_replica_groups,
        )
        if self._keep_dp_attention_pipeline_collectives_loaded():
            self._load_resident_product_kernel(kernel)

    def _run_product_embedding_mhc_pre_from_ids(
        self,
        bucket: Dsv4ProductBucket,
        hc_fn: Any,
        hc_scale: Any,
        hc_base: Any,
        norm_weight: Any,
        *,
        bsz: int,
        seqlen: int,
        is_decode: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        executor = self.runtime_surface
        embeddings_dev = _as_product_device_input(
            executor.w.embed,
            name="dsv4_product_embeddings",
        )
        vocab_range = self._sync_embedding_vocab_range(bucket)
        for value, where in (
            (bucket.input_ids_dev, "embedding_mhc_pre_from_ids/input_ids"),
            (embeddings_dev, "embedding_mhc_pre_from_ids/embeddings"),
            (vocab_range, "embedding_mhc_pre_from_ids/vocab_range"),
            (hc_fn, "embedding_mhc_pre_from_ids/hc_fn"),
            (hc_scale, "embedding_mhc_pre_from_ids/hc_scale"),
            (hc_base, "embedding_mhc_pre_from_ids/hc_base"),
            (norm_weight, "embedding_mhc_pre_from_ids/norm_weight"),
        ):
            _require_product_device_value(value, where=where)
        compile_bsz, compile_seqlen = self._product_compile_embedding_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            bucket_single_token=not bool(is_decode),
        )
        kernel = self._embedding_hc_mhc_pre_from_ids_kernel_for(
            bucket,
            bucket.input_ids_dev,
            embeddings_dev,
            vocab_range,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            bsz=int(compile_bsz),
            seqlen=int(compile_seqlen),
            hc_mult=int(executor.args.hc_mult),
            sinkhorn_iters=int(executor.args.hc_sinkhorn_iters),
            norm_eps=float(executor.args.norm_eps),
            hc_eps=float(executor.args.hc_eps),
            tp_degree=self.embed_tp_degree,
            tp_replica_groups=self.embed_tp_replica_groups,
        )
        hc_mult = int(executor.args.hc_mult)
        embed_shape = tuple(int(dim) for dim in getattr(embeddings_dev, "shape", ()))
        hidden_size = int(
            getattr(executor.model_config, "hidden_size", 0)
            or (embed_shape[-1] if embed_shape else 0)
        )
        if hidden_size <= 0:
            raise RuntimeError("DSV4 product embedding/mHC pre hidden size is unknown")
        dtype = _value_dtype(embeddings_dev, fallback=ml_dtypes.bfloat16)
        outputs = {
            "output0": self._bucket_scratch(
                bucket,
                "embedding_hc_active",
                (int(compile_bsz), int(compile_seqlen), hc_mult, hidden_size),
                dtype,
            ),
            "output1": self._bucket_scratch(
                bucket,
                "embedding_mhc_pre_y",
                (int(compile_bsz), int(compile_seqlen), hidden_size),
                dtype,
            ),
            "output2": self._bucket_scratch(
                bucket,
                "embedding_mhc_pre_post",
                (int(compile_bsz), int(compile_seqlen), hc_mult),
                np.float32,
            ),
            "output3": self._bucket_scratch(
                bucket,
                "embedding_mhc_pre_comb",
                (int(compile_bsz), int(compile_seqlen), hc_mult, hc_mult),
                np.float32,
            ),
        }
        _run_product_kernel(
            kernel,
            build_dir=self.build_dir,
            inputs={
                "input_ids": bucket.input_ids_dev,
                "local_embeddings": embeddings_dev,
                "vocab_range": vocab_range,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_weight": norm_weight,
            },
            outputs=outputs,
            unload_after_call=False,
        )
        return (
            self._product_active_alias(
                outputs["output0"],
                (int(bsz), int(seqlen), hc_mult, hidden_size),
            ),
            self._product_active_alias(
                outputs["output1"],
                (int(bsz), int(seqlen), hidden_size),
            ),
            self._product_active_alias(
                outputs["output2"],
                (int(bsz), int(seqlen), hc_mult),
            ),
            self._product_active_alias(
                outputs["output3"],
                (int(bsz), int(seqlen), hc_mult, hc_mult),
            ),
        )


# --- from frequency.py ---
class Dsv4ProductFrequencyMixin:
    def _product_freq_tables_for(
        self,
        cos_table: Any,
        sin_table: Any,
        *,
        name: str,
    ) -> tuple[Any, Any]:
        if _is_device_value(cos_table) and _is_device_value(sin_table):
            return cos_table, sin_table
        registry = getattr(self, "_product_freq_tables", None)
        if registry is None:
            registry = {}
            self._product_freq_tables = registry
        key = (
            str(name),
            _host_array_signature(cos_table),
            _host_array_signature(sin_table),
        )
        cached = registry.get(key)
        if cached is not None:
            return cached
        tensor_cls = _get_device_tensor_cls()
        cos_dev = tensor_cls.from_numpy(
            np.ascontiguousarray(np.asarray(cos_table, dtype=np.float32)),
            name=f"dsv4_product_{name}_cos_table",
        )
        sin_dev = tensor_cls.from_numpy(
            np.ascontiguousarray(np.asarray(sin_table, dtype=np.float32)),
            name=f"dsv4_product_{name}_sin_table",
        )
        registry[key] = (cos_dev, sin_dev)
        return cos_dev, sin_dev

    def _product_freq_positions_for(
        self,
        bucket: Dsv4ProductBucket,
        positions: Any,
        *,
        rows: int | None = None,
    ) -> Any:
        if _is_device_value(positions):
            pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
            n_pos = int(np.prod(pos_shape)) if pos_shape else 1
            target_rows = n_pos if rows is None else min(int(rows), n_pos)
            if target_rows <= 0:
                raise RuntimeError("DSV4 product frequency positions must be non-empty")
            if rows is not None and int(rows) > n_pos:
                full = self._product_full_value_for(positions, (int(rows),))
                if full is not None:
                    return full
            alias = _alias_device_value_shape(positions, (target_rows,))
            if alias is not None:
                return alias
            return positions
        pos = np.asarray(positions, dtype=np.int32).reshape(-1)
        if rows is not None:
            target_rows = int(rows)
            if target_rows < int(pos.shape[0]):
                pos = pos[:target_rows]
            elif target_rows > int(pos.shape[0]):
                pad_value = int(pos[0]) if int(pos.shape[0]) > 0 else 0
                pos = np.concatenate(
                    (
                        pos,
                        np.full(
                            (target_rows - int(pos.shape[0]),),
                            pad_value,
                            dtype=np.int32,
                        ),
                    ),
                    axis=0,
                )
        n_pos = int(pos.shape[0])
        if n_pos <= 0:
            raise RuntimeError("DSV4 product frequency positions must be non-empty")
        if n_pos > int(bucket.token_bucket):
            return _get_device_tensor_cls().from_numpy(
                np.ascontiguousarray(pos),
                name=f"dsv4_product_freq_positions_{n_pos}",
            )
        _overwrite_device_tensor_if_changed(
            bucket.freq_positions_dev,
            bucket.freq_positions_host,
            pos,
            prefix_len=n_pos,
            error_context="DSV4 product metadata sync",
        )
        alias = _alias_device_value_shape(bucket.freq_positions_dev, (n_pos,))
        if alias is not None:
            return alias
        return _get_device_tensor_cls().from_numpy(
            np.ascontiguousarray(pos),
            name=f"dsv4_product_freq_positions_{n_pos}",
        )


# --- from metadata.py ---
class Dsv4ProductMetadataMixin:
    def _prepare_embedding_input_ids(
        self,
        input_ids: np.ndarray,
    ) -> tuple[Dsv4ProductBucket, int, int, int]:
        ids = np.asarray(input_ids)
        bucket = self._require_active_product_bucket(where="embedding HC")
        if ids.ndim != 2:
            raise RuntimeError(
                "DSV4 product embedding HC expects input_ids with shape "
                f"[batch, seqlen], got shape={tuple(ids.shape)}"
            )

        bsz = int(ids.shape[0])
        seqlen = int(ids.shape[1])
        max_requests = int(bucket.max_requests)
        token_bucket = int(bucket.token_bucket)
        if bsz > max_requests:
            raise RuntimeError(
                "DSV4 product embedding batch exceeds configured max requests: "
                f"batch={bsz}, max_requests={max_requests}"
            )
        if max_requests <= 0:
            raise RuntimeError(
                "DSV4 product embedding HC requires a positive request bucket: "
                f"token_bucket={token_bucket}, max_requests={max_requests}"
            )
        if bsz * seqlen > token_bucket:
            raise RuntimeError(
                "DSV4 product embedding sequence exceeds configured token bucket: "
                f"batch={bsz}, seqlen={seqlen}, token_bucket={token_bucket}"
            )

        input_capacity = self._product_embedding_input_capacity(bucket)
        if seqlen > input_capacity:
            raise RuntimeError(
                "DSV4 product embedding sequence exceeds input-id capacity: "
                f"seqlen={seqlen}, input_capacity={input_capacity}"
            )
        if bsz == max_requests and seqlen == input_capacity:
            full_ids = ids.astype(np.int32, copy=False)
        else:
            full_ids = np.zeros((max_requests, input_capacity), dtype=ids.dtype)
            full_ids[:bsz, :seqlen] = ids
            flat_tokens = bsz * seqlen
            if 0 < flat_tokens <= input_capacity:
                full_ids[0, :flat_tokens] = ids.reshape(-1)
            full_ids = full_ids.astype(np.int32, copy=False)
        _overwrite_device_tensor_if_changed(
            bucket.input_ids_dev,
            bucket.input_ids_host,
            full_ids,
            error_context="DSV4 product metadata sync",
        )
        return bucket, bsz, seqlen, input_capacity

    def _sync_embedding_vocab_range(self, bucket: Dsv4ProductBucket) -> Any:
        start = int(getattr(self, "embed_vocab_offset", -1))
        end = int(getattr(self, "embed_vocab_end", -1))
        if start < 0 or end <= start:
            raise RuntimeError(
                "DSV4 product embedding requires a valid TP vocab range, "
                f"got start={start}, end={end}"
            )
        _overwrite_device_tensor_if_changed(
            bucket.vocab_range_dev,
            bucket.vocab_range_host,
            np.asarray((start, end), dtype=np.int32),
            error_context="DSV4 product metadata sync",
        )
        return bucket.vocab_range_dev

    def _sync_attention_dp_lane_start(
        self,
        bucket: Dsv4ProductBucket,
        start: int,
    ) -> Any:
        start_i = int(start)
        if start_i < 0:
            raise RuntimeError(
                f"DSV4 product attention DP lane start must be >= 0, got {start_i}"
            )
        _overwrite_device_tensor_if_changed(
            bucket.attention_dp_lane_start_dev,
            bucket.attention_dp_lane_start_host,
            np.asarray((start_i,), dtype=np.int32),
            error_context="DSV4 product metadata sync",
        )
        return bucket.attention_dp_lane_start_dev

    def _sync_attention_dp_token_range(
        self,
        bucket: Dsv4ProductBucket,
        *,
        token_start: int,
        token_count: int,
    ) -> tuple[Any, Any]:
        start_i = int(token_start)
        count_i = int(token_count)
        if start_i < 0 or count_i < 0:
            raise RuntimeError(
                "DSV4 product attention DP token range must be non-negative: "
                f"start={start_i}, count={count_i}"
            )
        _overwrite_device_tensor_if_changed(
            bucket.attention_dp_token_start_dev,
            bucket.attention_dp_token_start_host,
            np.asarray((start_i,), dtype=np.int32),
            error_context="DSV4 product metadata sync",
        )
        _overwrite_device_tensor_if_changed(
            bucket.attention_dp_token_count_dev,
            bucket.attention_dp_token_count_host,
            np.asarray((count_i,), dtype=np.int32),
            error_context="DSV4 product metadata sync",
        )
        return bucket.attention_dp_token_start_dev, bucket.attention_dp_token_count_dev

    def _last_token_indices_dev_for(
        self,
        input_ids: np.ndarray,
        *,
        metadata: Any | None,
        batch_size: int,
    ) -> Any:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(
                "DSV4 product sampled-head indices require an active token bucket"
            )
        bs = int(batch_size)
        if bs > int(bucket.max_requests):
            raise RuntimeError(
                "DSV4 product sampled head batch exceeds configured max requests: "
                f"batch={bs}, max={int(bucket.max_requests)}"
            )
        base = self._base_metadata(metadata)
        next_indices = np.zeros_like(bucket.last_token_indices_host)
        if base is not None and hasattr(base, "query_start_loc"):
            qsl = np.asarray(base.query_start_loc, dtype=np.int32)
            if qsl.shape[0] < bs + 1:
                raise RuntimeError(
                    "query_start_loc is too short for DSV4 sampled head: "
                    f"shape={qsl.shape}, batch={bs}"
                )
            next_indices[:bs] = qsl[1 : bs + 1] - 1
        else:
            ids = np.asarray(input_ids)
            if ids.ndim >= 2:
                seqlen = int(ids.shape[1])
                next_indices[:bs] = np.arange(bs, dtype=np.int32) * np.int32(
                    seqlen
                ) + np.int32(seqlen - 1)
            else:
                next_indices[0] = np.int32(ids.size - 1)
        _overwrite_device_tensor_if_changed(
            bucket.last_token_indices_dev,
            bucket.last_token_indices_host,
            next_indices,
            error_context="DSV4 product metadata sync",
        )
        return bucket.last_token_indices_dev

    def _compact_last_token_indices_dev_for(self, *, batch_size: int) -> Any:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(
                "DSV4 product compact sampled-head indices require an active token bucket"
            )
        bs = int(batch_size)
        if bs > int(bucket.max_requests):
            raise RuntimeError(
                "DSV4 product compact sampled head batch exceeds configured max requests: "
                f"batch={bs}, max={int(bucket.max_requests)}"
            )
        next_indices = np.zeros_like(bucket.last_token_indices_host)
        next_indices[:bs] = np.arange(bs, dtype=np.int32)
        _overwrite_device_tensor_if_changed(
            bucket.last_token_indices_dev,
            bucket.last_token_indices_host,
            next_indices,
            error_context="DSV4 product metadata sync",
        )
        return bucket.last_token_indices_dev


# --- from stage_profile.py ---
class Dsv4ProductProfileMixin:
    def _init_product_stage_profile_writer(self) -> None:
        self._product_stage_profile_writer: ProfileWriter | None = None
        if stage_profile_enabled():
            coord = _product_executor_coord(self)
            rank = int(getattr(coord, "rank", -1))
            if rank_trace_allowed(rank):
                self._product_stage_profile_writer = ProfileWriter(
                    f"dsv4_product_forward_rank_{rank}",
                    flush_every=1,
                )

    @contextmanager
    def _profile_product_stage(self, stage: str, **fields: Any):
        writer = getattr(self, "_product_stage_profile_writer", None)
        if writer is None:
            yield
            return
        t0 = time.perf_counter()
        status = "ok"
        error = ""
        try:
            yield
        except Exception as exc:
            status = "error"
            error = repr(exc)
            raise
        finally:
            self._write_product_stage_profile(
                stage,
                time.perf_counter() - t0,
                status=status,
                error=error,
                **fields,
            )

    def _write_product_stage_profile(
        self,
        stage: str,
        elapsed_s: float,
        *,
        status: str = "ok",
        error: str = "",
        **fields: Any,
    ) -> None:
        writer = getattr(self, "_product_stage_profile_writer", None)
        if writer is None:
            return
        active_layer_fields = getattr(
            self,
            "_active_product_layer_graph_profile_fields",
            None,
        )
        if active_layer_fields:
            fields = {**active_layer_fields, **fields}
        coord = _product_executor_coord(self)
        writer.write(
            {
                "ts": time.time(),
                "rank": int(getattr(coord, "rank", -1)),
                "tp": int(getattr(coord, "col", -1)),
                "ep": int(getattr(coord, "row_in_replica", -1)),
                "lane": int(getattr(coord, "attn_lane", -1)),
                "stage": str(stage),
                "elapsed_s": round(float(elapsed_s), 6),
                "status": str(status),
                **fields,
                **({"error": str(error)} if error else {}),
            }
        )
