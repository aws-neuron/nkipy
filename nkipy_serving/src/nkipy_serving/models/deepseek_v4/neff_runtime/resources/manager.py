"""NEFF-backed runtime resources for DeepSeek-V4."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.models.deepseek_v4.neff_runtime.lifecycle import (
    _product_executor_coord,
    _product_warmup_trace,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.kernel_cache import (
    _kernel_cache,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    _PRODUCT_KERNEL_CACHE_FIELDS,
    _PRODUCT_REQUIRED_GRAPH_KEYS,
    Dsv4ProductBucket,
    _TensorSpec,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)
from nkipy_serving.runtime.device_tensor import (
    normalize_dtype as _normalize_dtype,
)

_PRODUCT_LAYER_SCRATCH_SLOTS = 2


def _resource_value_dtype(value: Any, *, fallback: Any = np.float32) -> Any:
    return _normalize_dtype(getattr(value, "dtype", fallback))


def _scratch_cache_key(
    shape: tuple[int, ...], dtype: Any
) -> tuple[tuple[int, ...], str]:
    dtype_n = _normalize_dtype(dtype)
    return (tuple(int(dim) for dim in shape), str(np.dtype(dtype_n)))


def _scratch_from_cache(
    cache: dict[tuple[str, tuple[tuple[int, ...], str]], Any],
    *,
    tensor_cls: Any,
    token_bucket: int,
    kind: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    dtype_n = _normalize_dtype(dtype)
    key = (str(kind), _scratch_cache_key(shape, dtype_n))
    cached = cache.get(key)
    if cached is not None:
        return cached
    cached = tensor_cls.from_numpy(
        np.zeros(tuple(int(dim) for dim in shape), dtype=dtype_n),
        name=(f"dsv4_product_{str(kind)}_scratch_t{int(token_bucket)}_s{len(cache)}"),
    )
    cache[key] = cached
    return cached


class Dsv4ProductBucketManagerMixin:
    def _require_active_product_bucket(self, *, where: str) -> Dsv4ProductBucket:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            raise RuntimeError(f"DSV4 product {where} requires an active token bucket")
        return bucket

    def _require_runtime_token_bucket(self, token_bucket: int | None) -> int:
        if token_bucket is None:
            raise RuntimeError(
                "DSV4 product executor requires scheduler-provided token_bucket"
            )
        bucket = int(token_bucket)
        if bucket <= 0:
            raise RuntimeError(f"DSV4 product token_bucket must be > 0, got {bucket}")
        for configured in self._configured_product_token_buckets():
            if bucket <= configured:
                return int(configured)
        return bucket

    def precompile_token_bucket(self, token_bucket: int) -> Dsv4ProductBucket:
        """Register product-owned graph handles for a token bucket.

        This is the product-mode warmup boundary. The current implementation
        records the bucket-specific fused-stage handles; later steps can hang
        preallocated scratch and DeviceKernel objects off this object without
        changing the executor surface.
        """
        return self._ensure_product_bucket(
            self._require_runtime_token_bucket(int(token_bucket))
        )

    def precompile_token_buckets(
        self,
        token_buckets: tuple[int, ...] | list[int],
    ) -> tuple[Dsv4ProductBucket, ...]:
        return tuple(
            self.precompile_token_bucket(int(bucket)) for bucket in token_buckets
        )

    def _product_compile_batch_size(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
    ) -> int:
        """Use max-request batch kernels when the current seqlen fits the bucket."""
        bsz_i = int(bsz)
        seqlen_i = int(seqlen)
        max_requests = int(bucket.max_requests)
        if bsz_i <= 0 or seqlen_i <= 0 or max_requests <= bsz_i:
            return bsz_i
        if max_requests * seqlen_i <= int(bucket.token_bucket):
            return max_requests
        return bsz_i

    def _product_compile_batch_size_candidates(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
    ) -> tuple[int, ...]:
        promoted = self._product_compile_batch_size(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
        )
        candidates = [int(promoted)]
        if int(bsz) not in candidates:
            candidates.append(int(bsz))
        return tuple(candidates)

    def _product_compile_embedding_shape(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
        bucket_single_token: bool = False,
    ) -> tuple[int, int]:
        """Canonicalize first-layer embedding to the scheduler token bucket.

        The scheduler already pads flat prefill input IDs to ``token_bucket``.
        The sampled DSV4 runtime trims the active request metadata back to the real
        prompt length, so without this bucketing a short prompt such as 9 tokens
        would create a one-off embedding NEFF. Compile the embedding+mHC input
        as a rectangular slice of the selected product bucket instead.
        """
        return self._product_compile_sequence_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
            bucket_single_token=bool(bucket_single_token),
        )

    def _product_compile_sequence_shape(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
        bucket_single_token: bool = False,
    ) -> tuple[int, int]:
        """Canonicalize single-request sequence kernels to the token bucket.

        Decode must keep true ``s1`` kernels. One-token prefill can opt into
        bucketed shapes because active aliases trim padded rows at the layer
        boundaries, which removes a separate prefill-s1 NEFF family.
        """
        bsz_i = int(bsz)
        seq_i = int(seqlen)
        token_bucket = int(bucket.token_bucket)
        if bool(bucket_single_token) and bsz_i * seq_i <= token_bucket:
            return 1, token_bucket
        if bool(bucket_single_token) and bsz_i == 1 and seq_i <= 1:
            compile_bsz = 1
        else:
            compile_bsz = self._product_compile_batch_size(
                bucket,
                bsz=bsz_i,
                seqlen=seq_i,
            )
        compile_bsz = max(1, int(compile_bsz))
        if seq_i <= 1 and not bool(bucket_single_token):
            return compile_bsz, max(1, seq_i)
        if bsz_i * seq_i <= token_bucket:
            return 1, token_bucket
        compile_seqlen = max(seq_i, token_bucket // compile_bsz)
        input_capacity = self._product_embedding_input_capacity(bucket)
        compile_seqlen = min(int(compile_seqlen), int(input_capacity))
        if compile_bsz * compile_seqlen > token_bucket:
            raise RuntimeError(
                "DSV4 product embedding compile shape exceeds token bucket: "
                f"compile_bsz={compile_bsz}, compile_seqlen={compile_seqlen}, "
                f"token_bucket={token_bucket}, bsz={int(bsz)}, "
                f"seqlen={int(seqlen)}"
            )
        return compile_bsz, compile_seqlen

    def _dp_attention_post_pre_compile_shape(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
        rows: int,
        dispatch_context: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        seq_i = max(1, int(seqlen))
        bsz_i = max(1, int(bsz))
        if dispatch_context is not None and bool(dispatch_context.get("is_decode")):
            return (
                self._product_compile_batch_size(bucket, bsz=bsz_i, seqlen=seq_i),
                seq_i,
            )
        token_bucket = int(bucket.token_bucket)
        if bsz_i * seq_i <= token_bucket and int(rows) >= token_bucket:
            return 1, token_bucket
        compile_seqlen = self._dp_attention_post_pre_compile_seqlen(
            bucket,
            bsz=bsz_i,
            seqlen=seq_i,
            rows=int(rows),
            dispatch_context=dispatch_context,
        )
        return (
            self._product_compile_batch_size(
                bucket,
                bsz=bsz_i,
                seqlen=int(compile_seqlen),
            ),
            int(compile_seqlen),
        )

    def _product_compile_attention_qkv_shape(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
    ) -> tuple[int, int]:
        return self._product_compile_sequence_shape(
            bucket,
            bsz=int(bsz),
            seqlen=int(seqlen),
        )

    def _product_bucketed_prefill_offset(
        self,
        bucket: Dsv4ProductBucket,
        x: Any,
        *,
        bsz: int,
        seqlen: int,
        hidden_size: int,
        window_size: int,
        offset: int,
        bucketed: bool,
        max_compile_tokens: int | None = None,
    ) -> tuple[Any, int, bool, int]:
        """Shared bucketed-prefill re-alias + two-source offset for the QKV
        prologue runners (token-topk and all-KV indexer).

        When ``bucketed`` and the prologue compiles at a token bucket larger
        than the real prompt, re-alias ``x`` to the bucket buffer (if one backs
        it) and recompute ``offset`` == the downstream two-source ``primary_len``
        (the SWA window for short prefill, else the FINAL compiled seqlen). This
        is the single source of truth the host reads back via
        ``_product_last_qkv_compiled_offset``; callers must publish the returned
        ``offset``/``realiased``/``seqlen`` verbatim. Returns
        ``(x, seqlen, realiased, offset)`` — callers re-derive any NEFF-key-baked
        state tails from the returned ``seqlen`` themselves.
        """
        realiased = False
        if bucketed:
            _product_warmup_trace(
                _product_executor_coord(self),
                "bucketed_prefill_offset start "
                f"bsz={int(bsz)} seqlen={int(seqlen)} "
                f"hidden={int(hidden_size)} rows={max_compile_tokens} "
                f"offset={int(offset)} token_bucket={int(bucket.token_bucket)}",
            )
            compile_bsz, compile_seqlen = self._product_compile_attention_qkv_shape(
                bucket,
                bsz=int(bsz),
                seqlen=int(seqlen),
            )
            compile_bsz = int(compile_bsz)
            compile_seqlen = int(compile_seqlen)
            max_compile_tokens_i = (
                int(max_compile_tokens)
                if max_compile_tokens is not None and int(max_compile_tokens) > 0
                else None
            )
            if (
                max_compile_tokens_i is not None
                and compile_bsz * compile_seqlen > max_compile_tokens_i
                and int(bsz) == 1
            ):
                compile_bsz = 1
                compile_seqlen = int(max_compile_tokens_i)
            if compile_bsz != int(bsz) or compile_seqlen > int(seqlen):
                x_full = self._product_full_value_for(
                    x,
                    (compile_bsz, compile_seqlen, int(hidden_size)),
                )
                if x_full is None and int(bsz) == 1 and max_compile_tokens_i is None:
                    lane_bucket = self._attention_backend_bucket_for_tokens(
                        int(bsz) * int(seqlen),
                        int(bucket.token_bucket),
                        is_decode=False,
                    )
                    if int(lane_bucket) > int(seqlen):
                        lane_full = self._product_full_value_for(
                            x,
                            (1, int(lane_bucket), int(hidden_size)),
                        )
                        if lane_full is not None:
                            x_full = lane_full
                            compile_bsz = 1
                            compile_seqlen = int(lane_bucket)
                if (
                    x_full is None
                    and int(bsz) == 1
                    and int(compile_bsz) == 1
                    and int(compile_seqlen) > int(seqlen)
                ):
                    pad_hidden = getattr(self, "_run_product_sequence_hidden_pad", None)
                    if callable(pad_hidden):
                        _product_warmup_trace(
                            _product_executor_coord(self),
                            "bucketed_prefill_offset pad_hidden start "
                            f"bsz={int(bsz)} seqlen={int(seqlen)} "
                            f"compile_seqlen={int(compile_seqlen)} "
                            f"hidden={int(hidden_size)}",
                        )
                        x_full = pad_hidden(
                            bucket,
                            x,
                            rows=int(compile_seqlen),
                            hidden_size=int(hidden_size),
                        )
                        _product_warmup_trace(
                            _product_executor_coord(self),
                            "bucketed_prefill_offset pad_hidden done "
                            f"bsz={int(bsz)} seqlen={int(seqlen)} "
                            f"compile_seqlen={int(compile_seqlen)} "
                            f"x_full_shape={getattr(x_full, 'shape', None)}",
                        )
                # No bucket-sized backing buffer (warmup partial-batch lanes):
                # compile at the real seqlen with the passed offset.
                if x_full is not None:
                    realiased = True
                    x = x_full
                    seqlen = compile_seqlen
            offset = (
                int(window_size) if int(seqlen) <= int(window_size) else int(seqlen)
            )
            _product_warmup_trace(
                _product_executor_coord(self),
                "bucketed_prefill_offset done "
                f"compiled_seqlen={int(seqlen)} realiased={bool(realiased)} "
                f"offset={int(offset)}",
            )
        return x, int(seqlen), bool(realiased), int(offset)

    def _attention_out_dp_flat_compile_seqlen(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
        batch_size: int,
        start: int,
        size: int,
        rows: int,
        is_decode: bool = False,
    ) -> int:
        """Canonicalize single-request attention-output projection to a bucket.

        DP post/pre currently restores by slicing the first active
        ``batch_size * seqlen`` flat rows, so multi-request lanes must stay
        active-packed. For the common one-request prefill/decode path there is
        no cross-request row offset, and compiling at the selected product
        bucket avoids one-off NEFFs for prompt lengths such as 9 or 11 tokens.
        """
        seq_i = max(1, int(seqlen))
        if bool(is_decode):
            return seq_i
        if int(batch_size) * seq_i <= int(bucket.token_bucket) and int(rows) >= int(
            bucket.token_bucket
        ):
            return int(bucket.token_bucket)
        return seq_i

    def _dp_attention_post_pre_compile_seqlen(
        self,
        bucket: Dsv4ProductBucket,
        *,
        bsz: int,
        seqlen: int,
        rows: int,
        dispatch_context: dict[str, Any] | None = None,
    ) -> int:
        """Canonicalize single-request DP post/pre to the selected bucket.

        This stage consumes the DP-flat attention reduce buffer. For one
        prefill request, padded rows are isolated from active rows and the
        following active aliases trim them away, so compiling against the
        bucket removes prompt-length-specific kernels. Decode and multi-request
        batches keep active shapes because row packing is semantically tighter.
        """
        seq_i = max(1, int(seqlen))
        if dispatch_context is not None and bool(dispatch_context.get("is_decode")):
            return seq_i
        if int(bsz) == 1:
            return max(seq_i, min(int(bucket.token_bucket), int(rows)))
        return seq_i

    def _product_all_kv_decode_compile_kv_len(
        self,
        *,
        kv_len: int,
        seqlen: int,
        window_size: int,
        k_tile: int,
    ) -> int:
        """Bucket decode all-KV sparse-prep width without changing output shape."""
        actual = int(kv_len)
        if actual <= 0 or int(seqlen) != 1:
            return actual
        win_width = max(0, int(window_size))
        tile = max(1, int(k_tile))
        current_raw = win_width + actual
        current_padded = ((current_raw + tile - 1) // tile) * tile
        max_same_padded = max(actual, current_padded - win_width)
        return int(max_same_padded)

    def _configured_product_token_buckets(self) -> tuple[int, ...]:
        backend = getattr(self, "attention_backend", None)
        ladder = tuple(int(v) for v in (getattr(backend, "bucket_ladder", ()) or ()))
        if ladder:
            return tuple(sorted(set(ladder)))
        token_bucket = int(getattr(backend, "token_bucket", 0) or 0)
        return (token_bucket,) if token_bucket > 0 else ()

    def _configured_product_decode_buckets(self) -> tuple[int, ...]:
        backend = getattr(self, "attention_backend", None)
        ladder = tuple(
            int(v) for v in (getattr(backend, "decode_bucket_ladder", ()) or ())
        )
        return tuple(sorted({bucket for bucket in ladder if bucket > 0}))

    def _attention_backend_bucket_for_tokens(
        self,
        total_tokens: int,
        fallback_bucket: int,
        *,
        is_decode: bool = False,
    ) -> int:
        """Mirror sparse-attention backend bucket selection for product lanes.

        Delegates to ``backend.bucket_for_tokens`` (the prepare() routing) so
        product compile shapes cannot drift from the runtime bucket; falls back
        to the configured token buckets when no backend is installed.
        """
        required = max(2, int(total_tokens))
        backend = getattr(self, "attention_backend", None)
        route = getattr(backend, "bucket_for_tokens", None)
        if callable(route):
            chosen = int(route(required, is_decode=bool(is_decode)))
            if chosen >= required:
                return chosen
        for configured in sorted(set(self._configured_product_token_buckets())):
            if required <= int(configured):
                return int(configured)
        return max(required, int(fallback_bucket))

    def _bucket_scratch(
        self,
        bucket: Dsv4ProductBucket,
        kind: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any:
        # On-demand runtime scratch. Like the static per-layer buffers, only one
        # token bucket runs per step, so a bucket-proportional buffer can share
        # one max-bucket arena buffer ACROSS buckets, with a smaller bucket
        # aliasing a prefix slice.
        #
        # SAFETY: only alias when the LEADING dim equals the active token bucket
        # (shape[0] == bucket.token_bucket). Then a smaller bucket's buffer is
        # EXACTLY the first `token_bucket` rows of the max-bucket buffer -- a
        # true prefix, no scaling guesswork. Buffers whose leading dim is NOT the
        # token bucket (e.g. attention_topk_t=(k_padded,rows), mhc_pre_*=
        # (compile_bsz,...), or n_tokens when batch>1) fall through to the
        # per-bucket cache unchanged -- aliasing them by leading dim would be
        # wrong (and could over-allocate). The arena key carries the full tail
        # (shape[1:]) so distinct shapes the per-bucket cache keeps separate
        # (e.g. compressor_kv_bf16 comp vs idx tails) stay distinct here too.
        shape_t = tuple(int(d) for d in shape)
        cur = int(bucket.token_bucket)
        if len(shape_t) >= 2 and cur > 0 and int(shape_t[0]) == cur:
            configured = tuple(self._configured_product_token_buckets())
            max_bucket = max((*configured, cur))
            # Always route through the arena (not just smaller buckets): the
            # max bucket gets the base buffer (rows == max_rows), smaller buckets
            # alias a prefix of that SAME base. Guarding on max_bucket>cur would
            # leave the max bucket self-allocating in the per-bucket cache while
            # smaller buckets aliased a separate arena buffer -> no sharing AND a
            # double allocation at the max bucket.
            return self._arena_scratch(
                tensor_cls=_get_device_tensor_cls(),
                kind=f"ondemand:{kind}",
                layer_id=-2,
                max_rows=int(max_bucket),
                rows=int(cur),
                shape_tail=shape_t[1:],
                dtype=dtype,
            )
        return _scratch_from_cache(
            bucket.scratch_outputs,
            tensor_cls=_get_device_tensor_cls(),
            token_bucket=int(bucket.token_bucket),
            kind=kind,
            shape=shape,
            dtype=dtype,
        )

    def _alloc_mhc_post_output(
        self,
        bucket: Dsv4ProductBucket,
        *,
        residual_shape: tuple[int, ...],
        residual: Any,
        x: Any,
    ) -> Any:
        """Pick the double-buffered output buffer for a fused mHC-post kernel.

        Alternates between two scratch slots (``mhc_post_out_0``/``_1``) so a
        layer's output does not alias its own input; slot 1 reuses a published
        ``_mhc_post_slot1_alias`` when shape/dtype match and it is neither the
        current ``x`` nor ``residual``. Shared by every fused mHC-post runner
        (dp_attention, dp_attention_moe_fused, shared_expert).
        """
        scratch_index = int(getattr(self, "_mhc_post_scratch_index", 0))
        slot = scratch_index % 2
        self._mhc_post_scratch_index = scratch_index + 1
        dtype = _resource_value_dtype(residual, fallback=ml_dtypes.bfloat16)
        h = None
        if slot == 1:
            candidate = getattr(self, "_mhc_post_slot1_alias", None)
            if (
                candidate is not None
                and candidate is not x
                and candidate is not residual
                and _normalize_dtype(getattr(candidate, "dtype", None))
                == _normalize_dtype(dtype)
            ):
                h = _alias_device_value_shape(candidate, residual_shape)
        if h is None:
            h = self._bucket_scratch(
                bucket,
                f"mhc_post_out_{slot}",
                residual_shape,
                dtype,
            )
        return h

    def _alias_mhc_post_pre_outputs(
        self,
        outputs: dict[str, Any],
        *,
        bsz: int,
        seqlen: int,
        hc_mult: int,
        hidden_size: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Active-slice the 4 fused mHC-post/pre kernel outputs to their logical
        shapes. Shared by every fused mHC-post/pre runner (dp_attention,
        dp_attention_moe_fused, shared_expert)."""
        b, s, hc, h = int(bsz), int(seqlen), int(hc_mult), int(hidden_size)
        return (
            self._product_active_alias(outputs["output0"], (b, s, hc, h)),
            self._product_active_alias(outputs["output1"], (b, s, h)),
            self._product_active_alias(outputs["output2"], (b, s, hc)),
            self._product_active_alias(outputs["output3"], (b, s, hc, hc)),
        )

    def _product_hidden_size_for_bucket(self, bucket: Dsv4ProductBucket) -> int:
        hidden_shape = tuple(
            int(dim) for dim in getattr(bucket.head_hidden_output, "shape", ())
        )
        hidden_size = int(hidden_shape[-1]) if hidden_shape else 0
        if hidden_size > 0:
            return hidden_size
        hidden_size = int(
            getattr(
                getattr(self.runtime_surface, "model_config", None), "hidden_size", 0
            )
            or 0
        )
        if hidden_size > 0:
            return hidden_size
        embed = getattr(getattr(self.runtime_surface, "w", None), "embed", None)
        embed_shape = tuple(int(dim) for dim in getattr(embed, "shape", ()))
        if embed_shape:
            return int(embed_shape[-1])
        return 0

    def _product_embedding_input_capacity(self, bucket: Dsv4ProductBucket) -> int:
        # Input IDs are small compared with hidden activations. Allocate each
        # request enough columns for the full token bucket so bsz=1 can warm
        # long prompts while bsz=max_requests still respects the total-token
        # runtime bucket through the active bsz*seqlen checks below.
        capacity = int(bucket.token_bucket)
        if capacity <= 0:
            raise RuntimeError(
                "DSV4 product embedding input capacity must be positive: "
                f"token_bucket={int(bucket.token_bucket)}"
            )
        return capacity

    def _embedding_full_spec_for_bucket(
        self,
        bucket: Dsv4ProductBucket,
    ) -> _TensorSpec:
        max_requests = int(bucket.max_requests)
        if max_requests <= 0:
            raise RuntimeError(
                "DSV4 product embedding spec requires a positive request bucket: "
                f"token_bucket={int(bucket.token_bucket)}, max_requests={max_requests}"
            )
        hidden_size = self._product_hidden_size_for_bucket(bucket)
        if hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product embedding spec could not infer hidden size"
            )
        hc_mult = int(
            getattr(getattr(self.runtime_surface, "args", None), "hc_mult", 1)
        )
        dtype = _resource_value_dtype(
            getattr(getattr(self.runtime_surface, "w", None), "embed", None),
            fallback=ml_dtypes.bfloat16,
        )
        return _TensorSpec(
            (
                max_requests,
                self._product_embedding_input_capacity(bucket),
                hc_mult,
                hidden_size,
            ),
            dtype,
        )

    def _product_bucket_registry(self) -> dict[int, Dsv4ProductBucket]:
        registry = getattr(self, "_product_buckets", None)
        if registry is None:
            registry = {}
            self._product_buckets = registry
        return registry

    @staticmethod
    def _decode_moe_rows_for_requests(max_requests: int) -> int:
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            MOE_BLOCK_SIZE,
        )

        rows = max(int(max_requests), 1)
        block = int(MOE_BLOCK_SIZE)
        return ((rows + block - 1) // block) * block

    def _ensure_product_bucket(self, token_bucket: int) -> Dsv4ProductBucket:
        bucket = int(token_bucket)
        if bucket <= 0:
            raise RuntimeError(f"DSV4 product token_bucket must be > 0, got {bucket}")
        registry = self._product_bucket_registry()
        cached = registry.get(bucket)
        if cached is not None:
            return cached
        if bool(getattr(self, "_product_manifest_sealed", False)):
            known = ", ".join(str(k) for k in sorted(registry))
            raise RuntimeError(
                "DSV4 product token bucket was not precompiled before warmup "
                f"seal: token_bucket={bucket}, precompiled=[{known}]"
            )
        compiled = self._build_product_bucket(bucket)
        registry[bucket] = compiled
        return compiled

    def _product_scratch_arena(self) -> dict[Any, Any]:
        """Per-executor arena of row-major scratch buffers shared ACROSS buckets.

        DSV4 runs exactly one token bucket per step (``_active_product_bucket``)
        and these tensors are pure transient working memory (zeroed, fully
        overwritten each step), so the per-bucket scratch sets are never live
        simultaneously and can physically share HBM. Large per-layer outputs
        additionally rotate through a small slot ring; the forward path only
        needs the current layer plus the next-layer handoff, so two slots per
        kind cover the live range. Smaller buckets alias a first-dim slice of
        the max-bucket buffer. See device_tensor.alias_device_value_first_dim_slice.
        """
        arena = getattr(self, "_product_scratch_arena_cache", None)
        if arena is None:
            arena = {}
            self._product_scratch_arena_cache = arena
        return arena

    @staticmethod
    def _product_scratch_arena_layer_slot(layer_id: int) -> int:
        layer_i = int(layer_id)
        if layer_i < 0:
            return layer_i
        return layer_i % _PRODUCT_LAYER_SCRATCH_SLOTS

    def _arena_scratch(
        self,
        *,
        tensor_cls: Any,
        kind: str,
        layer_id: int,
        max_rows: int,
        rows: int,
        shape_tail: tuple[int, ...],
        dtype: Any,
    ) -> Any:
        """Return a ``(rows, *shape_tail)`` view of one ``(max_rows, *shape_tail)``
        arena buffer.

        Keyed by ``(kind, layer_slot, ...)`` so the buffer is shared across
        token buckets and across non-adjacent layers. Adjacent layers use
        distinct slots, and distinct ``kind`` values still never alias (for
        example, moe_prefill_out and moe_prefill_ep are both live during the
        EP->TP reduce). The arena always allocates the max-bucket size; smaller
        buckets alias a prefix slice with no new HBM allocation.
        """
        arena = self._product_scratch_arena()
        tail = tuple(int(d) for d in shape_tail)
        max_rows_i = int(max_rows)
        slot = self._product_scratch_arena_layer_slot(int(layer_id))
        key = (str(kind), int(slot), max_rows_i, tail, str(np.dtype(dtype)))
        base = arena.get(key)
        if base is None:
            base = tensor_cls.from_numpy(
                np.zeros((max_rows_i, *tail), dtype=dtype),
                name=(f"dsv4_product_{kind}_arena_s{int(slot)}_r{max_rows_i}"),
            )
            arena[key] = base
        rows_i = int(rows)
        if rows_i == max_rows_i:
            return base
        alias = _alias_device_value_first_dim_slice(base, start=0, size=rows_i)
        if alias is None:
            # Aliasing unsupported for this tensor backend (e.g. no tensor_ref);
            # fall back to a private allocation so correctness never depends on
            # the optimization being available.
            return tensor_cls.from_numpy(
                np.zeros((rows_i, *tail), dtype=dtype),
                name=f"dsv4_product_{kind}_l{int(layer_id)}_r{rows_i}",
            )
        return alias

    def _build_product_bucket(self, token_bucket: int) -> Dsv4ProductBucket:
        missing = [key for key in _PRODUCT_REQUIRED_GRAPH_KEYS if key not in self.graph]
        if missing:
            raise RuntimeError(
                "DSV4 product executor missing required trace functions: "
                + ", ".join(missing)
            )
        max_requests = int(getattr(self, "max_requests_per_step", 1))
        # Max configured token bucket sizes the shared scratch arena; smaller
        # buckets alias prefix slices. Fall back to this bucket if the ladder is
        # not yet known (then aliasing is a no-op and each bucket self-allocates).
        configured_buckets = tuple(self._configured_product_token_buckets())
        arena_max_token_bucket = max((*configured_buckets, int(token_bucket)))
        arena_max_prefill_rows = ((max(arena_max_token_bucket, 1) + 127) // 128) * 128
        attention_outputs = []
        moe_prefill_outputs = []
        moe_prefill_ep_outputs = []
        moe_prefill_tp_outputs = []
        moe_decode_outputs = []
        moe_decode_ep_outputs = []
        moe_decode_tp_outputs = []
        # Runtime on-demand scratch dict (consumed by the executor-level
        # _bucket_scratch method via bucket.scratch_outputs). The large STATIC
        # per-layer buffers below no longer use this -- they go through the
        # cross-bucket arena (_arena_scratch) instead.
        scratch_cache: dict[tuple[str, tuple[tuple[int, ...], str]], Any] = {}

        prefill_rows = ((max(int(token_bucket), 1) + 127) // 128) * 128
        decode_rows = self._decode_moe_rows_for_requests(max_requests)
        tensor_cls = _get_device_tensor_cls()
        blocks = tuple(getattr(self.runtime_surface, "blocks", ()))
        model_hidden_size = int(
            getattr(getattr(self.runtime_surface, "args", None), "dim", 0)
            or getattr(getattr(self.runtime_surface, "v4", None), "hidden_size", 0)
            or (getattr(getattr(blocks[0], "ffn", None), "dim", 0) if blocks else 0)
            or 0
        )
        if model_hidden_size <= 0:
            raise RuntimeError(
                "DSV4 product bucket requires model hidden size for sampled-head "
                "scratch; runtime_surface.args.dim/"
                "runtime_surface.v4.hidden_size/blocks[0].ffn.dim "
                "are missing"
            )
        input_id_capacity = int(token_bucket)
        for layer_id, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                raise RuntimeError(
                    "DSV4 product bucket requires per-layer attention metadata; "
                    f"layer {int(layer_id)} has no attn"
                )
            hidden_size = int(
                getattr(getattr(block, "ffn", None), "dim", 0)
                or getattr(getattr(self.runtime_surface, "args", None), "dim", 0)
                or 0
            )
            if hidden_size <= 0:
                raise RuntimeError(
                    "DSV4 product bucket requires per-layer MoE hidden size; "
                    f"layer {int(layer_id)} has no ffn.dim/runtime_surface.args.dim"
                )
            # Scratch shared ACROSS buckets via the arena: a bucket's buffer is a
            # prefix slice of the max-bucket buffer (only one bucket runs/step).
            # Keyed per (kind, layer_id) so layers and distinct kinds never alias.
            attention_outputs.append(
                self._arena_scratch(
                    tensor_cls=tensor_cls,
                    kind="attention_out",
                    layer_id=int(layer_id),
                    max_rows=int(arena_max_token_bucket),
                    rows=int(token_bucket),
                    shape_tail=(
                        int(getattr(attn, "n_heads")),
                        int(getattr(attn, "head_dim")),
                    ),
                    dtype=np.float32,
                )
            )
            prefill_out = self._arena_scratch(
                tensor_cls=tensor_cls,
                kind="moe_prefill_out",
                layer_id=int(layer_id),
                max_rows=int(arena_max_prefill_rows),
                rows=int(prefill_rows),
                shape_tail=(int(hidden_size),),
                dtype=ml_dtypes.bfloat16,
            )
            prefill_ep = self._arena_scratch(
                tensor_cls=tensor_cls,
                kind="moe_prefill_ep",
                layer_id=int(layer_id),
                max_rows=int(arena_max_prefill_rows),
                rows=int(prefill_rows),
                shape_tail=(int(hidden_size),),
                dtype=ml_dtypes.bfloat16,
            )
            # TP all-reduce runs after EP all-reduce, so the routed output
            # buffer is no longer a collective input and can hold the final TP
            # result. This saves one large persistent [rows, hidden] tensor per
            # product bucket without input/output aliasing inside a collective.
            prefill_tp = prefill_out
            moe_prefill_outputs.append(prefill_out)
            moe_prefill_ep_outputs.append(prefill_ep)
            moe_prefill_tp_outputs.append(prefill_tp)

            if int(decode_rows) == int(prefill_rows):
                moe_decode_outputs.append(prefill_out)
                moe_decode_ep_outputs.append(prefill_ep)
                moe_decode_tp_outputs.append(prefill_tp)
            else:
                # decode_rows (=MOE_BLOCK_SIZE, bucket-invariant) is the same for
                # every bucket, so a single arena entry serves all buckets; rows
                # == max_rows here => the base buffer is returned directly.
                moe_decode_outputs.append(
                    self._arena_scratch(
                        tensor_cls=tensor_cls,
                        kind="moe_decode_out",
                        layer_id=int(layer_id),
                        max_rows=int(decode_rows),
                        rows=int(decode_rows),
                        shape_tail=(int(hidden_size),),
                        dtype=ml_dtypes.bfloat16,
                    )
                )
                moe_decode_ep_outputs.append(
                    self._arena_scratch(
                        tensor_cls=tensor_cls,
                        kind="moe_decode_ep",
                        layer_id=int(layer_id),
                        max_rows=int(decode_rows),
                        rows=int(decode_rows),
                        shape_tail=(int(hidden_size),),
                        dtype=ml_dtypes.bfloat16,
                    )
                )
                moe_decode_tp_outputs.append(moe_decode_outputs[-1])
        return Dsv4ProductBucket(
            token_bucket=int(token_bucket),
            max_requests=max_requests,
            last_token_indices_host=np.zeros(
                (max_requests,),
                dtype=np.int32,
            ),
            last_token_indices_dev=tensor_cls.from_numpy(
                np.zeros(
                    (max_requests,),
                    dtype=np.int32,
                ),
                name=f"dsv4_product_last_token_indices_t{int(token_bucket)}",
            ),
            owner_ids_host=np.zeros((int(token_bucket),), dtype=np.int32),
            owner_ids_dev=tensor_cls.from_numpy(
                np.zeros((int(token_bucket),), dtype=np.int32),
                name=f"dsv4_product_owner_ids_t{int(token_bucket)}",
            ),
            input_ids_host=np.zeros(
                (max_requests, input_id_capacity),
                dtype=np.int32,
            ),
            input_ids_dev=tensor_cls.from_numpy(
                np.zeros((max_requests, input_id_capacity), dtype=np.int32),
                name=f"dsv4_product_input_ids_t{int(token_bucket)}",
            ),
            vocab_range_host=np.zeros((2,), dtype=np.int32),
            vocab_range_dev=tensor_cls.from_numpy(
                np.zeros((2,), dtype=np.int32),
                name=f"dsv4_product_vocab_range_t{int(token_bucket)}",
            ),
            freq_positions_host=np.zeros((int(token_bucket),), dtype=np.int32),
            freq_positions_dev=tensor_cls.from_numpy(
                np.zeros((int(token_bucket),), dtype=np.int32),
                name=f"dsv4_product_freq_positions_t{int(token_bucket)}",
            ),
            attention_dp_lane_start_host=np.zeros((1,), dtype=np.int32),
            attention_dp_lane_start_dev=tensor_cls.from_numpy(
                np.zeros((1,), dtype=np.int32),
                name=f"dsv4_product_attention_dp_lane_start_t{int(token_bucket)}",
            ),
            attention_dp_token_start_host=np.zeros((1,), dtype=np.int32),
            attention_dp_token_start_dev=tensor_cls.from_numpy(
                np.zeros((1,), dtype=np.int32),
                name=f"dsv4_product_attention_dp_token_start_t{int(token_bucket)}",
            ),
            attention_dp_token_count_host=np.zeros((1,), dtype=np.int32),
            attention_dp_token_count_dev=tensor_cls.from_numpy(
                np.zeros((1,), dtype=np.int32),
                name=f"dsv4_product_attention_dp_token_count_t{int(token_bucket)}",
            ),
            kernel_caches={
                cache_name: _kernel_cache()
                for cache_name in _PRODUCT_KERNEL_CACHE_FIELDS
            },
            head_hidden_output=self._arena_scratch(
                tensor_cls=tensor_cls,
                kind="head_hidden",
                layer_id=-1,
                max_rows=int(arena_max_token_bucket),
                rows=int(token_bucket),
                shape_tail=(int(model_hidden_size),),
                dtype=ml_dtypes.bfloat16,
            ),
            head_top1_values=tensor_cls.from_numpy(
                np.zeros((max_requests,), dtype=np.float32),
                name=f"dsv4_product_head_top1_values_t{int(token_bucket)}",
            ),
            head_top1_indices=tensor_cls.from_numpy(
                np.zeros((max_requests,), dtype=np.int32),
                name=f"dsv4_product_head_top1_indices_t{int(token_bucket)}",
            ),
            attention_outputs=tuple(attention_outputs),
            moe_prefill_outputs=tuple(moe_prefill_outputs),
            moe_prefill_ep_outputs=tuple(moe_prefill_ep_outputs),
            moe_prefill_tp_outputs=tuple(moe_prefill_tp_outputs),
            moe_decode_outputs=tuple(moe_decode_outputs),
            moe_decode_ep_outputs=tuple(moe_decode_ep_outputs),
            moe_decode_tp_outputs=tuple(moe_decode_tp_outputs),
            scratch_outputs=scratch_cache,
        )
