"""DeepSeek-V4 sparse attention backend orchestrator.

Metadata contracts, device-buffer helpers, and vanilla oracle helpers live in
smaller sibling modules so this backend class stays focused on orchestration.
"""

from __future__ import annotations

from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import FORWARD_MODE_DECODE
from nkipy_serving.attention.deepseek_v4.metadata import (
    SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
    SparseAttentionMetadata,
)
from nkipy_serving.attention.deepseek_v4.state import Dsv4DeviceState
from nkipy_serving.attention.deepseek_v4.types import (
    Dsv4AttentionMetadata,
    Dsv4DeviceAttentionInputs,
    allocate_dsv4_device_attention_inputs,
    build_positions_per_token,
    build_req_id_per_token,
    dsv4_device_sparse_attention_kernel_inputs,
    run_dsv4_device_sparse_attention,
    run_dsv4_swa_global_slots,
    tensor_to_step_field_name,
)
from nkipy_serving.attention.deepseek_v4.vanilla import (
    dsv4_vanilla_attn_fn,
    dsv4_vanilla_sparse_attention_core,
    dsv4_vanilla_update_kv_cache,
)
from nkipy_serving.runtime.device_tensor import (
    dtype_like,
    get_device_tensor_cls,
    is_device_tensor,
)

__all__ = [
    "Dsv4AttentionMetadata",
    "Dsv4DeviceAttentionInputs",
    "Dsv4DeviceState",
    "Dsv4SparseAttentionBackend",
    "allocate_dsv4_device_attention_inputs",
    "build_positions_per_token",
    "build_req_id_per_token",
    "dsv4_device_sparse_attention_kernel_inputs",
    "dsv4_vanilla_attn_fn",
    "dsv4_vanilla_sparse_attention_core",
    "dsv4_vanilla_update_kv_cache",
    "run_dsv4_device_sparse_attention",
    "run_dsv4_swa_global_slots",
]


class Dsv4SparseAttentionBackend:
    """DSV4 sparse attention backend: prep → write_kv → attention.

    The backend owns per-layer flat KV caches (``[num_slots, head_dim]``)
    and one set of per-bucket device metadata buffers. Three steps per
    forward:

    1. ``prepare(metadata, step_inputs)`` — upload scheduler metadata and
       run the unified SWA top-k kernel. Populates
       ``step_inputs.topk_global_t / topk_mask / topk_lens`` and
       ``step_inputs.slot_mapping / positions / req_id_per_token /
       block_tables / seq_lens / query_start_loc``.
    2. ``write_kv(layer_id, kv_new, step_inputs)`` — scatter fresh KV rows
       into the per-layer flat cache via the NKI scatter kernel.
    3. ``attention(layer_id, q_scaled_t, sink, step_inputs, output)`` —
       run the batched paged sparse attention kernel against the layer's
       flat cache.

    Tests and CPU dry-runs can drop in ``vanilla_mode=True`` to use the
    oracle on host instead of device kernels.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots_per_layer: int,
        head_dim: int,
        block_size: int,
        window_size: int,
        max_k: int,
        token_bucket: int,
        max_requests: int,
        max_blocks_per_request: int,
        alloc_device_scratch: Callable[..., Any] | None = None,
        alloc_device_cache: Callable[..., Any] | None = None,
        kv_caches: list[Any] | None = None,
        device_state: Dsv4DeviceState | None = None,
        artifacts_dir: str | None = None,
        vanilla_mode: bool = False,
        bucket_ladder: tuple[int, ...] | None = None,
        decode_bucket_ladder: tuple[int, ...] | None = None,
        fuse_swa_slots_in_attention: bool = False,
        _device_kernel_cls: Any | None = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_slots_per_layer <= 0:
            raise ValueError("num_slots_per_layer must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.num_layers = int(num_layers)
        self.num_slots_per_layer = int(num_slots_per_layer)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.window_size = int(window_size)
        self.max_k = int(max_k)
        self.token_bucket = int(token_bucket)
        self.max_requests = int(max_requests)
        self.max_blocks_per_request = int(max_blocks_per_request)
        self.artifacts_dir = artifacts_dir
        self.vanilla_mode = bool(vanilla_mode)
        self._fuse_swa_slots_in_attention = bool(
            fuse_swa_slots_in_attention and not vanilla_mode
        )
        self._use_fused_swa_attention_this_step = False
        self._device_kernel_cls = _device_kernel_cls
        self._swa_kernel_cache: dict[tuple, Any] = {}
        self._scatter_kernel_cache: dict[tuple, Any] = {}
        self._attn_kernel_cache: dict[tuple, Any] = {}
        self._topk_tail_kernel_cache: dict[tuple, Any] = {}
        self._device_state = device_state

        if device_state is not None:
            if kv_caches is not None:
                raise ValueError("pass either device_state or kv_caches, not both")
            if int(device_state.num_layers) != self.num_layers:
                raise ValueError(
                    f"device_state has {device_state.num_layers} layers, "
                    f"backend expects {self.num_layers}"
                )
            if int(device_state.head_dim) != self.head_dim:
                raise ValueError(
                    f"device_state head_dim={device_state.head_dim}, "
                    f"backend expects {self.head_dim}"
                )
            if int(device_state.num_slots_per_layer) != self.num_slots_per_layer:
                raise ValueError(
                    "device_state num_slots_per_layer="
                    f"{device_state.num_slots_per_layer}, backend expects "
                    f"{self.num_slots_per_layer}"
                )
            kv_caches = device_state.swa_kv_caches

        if self.vanilla_mode:
            if kv_caches is not None:
                if len(kv_caches) != self.num_layers:
                    raise ValueError(
                        f"kv_caches length {len(kv_caches)} != num_layers "
                        f"{self.num_layers}"
                    )
                self._kv_caches: list[Any] = list(kv_caches)
            else:
                self._kv_caches = [
                    np.zeros(
                        (self.num_slots_per_layer, self.head_dim),
                        dtype=np.float32,
                    )
                    for _ in range(self.num_layers)
                ]
            self._step_inputs: Dsv4DeviceAttentionInputs | None = None
        else:
            if kv_caches is not None:
                if len(kv_caches) != self.num_layers:
                    raise ValueError(
                        f"kv_caches length {len(kv_caches)} != num_layers "
                        f"{self.num_layers}"
                    )
                self._kv_caches = list(kv_caches)
            else:
                if alloc_device_cache is None:
                    raise ValueError(
                        "device mode requires either kv_caches or alloc_device_cache"
                    )
                self._kv_caches = [
                    alloc_device_cache(
                        (self.num_slots_per_layer, self.head_dim),
                        ml_dtypes.bfloat16,
                        name=f"dsv4_kv_cache_layer{li}",
                    )
                    for li in range(self.num_layers)
                ]
            if alloc_device_scratch is None:
                raise ValueError("device mode requires alloc_device_scratch")
            # Allocate step-input scratch per bucket so different request
            # sizes hit different (pre-compiled) NEFFs without a full re-
            # allocation on each call.
            #
            # Prefill pads on token count (`bucket_ladder` <- token_buckets);
            # decode pads on batch/request count (`decode_bucket_ladder` <-
            # request_buckets). prepare() routes by forward_mode. Scratch is
            # allocated over the UNION (one buffer set per distinct bucket size),
            # and token_bucket (the prepare() ceiling) is the union max so neither
            # mode can overflow. When no decode ladder is given, decode == prefill
            # (prior single-ladder behavior).
            ladder = (
                (self.token_bucket,)
                if bucket_ladder is None
                else tuple(sorted(set(int(b) for b in bucket_ladder)))
            )
            decode_ladder = (
                ladder
                if decode_bucket_ladder is None
                else tuple(sorted(set(int(b) for b in decode_bucket_ladder)))
            )
            if any(b <= 0 for b in ladder) or any(b <= 0 for b in decode_ladder):
                raise ValueError(
                    "bucket_ladder / decode_bucket_ladder entries must be "
                    f"positive, got prefill={ladder} decode={decode_ladder}"
                )
            union_ladder = tuple(sorted(set(ladder) | set(decode_ladder)))
            if max(union_ladder) != self.token_bucket:
                raise ValueError(
                    "largest bucket must equal token_bucket (union max="
                    f"{max(union_ladder)}, token_bucket={self.token_bucket})"
                )
            self._bucket_ladder = ladder
            self._decode_bucket_ladder = decode_ladder
            self._bucket_step_inputs: dict[int, Dsv4DeviceAttentionInputs] = {}
            for b in union_ladder:
                self._bucket_step_inputs[b] = allocate_dsv4_device_attention_inputs(
                    alloc_device_scratch,
                    token_bucket=b,
                    max_requests=self.max_requests,
                    max_blocks_per_request=self.max_blocks_per_request,
                    max_k=self.max_k,
                    prefix=f"backend_b{b}",
                )
            # Default to the largest bucket's scratch until prepare() picks.
            self._step_inputs = self._bucket_step_inputs[self.token_bucket]
            self._active_bucket = self.token_bucket

    @property
    def step_inputs(self) -> Dsv4DeviceAttentionInputs | None:
        return self._step_inputs

    @property
    def bucket_ladder(self) -> tuple[int, ...]:
        """Configured prefill token-bucket ladder (empty in vanilla mode)."""
        return tuple(getattr(self, "_bucket_ladder", ()) or ())

    @property
    def decode_bucket_ladder(self) -> tuple[int, ...]:
        """Configured decode request-bucket ladder (empty in vanilla mode)."""
        return tuple(getattr(self, "_decode_bucket_ladder", ()) or ())

    @property
    def active_bucket(self) -> int:
        """The bucket selected by the most recent ``prepare()`` (token bucket
        for prefill, request bucket for decode). Product kernel shapes must
        agree with this value."""
        return int(getattr(self, "_active_bucket", self.token_bucket))

    def uses_fused_swa_this_step(self) -> bool:
        """True when ``prepare()`` routed the step to the fused SWA KV-write
        path (the QKV kernel writes the SWA cache; ``write_kv`` is a no-op)."""
        return bool(getattr(self, "_use_fused_swa_attention_this_step", False))

    def step_inputs_for_bucket(self, bucket: int) -> Dsv4DeviceAttentionInputs | None:
        """Per-bucket device step inputs (None in vanilla mode / unknown bucket)."""
        buckets = getattr(self, "_bucket_step_inputs", None)
        if not isinstance(buckets, dict):
            return None
        return buckets.get(int(bucket))

    def bucket_for_tokens(self, total_tokens: int, *, is_decode: bool) -> int:
        """The bucket ``prepare()`` would select for ``total_tokens``.

        Single source of the routing logic so executor compile shapes cannot
        drift from the backend's runtime choice. Falls back to ``token_bucket``
        when no ladder fits.
        """
        # Decode pads on the request ladder; falls back to the prefill token
        # ladder when no decode ladder is configured.
        ladder = (
            self.decode_bucket_ladder
            if is_decode and self.decode_bucket_ladder
            else self.bucket_ladder
        )
        total = int(total_tokens)
        for b in ladder:
            if b >= total:
                return int(b)
        return int(self.token_bucket)

    def _require_step_inputs(self) -> Dsv4DeviceAttentionInputs:
        if self._step_inputs is None:
            raise RuntimeError(
                "Dsv4SparseAttentionBackend has no device step inputs "
                "(vanilla_mode=True)"
            )
        return self._step_inputs

    def kv_cache(self, layer_id: int) -> Any:
        return self._kv_caches[int(layer_id)]

    def prepare(
        self,
        metadata: Dsv4AttentionMetadata,
        *,
        upload_fn: Callable[[Any, np.ndarray], None] | None = None,
        extra_topk: np.ndarray | None = None,
        extra_topk_lens: np.ndarray | None = None,
    ) -> Dsv4AttentionMetadata:
        """Populate per-bucket device buffers from ``metadata`` + run SWA.

        For the device path ``upload_fn(tensor, host_array)`` writes
        ``host_array`` into ``tensor`` (typically a DeviceTensor.copy_from
        or an equivalent copy). In vanilla mode ``metadata`` is consumed
        directly by ``write_kv`` / ``attention`` without device buffers.
        """
        total = int(metadata.base.total_tokens)
        if total > self.token_bucket:
            raise ValueError(
                f"total_tokens={total} exceeds token_bucket={self.token_bucket}"
            )
        # Pick the smallest bucket >= total for device mode so we hit a
        # small NEFF when possible. Decode pads on batch (request_buckets);
        # prefill pads on token count (token_buckets) -- route by forward_mode.
        if not self.vanilla_mode and getattr(self, "_bucket_ladder", None):
            fm = getattr(metadata.base, "forward_mode", None)
            is_decode = fm is not None and int(fm) == int(FORWARD_MODE_DECODE)
            b = self.bucket_for_tokens(total, is_decode=is_decode)
            if b in self._bucket_step_inputs:
                self._step_inputs = self._bucket_step_inputs[b]
                self._active_bucket = b
        positions_host = metadata.positions
        if positions_host is None:
            positions_host = build_positions_per_token(
                metadata.base.query_start_loc,
                metadata.base.seq_lens,
            )
        positions_host = np.asarray(positions_host, dtype=np.int32).reshape(-1)
        req_id_host = build_req_id_per_token(metadata.base.query_start_loc)

        if self.vanilla_mode:
            from nkipy_serving.attention.deepseek_v4.kernels import (
                swa_global_slots_oracle,
            )

            swa_max_k = (
                self.max_k
                if extra_topk is None
                else self.max_k - int(extra_topk.shape[-1])
            )
            if swa_max_k <= 0:
                raise ValueError(
                    f"max_k={self.max_k} too small to hold extra_topk of "
                    f"width {0 if extra_topk is None else extra_topk.shape[-1]}"
                )
            swa_topk_t, _mask_unused, swa_lens = swa_global_slots_oracle(
                positions=positions_host,
                req_id_per_token=req_id_host,
                block_tables=metadata.base.block_tables,
                block_size=self.block_size,
                window_size=self.window_size,
                max_k=swa_max_k,
            )
            swa_topk = swa_topk_t.T.astype(np.int32, copy=False)
            if extra_topk is None:
                topk_indices = swa_topk
                topk_lens = swa_lens
            else:
                extra_arr = np.asarray(extra_topk, dtype=np.int32)
                if extra_arr.ndim != 2 or extra_arr.shape[0] != total:
                    raise ValueError(
                        f"extra_topk must be [{total}, k_extra], got {extra_arr.shape}"
                    )
                # Invalid entries must be -1; the oracle consumes that as
                # "missing" via gather_kv_and_mask.
                if extra_topk_lens is None:
                    extra_lens_arr = np.sum(
                        extra_arr >= 0,
                        axis=-1,
                    ).astype(np.int32)
                else:
                    extra_lens_arr = np.asarray(
                        extra_topk_lens,
                        dtype=np.int32,
                    ).reshape(-1)
                topk_indices = np.concatenate([swa_topk, extra_arr], axis=-1)
                topk_lens = (swa_lens + extra_lens_arr).astype(np.int32)
            refreshed = SparseAttentionMetadata(
                topk_indices=topk_indices,
                topk_lens=topk_lens,
                num_kv_positions=self.num_slots_per_layer,
                window_size=self.window_size,
                index_topk=(0 if extra_topk is None else int(extra_topk.shape[-1])),
                index_space=SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
            )
            return Dsv4AttentionMetadata(
                base=metadata.base,
                sparse=refreshed,
                positions=positions_host,
                state_owner_ids=metadata.state_owner_ids,
                dp_superstep=metadata.dp_superstep,
            )

        si = self._step_inputs
        if si is None:
            raise RuntimeError(
                "DSV4 sparse attention step inputs must be initialized before "
                "preparing device metadata"
            )
        self._use_fused_swa_attention_this_step = False

        def _replace(field_name: str, host: np.ndarray) -> None:
            """Replace ``si.<field_name>`` with a fresh DeviceTensor.

            ``DeviceTensor.numpy()`` returns a copy in the current
            runtime, so in-place writes via the host-side view do not
            propagate to the device. We upload by constructing a new
            DeviceTensor via ``from_numpy`` and patching the field on
            the frozen dataclass using ``object.__setattr__`` — same
            pattern the scratch allocator uses at init time.
            """
            cur = getattr(si, field_name)
            cur_shape = tuple(int(s) for s in cur.shape)
            if host.shape != cur_shape:
                raise ValueError(
                    f"upload shape mismatch for {field_name}: "
                    f"host={host.shape}, device={cur_shape}"
                )
            cur_dtype = cur.dtype
            if host.dtype != cur_dtype:
                host = host.astype(cur_dtype)
            new_tensor = get_device_tensor_cls().from_numpy(
                np.ascontiguousarray(host),
                name=getattr(cur, "name", field_name),
            )
            object.__setattr__(si, field_name, new_tensor)

        if upload_fn is None:

            def upload_fn_to_use(dev_tensor, host):
                return _replace(
                    tensor_to_step_field_name(si, dev_tensor),
                    host,
                )
        else:
            upload_fn_to_use = upload_fn

        tb = getattr(self, "_active_bucket", self.token_bucket)
        mr = self.max_requests
        mb = self.max_blocks_per_request

        def _pad(
            arr: np.ndarray,
            shape: tuple[int, ...],
            fill: int = 0,
        ) -> np.ndarray:
            out = np.full(shape, fill, dtype=arr.dtype)
            if arr.ndim == 1:
                out[: arr.shape[0]] = arr
            elif arr.ndim == 2:
                out[: arr.shape[0], : arr.shape[1]] = arr
            else:
                raise ValueError(f"cannot pad ndim={arr.ndim}")
            return out

        # Reserve the last slot as the "padding sink" so scatter writes to
        # padded rows don't corrupt live KV.
        pad_slot = self.num_slots_per_layer - 1

        slot_mapping = np.asarray(
            metadata.base.slot_mapping,
            dtype=np.int32,
        ).reshape(-1)
        seq_lens = np.asarray(metadata.base.seq_lens, dtype=np.int32).reshape(-1)
        qsl = np.asarray(metadata.base.query_start_loc, dtype=np.int32).reshape(-1)
        block_tables = np.asarray(metadata.base.block_tables, dtype=np.int32)
        bt_per_token_host = block_tables[req_id_host.astype(np.int64)]

        upload_fn_to_use(si.slot_mapping, _pad(slot_mapping, (tb,), fill=pad_slot))
        upload_fn_to_use(si.seq_lens, _pad(seq_lens.reshape(-1, 1), (mr, 1)))
        upload_fn_to_use(si.query_start_loc, _pad(qsl.reshape(-1, 1), (mr + 1, 1)))
        upload_fn_to_use(si.block_tables, _pad(block_tables, (mr, mb)))
        upload_fn_to_use(si.block_tables_per_token, _pad(bt_per_token_host, (tb, mb)))
        upload_fn_to_use(si.positions, _pad(positions_host.reshape(-1, 1), (tb, 1)))
        upload_fn_to_use(si.req_id_per_token, _pad(req_id_host.reshape(-1, 1), (tb, 1)))

        if self._fuse_swa_slots_in_attention and extra_topk is None:
            self._use_fused_swa_attention_this_step = True
        else:
            run_dsv4_swa_global_slots(
                step_inputs=si,
                block_size=self.block_size,
                window_size=self.window_size,
                artifacts_dir=self.artifacts_dir,
                _device_kernel_cls=self._device_kernel_cls,
                _kernel_cache=self._swa_kernel_cache,
            )
        if extra_topk is not None:
            extra_arr = np.asarray(extra_topk, dtype=np.int32)
            if extra_arr.ndim != 2 or extra_arr.shape[0] != total:
                raise ValueError(
                    f"extra_topk must be [{total}, k_extra], got {extra_arr.shape}"
                )
            k_extra = int(extra_arr.shape[-1])
            tail_start = self.max_k - k_extra
            if tail_start < 0:
                raise ValueError(f"max_k={self.max_k} < k_extra={k_extra}")
            # The SWA kernel fills all ``max_k`` rows; writing extras into
            # the K-tail overwrites them. To preserve SWA correctness we
            # require ``window_size <= tail_start`` so SWA never produced
            # valid entries past that cutoff.
            if self.window_size > tail_start:
                raise ValueError(
                    "union-mode device path requires window_size "
                    f"({self.window_size}) <= max_k - k_extra ({tail_start})"
                )
            valid = extra_arr >= 0
            safe = np.where(valid, extra_arr, pad_slot).astype(np.int32)
            if extra_topk_lens is None:
                extra_lens_arr = np.sum(valid, axis=-1).astype(np.int32)
            else:
                extra_lens_arr = np.asarray(
                    extra_topk_lens,
                    dtype=np.int32,
                ).reshape(-1)
                if extra_lens_arr.shape != (total,):
                    raise ValueError(
                        "extra_topk_lens must be [total_tokens], got "
                        f"{extra_lens_arr.shape}"
                    )
            safe_full = np.full((tb, k_extra), pad_slot, dtype=np.int32)
            safe_full[:total] = safe
            mask_dtype = dtype_like(si.topk_mask)
            mask_full = np.zeros((tb, k_extra), dtype=mask_dtype)
            mask_full[:total] = valid.astype(np.float32).astype(mask_dtype)
            lens_full = np.zeros((tb, 1), dtype=np.int32)
            lens_full[:total, 0] = extra_lens_arr

            from nkipy_serving.ops.deepseek_v4.topk_state import (
                run_topk_tail_insert_device,
            )

            DeviceTensor = get_device_tensor_cls()

            run_topk_tail_insert_device(
                topk_global_t=si.topk_global_t,
                topk_mask=si.topk_mask,
                topk_lens=si.topk_lens,
                safe_extra=DeviceTensor.from_numpy(
                    np.ascontiguousarray(safe_full),
                    name="dsv4_extra_topk",
                ),
                extra_mask=DeviceTensor.from_numpy(
                    np.ascontiguousarray(mask_full),
                    name="dsv4_extra_topk_mask",
                ),
                extra_lens=DeviceTensor.from_numpy(
                    np.ascontiguousarray(lens_full),
                    name="dsv4_extra_topk_lens",
                ),
                tail_start=tail_start,
                artifacts_dir=self.artifacts_dir,
                _device_kernel_cls=self._device_kernel_cls,
                _kernel_cache=self._topk_tail_kernel_cache,
            )
        return metadata

    def write_kv(
        self,
        layer_id: int,
        kv_new: Any,
        metadata: Dsv4AttentionMetadata,
    ) -> Any:
        """Scatter ``kv_new [total_tokens, head_dim]`` into the layer cache.

        ``kv_new`` may be a numpy array or an already-resident DeviceTensor.
        In device mode (and vanilla mode with a DeviceTensor cache), a
        DeviceTensor input is consumed directly without host round-trip.
        """
        cache = self._kv_caches[int(layer_id)]
        is_dev_kv = is_device_tensor(kv_new, require_numpy=True)
        if self.vanilla_mode:
            if hasattr(cache, "tensor_ref"):
                from nkipy_serving.attention.deepseek_v4.kernels import (
                    run_write_kv_slots_device,
                )

                DeviceTensor = get_device_tensor_cls()

                if is_dev_kv:
                    kv_shape = tuple(int(d) for d in kv_new.shape)
                    if len(kv_shape) != 2 or kv_shape[1] != self.head_dim:
                        raise ValueError(
                            f"kv_new must be [tokens, {self.head_dim}], got {kv_shape}"
                        )
                    kv_dev = kv_new
                else:
                    kv_arr = np.asarray(kv_new)
                    if kv_arr.ndim != 2 or kv_arr.shape[1] != self.head_dim:
                        raise ValueError(
                            f"kv_new must be [tokens, {self.head_dim}], "
                            f"got {kv_arr.shape}"
                        )
                    kv_dev = DeviceTensor.from_numpy(
                        np.ascontiguousarray(kv_arr.astype(ml_dtypes.bfloat16)),
                        name="vanilla_device_kv_new",
                    )
                run_write_kv_slots_device(
                    kv_cache=cache,
                    kv_new=kv_dev,
                    slot_mapping=DeviceTensor.from_numpy(
                        np.ascontiguousarray(
                            np.asarray(metadata.base.slot_mapping, dtype=np.int32),
                        ),
                        name="vanilla_device_slot_mapping",
                    ),
                    artifacts_dir=self.artifacts_dir,
                    _device_kernel_cls=self._device_kernel_cls,
                    _kernel_cache=self._scatter_kernel_cache,
                )
                return cache
            from nkipy_serving.attention.deepseek_v4.kernels import (
                write_kv_to_flat_cache_oracle,
            )

            kv_host = kv_new.numpy() if is_dev_kv else np.asarray(kv_new)
            write_kv_to_flat_cache_oracle(
                kv_new=kv_host,
                kv_cache=cache,
                slot_mapping=np.asarray(metadata.base.slot_mapping),
            )
            return cache
        from nkipy_serving.attention.deepseek_v4.kernels import (
            run_write_kv_slots_device,
        )

        active_bucket = getattr(self, "_active_bucket", self.token_bucket)
        if is_dev_kv:
            kv_shape = tuple(int(d) for d in kv_new.shape)
            if len(kv_shape) != 2 or kv_shape[1] != self.head_dim:
                raise ValueError(
                    f"kv_new must be [tokens, {self.head_dim}], got {kv_shape}"
                )
            if kv_shape[0] == active_bucket:
                kv_dev = kv_new
            else:
                slot_mapping = np.asarray(
                    metadata.base.slot_mapping,
                    dtype=np.int32,
                ).reshape(-1)
                n_new = int(kv_shape[0])
                if n_new > int(slot_mapping.shape[0]):
                    raise ValueError(
                        f"kv_new has {n_new} rows, but metadata only has "
                        f"{int(slot_mapping.shape[0])} slots"
                    )
                DeviceTensor = get_device_tensor_cls()

                run_write_kv_slots_device(
                    kv_cache=cache,
                    kv_new=kv_new,
                    slot_mapping=DeviceTensor.from_numpy(
                        np.ascontiguousarray(slot_mapping[:n_new]),
                        name="dsv4_unpadded_slot_mapping",
                    ),
                    artifacts_dir=self.artifacts_dir,
                    _device_kernel_cls=self._device_kernel_cls,
                    _kernel_cache=self._scatter_kernel_cache,
                )
                return cache
        if not is_dev_kv:
            # Pad kv_new to the compiled n_new == token_bucket so the
            # kernel NEFF has a single static shape. Padded rows scatter
            # to slots already clamped to a reserved zero-slot by
            # prepare()'s _pad, so the write is a redundant overwrite of
            # slot 0 with zeros. Callers relying on slot 0 should reserve
            # it explicitly.
            kv_host = np.asarray(kv_new)
            if kv_host.ndim != 2 or kv_host.shape[1] != self.head_dim:
                raise ValueError(
                    f"kv_new must be [tokens, {self.head_dim}], got {kv_host.shape}"
                )
            n_new = kv_host.shape[0]
            if n_new > active_bucket:
                raise ValueError(
                    f"kv_new has {n_new} rows, exceeds active bucket={active_bucket}"
                )
            if n_new < active_bucket:
                padded = np.zeros(
                    (active_bucket, self.head_dim),
                    dtype=kv_host.dtype,
                )
                padded[:n_new] = kv_host
                kv_host = padded
            kv_dev = self._as_device(kv_host, ml_dtypes.bfloat16, name="kv_new")
        run_write_kv_slots_device(
            kv_cache=cache,
            kv_new=kv_dev,
            slot_mapping=self._require_step_inputs().slot_mapping,
            artifacts_dir=self.artifacts_dir,
            _device_kernel_cls=self._device_kernel_cls,
            _kernel_cache=self._scatter_kernel_cache,
        )
        return cache

    def attention_ephemeral_paged(
        self,
        *,
        q: np.ndarray,  # [N_q, h, d]
        kv: np.ndarray,  # [N_kv, d]
        topk_indices: np.ndarray,  # [N_q, K] into kv rows, -1 = invalid
        sink: np.ndarray,  # [h]
        softmax_scale: float,
    ) -> np.ndarray:
        """Ephemeral paged sparse attention on device.

        Uploads the per-call ``kv`` buffer to a DeviceTensor, safe-clamps
        invalid topk indices, and runs the batched paged sparse-attention
        kernel. This avoids the host ``np.take`` that
        ``attention_ephemeral`` does via ``gather_kv_and_mask``.

        Shape contract:
        - ``q [N_q, h, d]`` fp32 on host
        - ``kv [N_kv, d]`` fp32/bf16 on host
        - ``topk_indices [N_q, K]`` int32 on host, -1 sentinels supported
        Returns ``[N_q, h, d]`` fp32.
        """
        import ml_dtypes as _ml

        from nkipy_serving.attention.deepseek_v4.kernels import (
            D_BLOCK,
            K_TILE,
            P_MAX,
            run_sparse_attention_paged_device,
        )

        if self.vanilla_mode:
            # Vanilla path — delegate to the oracle for parity.
            return self.attention_ephemeral(
                q=q,
                kv=kv,
                topk_indices=topk_indices,
                sink=sink,
                softmax_scale=softmax_scale,
            )
        DeviceTensor = get_device_tensor_cls()

        q_arr = np.asarray(q, dtype=np.float32)
        kv_arr = np.asarray(kv)
        topk = np.asarray(topk_indices, dtype=np.int32)
        if q_arr.ndim != 3 or kv_arr.ndim != 2 or topk.ndim != 2:
            raise ValueError(
                f"bad shapes: q={q_arr.shape}, kv={kv_arr.shape}, topk={topk.shape}"
            )
        N_q, h, d = q_arr.shape
        if h > P_MAX:
            raise ValueError(f"h={h} must be <= {P_MAX}")
        if d % D_BLOCK:
            raise NotImplementedError(f"d={d} not a multiple of {D_BLOCK}")
        # Pad K to a multiple of K_TILE (kernel requires static-K shape).
        K_raw = int(topk.shape[1])
        K = ((K_raw + K_TILE - 1) // K_TILE) * K_TILE
        if K != K_raw:
            pad = np.full((N_q, K - K_raw), -1, dtype=np.int32)
            topk = np.concatenate([topk, pad], axis=-1)

        # Safe-clamp -1 to 0; mask kills them.
        valid = topk >= 0
        safe = np.where(valid, topk, 0).astype(np.int32)
        topk_T = np.ascontiguousarray(safe.T)  # [K, N_q]
        mask_bf = np.ascontiguousarray(
            valid.astype(np.float32).astype(_ml.bfloat16),
        )  # [N_q, K]

        # Pre-scale q by softmax_scale on host.
        q_scaled = q_arr * np.float32(softmax_scale)
        q_T = np.ascontiguousarray(
            q_scaled.astype(_ml.bfloat16).transpose(0, 2, 1),
        )  # [N_q, d, h]
        sink_2d = np.ascontiguousarray(
            np.asarray(sink, dtype=np.float32).reshape(1, -1),
        )
        kv_bf = (
            np.ascontiguousarray(kv_arr)
            if kv_arr.dtype == _ml.bfloat16
            else np.ascontiguousarray(kv_arr.astype(_ml.bfloat16))
        )

        q_dev = DeviceTensor.from_numpy(q_T, name="q_T")
        kv_dev = DeviceTensor.from_numpy(kv_bf, name="kv_hbm")
        topk_dev = DeviceTensor.from_numpy(topk_T, name="topk_T")
        mask_dev = DeviceTensor.from_numpy(mask_bf, name="mask")
        sink_dev = DeviceTensor.from_numpy(sink_2d, name="sink")
        out_host = np.zeros((N_q, h, d), dtype=np.float32)
        out_dev = DeviceTensor.from_numpy(out_host, name="ephemeral_out")

        run_sparse_attention_paged_device(
            q_scaled_t=q_dev,
            kv_hbm=kv_dev,
            topk_t=topk_dev,
            mask=mask_dev,
            sink=sink_dev,
            output=out_dev,
            artifacts_dir=self.artifacts_dir,
            _device_kernel_cls=self._device_kernel_cls,
            _kernel_cache=self._attn_kernel_cache,
        )
        return out_dev.numpy()

    def attention_ephemeral_paged_two_source(
        self,
        *,
        q: np.ndarray | None = None,  # [N_q, h, d] fp32 host
        q_scaled_t: Any | None = None,  # [N_q, d, h] bf16 DeviceTensor
        q_shape: tuple[int, int, int] | None = None,
        kv_primary: np.ndarray,  # [B * primary_len, d]
        kv_secondary: Any,  # [B * secondary_stride, d]
        topk_indices: np.ndarray | None = None,  # [N_q, K] int32 host
        topk_t_dev: Any | None = None,  # [K, N_q] int32 DeviceTensor
        mask_dev: Any | None = None,  # [N_q, K] bf16 DeviceTensor
        owner_ids: np.ndarray,  # [N_q]
        owner_ids_dev: Any | None = None,  # [N_q] int32 DeviceTensor
        primary_owner_ids: np.ndarray | None = None,  # [N_q]
        primary_owner_ids_dev: Any | None = None,  # [N_q] int32 DeviceTensor
        primary_len: int,
        secondary_stride: int,
        primary_prefix_len: int | None = None,
        sink: np.ndarray,  # [h]
        softmax_scale: float,
        output: Any | None = None,
        return_device: bool = False,
    ) -> Any:
        """Ephemeral sparse attention over SWA and compressed KV sources.

        This keeps DSV4 compressed-attention indices request-local. The kernel
        maps ``idx < primary_len`` to the per-call SWA/prefill source and
        ``idx >= primary_len`` to the persistent compressed source, avoiding a
        host-side concat/upload of compressed KV.
        """
        import ml_dtypes as _ml

        from nkipy_serving.attention.deepseek_v4.kernels import (
            D_BLOCK,
            K_TILE,
            P_MAX,
            gather_two_source_kv_and_mask,
            run_sparse_attention_paged_two_source_device,
            sparse_attention_oracle,
        )

        if q_scaled_t is not None:
            if q_shape is None:
                raise ValueError("q_shape is required alongside q_scaled_t")
            q_arr = None
            N_q, h, d = (int(x) for x in q_shape)
        else:
            if q is None:
                raise ValueError("either q or (q_scaled_t, q_shape) is required")
            q_arr = np.asarray(q, dtype=np.float32)
            if q_arr.ndim != 3:
                raise ValueError(f"q must be [N_q, h, d], got {q_arr.shape}")
            N_q, h, d = q_arr.shape
        primary_shape = tuple(int(dim) for dim in getattr(kv_primary, "shape"))
        secondary_shape = tuple(int(dim) for dim in getattr(kv_secondary, "shape"))
        topk_device_path = topk_t_dev is not None or mask_dev is not None
        topk: np.ndarray | None
        if topk_device_path:
            if topk_t_dev is None or mask_dev is None:
                raise ValueError("topk_t_dev and mask_dev must be passed together")
            topk = None
        else:
            if topk_indices is None:
                raise ValueError(
                    "either topk_indices or (topk_t_dev, mask_dev) is required"
                )
            topk = np.asarray(topk_indices, dtype=np.int32)
            if topk.ndim != 2:
                raise ValueError(f"topk must be 2-D, got {topk.shape}")
        owners = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        primary_owners = (
            owners
            if primary_owner_ids is None
            else np.asarray(primary_owner_ids, dtype=np.int32).reshape(-1)
        )
        primary_len = int(primary_len)
        secondary_stride = int(secondary_stride)
        if len(primary_shape) != 2:
            raise ValueError(f"bad kv_primary shape: {primary_shape}")
        if owners.shape != (N_q,):
            raise ValueError(f"owner_ids must be [{N_q}], got {owners.shape}")
        if primary_owners.shape != (N_q,):
            raise ValueError(
                f"primary_owner_ids must be [{N_q}], got {primary_owners.shape}"
            )
        if len(secondary_shape) != 2:
            raise ValueError(f"kv_secondary must be 2-D, got {secondary_shape}")
        if primary_shape[1] != d:
            raise ValueError(
                f"kv_primary head_dim={primary_shape[1]} must match q d={d}"
            )
        if secondary_shape[1] != d:
            raise ValueError(
                f"kv_secondary head_dim={secondary_shape[1]} must match q d={d}"
            )
        if primary_len <= 0 or secondary_stride <= 0:
            raise ValueError("primary_len and secondary_stride must be positive")
        if owners.size:
            max_owner = int(owners.max())
            max_primary_owner = int(primary_owners.max())
            if primary_shape[0] < (max_primary_owner + 1) * primary_len:
                raise ValueError(
                    "kv_primary does not cover owner-local rows: "
                    f"rows={primary_shape[0]}, owner={max_primary_owner}, "
                    f"primary_len={primary_len}"
                )
            if secondary_shape[0] < (max_owner + 1) * secondary_stride:
                raise ValueError(
                    "kv_secondary does not cover owner-local rows: "
                    f"rows={secondary_shape[0]}, owner={max_owner}, "
                    f"secondary_stride={secondary_stride}"
                )

        if self.vanilla_mode:
            if return_device:
                raise ValueError("return_device=True requires device mode")
            if q_arr is None:
                q_arr = np.asarray(
                    q_scaled_t.numpy() / np.float32(softmax_scale),
                    dtype=np.float32,
                ).transpose(0, 2, 1)
            if topk_device_path:
                # Oracle wants [N_q, K] int32 host; reconstruct from the
                # prep fragment's outputs (safe topk_T + validity mask).
                topk_t_host = topk_t_dev.numpy()  # [K, N_q] int32
                mask_host = mask_dev.numpy()  # [N_q, K] bf16
                safe_h = topk_t_host.T
                valid_h = np.asarray(mask_host, dtype=np.float32) > 0
                topk = np.where(valid_h, safe_h, -1).astype(np.int32)
            primary_host = (
                kv_primary.numpy() if hasattr(kv_primary, "numpy") else kv_primary
            )
            secondary_host = (
                kv_secondary.numpy() if hasattr(kv_secondary, "numpy") else kv_secondary
            )
            gathered, valid_mask = gather_two_source_kv_and_mask(
                kv_primary=np.asarray(primary_host),
                kv_secondary=np.asarray(secondary_host),
                topk_idxs=topk,
                owner_ids=owners,
                primary_owner_ids=primary_owners,
                primary_len=primary_len,
                secondary_stride=secondary_stride,
            )
            return sparse_attention_oracle(
                q_arr,
                gathered,
                valid_mask,
                np.asarray(sink, dtype=np.float32),
                float(softmax_scale),
            ).astype(np.float32)

        if h > P_MAX:
            raise ValueError(f"h={h} must be <= {P_MAX}")
        if d % D_BLOCK:
            raise NotImplementedError(f"d={d} not a multiple of {D_BLOCK}")

        DeviceTensor = get_device_tensor_cls()

        if topk_device_path:
            topk_dev = topk_t_dev
            mask_device_tensor = mask_dev
        else:
            K_raw = int(topk.shape[1])
            K = ((K_raw + K_TILE - 1) // K_TILE) * K_TILE
            if K != K_raw:
                pad = np.full((N_q, K - K_raw), -1, dtype=np.int32)
                topk = np.concatenate([topk, pad], axis=-1)
            valid = topk >= 0
            safe = np.where(valid, topk, 0).astype(np.int32)
            topk_T = np.ascontiguousarray(safe.T)
            mask_bf = np.ascontiguousarray(
                valid.astype(np.float32).astype(_ml.bfloat16),
            )
            topk_dev = DeviceTensor.from_numpy(topk_T, name="topk_T")
            mask_device_tensor = DeviceTensor.from_numpy(mask_bf, name="mask")

        sink_is_device = hasattr(sink, "shape") and not isinstance(sink, np.ndarray)
        if sink_is_device:
            sink_shape = tuple(int(dim) for dim in getattr(sink, "shape"))
            if sink_shape != (1, h):
                raise ValueError(f"device sink must be [1,{h}], got {sink_shape}")
            sink_dev = sink
        else:
            sink_2d = np.ascontiguousarray(
                np.asarray(sink, dtype=np.float32).reshape(1, -1),
            )
            sink_dev = DeviceTensor.from_numpy(sink_2d, name="sink")

        if q_scaled_t is not None:
            q_dev = q_scaled_t
        else:
            q_scaled = q_arr * np.float32(softmax_scale)
            q_T = np.ascontiguousarray(
                q_scaled.astype(_ml.bfloat16).transpose(0, 2, 1),
            )
            q_dev = DeviceTensor.from_numpy(q_T, name="q_T")
        primary_dev = self._as_device(
            kv_primary,
            _ml.bfloat16,
            name="kv_primary",
        )
        secondary_dev = self._as_device(
            kv_secondary,
            _ml.bfloat16,
            name="kv_secondary",
        )
        if owner_ids_dev is None:
            owner_dev = DeviceTensor.from_numpy(
                np.ascontiguousarray(owners.astype(np.int32)),
                name="owner_ids",
            )
        else:
            owner_shape = tuple(int(dim) for dim in getattr(owner_ids_dev, "shape", ()))
            if owner_shape != (N_q,):
                raise ValueError(f"owner_ids_dev must be [{N_q}], got {owner_shape}")
            owner_dev = owner_ids_dev
        if primary_owner_ids is None and primary_owner_ids_dev is None:
            primary_owner_dev = None
        elif primary_owner_ids_dev is None:
            primary_owner_dev = DeviceTensor.from_numpy(
                np.ascontiguousarray(primary_owners.astype(np.int32)),
                name="primary_owner_ids",
            )
        else:
            primary_owner_shape = tuple(
                int(dim) for dim in getattr(primary_owner_ids_dev, "shape", ())
            )
            if primary_owner_shape != (N_q,):
                raise ValueError(
                    f"primary_owner_ids_dev must be [{N_q}], got {primary_owner_shape}"
                )
            primary_owner_dev = primary_owner_ids_dev
        if output is None:
            out_host = np.zeros((N_q, h, d), dtype=np.float32)
            out_dev = DeviceTensor.from_numpy(
                out_host,
                name="ephemeral_two_source_out",
            )
        else:
            output_shape = tuple(int(dim) for dim in getattr(output, "shape", ()))
            expected_shape = (N_q, h, d)
            if output_shape != expected_shape:
                raise ValueError(
                    "two-source attention output must be "
                    f"{expected_shape}, got {output_shape}"
                )
            out_dev = output

        run_sparse_attention_paged_two_source_device(
            q_scaled_t=q_dev,
            kv_primary=primary_dev,
            kv_secondary=secondary_dev,
            topk_t=topk_dev,
            mask=mask_device_tensor,
            owner_ids=owner_dev,
            primary_owner_ids=primary_owner_dev,
            sink=sink_dev,
            output=out_dev,
            primary_len=primary_len,
            secondary_stride=secondary_stride,
            primary_prefix_len=primary_prefix_len,
            artifacts_dir=self.artifacts_dir,
            _device_kernel_cls=self._device_kernel_cls,
            _kernel_cache=self._attn_kernel_cache,
        )
        if return_device:
            return out_dev
        return out_dev.numpy()

    def attention_ephemeral(
        self,
        *,
        q: np.ndarray,  # [N_q, h, d]
        kv: np.ndarray,  # [N_kv, d]
        topk_indices: np.ndarray,  # [N_q, K] request-local, -1 = invalid
        sink: np.ndarray,  # [h]
        softmax_scale: float,
    ) -> np.ndarray:
        """Sparse attention against a per-call KV buffer.

        This path is the device replacement for the eager ``_sparse_attn``
        Python loop. It does not use the backend's persistent per-layer
        KV cache — instead the caller provides a fresh ``kv`` matrix (as
        compressed layers do today, via ``state.kv_cache`` + compressed
        KV concat) and request-local indices.

        Returns ``[N_q, h, d]`` fp32.
        """
        from nkipy_serving.attention.deepseek_v4.kernels import (
            K_TILE,
            sparse_attention_host_gather,
        )

        q_arr = np.asarray(q)
        kv_arr = np.asarray(kv)
        topk = np.asarray(topk_indices, dtype=np.int32)
        if q_arr.ndim != 3 or kv_arr.ndim != 2 or topk.ndim != 2:
            raise ValueError(
                f"bad shapes: q={q_arr.shape}, kv={kv_arr.shape}, topk={topk.shape}"
            )
        if topk.shape[0] != q_arr.shape[0]:
            raise ValueError(f"N_q mismatch: q={q_arr.shape[0]}, topk={topk.shape[0]}")

        if self.vanilla_mode:
            # Oracle path — same as eager ``_sparse_attn`` semantics.
            from nkipy_serving.attention.deepseek_v4.kernels import (
                gather_kv_and_mask,
                sparse_attention_oracle,
            )

            gathered, valid_mask = gather_kv_and_mask(kv_arr, topk)
            return sparse_attention_oracle(
                q_arr.astype(np.float32),
                gathered,
                valid_mask,
                np.asarray(sink, dtype=np.float32),
                float(softmax_scale),
            ).astype(np.float32)

        # Device path reuses the host-gather wrapper, which pads K to a
        # multiple of K_TILE and runs the batched multi-K kernel. The wrapper
        # gathers on host because no persistent KV buffer exists; that matches
        # the ephemeral contract.
        K_max = int(topk.shape[1])
        target_K = ((K_max + K_TILE - 1) // K_TILE) * K_TILE
        if target_K != K_max:
            pad = np.full(
                (topk.shape[0], target_K - K_max),
                -1,
                dtype=np.int32,
            )
            topk = np.concatenate([topk, pad], axis=-1)
        return sparse_attention_host_gather(
            q_arr.astype(np.float32),
            kv_arr.astype(np.float32),
            topk,
            np.asarray(sink, dtype=np.float32),
            float(softmax_scale),
            use_device=True,
            artifacts_dir=self.artifacts_dir,
        )

    def attention(
        self,
        layer_id: int,
        *,
        q: Any | None = None,
        q_scaled_t: Any | None = None,
        sink: Any,
        metadata: Dsv4AttentionMetadata,
        softmax_scale: float,
        output: Any | None = None,
    ) -> Any:
        """Run sparse attention against the layer's KV cache.

        Vanilla mode: ``q [total_tokens, h, d]`` host ndarray.
        Device mode: either ``q_scaled_t`` preallocated DeviceTensor
        (caller owns the transpose+scale) or ``q`` as a host ndarray
        (we upload per-call).
        """
        cache = self._kv_caches[int(layer_id)]
        if self.vanilla_mode:
            if q is None:
                raise ValueError("vanilla mode requires q (host ndarray)")
            cache_host = cache.numpy() if hasattr(cache, "numpy") else cache
            return dsv4_vanilla_sparse_attention_core(
                np.asarray(q),
                cache_host,
                metadata,
                np.asarray(sink),
                float(softmax_scale),
            )
        if q_scaled_t is None:
            if q is None:
                raise ValueError("device mode requires q or q_scaled_t")
            q_scaled_t, output_dev = self._upload_q_and_output(
                q,
                softmax_scale,
                output,
            )
            output = output_dev
        elif output is None:
            raise ValueError("preallocated output required with q_scaled_t")

        sink_dev = self._as_device(sink, np.float32, name="sink", shape_2d=True)
        step_inputs = self._require_step_inputs()
        if self._use_fused_swa_attention_this_step:
            from nkipy_serving.attention.deepseek_v4.kernels import (
                run_sparse_attention_paged_swa_device,
            )

            run_sparse_attention_paged_swa_device(
                q_scaled_t=q_scaled_t,
                kv_hbm=cache,
                positions=step_inputs.positions,
                block_tables_per_token=step_inputs.block_tables_per_token,
                sink=sink_dev,
                output=output,
                block_size=self.block_size,
                window_size=self.window_size,
                max_k=self.max_k,
                artifacts_dir=self.artifacts_dir,
                _device_kernel_cls=self._device_kernel_cls,
                _kernel_cache=self._attn_kernel_cache,
            )
        else:
            from nkipy_serving.attention.deepseek_v4.kernels import (
                run_sparse_attention_paged_device,
            )

            run_sparse_attention_paged_device(
                q_scaled_t=q_scaled_t,
                kv_hbm=cache,
                topk_t=step_inputs.topk_global_t,
                mask=step_inputs.topk_mask,
                sink=sink_dev,
                output=output,
                artifacts_dir=self.artifacts_dir,
                _device_kernel_cls=self._device_kernel_cls,
                _kernel_cache=self._attn_kernel_cache,
            )
        return output

    def _as_device(
        self,
        value: Any,
        dtype: Any,
        *,
        name: str,
        shape_2d: bool = False,
    ) -> Any:
        if is_device_tensor(value, require_numpy=True):
            return value
        DeviceTensor = get_device_tensor_cls()
        arr = np.asarray(value)
        if arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        if shape_2d and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return DeviceTensor.from_numpy(np.ascontiguousarray(arr), name=name)

    def _upload_q_and_output(
        self,
        q: np.ndarray,
        softmax_scale: float,
        output: Any | None,
    ) -> tuple[Any, Any]:
        DeviceTensor = get_device_tensor_cls()
        qarr = np.asarray(q, dtype=np.float32) * np.float32(softmax_scale)
        bf = ml_dtypes.bfloat16
        if qarr.ndim != 3:
            raise ValueError(f"q must be [tokens, h, d], got {qarr.shape}")
        tokens, h, d = qarr.shape
        # Pad to the active bucket so kernel NEFF shape is stable per bucket.
        target = getattr(self, "_active_bucket", self.token_bucket)
        if tokens > target:
            raise ValueError(f"q has {tokens} rows, exceeds active bucket={target}")
        if tokens < target:
            padded = np.zeros((target, h, d), dtype=qarr.dtype)
            padded[:tokens] = qarr
            qarr = padded
            tokens = target
        q_T = np.ascontiguousarray(
            qarr.astype(bf).transpose(0, 2, 1),
        )
        q_dev = DeviceTensor.from_numpy(q_T, name="q_T")
        if output is None:
            out_host = np.zeros((tokens, h, d), dtype=np.float32)
            output = DeviceTensor.from_numpy(out_host, name="paged_out")
        return q_dev, output
