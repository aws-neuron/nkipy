"""Product support-kernel warmup helpers for DSV4."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.attention.deepseek_v4.kernels import (
    K_TILE,
    run_sparse_attention_paged_swa_device,
    run_sparse_attention_paged_two_source_device,
    run_write_kv_owner_window_device,
    run_write_kv_slots_device,
)
from nkipy_serving.models._device_utils import _get_device_tensor_cls
from nkipy_serving.ops.deepseek_v4.compressor_state import (
    run_prefill_pool_from_slab_device,
    run_write_kv_score_state_device,
    run_write_swa_dual_kv_score_state_device,
    run_write_swa_kv_score_state_owner_clen_device,
)


def _normalize_two_source_primary_prefix_len(primary_prefix_len: int) -> int:
    """Mirror the two-source kernel's cache-key normalization."""

    prefix = int(primary_prefix_len)
    if 0 < prefix < int(K_TILE):
        return 0
    return prefix


def _bucketed_two_source_primary_prefix_variants(
    *,
    window_size: int,
    k_padded: int,
) -> tuple[int, ...]:
    """Prefix variants serving can use for bucketed prefill two-source attention."""

    max_prefix = min(int(window_size), int(k_padded))
    if max_prefix <= 0:
        return ()
    return tuple(
        sorted(
            {
                _normalize_two_source_primary_prefix_len(0),
                _normalize_two_source_primary_prefix_len(max_prefix),
            }
        )
    )


def _bucketed_two_source_warmup_specs(
    token_buckets: tuple[int, ...] | list[int],
    *,
    exact_query_rows: tuple[int, ...] | list[int] = (),
    window_size: int,
    ratio: int,
    k_tile: int,
    index_topk: int = 0,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(query_rows, primary_len, k_padded)`` warmup specs."""

    win = int(window_size)
    ratio_i = int(ratio)
    tile = int(k_tile)
    index_topk_i = int(index_topk)
    if win <= 0 or ratio_i <= 0 or tile <= 0:
        return ()

    specs: set[tuple[int, int, int]] = set()
    short_k = ((win + tile - 1) // tile) * tile
    for rows in sorted({int(row) for row in exact_query_rows if int(row) > 0}):
        if rows <= win:
            specs.add((int(rows), win, int(short_k)))

    for bucket in sorted({int(bucket) for bucket in token_buckets if int(bucket) > 0}):
        rows = int(bucket)
        if rows <= win:
            continue
        comp_width = max(1, rows // ratio_i)
        sparse_width = min(index_topk_i, comp_width) if index_topk_i > 0 else comp_width
        k_raw = min(rows, win) + int(sparse_width)
        k_padded = ((k_raw + tile - 1) // tile) * tile
        specs.add((rows, rows, k_padded))

    return tuple(sorted(specs))


def _bucketed_state_write_warmup_rows(
    buckets: tuple[int, ...] | list[int],
    *,
    max_rows: int,
) -> tuple[int, ...]:
    """Rows that bucketed state-write warmup must compile."""

    requested = sorted({int(bucket) for bucket in buckets if int(bucket) > 0})
    max_rows_i = int(max_rows)
    if max_rows_i <= 0:
        return ()

    rows = {min(int(bucket), max_rows_i) for bucket in requested}
    rows.add(max_rows_i)

    # Runtime uses bucket rows when QKV exposes a bucket-backed tensor. Some
    # QKV/flatten aliases are exact active-request tensors, so compile those
    # short exact counts up to the smallest configured state-write bucket too.
    if requested:
        short_exact_limit = min(int(requested[0]), max_rows_i)
        rows.update(range(1, short_exact_limit + 1))

    return tuple(sorted(row for row in rows if row > 0))


class Dsv4ProductSupportKernelWarmupMixin:
    """Precompile device support kernels that serving may call outside graphs."""

    def seal_blockwise_moe_precompiled_kernels(self) -> None:
        seal = getattr(self.blockwise_moe_state, "seal_precompiled_kernels", None)
        if callable(seal):
            seal()

    def precompile_bucketed_prefill_swa_attention(
        self,
        token_buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile fused bucketed SWA sparse-attention kernels."""

        backend = getattr(self, "attention_backend", None)
        device_state = getattr(self, "device_state", None)
        runtime_surface = getattr(self, "runtime_surface", None)
        if backend is None or device_state is None or runtime_surface is None:
            return
        if bool(getattr(backend, "vanilla_mode", False)):
            return
        if not bool(getattr(backend, "_fuse_swa_slots_in_attention", False)):
            return

        requested_buckets = sorted(
            {int(bucket) for bucket in token_buckets if int(bucket) > 0}
        )
        if not requested_buckets:
            return

        DeviceTensor = _get_device_tensor_cls()
        kernel_cache = getattr(backend, "_attn_kernel_cache", None)
        device_kernel_cls = getattr(backend, "_device_kernel_cls", None)
        artifacts_dir = getattr(backend, "artifacts_dir", None) or getattr(
            self,
            "build_dir",
            None,
        )
        step_inputs_by_bucket = getattr(backend, "_bucket_step_inputs", {})
        block_size = int(getattr(backend, "block_size", 0) or 0)
        window_size = int(getattr(backend, "window_size", 0) or 0)
        max_k = int(getattr(backend, "max_k", 0) or 0)
        if block_size <= 0 or window_size <= 0 or max_k <= 0:
            return

        seen: set[tuple[Any, ...]] = set()
        for layer_id, block in enumerate(getattr(runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            n_heads = int(getattr(attn, "n_heads", 0) or 0)
            head_dim = int(getattr(attn, "head_dim", 0) or 0)
            if n_heads <= 0 or head_dim <= 0:
                continue
            try:
                layer_state = device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            swa_kv_cache = getattr(layer_state, "swa_kv_cache", None)
            kv_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ()))
            if len(kv_shape) != 2 or int(kv_shape[1]) != head_dim:
                continue

            sink = DeviceTensor.from_numpy(
                np.zeros((1, n_heads), dtype=np.float32),
                name=f"dsv4_warmup_swa_attention_sink_l{int(layer_id)}",
            )
            for bucket in requested_buckets:
                rows = int(bucket)
                step_inputs = None
                if isinstance(step_inputs_by_bucket, dict):
                    step_inputs = step_inputs_by_bucket.get(rows)
                if step_inputs is None:
                    step_inputs = getattr(backend, "_step_inputs", None)
                    if (
                        step_inputs is not None
                        and int(getattr(step_inputs, "token_bucket", -1)) != rows
                    ):
                        step_inputs = None
                if step_inputs is None:
                    continue

                positions = getattr(step_inputs, "positions", None)
                block_tables_per_token = getattr(
                    step_inputs,
                    "block_tables_per_token",
                    None,
                )
                pos_shape = tuple(int(dim) for dim in getattr(positions, "shape", ()))
                bt_shape = tuple(
                    int(dim) for dim in getattr(block_tables_per_token, "shape", ())
                )
                if pos_shape != (rows, 1) or len(bt_shape) != 2 or bt_shape[0] != rows:
                    continue

                key = (
                    int(rows),
                    int(head_dim),
                    int(n_heads),
                    kv_shape,
                    str(getattr(swa_kv_cache, "dtype", "")),
                    pos_shape,
                    str(getattr(positions, "dtype", "")),
                    bt_shape,
                    str(getattr(block_tables_per_token, "dtype", "")),
                    int(block_size),
                    int(window_size),
                    int(max_k),
                )
                if key in seen:
                    continue
                seen.add(key)

                q_t = DeviceTensor.from_numpy(
                    np.zeros((rows, head_dim, n_heads), dtype=ml_dtypes.bfloat16),
                    name=f"dsv4_warmup_swa_attention_q_t{rows}",
                )
                output = DeviceTensor.from_numpy(
                    np.zeros((rows, n_heads, head_dim), dtype=np.float32),
                    name=f"dsv4_warmup_swa_attention_out_t{rows}",
                )
                run_sparse_attention_paged_swa_device(
                    q_scaled_t=q_t,
                    kv_hbm=swa_kv_cache,
                    positions=positions,
                    block_tables_per_token=block_tables_per_token,
                    sink=sink,
                    output=output,
                    block_size=int(block_size),
                    window_size=int(window_size),
                    max_k=int(max_k),
                    artifacts_dir=artifacts_dir,
                    _device_kernel_cls=device_kernel_cls,
                    _kernel_cache=kernel_cache,
                )

    def precompile_bucketed_prefill_two_source_attention(
        self,
        token_buckets: tuple[int, ...] | list[int],
        *,
        exact_query_rows: tuple[int, ...] | list[int] = (),
    ) -> None:
        """Compile bucketed long-prefill two-source sparse-attention kernels."""

        backend = getattr(self, "attention_backend", None)
        device_state = getattr(self, "device_state", None)
        runtime_surface = getattr(self, "runtime_surface", None)
        if backend is None or device_state is None or runtime_surface is None:
            return
        if bool(getattr(backend, "vanilla_mode", False)):
            return

        requested_buckets = sorted(
            {int(bucket) for bucket in token_buckets if int(bucket) > 0}
        )
        if not requested_buckets:
            return

        DeviceTensor = _get_device_tensor_cls()
        kernel_cache = getattr(backend, "_attn_kernel_cache", None)
        device_kernel_cls = getattr(backend, "_device_kernel_cls", None)
        artifacts_dir = getattr(backend, "artifacts_dir", None) or getattr(
            self,
            "build_dir",
            None,
        )
        seen: set[tuple[Any, ...]] = set()

        for layer_id, block in enumerate(getattr(runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            compressor = getattr(attn, "compressor", None)
            ratio = int(
                getattr(attn, "compress_ratio", 0)
                or getattr(compressor, "compress_ratio", 0)
                or 0
            )
            if attn is None or compressor is None or ratio <= 1:
                continue
            indexer = getattr(attn, "indexer", None)
            index_topk = int(
                getattr(indexer, "index_topk", 0) or getattr(attn, "index_topk", 0) or 0
            )
            window_size = int(getattr(attn, "window_size", 0) or 0)
            n_heads = int(getattr(attn, "n_heads", 0) or 0)
            head_dim = int(getattr(attn, "head_dim", 0) or 0)
            if window_size <= 0 or n_heads <= 0 or head_dim <= 0:
                continue
            try:
                layer_state = device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            comp_state = getattr(layer_state, "compressor", None)
            if comp_state is None:
                continue
            kv_secondary = getattr(comp_state, "compressed_kv_cache", None)
            secondary_shape = tuple(
                int(dim) for dim in getattr(kv_secondary, "shape", ())
            )
            if len(secondary_shape) != 2 or int(secondary_shape[1]) != head_dim:
                continue
            spec = getattr(comp_state, "spec", None)
            secondary_stride = int(getattr(spec, "max_compressed_len", 0) or 0)
            if secondary_stride <= 0:
                secondary_stride = int(
                    getattr(comp_state, "max_compressed_len", 0) or 0
                )
            if secondary_stride <= 0:
                continue
            swa_kv_cache = getattr(layer_state, "swa_kv_cache", None)
            swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ()))

            for query_rows, primary_len, k_padded in _bucketed_two_source_warmup_specs(
                requested_buckets,
                exact_query_rows=exact_query_rows,
                window_size=int(window_size),
                ratio=int(ratio),
                k_tile=int(K_TILE),
                index_topk=int(index_topk),
            ):
                rows = int(query_rows)
                use_swa_primary = int(primary_len) == int(window_size) and rows <= int(
                    window_size
                )
                if use_swa_primary and (
                    len(swa_shape) != 2 or int(swa_shape[1]) != head_dim
                ):
                    continue
                prefix_variants = (
                    (0,)
                    if use_swa_primary and int(rows) < int(K_TILE)
                    else _bucketed_two_source_primary_prefix_variants(
                        window_size=int(window_size),
                        k_padded=int(k_padded),
                    )
                )
                for primary_prefix_len in prefix_variants:
                    key = (
                        int(rows),
                        int(n_heads),
                        int(head_dim),
                        int(k_padded),
                        int(primary_len),
                        int(secondary_stride),
                        tuple(secondary_shape),
                        str(getattr(kv_secondary, "dtype", "")),
                        tuple(swa_shape) if use_swa_primary else (),
                        int(primary_prefix_len),
                        bool(use_swa_primary),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    q_t = DeviceTensor.from_numpy(
                        np.zeros(
                            (rows, head_dim, n_heads),
                            dtype=ml_dtypes.bfloat16,
                        ),
                        name=(
                            "dsv4_warmup_two_source_q_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    if use_swa_primary:
                        kv_primary = swa_kv_cache
                    else:
                        kv_primary = DeviceTensor.from_numpy(
                            np.zeros((int(primary_len), head_dim), dtype=np.float32),
                            name=(
                                "dsv4_warmup_two_source_kv_primary_"
                                f"t{rows}_p{int(primary_len)}_"
                                f"k{k_padded}_pp{primary_prefix_len}"
                            ),
                        )
                    topk_t = DeviceTensor.from_numpy(
                        np.zeros((k_padded, rows), dtype=np.int32),
                        name=(
                            "dsv4_warmup_two_source_topk_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    mask = DeviceTensor.from_numpy(
                        np.zeros((rows, k_padded), dtype=ml_dtypes.bfloat16),
                        name=(
                            "dsv4_warmup_two_source_mask_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    owner_ids = DeviceTensor.from_numpy(
                        np.zeros((rows,), dtype=np.int32),
                        name=(
                            "dsv4_warmup_two_source_owner_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    primary_owner_ids = DeviceTensor.from_numpy(
                        np.zeros((rows,), dtype=np.int32),
                        name=(
                            "dsv4_warmup_two_source_primary_owner_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    sink = DeviceTensor.from_numpy(
                        np.zeros((1, n_heads), dtype=np.float32),
                        name=(
                            "dsv4_warmup_two_source_sink_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    output = DeviceTensor.from_numpy(
                        np.zeros((rows, n_heads, head_dim), dtype=np.float32),
                        name=(
                            "dsv4_warmup_two_source_out_"
                            f"t{rows}_k{k_padded}_pp{primary_prefix_len}"
                        ),
                    )
                    run_sparse_attention_paged_two_source_device(
                        q_scaled_t=q_t,
                        kv_primary=kv_primary,
                        kv_secondary=kv_secondary,
                        topk_t=topk_t,
                        mask=mask,
                        owner_ids=owner_ids,
                        primary_owner_ids=(
                            None if use_swa_primary else primary_owner_ids
                        ),
                        sink=sink,
                        output=output,
                        primary_len=int(primary_len),
                        secondary_stride=secondary_stride,
                        primary_prefix_len=primary_prefix_len,
                        artifacts_dir=artifacts_dir,
                        _device_kernel_cls=device_kernel_cls,
                        _kernel_cache=kernel_cache,
                    )

    def precompile_swa_owner_window_write_buckets(
        self,
        buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile SWA ring-cache device-row write kernels before namespace seal.

        Writes are compiled at bucket shape and receive ``live_rows`` at
        runtime, so one bucket covers partial small prompts that previously
        required one NEFF per exact row count.
        """

        requested = sorted({int(bucket) for bucket in buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        scratch_cache_by_sig: dict[tuple[tuple[int, ...], str], Any] = {}
        seen: set[tuple[tuple[int, ...], str, str, int, int, tuple[int, ...]]] = set()
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            window_size = int(getattr(attn, "window_size", 0) or 0)
            if window_size <= 0:
                continue
            device_layer_state = self.device_state.layer(int(layer_id))
            swa_kv_cache = getattr(device_layer_state, "swa_kv_cache", None)
            cache_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ()))
            if len(cache_shape) != 2:
                continue
            head_dim = int(cache_shape[1])
            cache_dtype = getattr(swa_kv_cache, "dtype", ml_dtypes.bfloat16)
            cache_sig = (cache_shape, str(cache_dtype))
            scratch_cache = scratch_cache_by_sig.get(cache_sig)
            if scratch_cache is None:
                try:
                    cache_zeros = np.zeros(cache_shape, dtype=cache_dtype)
                except TypeError:
                    cache_zeros = np.zeros(cache_shape, dtype=ml_dtypes.bfloat16)
                scratch_cache = DeviceTensor.from_numpy(
                    cache_zeros,
                    name=f"dsv4_warmup_swa_owner_window_cache_l{int(layer_id)}",
                )
                scratch_cache_by_sig[cache_sig] = scratch_cache
            comp_obj = getattr(attn, "compressor", None)
            swa_comp_ratio = int(getattr(comp_obj, "compress_ratio", 0) or 0)
            swa_comp_overlap = bool(getattr(comp_obj, "overlap", False))
            kv_new_dtypes = (ml_dtypes.bfloat16, np.float32)
            for n_rows in _bucketed_state_write_warmup_rows(
                requested,
                max_rows=int(window_size),
            ):
                if n_rows <= 0:
                    continue
                if not swa_comp_overlap and swa_comp_ratio > 1:
                    continue
                live_rows = DeviceTensor.from_numpy(
                    np.asarray([[int(n_rows)]], dtype=np.int32),
                    name=f"dsv4_warmup_swa_owner_window_live_{n_rows}",
                )
                owner_ids = DeviceTensor.from_numpy(
                    np.zeros((n_rows,), dtype=np.int32),
                    name=f"dsv4_warmup_swa_owner_window_owners_{n_rows}",
                )
                position_arrays = (
                    np.arange(n_rows, dtype=np.int32),
                    np.arange(n_rows, dtype=np.int32).reshape(n_rows, 1),
                )
                for position_array in position_arrays:
                    pos_shape = tuple(int(dim) for dim in position_array.shape)
                    positions = DeviceTensor.from_numpy(
                        position_array,
                        name=(
                            "dsv4_warmup_swa_owner_window_positions_"
                            f"{n_rows}_{'x'.join(str(dim) for dim in pos_shape)}"
                        ),
                    )
                    for kv_new_dtype in kv_new_dtypes:
                        key = (
                            cache_shape,
                            str(cache_dtype),
                            str(kv_new_dtype),
                            int(window_size),
                            int(n_rows),
                            pos_shape,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        kv_new = DeviceTensor.from_numpy(
                            np.zeros((n_rows, head_dim), dtype=kv_new_dtype),
                            name=(
                                "dsv4_warmup_swa_owner_window_rows_"
                                f"{n_rows}_{np.dtype(kv_new_dtype).name}"
                            ),
                        )
                        run_write_kv_owner_window_device(
                            kv_cache=scratch_cache,
                            kv_new=kv_new,
                            owner_ids=owner_ids,
                            positions=positions,
                            live_rows=live_rows,
                            window_size=int(window_size),
                            artifacts_dir=self.build_dir,
                        )

    def precompile_compressor_state_write_buckets(
        self,
        buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile compressor ring-state write kernels before namespace seal."""

        requested = sorted({int(bucket) for bucket in buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        scratch_state_by_sig: dict[tuple[tuple[int, ...], str], Any] = {}
        scratch_ape_by_sig: dict[tuple[tuple[int, ...], str], Any] = {}
        seen: set[tuple[tuple[int, ...], str, tuple[int, ...], str, int, str, int]] = (
            set()
        )
        kv_score_dtypes = (ml_dtypes.bfloat16, np.float32)
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            try:
                device_layer_state = self.device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            compressors: list[tuple[str, Any, Any]] = []
            main_compressor = getattr(attn, "compressor", None)
            main_state = getattr(device_layer_state, "compressor", None)
            if main_compressor is not None and main_state is not None:
                compressors.append(("main", main_compressor, main_state))
            indexer = getattr(attn, "indexer", None)
            indexer_compressor = getattr(indexer, "compressor", None)
            indexer_state = getattr(device_layer_state, "indexer", None)
            if indexer_compressor is not None and indexer_state is not None:
                compressors.append(("indexer", indexer_compressor, indexer_state))

            for role, compressor, device_comp_state in compressors:
                kv_score_state = getattr(device_comp_state, "kv_score_state", None)
                state_shape = tuple(
                    int(dim) for dim in getattr(kv_score_state, "shape", ())
                )
                if len(state_shape) != 2 or state_shape[1] % 2 != 0:
                    continue
                width = int(state_shape[1]) // 2
                ring_size = int(getattr(device_comp_state, "ring_size", 0) or 0)
                if ring_size <= 0:
                    state_spec = getattr(device_comp_state, "spec", None)
                    ring_size = int(getattr(state_spec, "ring_size", 0) or 0)
                if ring_size <= 0 or state_shape[0] % ring_size != 0:
                    continue
                ape = getattr(compressor, "ape", None)
                ape_shape = tuple(int(dim) for dim in getattr(ape, "shape", ()))
                if len(ape_shape) != 2 or int(ape_shape[1]) != width:
                    continue

                state_dtype = getattr(kv_score_state, "dtype", np.float32)
                state_sig = (state_shape, str(state_dtype))
                scratch_state = scratch_state_by_sig.get(state_sig)
                if scratch_state is None:
                    try:
                        state_zeros = np.zeros(state_shape, dtype=state_dtype)
                    except TypeError:
                        state_zeros = np.zeros(state_shape, dtype=np.float32)
                    scratch_state = DeviceTensor.from_numpy(
                        state_zeros,
                        name=(f"dsv4_warmup_compressor_state_{role}_l{int(layer_id)}"),
                    )
                    scratch_state_by_sig[state_sig] = scratch_state

                ape_dtype = getattr(ape, "dtype", np.float32)
                ape_sig = (ape_shape, str(ape_dtype))
                scratch_ape = scratch_ape_by_sig.get(ape_sig)
                if scratch_ape is None:
                    try:
                        ape_zeros = np.zeros(ape_shape, dtype=ape_dtype)
                    except TypeError:
                        ape_zeros = np.zeros(ape_shape, dtype=np.float32)
                    scratch_ape = DeviceTensor.from_numpy(
                        ape_zeros,
                        name=(f"dsv4_warmup_compressor_ape_{role}_l{int(layer_id)}"),
                    )
                    scratch_ape_by_sig[ape_sig] = scratch_ape

                comp_ratio = int(getattr(compressor, "compress_ratio", 0) or 0)
                comp_overlap = bool(
                    getattr(
                        getattr(device_comp_state, "spec", None),
                        "overlap",
                        getattr(compressor, "overlap", False),
                    )
                )
                for n_rows in _bucketed_state_write_warmup_rows(
                    requested,
                    max_rows=int(ring_size),
                ):
                    if n_rows <= 0:
                        continue
                    # For non-overlap layers (ratio-128): ALL prefill state writes
                    # are dead at serve. Decode writes are at n=bsz which is
                    # already covered by the decode warmup steps.
                    if not comp_overlap and comp_ratio > 1:
                        continue
                    live_rows = DeviceTensor.from_numpy(
                        np.asarray([[int(n_rows)]], dtype=np.int32),
                        name=f"dsv4_warmup_compressor_live_{n_rows}",
                    )
                    owner_ids = DeviceTensor.from_numpy(
                        np.zeros((n_rows,), dtype=np.int32),
                        name=f"dsv4_warmup_compressor_owner_ids_{n_rows}",
                    )
                    positions = DeviceTensor.from_numpy(
                        np.arange(n_rows, dtype=np.int32),
                        name=f"dsv4_warmup_compressor_positions_{n_rows}",
                    )
                    for kv_score_dtype in kv_score_dtypes:
                        key = (
                            state_shape,
                            str(state_dtype),
                            ape_shape,
                            str(ape_dtype),
                            int(ring_size),
                            str(kv_score_dtype),
                            int(n_rows),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        kv_new = DeviceTensor.from_numpy(
                            np.zeros((n_rows, width), dtype=kv_score_dtype),
                            name=(
                                "dsv4_warmup_compressor_kv_"
                                f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                            ),
                        )
                        score_new = DeviceTensor.from_numpy(
                            np.zeros((n_rows, width), dtype=kv_score_dtype),
                            name=(
                                "dsv4_warmup_compressor_score_"
                                f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                            ),
                        )
                        run_write_kv_score_state_device(
                            kv_score_state=scratch_state,
                            kv_new=kv_new,
                            score_new=score_new,
                            owner_ids=owner_ids,
                            positions=positions,
                            ape=scratch_ape,
                            ring_size=int(ring_size),
                            live_rows=live_rows,
                            artifacts_dir=self.build_dir,
                        )

    def precompile_compressor_prefill_pool_buckets(
        self,
        token_buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile bucket-sized deferred-indexer prefill pool kernels before seal."""

        requested = sorted({int(bucket) for bucket in token_buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        scratch_ape_by_sig: dict[tuple[tuple[int, ...], str], Any] = {}
        seen: set[tuple] = set()
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            try:
                device_layer_state = self.device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue

            indexer = getattr(attn, "indexer", None)
            candidates = (
                (
                    "compressor",
                    getattr(attn, "compressor", None),
                    getattr(device_layer_state, "compressor", None),
                ),
                (
                    "indexer",
                    getattr(indexer, "compressor", None),
                    getattr(device_layer_state, "indexer", None),
                ),
            )
            for role, compressor, device_comp_state in candidates:
                if device_comp_state is None:
                    continue

                state_spec = getattr(device_comp_state, "spec", None)
                comp_ratio = int(
                    getattr(
                        state_spec,
                        "compress_ratio",
                        getattr(compressor, "compress_ratio", 0),
                    )
                    or 0
                )
                head_dim = int(
                    getattr(
                        state_spec,
                        "head_dim",
                        getattr(compressor, "head_dim", 0),
                    )
                    or 0
                )
                if comp_ratio <= 0 or head_dim <= 0:
                    continue
                comp_overlap = bool(
                    getattr(
                        state_spec,
                        "overlap",
                        getattr(compressor, "overlap", False),
                    )
                )
                state_width = int(
                    getattr(
                        state_spec,
                        "state_width",
                        int(head_dim) * (2 if comp_overlap else 1),
                    )
                )
                if state_width <= 0:
                    continue
                ape_shape = (int(comp_ratio), int(state_width))

                ape = getattr(compressor, "ape", None)
                ape_dtype = getattr(ape, "dtype", np.float32)
                ape_sig = (ape_shape, str(ape_dtype))
                scratch_ape = scratch_ape_by_sig.get(ape_sig)
                if scratch_ape is None:
                    try:
                        ape_zeros = np.zeros(ape_shape, dtype=ape_dtype)
                    except TypeError:
                        ape_zeros = np.zeros(ape_shape, dtype=np.float32)
                    scratch_ape = DeviceTensor.from_numpy(
                        ape_zeros,
                        name=(f"dsv4_warmup_prefill_pool_ape_{role}_l{int(layer_id)}"),
                    )
                    scratch_ape_by_sig[ape_sig] = scratch_ape

                for token_bucket in requested:
                    token_bucket_i = int(token_bucket)
                    if token_bucket_i % comp_ratio:
                        continue
                    out_rows = token_bucket_i // comp_ratio
                    key = (
                        int(token_bucket_i),
                        int(comp_ratio),
                        int(head_dim),
                        int(state_width),
                        bool(comp_overlap),
                        ape_shape,
                        str(ape_dtype),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    kv_new = DeviceTensor.from_numpy(
                        np.zeros(
                            (token_bucket_i, state_width),
                            dtype=ml_dtypes.bfloat16,
                        ),
                        name=(
                            "dsv4_warmup_prefill_pool_kv_"
                            f"{role}_t{token_bucket_i}_bfloat16"
                        ),
                    )
                    score_new = DeviceTensor.from_numpy(
                        np.zeros(
                            (token_bucket_i, state_width),
                            dtype=ml_dtypes.bfloat16,
                        ),
                        name=(
                            "dsv4_warmup_prefill_pool_score_"
                            f"{role}_t{token_bucket_i}_bfloat16"
                        ),
                    )
                    output = DeviceTensor.from_numpy(
                        np.zeros((out_rows, head_dim), dtype=np.float32),
                        name=(f"dsv4_warmup_prefill_pool_out_{role}_t{token_bucket_i}"),
                    )
                    run_prefill_pool_from_slab_device(
                        kv_new=kv_new,
                        score_new=score_new,
                        ape=scratch_ape,
                        bsz=1,
                        seqlen=token_bucket_i,
                        ratio=comp_ratio,
                        head_dim=head_dim,
                        overlap=comp_overlap,
                        artifacts_dir=self.build_dir,
                        output=output,
                    )

    def precompile_compressor_slot_write_buckets(
        self,
        token_buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile bucket-sized deferred-indexer compressed-cache slot writes."""

        requested = sorted({int(bucket) for bucket in token_buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        scratch_cache_by_sig: dict[tuple[tuple[int, ...], str], Any] = {}
        seen: set[tuple] = set()
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            try:
                device_layer_state = self.device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            indexer = getattr(attn, "indexer", None)
            candidates = (
                (
                    "compressor",
                    getattr(attn, "compressor", None),
                    getattr(device_layer_state, "compressor", None),
                ),
                (
                    "indexer",
                    getattr(indexer, "compressor", None),
                    getattr(device_layer_state, "indexer", None),
                ),
            )
            for role, compressor, device_comp_state in candidates:
                if device_comp_state is None:
                    continue

                state_spec = getattr(device_comp_state, "spec", None)
                comp_ratio = int(
                    getattr(
                        state_spec,
                        "compress_ratio",
                        getattr(compressor, "compress_ratio", 0),
                    )
                    or 0
                )
                compressed_kv_cache = getattr(
                    device_comp_state,
                    "compressed_kv_cache",
                    None,
                )
                cache_shape = tuple(
                    int(dim) for dim in getattr(compressed_kv_cache, "shape", ())
                )
                if comp_ratio <= 0 or len(cache_shape) != 2:
                    continue
                head_dim = int(cache_shape[1])
                if head_dim <= 0:
                    continue
                cache_dtype = getattr(compressed_kv_cache, "dtype", ml_dtypes.bfloat16)
                cache_sig = (cache_shape, str(cache_dtype))
                scratch_cache = scratch_cache_by_sig.get(cache_sig)
                if scratch_cache is None:
                    try:
                        cache_zeros = np.zeros(cache_shape, dtype=cache_dtype)
                    except TypeError:
                        cache_zeros = np.zeros(cache_shape, dtype=ml_dtypes.bfloat16)
                    scratch_cache = DeviceTensor.from_numpy(
                        cache_zeros,
                        name=(
                            "dsv4_warmup_compressor_slots_cache_"
                            f"{role}_l{int(layer_id)}"
                        ),
                    )
                    scratch_cache_by_sig[cache_sig] = scratch_cache

                for token_bucket in requested:
                    token_bucket_i = int(token_bucket)
                    if token_bucket_i % comp_ratio:
                        continue
                    n_new = token_bucket_i // comp_ratio
                    if n_new <= 0 or n_new > int(cache_shape[0]):
                        continue
                    key = (
                        cache_shape,
                        str(cache_dtype),
                        int(n_new),
                        int(head_dim),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    kv_new = DeviceTensor.from_numpy(
                        np.zeros((n_new, head_dim), dtype=ml_dtypes.bfloat16),
                        name=(
                            f"dsv4_warmup_compressor_slots_kv_{role}_t{token_bucket_i}"
                        ),
                    )
                    slot_mapping = DeviceTensor.from_numpy(
                        np.arange(n_new, dtype=np.int32),
                        name=(f"dsv4_warmup_compressor_slots_{role}_t{token_bucket_i}"),
                    )
                    run_write_kv_slots_device(
                        kv_cache=scratch_cache,
                        kv_new=kv_new,
                        slot_mapping=slot_mapping,
                        artifacts_dir=self.build_dir,
                    )

    def precompile_dual_state_swa_write_buckets(
        self,
        buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile the standalone fused SWA+main+indexer dual-state write kernel."""

        requested = sorted({int(bucket) for bucket in buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        kv_score_dtypes = (ml_dtypes.bfloat16, np.float32)
        seen: set[tuple] = set()
        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            compressor = getattr(attn, "compressor", None)
            indexer_obj = getattr(attn, "indexer", None)
            indexer_compressor = getattr(indexer_obj, "compressor", None)
            if compressor is None or indexer_obj is None or indexer_compressor is None:
                continue
            try:
                device_layer_state = self.device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            device_comp_state = getattr(device_layer_state, "compressor", None)
            device_idx_state = getattr(device_layer_state, "indexer", None)
            swa_kv_cache = getattr(device_layer_state, "swa_kv_cache", None)
            if (
                device_comp_state is None
                or device_idx_state is None
                or swa_kv_cache is None
            ):
                continue
            if not hasattr(swa_kv_cache, "tensor_ref"):
                continue
            kv_score_state = getattr(device_comp_state, "kv_score_state", None)
            indexer_kv_score_state = getattr(device_idx_state, "kv_score_state", None)
            if not (
                hasattr(kv_score_state, "tensor_ref")
                and hasattr(indexer_kv_score_state, "tensor_ref")
            ):
                continue
            swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ()))
            state_shape = tuple(
                int(dim) for dim in getattr(kv_score_state, "shape", ())
            )
            idx_state_shape = tuple(
                int(dim) for dim in getattr(indexer_kv_score_state, "shape", ())
            )
            if (
                len(swa_shape) != 2
                or len(state_shape) != 2
                or len(idx_state_shape) != 2
                or state_shape[1] % 2 != 0
                or idx_state_shape[1] % 2 != 0
            ):
                continue
            swa_head_dim = int(swa_shape[1])
            width = int(state_shape[1]) // 2
            indexer_width = int(idx_state_shape[1]) // 2
            ring_size = int(getattr(device_comp_state, "ring_size", 0) or 0)
            indexer_ring_size = int(getattr(device_idx_state, "ring_size", 0) or 0)
            if ring_size <= 0 or indexer_ring_size <= 0:
                continue
            if state_shape[0] % ring_size or idx_state_shape[0] % indexer_ring_size:
                continue
            window_size = int(
                getattr(self.device_state, "window_size", 0)
                or getattr(attn, "window_size", 0)
                or 0
            )
            if window_size <= 0:
                continue
            ape = getattr(compressor, "ape", None)
            indexer_ape = getattr(indexer_compressor, "ape", None)
            ape_shape = tuple(int(dim) for dim in getattr(ape, "shape", ()))
            indexer_ape_shape = tuple(
                int(dim) for dim in getattr(indexer_ape, "shape", ())
            )
            if (
                len(ape_shape) != 2
                or int(ape_shape[1]) != width
                or len(indexer_ape_shape) != 2
                or int(indexer_ape_shape[1]) != indexer_width
            ):
                continue

            def _ape_device(value: Any, *, name: str) -> Any:
                if hasattr(value, "tensor_ref"):
                    return value
                return DeviceTensor.from_numpy(
                    np.ascontiguousarray(np.asarray(value, dtype=ml_dtypes.bfloat16)),
                    name=name,
                )

            ape_dev = _ape_device(ape, name=f"dsv4_warmup_dual_ape_l{int(layer_id)}")
            indexer_ape_dev = _ape_device(
                indexer_ape, name=f"dsv4_warmup_dual_idx_ape_l{int(layer_id)}"
            )
            swa_rows_dtypes = (ml_dtypes.bfloat16, np.float32)
            for bucket in requested:
                n_rows = int(bucket)
                if n_rows <= 0:
                    continue
                live_rows = DeviceTensor.from_numpy(
                    np.asarray([[max(1, n_rows)]], dtype=np.int32),
                    name=f"dsv4_warmup_dual_live_{n_rows}",
                )
                owner_ids = DeviceTensor.from_numpy(
                    np.zeros((n_rows,), dtype=np.int32),
                    name=f"dsv4_warmup_dual_owner_ids_{n_rows}",
                )
                position_arrays = (
                    np.arange(n_rows, dtype=np.int32),
                    np.arange(n_rows, dtype=np.int32).reshape(n_rows, 1),
                )
                for position_array in position_arrays:
                    pos_shape = tuple(int(dim) for dim in position_array.shape)
                    positions = DeviceTensor.from_numpy(
                        position_array,
                        name=(
                            "dsv4_warmup_dual_positions_"
                            f"{n_rows}_{'x'.join(str(dim) for dim in pos_shape)}"
                        ),
                    )
                    for swa_rows_dtype in swa_rows_dtypes:
                        swa_rows = DeviceTensor.from_numpy(
                            np.zeros((n_rows, swa_head_dim), dtype=swa_rows_dtype),
                            name=(
                                "dsv4_warmup_dual_swa_rows_"
                                f"{n_rows}_{np.dtype(swa_rows_dtype).name}"
                            ),
                        )
                        for kv_score_dtype in kv_score_dtypes:
                            key = (
                                swa_shape,
                                state_shape,
                                idx_state_shape,
                                ape_shape,
                                indexer_ape_shape,
                                int(window_size),
                                int(ring_size),
                                int(indexer_ring_size),
                                str(np.dtype(swa_rows_dtype).name),
                                str(np.dtype(kv_score_dtype).name),
                                int(n_rows),
                                pos_shape,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            kv_new = DeviceTensor.from_numpy(
                                np.zeros((n_rows, width), dtype=kv_score_dtype),
                                name=(
                                    "dsv4_warmup_dual_kv_"
                                    f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                                ),
                            )
                            score_new = DeviceTensor.from_numpy(
                                np.zeros((n_rows, width), dtype=kv_score_dtype),
                                name=(
                                    "dsv4_warmup_dual_score_"
                                    f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                                ),
                            )
                            indexer_kv_new = DeviceTensor.from_numpy(
                                np.zeros((n_rows, indexer_width), dtype=kv_score_dtype),
                                name=(
                                    "dsv4_warmup_dual_idx_kv_"
                                    f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                                ),
                            )
                            indexer_score_new = DeviceTensor.from_numpy(
                                np.zeros(
                                    (n_rows, indexer_width),
                                    dtype=kv_score_dtype,
                                ),
                                name=(
                                    "dsv4_warmup_dual_idx_score_"
                                    f"{n_rows}_{np.dtype(kv_score_dtype).name}"
                                ),
                            )
                            run_write_swa_dual_kv_score_state_device(
                                swa_kv_cache=swa_kv_cache,
                                kv_score_state=kv_score_state,
                                indexer_kv_score_state=indexer_kv_score_state,
                                swa_rows=swa_rows,
                                kv_new=kv_new,
                                score_new=score_new,
                                indexer_kv_new=indexer_kv_new,
                                indexer_score_new=indexer_score_new,
                                owner_ids=owner_ids,
                                positions=positions,
                                ape=ape_dev,
                                indexer_ape=indexer_ape_dev,
                                live_rows=live_rows,
                                window_size=int(window_size),
                                ring_size=int(ring_size),
                                indexer_ring_size=int(indexer_ring_size),
                                artifacts_dir=self.build_dir,
                            )

    def precompile_bucketed_single_state_swa_cache_write_buckets(
        self,
        token_buckets: tuple[int, ...] | list[int],
    ) -> None:
        """Compile bucketed no-indexer token-topk SWA+state+cache writers."""

        requested = sorted({int(bucket) for bucket in token_buckets if int(bucket) > 0})
        if not requested:
            return

        DeviceTensor = _get_device_tensor_cls()
        seen: set[tuple] = set()

        def _full_width_ape_device(
            ape: Any,
            *,
            ratio: int,
            width: int,
            layer_id: int,
        ) -> Any:
            if ape is None:
                ape_arr = np.zeros((int(ratio), int(width)), dtype=ml_dtypes.bfloat16)
            else:
                source = ape.numpy() if hasattr(ape, "numpy") else np.asarray(ape)
                ape_arr = np.asarray(source, dtype=ml_dtypes.bfloat16).reshape(
                    int(ratio),
                    -1,
                )
                if int(ape_arr.shape[1]) != int(width):
                    ape_arr = np.broadcast_to(ape_arr, (int(ratio), int(width)))
            return DeviceTensor.from_numpy(
                np.ascontiguousarray(ape_arr),
                name=f"dsv4_warmup_single_ape_l{int(layer_id)}",
            )

        for layer_id, block in enumerate(getattr(self.runtime_surface, "blocks", ())):
            attn = getattr(block, "attn", None)
            compressor = getattr(attn, "compressor", None)
            indexer_obj = getattr(attn, "indexer", None)
            if compressor is None or indexer_obj is not None:
                continue
            try:
                device_layer_state = self.device_state.layer(int(layer_id))
            except (AttributeError, IndexError, TypeError):
                continue
            device_comp_state = getattr(device_layer_state, "compressor", None)
            swa_kv_cache = getattr(device_layer_state, "swa_kv_cache", None)
            if device_comp_state is None or swa_kv_cache is None:
                continue
            kv_score_state = getattr(device_comp_state, "kv_score_state", None)
            compressed_kv_cache = getattr(
                device_comp_state, "compressed_kv_cache", None
            )
            if not (
                hasattr(swa_kv_cache, "tensor_ref")
                and hasattr(kv_score_state, "tensor_ref")
                and hasattr(compressed_kv_cache, "tensor_ref")
            ):
                continue

            spec = getattr(device_comp_state, "spec", None)
            ratio = int(
                getattr(spec, "compress_ratio", 0)
                or getattr(compressor, "compress_ratio", 0)
                or 0
            )
            if ratio <= 1:
                continue
            overlap = bool(
                getattr(spec, "overlap", getattr(compressor, "overlap", False))
            )
            max_keep = (2 * ratio - 1) if overlap else (ratio - 1 or 1)
            if max_keep <= 0:
                continue

            window_size = int(
                getattr(self.device_state, "window_size", 0)
                or getattr(attn, "window_size", 0)
                or 0
            )
            ring_size = int(
                getattr(spec, "ring_size", 0)
                or getattr(device_comp_state, "ring_size", 0)
                or 0
            )
            max_clen = int(
                getattr(spec, "max_compressed_len", 0)
                or getattr(device_comp_state, "max_compressed_len", 0)
                or 0
            )
            num_state_owners = int(getattr(spec, "num_state_owners", 0) or 0)
            guard_owner = int(
                getattr(spec, "guard_owner", max(0, num_state_owners - 1))
            )
            if (
                window_size <= 0
                or ring_size <= 0
                or max_clen <= 0
                or num_state_owners <= 0
            ):
                continue

            swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape", ()))
            state_shape = tuple(
                int(dim) for dim in getattr(kv_score_state, "shape", ())
            )
            cache_shape = tuple(
                int(dim) for dim in getattr(compressed_kv_cache, "shape", ())
            )
            if (
                len(swa_shape) != 2
                or len(state_shape) != 2
                or len(cache_shape) != 2
                or state_shape[1] % 2 != 0
            ):
                continue
            swa_head_dim = int(swa_shape[1])
            state_width = int(state_shape[1]) // 2
            cache_head_dim = int(cache_shape[1])
            if state_width <= 0 or cache_head_dim <= 0:
                continue

            ape_dev = _full_width_ape_device(
                getattr(compressor, "ape", None),
                ratio=int(ratio),
                width=int(state_width),
                layer_id=int(layer_id),
            )
            cache_owner_ids = DeviceTensor.from_numpy(
                np.full(num_state_owners, guard_owner, dtype=np.int32),
                name=f"dsv4_warmup_single_cache_owner_l{int(layer_id)}",
            )
            state_owner_ids = DeviceTensor.from_numpy(
                np.full(max_keep, guard_owner, dtype=np.int32),
                name=f"dsv4_warmup_single_state_owner_l{int(layer_id)}",
            )
            state_positions = DeviceTensor.from_numpy(
                np.zeros(max_keep, dtype=np.int32),
                name=f"dsv4_warmup_single_state_pos_l{int(layer_id)}",
            )

            for bucket in requested:
                bucket_i = int(bucket)
                if bucket_i < max_keep or bucket_i < (2 * ratio - 1):
                    continue
                clen = bucket_i // ratio
                if clen <= 0:
                    continue
                cache_real_clen = max(1, min(clen, clen - 1 if clen > 1 else clen))
                swa_owner_ids = DeviceTensor.from_numpy(
                    np.full(bucket_i, guard_owner, dtype=np.int32),
                    name=(f"dsv4_warmup_single_swa_owner_l{int(layer_id)}_t{bucket_i}"),
                )
                swa_positions = DeviceTensor.from_numpy(
                    np.zeros(bucket_i, dtype=np.int32),
                    name=(f"dsv4_warmup_single_swa_pos_l{int(layer_id)}_t{bucket_i}"),
                )
                compressed_rows = DeviceTensor.from_numpy(
                    np.zeros((clen, cache_head_dim), dtype=ml_dtypes.bfloat16),
                    name=(f"dsv4_warmup_single_comp_rows_l{int(layer_id)}_t{bucket_i}"),
                )
                kv_new = DeviceTensor.from_numpy(
                    np.zeros((max_keep, state_width), dtype=ml_dtypes.bfloat16),
                    name=(f"dsv4_warmup_single_kv_l{int(layer_id)}_t{bucket_i}"),
                )
                score_new = DeviceTensor.from_numpy(
                    np.zeros((max_keep, state_width), dtype=ml_dtypes.bfloat16),
                    name=(f"dsv4_warmup_single_score_l{int(layer_id)}_t{bucket_i}"),
                )
                key = (
                    swa_shape,
                    state_shape,
                    cache_shape,
                    int(window_size),
                    int(ring_size),
                    int(max_clen),
                    int(guard_owner),
                    int(bucket_i),
                    int(max_keep),
                    int(clen),
                    str(np.dtype(np.float32).name),
                )
                if key in seen:
                    continue
                seen.add(key)
                swa_rows = DeviceTensor.from_numpy(
                    np.zeros((bucket_i, swa_head_dim), dtype=np.float32),
                    name=f"dsv4_warmup_single_swa_rows_t{bucket_i}_float32",
                )
                run_write_swa_kv_score_state_owner_clen_device(
                    swa_kv_cache=swa_kv_cache,
                    kv_score_state=kv_score_state,
                    compressed_kv_cache=compressed_kv_cache,
                    swa_rows=swa_rows,
                    kv_new=kv_new,
                    score_new=score_new,
                    compressed_rows=compressed_rows,
                    swa_owner_ids=swa_owner_ids,
                    swa_positions=swa_positions,
                    state_owner_ids=state_owner_ids,
                    state_positions=state_positions,
                    cache_owner_ids=cache_owner_ids,
                    ape=ape_dev,
                    window_size=int(window_size),
                    ring_size=int(ring_size),
                    clen=int(clen),
                    owner_id_stride=1,
                    max_clen=int(max_clen),
                    cache_real_clen=int(cache_real_clen),
                    guard_owner=int(guard_owner),
                    artifacts_dir=self.build_dir,
                )


__all__ = ["Dsv4ProductSupportKernelWarmupMixin"]
