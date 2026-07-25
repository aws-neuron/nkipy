"""DeepSeek-V4 sparse-attention metadata and device buffer contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import ml_dtypes
import numpy as np

from nkipy_serving.attention.base import AttentionMetadata
from nkipy_serving.attention.deepseek_v4.metadata import (
    SPARSE_INDEX_SPACE_GLOBAL_SLOTS,
    SparseAttentionMetadata,
)


@dataclass(frozen=True)
class Dsv4DpAttentionSuperstepMetadata:
    """Replica-local lane layout for a DP-attention superstep."""

    num_lanes: int
    lane_token_counts: np.ndarray
    lane_batch_sizes: np.ndarray
    lane_token_offsets: np.ndarray
    lane_batch_offsets: np.ndarray

    def __post_init__(self) -> None:
        lanes = int(self.num_lanes)
        if lanes <= 0:
            raise ValueError(f"num_lanes must be positive, got {lanes}")
        object.__setattr__(self, "num_lanes", lanes)
        arrays = {
            "lane_token_counts": np.asarray(
                self.lane_token_counts,
                dtype=np.int32,
            ).reshape(-1),
            "lane_batch_sizes": np.asarray(
                self.lane_batch_sizes,
                dtype=np.int32,
            ).reshape(-1),
            "lane_token_offsets": np.asarray(
                self.lane_token_offsets,
                dtype=np.int32,
            ).reshape(-1),
            "lane_batch_offsets": np.asarray(
                self.lane_batch_offsets,
                dtype=np.int32,
            ).reshape(-1),
        }
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)
        for name in ("lane_token_counts", "lane_batch_sizes"):
            arr = arrays[name]
            if arr.shape != (lanes,):
                raise ValueError(f"{name} must be [{lanes}], got {arr.shape}")
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative")
        for name in ("lane_token_offsets", "lane_batch_offsets"):
            arr = arrays[name]
            if arr.shape != (lanes + 1,):
                raise ValueError(f"{name} must be [{lanes + 1}], got {arr.shape}")
            if int(arr[0]) != 0:
                raise ValueError(f"{name}[0] must be 0")
            if np.any(arr[1:] < arr[:-1]):
                raise ValueError(f"{name} must be monotonic")
        expected_token_offsets = np.zeros((lanes + 1,), dtype=np.int32)
        expected_token_offsets[1:] = np.cumsum(
            arrays["lane_token_counts"],
            dtype=np.int32,
        )
        if not np.array_equal(arrays["lane_token_offsets"], expected_token_offsets):
            raise ValueError("lane_token_offsets must be cumulative lane_token_counts")
        expected_batch_offsets = np.zeros((lanes + 1,), dtype=np.int32)
        expected_batch_offsets[1:] = np.cumsum(
            arrays["lane_batch_sizes"],
            dtype=np.int32,
        )
        if not np.array_equal(arrays["lane_batch_offsets"], expected_batch_offsets):
            raise ValueError("lane_batch_offsets must be cumulative lane_batch_sizes")

    @property
    def total_tokens(self) -> int:
        return int(self.lane_token_offsets[-1])

    @property
    def batch_size(self) -> int:
        return int(self.lane_batch_offsets[-1])

    def token_range(self, lane: int) -> tuple[int, int]:
        lane_i = int(lane)
        if lane_i < 0 or lane_i >= int(self.num_lanes):
            raise ValueError(
                f"lane must be in [0, {int(self.num_lanes)}), got {lane_i}"
            )
        return (
            int(self.lane_token_offsets[lane_i]),
            int(self.lane_token_offsets[lane_i + 1]),
        )

    def batch_range(self, lane: int) -> tuple[int, int]:
        lane_i = int(lane)
        if lane_i < 0 or lane_i >= int(self.num_lanes):
            raise ValueError(
                f"lane must be in [0, {int(self.num_lanes)}), got {lane_i}"
            )
        return (
            int(self.lane_batch_offsets[lane_i]),
            int(self.lane_batch_offsets[lane_i + 1]),
        )


@dataclass(frozen=True)
class Dsv4AttentionMetadata:
    """DSV4 sparse-attention metadata layered on the generic contract.

    ``base`` remains the serving source of truth for request boundaries and
    KV slots. ``sparse.topk_indices`` must already be global KV-cache slot
    IDs for the device path.
    """

    base: AttentionMetadata
    sparse: SparseAttentionMetadata
    positions: np.ndarray | None = None
    state_owner_ids: np.ndarray | None = None
    dp_superstep: Dsv4DpAttentionSuperstepMetadata | None = None

    def __post_init__(self) -> None:
        if int(self.sparse.total_tokens) != int(self.base.total_tokens):
            raise ValueError(
                "sparse metadata token count must match base metadata: "
                f"sparse={self.sparse.total_tokens}, base={self.base.total_tokens}"
            )
        if self.sparse.index_space != SPARSE_INDEX_SPACE_GLOBAL_SLOTS:
            raise ValueError(
                "Dsv4AttentionMetadata requires sparse.index_space="
                f"{SPARSE_INDEX_SPACE_GLOBAL_SLOTS!r}; got "
                f"{self.sparse.index_space!r}"
            )
        if self.positions is not None:
            positions = np.asarray(self.positions)
            if positions.shape != (int(self.base.total_tokens),):
                raise ValueError(
                    "positions must be [total_tokens="
                    f"{self.base.total_tokens}], got {positions.shape}"
                )
        if self.state_owner_ids is not None:
            owners = np.asarray(self.state_owner_ids)
            if owners.shape != (int(self.base.total_tokens),):
                raise ValueError(
                    "state_owner_ids must be [total_tokens="
                    f"{self.base.total_tokens}], got {owners.shape}"
                )
        if self.dp_superstep is not None:
            if int(self.dp_superstep.total_tokens) != int(self.base.total_tokens):
                raise ValueError(
                    "dp_superstep token count must match base metadata: "
                    f"dp={self.dp_superstep.total_tokens}, "
                    f"base={self.base.total_tokens}"
                )
            if int(self.dp_superstep.batch_size) != int(self.base.batch_size):
                raise ValueError(
                    "dp_superstep batch size must match base metadata: "
                    f"dp={self.dp_superstep.batch_size}, "
                    f"base={self.base.batch_size}"
                )

    @property
    def total_tokens(self) -> int:
        return int(self.base.total_tokens)

    @property
    def query_lens(self) -> np.ndarray:
        return np.diff(np.asarray(self.base.query_start_loc, dtype=np.int64))

    def slice_for_dp_lane(self, lane: int) -> "Dsv4AttentionMetadata":
        """Return lane-local metadata for DP-attention execution."""
        dp = self.dp_superstep
        if dp is None:
            raise ValueError("slice_for_dp_lane requires dp_superstep metadata")
        token_start, token_end = dp.token_range(lane)
        batch_start, batch_end = dp.batch_range(lane)
        lane_tokens = int(token_end - token_start)
        lane_bs = int(batch_end - batch_start)
        base = self.base
        qsl = np.asarray(base.query_start_loc, dtype=np.int64).reshape(-1)
        if qsl.shape[0] < batch_end + 1:
            raise ValueError(
                "query_start_loc too short for DP-attention lane slice: "
                f"shape={qsl.shape}, batch_end={batch_end}"
            )
        lane_qsl = qsl[batch_start : batch_end + 1] - np.int64(token_start)
        lane_base = AttentionMetadata(
            forward_mode=int(base.forward_mode),
            seq_lens=np.asarray(base.seq_lens)[batch_start:batch_end],
            slot_mapping=np.asarray(base.slot_mapping)[token_start:token_end],
            block_tables=np.asarray(base.block_tables)[batch_start:batch_end],
            query_start_loc=lane_qsl.astype(np.int64, copy=False),
            total_tokens=lane_tokens,
            batch_size=lane_bs,
            max_seq_len=int(base.max_seq_len),
            num_kv_heads=int(base.num_kv_heads),
            head_dim=int(base.head_dim),
            block_size=int(base.block_size),
        )
        lane_sparse = SparseAttentionMetadata(
            topk_indices=np.asarray(self.sparse.topk_indices)[token_start:token_end],
            topk_lens=np.asarray(self.sparse.topk_lens)[token_start:token_end],
            compress_ratio=int(self.sparse.compress_ratio),
            num_kv_positions=int(self.sparse.num_kv_positions),
            window_size=int(self.sparse.window_size),
            index_topk=int(self.sparse.index_topk),
            index_space=str(self.sparse.index_space),
        )
        positions = (
            None
            if self.positions is None
            else np.asarray(self.positions)[token_start:token_end]
        )
        state_owner_ids = (
            None
            if self.state_owner_ids is None
            else np.asarray(self.state_owner_ids)[token_start:token_end]
        )
        return Dsv4AttentionMetadata(
            base=lane_base,
            sparse=lane_sparse,
            positions=positions,
            state_owner_ids=state_owner_ids,
            dp_superstep=None,
        )

    @property
    def has_prefill(self) -> bool:
        return bool(np.any(self.query_lens > 1))

    @property
    def has_decode(self) -> bool:
        return bool(np.any(self.query_lens == 1))


@dataclass(frozen=True)
class Dsv4DeviceAttentionInputs:
    """Device-resident per-bucket tensors for DSV4 attention kernels.

    These are allocation handles, not CPU-prepared tile plans. Runtime code
    may upload scheduler metadata into the base tensors, but SWA generation,
    indexer/top-k, local-to-global slot conversion, masking, KV scatter, and
    sparse attention gather are expected to run on device.
    """

    token_bucket: int
    max_requests: int
    max_blocks_per_request: int
    max_k: int
    slot_mapping: Any
    # NKI-friendly column vectors for scalar metadata.
    seq_lens: Any
    query_start_loc: Any
    block_tables: Any
    # Per-token block-table view used by the current SWA kernel ABI.
    block_tables_per_token: Any
    positions: Any
    # Per-token owning request index, retained as scheduler metadata.
    req_id_per_token: Any
    topk_global_t: Any
    topk_lens: Any
    # Numeric 0/1 mask in the layout consumed by sparse attention kernels.
    topk_mask: Any

    def as_kernel_inputs(self) -> dict[str, Any]:
        return {
            "slot_mapping": self.slot_mapping,
            "seq_lens": self.seq_lens,
            "query_start_loc": self.query_start_loc,
            "block_tables": self.block_tables,
            "block_tables_per_token": self.block_tables_per_token,
            "positions": self.positions,
            "req_id_per_token": self.req_id_per_token,
            "topk_global_t": self.topk_global_t,
            "topk_lens": self.topk_lens,
            "topk_mask": self.topk_mask,
        }


def allocate_dsv4_device_attention_inputs(
    alloc_device_scratch: Callable[..., Any],
    *,
    token_bucket: int,
    max_requests: int,
    max_blocks_per_request: int,
    max_k: int,
    prefix: str,
) -> Dsv4DeviceAttentionInputs:
    """Allocate DSV4 device attention metadata/scratch buffers."""
    tb = int(token_bucket)
    mr = int(max_requests)
    mb = int(max_blocks_per_request)
    mk = int(max_k)
    if tb <= 0 or mr <= 0 or mb <= 0 or mk <= 0:
        raise ValueError(
            "token_bucket, max_requests, max_blocks_per_request, and max_k "
            "must all be positive"
        )

    return Dsv4DeviceAttentionInputs(
        token_bucket=tb,
        max_requests=mr,
        max_blocks_per_request=mb,
        max_k=mk,
        slot_mapping=alloc_device_scratch(
            (tb,),
            np.int32,
            name=f"{prefix}_dsv4_slot_mapping_t{tb}",
        ),
        seq_lens=alloc_device_scratch(
            (mr, 1),
            np.int32,
            name=f"{prefix}_dsv4_seq_lens_t{tb}",
        ),
        query_start_loc=alloc_device_scratch(
            (mr + 1, 1),
            np.int32,
            name=f"{prefix}_dsv4_query_start_loc_t{tb}",
        ),
        block_tables=alloc_device_scratch(
            (mr, mb),
            np.int32,
            name=f"{prefix}_dsv4_block_tables_t{tb}",
        ),
        block_tables_per_token=alloc_device_scratch(
            (tb, mb),
            np.int32,
            name=f"{prefix}_dsv4_block_tables_per_token_t{tb}",
        ),
        positions=alloc_device_scratch(
            (tb, 1),
            np.int32,
            name=f"{prefix}_dsv4_positions_t{tb}",
        ),
        req_id_per_token=alloc_device_scratch(
            (tb, 1),
            np.int32,
            name=f"{prefix}_dsv4_req_id_per_token_t{tb}",
        ),
        # K-major layout matches the paged sparse attention kernel input
        # ``topk_T [K, tokens]`` and avoids a host transpose.
        topk_global_t=alloc_device_scratch(
            (mk, tb),
            np.int32,
            name=f"{prefix}_dsv4_topk_global_T_t{tb}_k{mk}",
        ),
        topk_lens=alloc_device_scratch(
            (tb, 1),
            np.int32,
            name=f"{prefix}_dsv4_topk_lens_t{tb}",
        ),
        topk_mask=alloc_device_scratch(
            (tb, mk),
            ml_dtypes.bfloat16,
            name=f"{prefix}_dsv4_topk_mask_t{tb}_k{mk}",
        ),
    )


def dsv4_device_sparse_attention_kernel_inputs(
    *,
    q_scaled_t: Any,
    kv_hbm: Any,
    step_inputs: Dsv4DeviceAttentionInputs,
    sink: Any,
) -> dict[str, Any]:
    """Bind DSV4 device buffers to the paged sparse attention kernel ABI."""
    return {
        "q_T": q_scaled_t,
        "kv_hbm": kv_hbm,
        "topk_T": step_inputs.topk_global_t,
        "mask": step_inputs.topk_mask,
        "sink": sink,
    }


def run_dsv4_device_sparse_attention(
    *,
    q_scaled_t: Any,
    kv_hbm: Any,
    step_inputs: Dsv4DeviceAttentionInputs,
    sink: Any,
    output: Any,
    artifacts_dir: str | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Run DSV4 sparse attention using only device-resident tensors."""
    from nkipy_serving.attention.deepseek_v4.kernels import (
        run_sparse_attention_paged_device,
    )

    return run_sparse_attention_paged_device(
        q_scaled_t=q_scaled_t,
        kv_hbm=kv_hbm,
        topk_t=step_inputs.topk_global_t,
        mask=step_inputs.topk_mask,
        sink=sink,
        output=output,
        artifacts_dir=artifacts_dir,
        _device_kernel_cls=_device_kernel_cls,
        _kernel_cache=_kernel_cache,
    )


def run_dsv4_swa_global_slots(
    *,
    step_inputs: Dsv4DeviceAttentionInputs,
    block_size: int,
    window_size: int,
    artifacts_dir: str | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Fill DSV4 SWA global-slot buffers on device (prefill + decode)."""
    from nkipy_serving.attention.deepseek_v4.kernels import (
        run_swa_global_slots_device,
    )

    return run_swa_global_slots_device(
        positions=step_inputs.positions,
        block_tables_per_token=step_inputs.block_tables_per_token,
        topk_t=step_inputs.topk_global_t,
        topk_lens=step_inputs.topk_lens,
        topk_mask=step_inputs.topk_mask,
        block_size=int(block_size),
        window_size=int(window_size),
        artifacts_dir=artifacts_dir,
        _device_kernel_cls=_device_kernel_cls,
        _kernel_cache=_kernel_cache,
    )


def tensor_to_step_field_name(
    step_inputs: Dsv4DeviceAttentionInputs,
    dev_tensor: Any,
) -> str:
    """Map a ``Dsv4DeviceAttentionInputs`` field value back to its name."""
    for f in fields(step_inputs):
        if getattr(step_inputs, f.name) is dev_tensor:
            return f.name
    raise ValueError("dev_tensor is not a field of step_inputs")


def build_req_id_per_token(query_start_loc: np.ndarray) -> np.ndarray:
    """Expand ``query_start_loc`` into a ``[total_tokens]`` per-token req id."""
    qsl = np.asarray(query_start_loc, dtype=np.int64).reshape(-1)
    if qsl.ndim != 1 or qsl.shape[0] < 1:
        raise ValueError(f"query_start_loc must be 1D, got {qsl.shape}")
    total = int(qsl[-1])
    req_id = np.zeros((total,), dtype=np.int32)
    for ri in range(qsl.shape[0] - 1):
        start = int(qsl[ri])
        end = int(qsl[ri + 1])
        if end > start:
            req_id[start:end] = ri
    return req_id


def build_positions_per_token(
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
) -> np.ndarray:
    """Expand per-request ``seq_lens`` into absolute per-token positions.

    For each request ``r`` with query-length ``q_len_r`` and kv-length
    ``seq_lens[r]``, the query tokens have absolute positions
    ``[seq_lens[r] - q_len_r .. seq_lens[r] - 1]``.
    """
    qsl = np.asarray(query_start_loc, dtype=np.int64).reshape(-1)
    sl = np.asarray(seq_lens, dtype=np.int64).reshape(-1)
    if qsl.shape[0] != sl.shape[0] + 1:
        raise ValueError(
            f"query_start_loc shape {qsl.shape} inconsistent with "
            f"seq_lens shape {sl.shape}"
        )
    total = int(qsl[-1])
    positions = np.zeros((total,), dtype=np.int32)
    for ri in range(sl.shape[0]):
        start = int(qsl[ri])
        end = int(qsl[ri + 1])
        q_len = end - start
        if q_len == 0:
            continue
        seq_len = int(sl[ri])
        if q_len > seq_len:
            raise ValueError(f"request {ri}: q_len={q_len} exceeds seq_len={seq_len}")
        positions[start:end] = np.arange(seq_len - q_len, seq_len, dtype=np.int32)
    return positions


__all__ = [
    "Dsv4AttentionMetadata",
    "Dsv4DpAttentionSuperstepMetadata",
    "Dsv4DeviceAttentionInputs",
    "allocate_dsv4_device_attention_inputs",
    "build_positions_per_token",
    "build_req_id_per_token",
    "dsv4_device_sparse_attention_kernel_inputs",
    "run_dsv4_device_sparse_attention",
    "run_dsv4_swa_global_slots",
    "tensor_to_step_field_name",
]
