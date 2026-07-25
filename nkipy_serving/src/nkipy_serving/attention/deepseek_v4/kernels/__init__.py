"""Sparse-attention kernels for the DSV4 production backend.

Implementation modules are grouped by kernel family while this package keeps the
legacy import surface used by the backend and tests.
"""

from __future__ import annotations

from nkipy_serving.ops.attention.sparse_mla import (
    D_BLOCK,
    K_TILE,
    P_MAX,
    gather_kv_and_mask,
)
from nkipy_serving.ops.attention.sparse_mla import (
    sparse_mla_attention_host_gather as sparse_attention_host_gather,
)
from nkipy_serving.ops.attention.sparse_mla import (
    sparse_mla_attention_oracle as sparse_attention_oracle,
)
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock

from .kv_write import (
    run_write_kv_owner_clen_device,
    run_write_kv_owner_pos_device,
    run_write_kv_owner_window_device,
    run_write_kv_slots_device,
    write_kv_to_flat_cache_oracle,
)
from .paged import run_sparse_attention_paged_device
from .swa import (
    run_sparse_attention_paged_swa_device,
    run_swa_global_slots_device,
    swa_global_slots_oracle,
)
from .two_source import (
    gather_two_source_kv_and_mask,
    run_sparse_attention_paged_two_source_device,
)

__all__ = [
    "D_BLOCK",
    "K_TILE",
    "P_MAX",
    "compile_and_load_with_lock",
    "gather_kv_and_mask",
    "gather_two_source_kv_and_mask",
    "run_sparse_attention_paged_device",
    "run_sparse_attention_paged_swa_device",
    "run_sparse_attention_paged_two_source_device",
    "run_swa_global_slots_device",
    "run_write_kv_owner_clen_device",
    "run_write_kv_owner_pos_device",
    "run_write_kv_owner_window_device",
    "run_write_kv_slots_device",
    "sparse_attention_host_gather",
    "sparse_attention_oracle",
    "swa_global_slots_oracle",
    "write_kv_to_flat_cache_oracle",
]
