# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored nkilib ``indexed_flatten`` subkernel for MoE scheduling."""

from __future__ import annotations

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from .kernel_assert import kernel_assert
from .kernel_helpers import div_ceil


@nki.jit
def indexed_flatten(
    input_tensor: nl.ndarray,
    f_len: int,
    output_len: int,
    row_offsets: nl.ndarray,
    row_offsets_start: Optional[nl.ndarray] = None,
    padding_val: int = -1,
) -> nl.ndarray:
    """Flatten ``[E, T]`` rows into blockwise token-position slots.

    This is copied from ``nkilib.core.subkernels.indexed_flatten`` with a
    small LNC=1 adaptation: when the launch grid has one program, the final
    cross-NC ``sendrecv`` is skipped.
    """
    index_dtype = input_tensor.dtype
    P_MAX = nl.tile_size.pmax
    E, T = input_tensor.shape
    N = row_offsets.shape[0]

    kernel_assert(
        output_len % P_MAX == 0,
        f"output_len must be divisible by P_MAX ({P_MAX}), got {output_len}",
    )
    kernel_assert(
        output_len % f_len == 0,
        f"output_len must be divisible by f_len, got {output_len=}, {f_len=}",
    )
    kernel_assert(T % f_len == 0, f"T must be divisible by f_len, got {T=}, {f_len=}")
    kernel_assert(
        (T // f_len) % 16 == 0,
        f"(T // f_len) must be divisible by 16, got {T // f_len}",
    )

    if row_offsets_start is None:
        kernel_assert(
            N == E, f"When row_offsets_start is None, N ({N}) must equal E ({E})"
        )
        row_offsets_start_val = 0
    else:
        kernel_assert(
            N >= E, f"When row_offsets_start is provided, N ({N}) must be >= E ({E})"
        )
        row_offsets_start_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_offsets_start_sb, src=row_offsets_start.reshape((1, 1)))
        row_offsets_start_val = row_offsets_start_sb

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    kernel_assert(
        num_shards == 1 or num_shards == 2,
        f"indexed_flatten supports LNC 1/2, got {num_shards}",
    )

    if num_shards == 1:
        E_per_shard_0 = E
        E_per_shard_1 = E
        E_per_shard = E
        E_offset = 0
    else:
        E_per_shard_0 = div_ceil(E, num_shards)
        E_per_shard_1 = E // num_shards
        E_per_shard = E_per_shard_0 if shard_id == 0 else E_per_shard_1
        E_offset = 0 if shard_id == 0 else E_per_shard_0

    num_output_blocks = output_len // f_len
    partitions_per_row = T // f_len
    partition_tile_count = div_ceil(partitions_per_row, P_MAX)

    flattened_array_partial = nl.ndarray(
        (num_output_blocks, f_len),
        dtype=index_dtype,
        buffer=nl.private_hbm,
    )

    sbuf_init = nl.ndarray(
        (P_MAX, output_len // P_MAX),
        dtype=index_dtype,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=sbuf_init, value=padding_val)
    nisa.dma_copy(
        dst=flattened_array_partial.reshape((P_MAX, output_len // P_MAX)),
        src=sbuf_init,
    )

    input_tensor_reshape = input_tensor.reshape((E, partitions_per_row, f_len))
    row_offsets_2d = row_offsets.reshape((1, N))

    max_E_per_shard = E_per_shard_0
    if num_shards != 1:
        INVALID_OFFSET_VALUE = -1000000
        row_offsets_local = nl.ndarray(
            (1, max_E_per_shard),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        nisa.memset(dst=row_offsets_local, value=INVALID_OFFSET_VALUE)
        if E_per_shard > 0:
            if row_offsets_start is None:
                nisa.dma_copy(
                    dst=row_offsets_local[0:1, 0:E_per_shard],
                    src=row_offsets_2d[0:1, E_offset : E_offset + E_per_shard],
                )
            else:
                nisa.dma_copy(
                    dst=row_offsets_local[0:1, 0:E_per_shard],
                    src=row_offsets_2d.ap(
                        pattern=[[N, 1], [1, E_per_shard]],
                        offset=E_offset,
                        scalar_offset=row_offsets_start_val,
                        indirect_dim=1,
                    ),
                )

    for row_idx_local in nl.sequential_range(max_E_per_shard):
        row_offset_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        if num_shards == 1:
            if row_offsets_start is None:
                nisa.dma_copy(
                    dst=row_offset_sb,
                    src=row_offsets_2d[0:1, row_idx_local : row_idx_local + 1],
                )
            else:
                nisa.dma_copy(
                    dst=row_offset_sb,
                    src=row_offsets_2d.ap(
                        pattern=[[N, 1], [1, 1]],
                        offset=row_idx_local,
                        scalar_offset=row_offsets_start_val,
                        indirect_dim=1,
                    ),
                )
            row_idx = row_idx_local
        else:
            nisa.dma_copy(
                dst=row_offset_sb,
                src=row_offsets_local[0:1, row_idx_local : row_idx_local + 1],
            )
            row_idx = min(row_idx_local + E_offset, E - 1)

        for partition_tile_idx in nl.sequential_range(partition_tile_count):
            partition_start = partition_tile_idx * P_MAX
            partition_count = min(P_MAX, partitions_per_row - partition_start)

            if partition_count > 0:
                input_tile = nl.ndarray(
                    (partition_count, f_len),
                    dtype=index_dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=input_tile,
                    src=input_tensor_reshape[
                        row_idx,
                        partition_start : partition_start + partition_count,
                        0:f_len,
                    ],
                )

                nisa.dma_copy(
                    dst=flattened_array_partial.ap(
                        pattern=[[f_len, partition_count], [1, f_len]],
                        offset=partition_start * f_len,
                        scalar_offset=row_offset_sb,
                        indirect_dim=0,
                    ),
                    src=input_tile,
                    oob_mode=nisa.oob_mode.skip,
                )

    reshaped_reload = flattened_array_partial.reshape((P_MAX, output_len // P_MAX))
    reshaped_reload_local = nl.ndarray(
        (P_MAX, output_len // P_MAX),
        dtype=index_dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=reshaped_reload_local, src=reshaped_reload)

    if num_shards == 1:
        result_sb = reshaped_reload_local
    else:
        reshaped_reload_remote = nl.ndarray(
            (P_MAX, output_len // P_MAX),
            dtype=index_dtype,
            buffer=nl.sbuf,
        )
        nisa.sendrecv(
            src=reshaped_reload_local,
            dst=reshaped_reload_remote,
            send_to_rank=(1 - shard_id),
            recv_from_rank=(1 - shard_id),
            pipe_id=0,
        )

        result_sb = nl.ndarray(
            (P_MAX, output_len // P_MAX),
            dtype=index_dtype,
            buffer=nl.sbuf,
        )
        nisa.tensor_tensor(
            dst=result_sb,
            data1=reshaped_reload_local,
            data2=reshaped_reload_remote,
            op=nl.maximum,
        )

    flattened_array = nl.ndarray((output_len,), dtype=index_dtype, buffer=nl.shared_hbm)
    if shard_id == 0:
        nisa.dma_copy(
            dst=flattened_array.reshape((P_MAX, output_len // P_MAX)),
            src=result_sb,
        )

    return flattened_array
