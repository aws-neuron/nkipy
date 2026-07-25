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

"""Vendored nkilib ``find_nonzero_indices`` subkernel for MoE scheduling.

Source: ``nkilib.core.subkernels.find_nonzero_indices`` from the local
``ref_repos/nki-library`` snapshot. This copy is intentionally kept local so
DSV4 scheduling does not depend on an external nkilib package at runtime.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import constants as nisa_constants

from .kernel_assert import kernel_assert
from .kernel_helpers import div_ceil

_QUADRANT_SIZE = 32
_NUM_QUADRANTS = 4
_NUM_GPSIMD_CORES = 8
_GPSIMD_CORES_PER_QUADRANT = 2
_PARTITIONS_PER_GPSIMD = 16


@nki.jit
def find_nonzero_indices(
    input_tensor: nl.ndarray,
    col_start_id: nl.ndarray = None,
    n_cols: int = None,
    chunk_size: int = None,
    index_dtype: nki.dtype = nl.int32,
):
    """Find token indices of nonzero elements per expert column.

    Args:
        input_tensor: ``[T, C]`` HBM tensor. Nonzeros are found along ``T``.
        col_start_id: optional ``[1]`` HBM scalar for a global column offset.
        n_cols: number of columns to process when ``col_start_id`` is set.
        chunk_size: token chunk size. DSV4 passes the full token bucket today.
        index_dtype: output index dtype.

    Returns:
        ``indices [C, T]`` padded with ``-1`` and ``nonzero_counts [C]``.
    """
    T_DIM, C_DIM = input_tensor.shape
    if col_start_id is not None and n_cols is not None:
        col_start_id_sbuf = nl.ndarray(
            (1, 1), dtype=nl.int32, buffer=nl.sbuf, name="col_start_id_sbuf"
        )
        nisa.dma_copy(dst=col_start_id_sbuf, src=col_start_id[0:1])
        C = n_cols
    else:
        col_start_id_sbuf = None
        C = C_DIM

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    kernel_assert(C % num_shards == 0, f"C ({C}) must divide num_shards ({num_shards})")
    C_per_shard = C // num_shards
    C_offset = C_per_shard * shard_id

    P_MAX = nl.tile_size.pmax
    T_TILE_SIZE = P_MAX
    C_TILE_SIZE = P_MAX

    if chunk_size is None:
        chunk_size = T_DIM
    kernel_assert(
        T_DIM % chunk_size == 0,
        f"T_DIM ({T_DIM}) must be divisible by chunk_size ({chunk_size})",
    )
    CHUNK_T_TILES = chunk_size // T_TILE_SIZE
    NUM_CHUNKS = T_DIM // chunk_size

    indices = nl.ndarray((C, T_DIM), dtype=index_dtype, buffer=nl.shared_hbm)

    sbuf_init = nl.ndarray(
        (P_MAX, C_per_shard * T_DIM // P_MAX),
        dtype=index_dtype,
        buffer=nl.sbuf,
        name="sbuf_init",
    )
    nisa.memset(dst=sbuf_init, value=-1)
    reshaped_dst = indices.reshape((P_MAX * num_shards, C_per_shard * T_DIM // P_MAX))
    nisa.dma_copy(
        dst=reshaped_dst[P_MAX * shard_id : P_MAX * (shard_id + 1), :],
        src=sbuf_init,
    )

    nonzero_counts = nl.ndarray((C,), dtype=nl.int32, buffer=nl.shared_hbm)
    nonzero_counts_local = nl.ndarray(
        (1, C_per_shard),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name="nonzero_counts_local",
    )
    nisa.memset(dst=nonzero_counts_local, value=0)

    n_column_rounds = div_ceil(C_per_shard, _NUM_GPSIMD_CORES)
    identity_sb = nl.shared_identity_matrix(P_MAX, dtype=nl.float32)

    for column_round_idx in range(n_column_rounds):
        n_columns_this_round = min(
            _NUM_GPSIMD_CORES,
            C_per_shard - _NUM_GPSIMD_CORES * column_round_idx,
        )
        column_start_offset = column_round_idx * _NUM_GPSIMD_CORES + C_offset

        offsets = nl.ndarray(
            (1, _NUM_GPSIMD_CORES),
            dtype=nl.int32,
            buffer=nl.sbuf,
            name=f"offsets_er-{column_round_idx}",
        )
        nisa.memset(dst=offsets, value=0)
        for chunk_idx in range(NUM_CHUNKS):
            input_sbuf = nl.ndarray(
                (T_TILE_SIZE, CHUNK_T_TILES, _NUM_GPSIMD_CORES),
                dtype=input_tensor.dtype,
                buffer=nl.sbuf,
            )
            input_gpsimd_aligned_sbuf = nl.ndarray(
                (T_TILE_SIZE, CHUNK_T_TILES, C_TILE_SIZE),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            input_gpsimd_aligned_transposed_sbuf = nl.ndarray(
                (C_TILE_SIZE, CHUNK_T_TILES, T_TILE_SIZE),
                dtype=input_tensor.dtype,
                buffer=nl.sbuf,
            )
            indices_sbuf = nl.ndarray(
                (C_TILE_SIZE, 1, chunk_size + 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            t_chunk_start = chunk_idx * chunk_size

            if col_start_id_sbuf is not None:
                nisa.dma_copy(
                    dst=input_sbuf[:, 0:CHUNK_T_TILES, 0:n_columns_this_round],
                    src=input_tensor.ap(
                        pattern=[
                            [C_DIM, T_TILE_SIZE],
                            [C_DIM * T_TILE_SIZE, CHUNK_T_TILES],
                            [1, n_columns_this_round],
                        ],
                        offset=column_start_offset + (t_chunk_start * C_DIM),
                        scalar_offset=col_start_id_sbuf,
                        indirect_dim=1,
                    ),
                    dge_mode=nisa_constants.dge_mode.hwdge,
                )
            else:
                nisa.dma_copy(
                    dst=input_sbuf[:, 0:CHUNK_T_TILES, 0:n_columns_this_round],
                    src=input_tensor.ap(
                        pattern=[
                            [C_DIM, T_TILE_SIZE],
                            [C_DIM * T_TILE_SIZE, CHUNK_T_TILES],
                            [1, n_columns_this_round],
                        ],
                        offset=column_start_offset + (t_chunk_start * C_DIM),
                    ),
                )

            for column_idx in range(n_columns_this_round):
                nisa.tensor_copy(
                    dst=input_gpsimd_aligned_sbuf[
                        :, :, column_idx * _PARTITIONS_PER_GPSIMD
                    ],
                    src=input_sbuf[:, :, column_idx],
                    engine=nisa.engine.scalar,
                )

            for t_tile_idx in range(CHUNK_T_TILES):
                transposed_psum = nl.ndarray(
                    (C_TILE_SIZE, T_TILE_SIZE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_matmul(
                    dst=transposed_psum,
                    stationary=input_gpsimd_aligned_sbuf[:, t_tile_idx, :],
                    moving=identity_sb[0:P_MAX, 0:P_MAX],
                    is_transpose=True,
                )
                nisa.tensor_copy(
                    dst=input_gpsimd_aligned_transposed_sbuf[:, t_tile_idx, :],
                    src=transposed_psum,
                )

            nisa.nonzero_with_count(
                dst=indices_sbuf,
                src=input_gpsimd_aligned_transposed_sbuf,
                index_offset=chunk_idx * chunk_size,
                padding_val=-1,
            )

            for quadrant_idx in range(_NUM_QUADRANTS):
                column_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT
                _store_indices_and_count(
                    indices_sbuf=indices_sbuf,
                    indices=indices,
                    offsets=offsets,
                    column_round_idx=column_round_idx,
                    chunk_idx=chunk_idx,
                    quadrant_idx=quadrant_idx,
                    column_idx=column_idx,
                    n_columns_this_round=n_columns_this_round,
                    C_offset=C_offset,
                    chunk_size=chunk_size,
                    T_DIM=T_DIM,
                    name_prefix="even",
                )

            quad_mask = [_PARTITIONS_PER_GPSIMD] + [255] * (_QUADRANT_SIZE - 1)
            nisa.nc_stream_shuffle(
                dst=indices_sbuf,
                src=indices_sbuf,
                shuffle_mask=quad_mask,
            )

            for quadrant_idx in range(_NUM_QUADRANTS):
                column_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + 1
                _store_indices_and_count(
                    indices_sbuf=indices_sbuf,
                    indices=indices,
                    offsets=offsets,
                    column_round_idx=column_round_idx,
                    chunk_idx=chunk_idx,
                    quadrant_idx=quadrant_idx,
                    column_idx=column_idx,
                    n_columns_this_round=n_columns_this_round,
                    C_offset=C_offset,
                    chunk_size=chunk_size,
                    T_DIM=T_DIM,
                    name_prefix="odd",
                )

        nisa.tensor_copy(
            dst=nonzero_counts_local[
                0:1,
                column_round_idx * _NUM_GPSIMD_CORES : column_round_idx
                * _NUM_GPSIMD_CORES
                + n_columns_this_round,
            ],
            src=offsets[0:1, 0:n_columns_this_round],
        )

    nonzero_counts_reshape = nonzero_counts.reshape((1, C))
    nisa.dma_copy(
        dst=nonzero_counts_reshape[0:1, C_offset : C_offset + C_per_shard],
        src=nonzero_counts_local,
    )

    return indices, nonzero_counts


def _store_indices_and_count(
    indices_sbuf: nl.ndarray,
    indices: nl.ndarray,
    offsets: nl.ndarray,
    column_round_idx: int,
    chunk_idx: int,
    quadrant_idx: int,
    column_idx: int,
    n_columns_this_round: int,
    C_offset: int,
    chunk_size: int,
    T_DIM: int,
    name_prefix: str,
):
    """Store one GpSimd core's nonzero indices/count into HBM."""
    if column_idx >= n_columns_this_round:
        return

    offset_tile = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_offset_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-{quadrant_idx}",
    )
    nisa.tensor_copy(dst=offset_tile, src=offsets[0:1, column_idx : column_idx + 1])

    out_col = C_offset + column_round_idx * _NUM_GPSIMD_CORES + column_idx
    src_data = nl.ndarray(
        (1, chunk_size),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_src_data_er-{column_round_idx}_ch-{chunk_idx}_qi-{quadrant_idx}",
    )
    nisa.tensor_copy(
        dst=src_data,
        src=indices_sbuf[
            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1,
            0,
            0:chunk_size,
        ],
    )
    nisa.dma_copy(
        dst=indices.ap(
            pattern=[[T_DIM, 1], [1, chunk_size]],
            offset=out_col * T_DIM,
            scalar_offset=offset_tile,
            indirect_dim=1,
        ),
        src=src_data,
    )

    count_tile = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_count_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-{quadrant_idx}",
    )
    nisa.tensor_copy(
        dst=count_tile,
        src=indices_sbuf[
            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1,
            0,
            chunk_size : chunk_size + 1,
        ],
    )
    nisa.tensor_tensor(
        dst=offsets[0:1, column_idx : column_idx + 1],
        data1=offsets[0:1, column_idx : column_idx + 1],
        data2=count_tile,
        op=nl.add,
    )
