# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
kernels - Builtin high performance attention kernels
"""
# ruff: noqa

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FlashPAPlan:
    tile_q_indices: npt.NDArray[np.int32]
    tile_block_table_offsets: npt.NDArray[np.int32]
    tile_q_seq_ids: npt.NDArray[np.int32]
    tile_kv_seq_ids: npt.NDArray[np.int32]
    tile_kv_skip_indices: npt.NDArray[np.int32]
    block_size: int
    tile_size_q: int
    tile_size_kv: int
    num_real_tiles: int = None

    def __post_init__(self):
        if self.num_real_tiles is None:
            object.__setattr__(self, "num_real_tiles", self.num_tiles)
        for arg in [
            self.tile_q_indices,
            self.tile_block_table_offsets,
            self.tile_q_seq_ids,
            self.tile_kv_seq_ids,
        ]:
            assert isinstance(arg, np.ndarray) and arg.dtype == np.int32, type(arg)
            assert arg.shape[0] == self.num_tiles, (arg.shape, self.num_tiles)

        if self.tile_kv_skip_indices.size > 0:
            assert self.tile_kv_skip_indices[0] > 0, (
                f"We will never expect to skip first tile"
            )
            assert np.all(self.tile_kv_skip_indices < self.num_tiles)

    @property
    def num_tiles(self):
        return len(self.tile_q_indices)

    def pad_plan(self, pad_num_tile_to, q_pad_value=0):
        num_tiles = self.num_tiles
        if pad_num_tile_to <= num_tiles:
            return self

        def pad(x, pad_to, pad_value=0):
            shape = x.shape
            pad_width = [(0, pad_to - shape[0])] + [(0, 0)] * (len(shape) - 1)
            return np.pad(x, pad_width, mode="constant", constant_values=pad_value)

        tile_q_indices = pad(
            self.tile_q_indices, pad_num_tile_to, pad_value=q_pad_value
        )
        tile_block_tables_offsets = pad(self.tile_block_table_offsets, pad_num_tile_to)
        # pad different value for q and kv seq ids so that sequence affiliation mask is False
        tile_q_seq_ids = pad(self.tile_q_seq_ids, pad_num_tile_to, pad_value=0)
        tile_kv_seq_ids = pad(self.tile_kv_seq_ids, pad_num_tile_to, pad_value=1)
        tile_kv_skip_indices = np.array(
            self.tile_kv_skip_indices.tolist()
            + list(range(num_tiles, pad_num_tile_to)),
            dtype=np.int32,
        )
        return self.__class__(
            tile_q_indices,
            tile_block_tables_offsets,
            tile_q_seq_ids,
            tile_kv_seq_ids,
            tile_kv_skip_indices,
            self.block_size,
            self.tile_size_q,
            self.tile_size_kv,
            self.num_real_tiles,
        )

    def build_tile_q_indices(self, skip_value=None):
        raise NotImplementedError

    def build_tile_block_tables(self, block_tables, skip_value, dynamic_loop_unrolling):
        raise NotImplementedError

    def build_tile_masks(self, decode_kq_matmul):
        tile_kv_seq_ids = self.tile_kv_seq_ids
        B_P_SIZE = 128
        num_tiles, tile_size_kv = tile_kv_seq_ids.shape
        assert tile_size_kv % B_P_SIZE == 0 and tile_size_kv % self.block_size == 0
        num_tiled_blocks = max(B_P_SIZE, tile_size_kv // self.block_size)
        tiled_block_size = tile_size_kv // num_tiled_blocks
        if tiled_block_size > 1:
            tile_kv_seq_ids = tile_kv_seq_ids.reshape(
                (
                    num_tiles,
                    num_tiled_blocks // B_P_SIZE,
                    B_P_SIZE,
                    tiled_block_size,
                )
            )
            tile_kv_seq_ids = tile_kv_seq_ids.transpose(0, 1, 3, 2).reshape(
                (num_tiles, tile_size_kv)
            )
        if decode_kq_matmul:
            tile_masks = np.expand_dims(self.tile_q_seq_ids, 1) == np.expand_dims(
                tile_kv_seq_ids, 2
            )
            tile_masks = tile_masks.reshape(
                (num_tiles, tile_size_kv // B_P_SIZE, B_P_SIZE)
            )
            # Transpose for efficient load
            # New layout: (B_P_SIZE, num_tiles, tile_size_kv // B_P_SIZE)
            tile_masks = tile_masks.transpose(2, 0, 1)
        else:
            tile_masks = np.expand_dims(self.tile_q_seq_ids, 2) == np.expand_dims(
                tile_kv_seq_ids, 1
            )
        return tile_masks

    def build_decode_tile_update_indices(
        self, padded_seqlen_q, build_update_pred=False
    ):
        tile_q_indices = self.tile_q_indices[: self.num_real_tiles].flatten().copy()
        _, num_tiles_per_seq = np.unique(tile_q_indices, return_counts=True)
        last_tile_indices = np.cumsum(num_tiles_per_seq) - 1

        if build_update_pred:
            update_indices = np.ones(self.num_tiles)
            update_indices[self.num_real_tiles :] = 0
            update_indices[last_tile_indices] = 0
            update_indices = update_indices.astype(np.uint8)
        else:
            # let kernel build update_indices using last_tile_indices
            update_indices = None

        # pad last_tile_indices to match padded_seqlen_q
        padded_last_tile_indices = np.empty(padded_seqlen_q, dtype=np.int32)
        padded_last_tile_indices[: len(last_tile_indices)] = last_tile_indices
        padded_last_tile_indices[len(last_tile_indices) :] = last_tile_indices[-1]
        return update_indices, padded_last_tile_indices


class FlashPagedAttentionSchedulerBase:
    """
    Generate schedule for flash attention
    """

    def __init__(
        self, prompt_lens, context_lens, tile_size_q, tile_size_kv, block_size
    ):
        assert tile_size_kv % block_size == 0, (
            "tile_size_kv must be multiple of block_size"
        )

        def _check_np_int_array(*arrays):
            for a in arrays:
                if not isinstance(a, np.ndarray) or a.dtype not in (np.int32, np.int64):
                    return False
            return True

        assert _check_np_int_array(prompt_lens, context_lens)
        assert len(prompt_lens) == len(context_lens), (
            "prompt_lens and context_lens must have the same length"
        )
        self.num_seq = len(prompt_lens)
        assert self.num_seq > 0, "prompt_lens and context_lens must be non-empty"
        self.prompt_lens = prompt_lens.astype(np.int32)
        self.context_lens = context_lens.astype(np.int32)
        self.tile_size_q = tile_size_q
        self.tile_size_kv = tile_size_kv
        self.block_size = block_size

    def generate_plan(self):
        raise NotImplementedError


def _ceil_div(a, b):
    assert b != 0
    return (a + b - 1) // b


def _get_seq_start_end(seqlens, padded_seqlens=None):
    if padded_seqlens is None:
        padded_seqlens = seqlens
    cu_seqlen = np.cumsum(padded_seqlens)
    seqlens_starts = np.concatenate(([0], cu_seqlen[:-1]))
    seqlens_ends = seqlens_starts + seqlens
    return seqlens_starts, seqlens_ends, cu_seqlen[-1]


class GridAlignedPlan(FlashPAPlan):
    def build_tile_q_indices(self):
        in_tile_offset = np.arange(self.tile_size_q, dtype=np.int32)
        q_tile_starts = self.tile_q_indices * self.tile_size_q
        q_indices = q_tile_starts.reshape(-1, 1) + in_tile_offset.reshape(1, -1)
        return q_indices

    def build_tile_block_tables(
        self, block_tables, skip_value, dynamic_loop_unrolling=1
    ):
        tile_size_kv = self.tile_kv_seq_ids.shape[1]
        assert tile_size_kv % self.block_size == 0
        num_blocks_per_tile = tile_size_kv // self.block_size
        block_tables = block_tables.squeeze()
        in_tile_offset = np.arange(num_blocks_per_tile)
        indices = self.tile_block_table_offsets.reshape(-1, 1) + in_tile_offset.reshape(
            1, -1
        )
        tile_block_tables = block_tables[indices]
        if len(self.tile_kv_skip_indices) > 0:
            tile_kv_skip_indices = self.tile_kv_skip_indices
            if dynamic_loop_unrolling is not None:
                indices_not_at_loop_start = (
                    tile_kv_skip_indices % dynamic_loop_unrolling != 0
                )
                tile_kv_skip_indices = tile_kv_skip_indices[indices_not_at_loop_start]
            tile_block_tables[tile_kv_skip_indices, :] = skip_value
        return tile_block_tables


class GridAlignedScheduler(FlashPagedAttentionSchedulerBase):
    def __init__(
        self,
        prompt_lens,
        context_lens,
        tile_size_q,
        tile_size_kv,
        block_size,
        column_order,
        kv_dma_skipping,
    ):
        super(__class__, self).__init__(
            prompt_lens,
            context_lens,
            tile_size_q,
            tile_size_kv,
            block_size,
        )
        self.column_order = column_order
        self.kv_dma_skipping = kv_dma_skipping

    def generate_plan(self):
        num_context_blocks = _ceil_div(self.context_lens, self.block_size)
        padded_context_lens = num_context_blocks * self.block_size
        prompt_starts, prompt_ends, max_seqlen_q = _get_seq_start_end(self.prompt_lens)
        context_starts, context_ends, max_seqlen_kv = _get_seq_start_end(
            self.context_lens, padded_seqlens=padded_context_lens
        )

        # q dimension seq id
        tile_q_starts = np.arange(0, max_seqlen_q, self.tile_size_q)
        tile_q_ends = tile_q_starts + self.tile_size_q
        tile_q_seq_starts = np.searchsorted(prompt_ends, tile_q_starts, side="right")
        tile_q_seq_ends = np.searchsorted(prompt_starts, tile_q_ends, side="left")

        # kv dimension seq id
        tile_kv_starts = np.arange(0, max_seqlen_kv, self.tile_size_kv)
        tile_kv_ends = tile_kv_starts + self.tile_size_kv
        tile_kv_seq_starts = np.searchsorted(context_ends, tile_kv_starts, side="right")
        tile_kv_seq_ends = np.searchsorted(context_starts, tile_kv_ends, side="left")

        # tile_needed = max(q_id_start, kv_id_start) < min(q_id_end, kv_id_end)
        tile_seq_starts = np.maximum(
            tile_q_seq_starts.reshape(-1, 1), tile_kv_seq_starts.reshape(1, -1)
        )
        tile_seq_ends = np.minimum(
            tile_q_seq_ends.reshape(-1, 1), tile_kv_seq_ends.reshape(1, -1)
        )
        tile_needed = tile_seq_starts < tile_seq_ends
        tile_q_indices, tile_kv_indices = np.nonzero(tile_needed)

        num_q_tiles = len(tile_q_starts)
        num_kv_tiles = len(tile_kv_starts)
        q_seq_ids = np.repeat(
            np.arange(self.num_seq + 1, dtype=np.int32),  # use num_seq as padding value
            np.concatenate(
                (self.prompt_lens, [num_q_tiles * self.tile_size_q - max_seqlen_q])
            ),
        ).reshape((num_q_tiles, self.tile_size_q))
        kv_seq_ids = np.repeat(
            np.stack(
                (
                    np.arange(self.num_seq, dtype=np.int32),
                    np.full(
                        (self.num_seq,), self.num_seq + 1, dtype=np.int32
                    ),  # use num_seq + 1 as padding
                )
            ).flatten("F"),
            np.stack(
                (
                    self.context_lens,
                    padded_context_lens - self.context_lens,
                )
            ).flatten("F"),
        )
        kv_seq_ids = np.concatenate(
            (
                kv_seq_ids,
                np.full(
                    (num_kv_tiles * self.tile_size_kv - max_seqlen_kv,),
                    self.num_seq + 1,
                    dtype=np.int32,
                ),
            )
        ).reshape((num_kv_tiles, self.tile_size_kv))

        if self.column_order:
            sort_indices = np.argsort(tile_kv_indices, kind="stable")
            tile_q_indices = tile_q_indices[sort_indices]
            tile_kv_indices = tile_kv_indices[sort_indices]

        tile_q_indices = tile_q_indices.astype(np.int32)
        tile_kv_offsets = tile_kv_indices.astype(np.int32) * self.tile_size_kv
        tile_bt_offsets = tile_kv_offsets // self.block_size
        tile_q_seq_ids = q_seq_ids[tile_q_indices]
        tile_kv_seq_ids = kv_seq_ids[tile_kv_indices]

        # calculate load mask for kv
        if self.kv_dma_skipping:
            prev_indices = np.concatenate(([-1], tile_kv_indices[:-1]))
            tile_kv_skip_indices = np.nonzero(tile_kv_indices == prev_indices)[
                0
            ].astype(np.int32)
        else:
            tile_kv_skip_indices = np.array([], dtype=np.int32)

        return GridAlignedPlan(
            tile_q_indices,
            tile_bt_offsets,
            tile_q_seq_ids,
            tile_kv_seq_ids,
            tile_kv_skip_indices,
            self.block_size,
            self.tile_size_q,
            self.tile_size_kv,
        )


class SequenceAlignedPlan(FlashPAPlan):
    def build_tile_q_indices(self, skip_value=None):
        tile_q_indices = self.tile_q_indices.copy()
        if skip_value is not None:
            tile_q_indices[tile_q_indices == -1] = skip_value
        return tile_q_indices

    def build_tile_block_tables(
        self, block_tables, skip_value, dynamic_loop_unrolling=1
    ):
        block_tables = np.concatenate(
            [block_tables.squeeze(), np.array([skip_value], dtype=np.int32)]
        )
        tile_block_tables = block_tables[self.tile_block_table_offsets]
        if self.tile_kv_skip_indices.size > 0:
            tile_kv_skip_indices = self.tile_kv_skip_indices
            if dynamic_loop_unrolling is not None:
                indices_not_at_loop_start = (
                    tile_kv_skip_indices % dynamic_loop_unrolling != 0
                )
                tile_kv_skip_indices = tile_kv_skip_indices[indices_not_at_loop_start]
            tile_block_tables[tile_kv_skip_indices, :] = skip_value
        return tile_block_tables


class SequenceAlignedScheduler(FlashPagedAttentionSchedulerBase):
    def __init__(
        self,
        prompt_lens,
        context_lens,
        tile_size_q,
        tile_size_kv,
        block_size,
        column_order,
        kv_dma_skipping,
    ):
        super(__class__, self).__init__(
            prompt_lens,
            context_lens,
            tile_size_q,
            tile_size_kv,
            block_size,
        )
        self.column_order = column_order
        self.kv_dma_skipping = kv_dma_skipping

    def generate_plan(self):
        prompt_starts, _, _ = _get_seq_start_end(self.prompt_lens)
        num_context_blocks = _ceil_div(self.context_lens, self.block_size)
        context_block_starts, _, _ = _get_seq_start_end(num_context_blocks)

        # q dimension seq id
        num_seq_q_tiles = _ceil_div(self.prompt_lens, self.tile_size_q)
        num_seq_kv_tiles = _ceil_div(self.context_lens, self.tile_size_kv)

        def _build_seq_ids(seq_id, seqlen, tile_size, indices, pad_value):
            num_tiles = _ceil_div(seqlen, tile_size)
            seq_ids = np.full((num_tiles * tile_size), seq_id, dtype=np.int32)
            seq_ids[seqlen:] = pad_value
            seq_ids = seq_ids.reshape((num_tiles, tile_size))
            return seq_ids[indices]

        def _build_load_offsets(start_id, seqlen, tile_size, indices, pad_value):
            num_tiles = _ceil_div(seqlen, tile_size)
            load_indices = np.arange(num_tiles * tile_size, dtype=np.int32) + start_id
            load_indices[seqlen:] = pad_value
            load_indices = load_indices.reshape((num_tiles, tile_size))
            return load_indices[indices]

        tile_q_offsets = []
        tile_bt_offsets = []
        tile_q_seq_ids = []
        tile_kv_seq_ids = []
        tile_kv_skip_indices = []
        total_num_tiles = 0
        for seq_id, (num_q_tiles, num_kv_tiles) in enumerate(
            zip(num_seq_q_tiles, num_seq_kv_tiles)
        ):
            if num_q_tiles == 0 or num_kv_tiles == 0:
                continue
            num_tiles = (num_q_tiles, num_kv_tiles)
            q_indices = np.broadcast_to(
                np.arange(num_q_tiles, dtype=np.int32).reshape(-1, 1), num_tiles
            )
            kv_indices = np.broadcast_to(
                np.arange(num_kv_tiles, dtype=np.int32).reshape(1, -1), num_tiles
            )
            if self.column_order:
                q_indices = q_indices.transpose()
                kv_indices = kv_indices.transpose()
            q_indices = q_indices.flatten()
            kv_indices = kv_indices.flatten()

            q_offsets = _build_load_offsets(
                prompt_starts[seq_id],
                self.prompt_lens[seq_id],
                self.tile_size_q,
                q_indices,
                -1,
            )
            bt_offsets = _build_load_offsets(
                context_block_starts[seq_id],
                num_context_blocks[seq_id],
                self.tile_size_kv // self.block_size,
                kv_indices,
                -1,
            )

            q_seq_ids = _build_seq_ids(
                seq_id,
                self.prompt_lens[seq_id],
                self.tile_size_q,
                q_indices,
                self.num_seq,
            )
            kv_seq_ids = _build_seq_ids(
                seq_id,
                self.context_lens[seq_id],
                self.tile_size_kv,
                kv_indices,
                self.num_seq + 1,
            )

            tile_q_offsets.append(q_offsets)
            tile_bt_offsets.append(bt_offsets)
            tile_q_seq_ids.append(q_seq_ids)
            tile_kv_seq_ids.append(kv_seq_ids)
            if self.kv_dma_skipping:
                # calculate load mask for kv
                prev_kv_indices = np.concatenate(([-1], kv_indices[:-1]))
                kv_skip_indices = (
                    np.nonzero(kv_indices == prev_kv_indices)[0] + total_num_tiles
                ).astype(np.int32)
                tile_kv_skip_indices.append(kv_skip_indices)

            total_num_tiles += num_q_tiles * num_kv_tiles

        if len(tile_q_offsets) > 0:
            tile_q_offsets = np.concatenate(tile_q_offsets)
            tile_bt_offsets = np.concatenate(tile_bt_offsets)
            tile_q_seq_ids = np.concatenate(tile_q_seq_ids)
            tile_kv_seq_ids = np.concatenate(tile_kv_seq_ids)
            if self.kv_dma_skipping:
                tile_kv_skip_indices = np.concatenate(tile_kv_skip_indices)
                if self.tile_size_q == 1:
                    assert len(tile_kv_skip_indices) == 0
            else:
                tile_kv_skip_indices = np.array([], dtype=np.int32)
        else:
            tile_q_offsets = np.empty((0, self.tile_size_q), dtype=np.int32)
            tile_bt_offsets = np.empty(
                (0, self.tile_size_kv // self.block_size), dtype=np.int32
            )
            tile_q_seq_ids = np.empty((0, self.tile_size_q), dtype=np.int32)
            tile_kv_seq_ids = np.empty((0, self.tile_size_kv), dtype=np.int32)
            tile_kv_skip_indices = np.array([], dtype=np.int32)

        return SequenceAlignedPlan(
            tile_q_offsets,
            tile_bt_offsets,
            tile_q_seq_ids,
            tile_kv_seq_ids,
            tile_kv_skip_indices,
            self.block_size,
            self.tile_size_q,
            self.tile_size_kv,
        )
