# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Flash Paged Attention kernels with variable-length sequence inputs.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .constants import B_P_SIZE


def _pad_dim(x, pad_to, dim=0, pad_value=0):
    shape = x.shape
    old_dim = dim
    if dim < 0:
        dim += len(shape)
    assert 0 <= dim < len(x.shape), f"padding on dim {old_dim} of {len(shape)}-D tensor"
    padded_amount = pad_to - shape[dim]
    if padded_amount == 0:
        return x
    assert padded_amount > 0, f"Padding dim size {shape[dim]=} to {pad_to}"
    pad_width = (
        [(0, 0)] * dim + [(0, padded_amount)] + [(0, 0)] * (len(shape) - dim - 1)
    )
    return np.pad(x, pad_width, mode="constant", constant_values=pad_value)


def _kv_token_reorder_for_dge(kv_tokens, tile_size_kv, block_size):
    num_tiled_blocks = max(B_P_SIZE, tile_size_kv // block_size)
    tiled_block_size = tile_size_kv // num_tiled_blocks
    if tiled_block_size > 1:
        original_shape = kv_tokens.shape
        kv_tokens = kv_tokens.reshape(
            (
                -1,
                num_tiled_blocks // B_P_SIZE,
                B_P_SIZE,
                tiled_block_size,
            )
        )
        kv_tokens = kv_tokens.transpose(0, 1, 3, 2).reshape(original_shape)
    return kv_tokens


def _build_mask_from_position_ids(
    q_pos,
    kv_pos,
    q_max_pos,
    kv_max_pos,
    decode_kq_layout,
    tile_size_q,
    tile_size_kv,
    improve_dma_layout,
    sliding_window,
):
    num_tiles = q_max_pos.size
    assert kv_max_pos.size == num_tiles
    q_in_range = q_pos < q_max_pos.reshape((num_tiles, 1))
    kv_in_range = kv_pos < kv_max_pos.reshape((num_tiles, 1))
    if decode_kq_layout:
        tile_masks = np.expand_dims(q_in_range, 1) & np.expand_dims(kv_in_range, 2)
        if sliding_window is not None:
            window_mask = np.expand_dims(kv_pos, 2) > np.expand_dims(
                q_pos - sliding_window, 1
            )
            tile_masks = tile_masks & window_mask
    else:
        causal_mask = np.expand_dims(q_pos, 2) >= np.expand_dims(kv_pos, 1)
        tile_masks = (
            np.expand_dims(q_in_range, 2) & np.expand_dims(kv_in_range, 1) & causal_mask
        )
        if sliding_window is not None:
            window_mask = np.expand_dims(kv_pos, 1) > np.expand_dims(
                q_pos - sliding_window, 2
            )
            print(f"{window_mask.shape=}, {tile_masks.shape=}")
            tile_masks = tile_masks & window_mask

    tile_masks = tile_masks.astype(np.uint8)
    if improve_dma_layout:
        if decode_kq_layout:
            assert tile_size_kv % B_P_SIZE == 0, (
                f"{tile_size_kv=} not multiple of {B_P_SIZE=}"
            )
            tile_masks = tile_masks.reshape(
                (num_tiles, tile_size_kv // B_P_SIZE, B_P_SIZE)
            )
            # Transpose for efficient load
            # New layout: (B_P_SIZE, num_tiles, tile_size_kv // B_P_SIZE)
            tile_masks = tile_masks.transpose(2, 0, 1)
        else:
            # Transpose for efficient load
            # New layout: (B_P_SIZE, num_tiles, tile_size_q // B_P_SIZE, tile_size_kv)
            inner_tile_size_q = min(tile_size_q, B_P_SIZE)
            assert tile_size_q % inner_tile_size_q == 0, (
                f"{tile_size_q=} not multiple of {inner_tile_size_q=}"
            )
            tile_masks = tile_masks.reshape(
                (
                    num_tiles,
                    tile_size_q // inner_tile_size_q,
                    inner_tile_size_q,
                    tile_size_kv,
                )
            )
            tile_masks = tile_masks.transpose(2, 0, 1, 3)
    return tile_masks


@dataclass(frozen=True)
class FlashTilePlan:
    tile_q_indices: npt.NDArray[np.int32]
    tile_block_table_offsets: npt.NDArray[np.int32]
    # max q pos of request
    tile_q_max_pos: npt.NDArray[np.int32]
    # max kv pos of request
    tile_kv_max_pos: npt.NDArray[np.int32]
    tile_q_start_pos: npt.NDArray[np.int32]
    tile_kv_start_pos: npt.NDArray[np.int32]
    tile_kv_skip_indices: npt.NDArray[np.int32]
    block_size: int
    tile_size_q: int
    tile_size_kv: int
    req_num_real_tiles: npt.NDArray[np.int32]

    def __post_init__(self):
        def _check_int_array(arg):
            assert isinstance(arg, np.ndarray) and arg.dtype == np.int32, type(arg)

        for arg in [
            self.tile_q_indices,
            self.tile_block_table_offsets,
            self.tile_q_max_pos,
            self.tile_kv_max_pos,
            self.tile_q_start_pos,
            self.tile_kv_start_pos,
        ]:
            if arg is not None:
                _check_int_array(arg)
                assert arg.shape[0] == self.num_tiles, (arg.shape, self.num_tiles)
        _check_int_array(self.req_num_real_tiles)
        assert len(self.tile_kv_skip_indices) == self.num_tiles - self.num_real_tiles
        if self.tile_kv_skip_indices.size > 0:
            assert self.tile_kv_skip_indices[0] > 0, (
                "We will never expect to skip first tile"
            )
            assert np.all(self.tile_kv_skip_indices < self.num_tiles)
        assert (
            self.tile_size_kv % B_P_SIZE == 0
            and self.tile_size_kv % self.block_size == 0
        )
        assert self.tile_size_kv % self.block_size == 0

    def create_sharded_plan(self, num_shards: int):
        assert self.block_size % num_shards == 0
        return ShardedTilePlan(**self.__dict__, num_shards=num_shards)

    @property
    def num_real_tiles(self):
        return self.req_num_real_tiles.sum()

    @property
    def num_tiles(self):
        return len(self.tile_q_indices)

    def pad_plan(self, pad_num_tile_to, q_pad_value=0):
        num_tiles = self.num_tiles
        if pad_num_tile_to <= num_tiles:
            return self

        tile_q_indices = _pad_dim(
            self.tile_q_indices,
            pad_num_tile_to,
            dim=0,
            pad_value=q_pad_value,
        )

        # zero pad dim 0 of the following
        tile_block_tables_offsets = _pad_dim(
            self.tile_block_table_offsets, pad_num_tile_to
        )
        tile_q_max_pos = _pad_dim(self.tile_q_max_pos, pad_num_tile_to)
        tile_kv_max_pos = _pad_dim(self.tile_kv_max_pos, pad_num_tile_to)
        tile_q_start_pos = _pad_dim(self.tile_q_start_pos, pad_num_tile_to)
        tile_kv_start_pos = _pad_dim(self.tile_kv_start_pos, pad_num_tile_to)

        tile_kv_skip_indices = np.array(
            self.tile_kv_skip_indices.tolist()
            + list(range(num_tiles, pad_num_tile_to)),
            dtype=np.int32,
        )
        return FlashTilePlan(
            tile_q_indices=tile_q_indices,
            tile_block_table_offsets=tile_block_tables_offsets,
            tile_q_max_pos=tile_q_max_pos,
            tile_kv_max_pos=tile_kv_max_pos,
            tile_q_start_pos=tile_q_start_pos,
            tile_kv_start_pos=tile_kv_start_pos,
            tile_kv_skip_indices=tile_kv_skip_indices,
            block_size=self.block_size,
            tile_size_q=self.tile_size_q,
            tile_size_kv=self.tile_size_kv,
            req_num_real_tiles=self.req_num_real_tiles,
        )

    def build_tile_q_indices(self, skip_value=None):
        tile_q_indices = self.tile_q_indices.copy()
        if skip_value is not None:
            tile_q_indices[tile_q_indices == -1] = skip_value
        return tile_q_indices

    def build_tile_block_tables(self, block_tables, skip_value):
        block_tables = np.concatenate(
            [block_tables.squeeze(), np.array([skip_value], dtype=np.int32)]
        )
        tile_block_tables = block_tables[self.tile_block_table_offsets]
        if self.tile_kv_skip_indices.size > 0:
            tile_kv_skip_indices = self.tile_kv_skip_indices
            tile_block_tables[tile_kv_skip_indices, :] = skip_value
        return tile_block_tables

    def build_tile_masks(
        self, decode_kq_layout, improve_dma_layout=True, sliding_window=None
    ):
        num_tiles = self.num_tiles
        q_pos = np.arange(self.tile_size_q, dtype=np.int32).reshape(
            (1, self.tile_size_q)
        )
        q_pos = q_pos + self.tile_q_start_pos.reshape((num_tiles, 1))
        kv_pos = np.arange(self.tile_size_kv, dtype=np.int32).reshape(
            (1, self.tile_size_kv)
        )
        kv_pos = _kv_token_reorder_for_dge(
            kv_pos, tile_size_kv=self.tile_size_kv, block_size=self.block_size
        ) + self.tile_kv_start_pos.reshape((num_tiles, 1))
        return _build_mask_from_position_ids(
            q_pos=q_pos,
            kv_pos=kv_pos,
            q_max_pos=self.tile_q_max_pos,
            kv_max_pos=self.tile_kv_max_pos,
            decode_kq_layout=decode_kq_layout,
            tile_size_q=self.tile_size_q,
            tile_size_kv=self.tile_size_kv,
            improve_dma_layout=improve_dma_layout,
            sliding_window=sliding_window,
        )

    def build_tile_update_indices(self, max_num_q_tiles, build_update_pred=True):
        tile_q_offsets = self.tile_q_indices[: self.num_real_tiles, :1].flatten().copy()
        _, num_tiles_per_seq = np.unique(tile_q_offsets, return_counts=True)
        last_tile_indices = np.cumsum(num_tiles_per_seq) - 1
        assert len(last_tile_indices) <= max_num_q_tiles, (
            f"Number of q tiles {len(last_tile_indices)} is > {max_num_q_tiles=}"
        )

        if build_update_pred:
            update_indices = np.ones(self.num_tiles)
            update_indices[self.num_real_tiles :] = 0
            update_indices[last_tile_indices] = 0
            update_indices = update_indices.astype(np.uint8)
        else:
            # let kernel build update_indices using last_tile_indices
            update_indices = None

        # pad last_tile_indices to match padded_seqlen
        padded_last_tile_indices = np.empty(max_num_q_tiles, dtype=np.int32)
        padded_last_tile_indices[: len(last_tile_indices)] = last_tile_indices
        padded_last_tile_indices[len(last_tile_indices) :] = (
            0 if last_tile_indices.size == 0 else last_tile_indices[-1]
        )
        return update_indices, padded_last_tile_indices


@dataclass(frozen=True)
class ShardedTilePlan(FlashTilePlan):
    num_shards: int

    def build_tile_masks(self, decode_kq_layout, improve_dma_layout=True):
        num_tiles = self.num_tiles
        assert self.block_size % self.num_shards == 0
        num_blocks = self.tile_size_kv // self.block_size
        sharded_block_size = self.block_size // self.num_shards
        sharded_tile_size_kv = self.tile_size_kv // self.num_shards
        q_pos = np.arange(self.tile_size_q, dtype=np.int32).reshape(
            (1, self.tile_size_q)
        )
        q_pos = q_pos + self.tile_q_start_pos.reshape((num_tiles, 1))
        kv_pos = np.arange(self.tile_size_kv, dtype=np.int32).reshape(
            (1, self.tile_size_kv)
        )
        kv_pos = (
            kv_pos.reshape((num_blocks, self.num_shards, sharded_block_size))
            .transpose(1, 0, 2)
            .reshape((self.num_shards, sharded_tile_size_kv))
        )
        tile_masks = []
        for shard_id in range(self.num_shards):
            sharded_kv_pos = _kv_token_reorder_for_dge(
                kv_pos[shard_id],
                tile_size_kv=sharded_tile_size_kv,
                block_size=sharded_block_size,
            ) + self.tile_kv_start_pos.reshape((num_tiles, 1))
            mask = _build_mask_from_position_ids(
                q_pos=q_pos,
                kv_pos=sharded_kv_pos,
                q_max_pos=self.tile_q_max_pos,
                kv_max_pos=self.tile_kv_max_pos,
                decode_kq_layout=decode_kq_layout,
                tile_size_q=self.tile_size_q,
                tile_size_kv=sharded_tile_size_kv,
                improve_dma_layout=improve_dma_layout,
            )
            tile_masks.append(mask)
        return tile_masks


def _ceil_div(a, b):
    assert b != 0
    return (a + b - 1) // b


class FlashAttentionPlanner:
    """
    Generate schedule for flash attention
    """

    def __init__(
        self,
        *,
        prompt_lens,
        prior_context_lens,
        tile_size_q,
        tile_size_kv,
        block_size,
        include_prompt_in_context=False,
        max_seq_len=None,
    ):
        assert tile_size_kv % block_size == 0, (
            "tile_size_kv must be multiple of block_size"
        )

        def _check_np_int_array(*arrays):
            for a in arrays:
                if not isinstance(a, np.ndarray) or a.dtype not in (np.int32, np.int64):
                    return False
            return True

        assert _check_np_int_array(prompt_lens, prior_context_lens)
        assert len(prompt_lens) == len(prior_context_lens), (
            "prompt_lens and prior_context_lens must have the same length"
        )
        self.num_seq = len(prompt_lens)
        assert self.num_seq > 0, "prompt_lens and context_lens must be non-empty"
        self.prompt_lens = prompt_lens.astype(np.int32)
        self.prior_context_lens = prior_context_lens.astype(np.int32)
        self.tile_size_q = tile_size_q
        self.tile_size_kv = tile_size_kv
        self.block_size = block_size
        self.include_prompt_in_context = include_prompt_in_context
        self.max_seq_len = max_seq_len
        if self.max_seq_len is not None:
            if self.include_prompt_in_context:
                # user specified total sequence len for each request
                assert np.all(
                    self.prior_context_lens + self.prompt_lens <= self.max_seq_len
                ), f"{self.prior_context_lens=} {self.prompt_lens=} {self.max_seq_len=}"
            else:
                # user specified total sequence len for each request
                assert np.all(self.prior_context_lens <= self.max_seq_len)

    def generate_plan(self):
        def _get_cu_start(seqlens):
            cu_seqlen = np.cumsum(seqlens)
            seqlens_starts = np.concatenate(([0], cu_seqlen[:-1]))
            return seqlens_starts

        if self.include_prompt_in_context:
            full_context_lens = self.prompt_lens + self.prior_context_lens
        else:
            full_context_lens = self.prior_context_lens
        if self.max_seq_len is None:
            seq_lens = full_context_lens
        else:
            seq_lens = np.full(
                (len(full_context_lens),), self.max_seq_len, dtype=np.int32
            )

        prompt_starts = _get_cu_start(self.prompt_lens)
        num_seq_blocks = _ceil_div(seq_lens, self.block_size)
        context_block_starts = _get_cu_start(num_seq_blocks)
        num_context_blocks = _ceil_div(full_context_lens, self.block_size)

        # q dimension seq id
        num_seq_q_tiles = _ceil_div(self.prompt_lens, self.tile_size_q)
        num_seq_kv_tiles = _ceil_div(full_context_lens, self.tile_size_kv)

        def _build_load_offsets(start_idx, seqlen, tile_size, indices, pad_value):
            num_tiles = _ceil_div(seqlen, tile_size)
            load_indices = np.arange(num_tiles * tile_size, dtype=np.int32) + start_idx
            load_indices[seqlen:] = pad_value
            load_indices = load_indices.reshape((num_tiles, tile_size))
            return load_indices[indices]

        def _build_start_pos(seqlen, tile_size, indices):
            num_tiles = _ceil_div(seqlen, tile_size)
            # tile pos
            tile_start_pos = np.arange(num_tiles, dtype=np.int32) * tile_size
            # valid seq len
            return tile_start_pos[indices]

        tile_q_offsets = []
        tile_bt_offsets = []
        tile_q_max_pos = []
        tile_kv_max_pos = []
        tile_q_start_pos = []
        tile_kv_start_pos = []
        total_num_tiles = 0
        req_num_tiles = np.zeros_like(self.prior_context_lens)
        for seq_id, (num_q_tiles, num_kv_tiles) in enumerate(
            zip(num_seq_q_tiles, num_seq_kv_tiles)
        ):
            req_num_tiles[seq_id] = num_q_tiles * num_kv_tiles
            if req_num_tiles[seq_id] == 0:
                continue

            num_tiles = (num_q_tiles, num_kv_tiles)
            local_q_tile_indices = np.broadcast_to(
                np.arange(num_q_tiles, dtype=np.int32).reshape(-1, 1), num_tiles
            )
            local_kv_tile_indices = np.broadcast_to(
                np.arange(num_kv_tiles, dtype=np.int32).reshape(1, -1), num_tiles
            )
            local_q_tile_indices = local_q_tile_indices.flatten()
            local_kv_tile_indices = local_kv_tile_indices.flatten()

            q_offsets = _build_load_offsets(
                prompt_starts[seq_id],
                self.prompt_lens[seq_id],
                self.tile_size_q,
                local_q_tile_indices,
                -1,
            )
            q_start_pos = (
                _build_start_pos(
                    self.prompt_lens[seq_id],
                    self.tile_size_q,
                    local_q_tile_indices,
                )
                + self.prior_context_lens[seq_id]
            )
            bt_offsets = _build_load_offsets(
                context_block_starts[seq_id],
                num_context_blocks[seq_id],
                self.tile_size_kv // self.block_size,
                local_kv_tile_indices,
                -1,
            )
            kv_start_pos = _build_start_pos(
                full_context_lens[seq_id],
                self.tile_size_kv,
                local_kv_tile_indices,
            )
            tile_q_offsets.append(q_offsets)
            tile_bt_offsets.append(bt_offsets)
            tile_q_max_pos.append(
                np.full(
                    (req_num_tiles[seq_id],),
                    self.prompt_lens[seq_id] + self.prior_context_lens[seq_id],
                    dtype=np.int32,
                )
            )
            tile_kv_max_pos.append(
                np.full(
                    (req_num_tiles[seq_id],),
                    full_context_lens[seq_id],
                    dtype=np.int32,
                )
            )
            tile_q_start_pos.append(q_start_pos)
            tile_kv_start_pos.append(kv_start_pos)

            total_num_tiles += num_q_tiles * num_kv_tiles

        if len(tile_q_offsets) > 0:
            tile_q_offsets = np.concatenate(tile_q_offsets)
            tile_bt_offsets = np.concatenate(tile_bt_offsets)
            tile_q_max_pos = np.concatenate(tile_q_max_pos)
            tile_kv_max_pos = np.concatenate(tile_kv_max_pos)
            tile_q_start_pos = np.concatenate(tile_q_start_pos)
            tile_kv_start_pos = np.concatenate(tile_kv_start_pos)
        else:
            tile_q_offsets = np.empty((0, self.tile_size_q), dtype=np.int32)
            tile_bt_offsets = np.empty(
                (0, self.tile_size_kv // self.block_size), dtype=np.int32
            )
            tile_q_max_pos = np.array([], dtype=np.int32)
            tile_kv_max_pos = np.array([], dtype=np.int32)
            tile_q_start_pos = np.array([], dtype=np.int32)
            tile_kv_start_pos = np.array([], dtype=np.int32)
        tile_kv_skip_indices = np.array([], dtype=np.int32)

        return FlashTilePlan(
            tile_q_indices=tile_q_offsets,
            tile_block_table_offsets=tile_bt_offsets,
            tile_q_max_pos=tile_q_max_pos,
            tile_kv_max_pos=tile_kv_max_pos,
            tile_q_start_pos=tile_q_start_pos,
            tile_kv_start_pos=tile_kv_start_pos,
            tile_kv_skip_indices=tile_kv_skip_indices,
            block_size=self.block_size,
            tile_size_q=self.tile_size_q,
            tile_size_kv=self.tile_size_kv,
            req_num_real_tiles=req_num_tiles,
        )
