# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
#
# Vendored from vllm-nkipy: NKI kernel for scatter-writing K,V into paged cache.
# Removed NKIOpRegistry dependency — decorated with @nki.jit so nkipy tracing
# can capture it as an NKI custom call rather than executing NKI ops directly.
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

# Define tile size for num_tokens dimension
MAX_TOKENS_PER_TILE = 128


def cdiv(a, b):
    return (a + b - 1) // b


@nki.jit
def update_kv_cache(
    key: nt.tensor,
    value: nt.tensor,
    kv_cache: nt.tensor[nt.mutable],
    slot_mapping: nt.tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (nt.tensor): Key tensor with shape (num_tokens, n_kv_head, d_head)
        value (nt.tensor): Value tensor with shape (num_tokens, n_kv_head, d_head)
        kv_cache (nt.tensor[nt.mutable]): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (nt.tensor): Mapping tensor indicating cache positions
            with shape (num_tokens,)
    """
    _, num_blocks, n_kv_head, block_size, d_head = kv_cache.shape
    num_tokens, n_kv_head, d_head = key.shape

    # Total cache size for linearized indexing
    total_cache_size = num_blocks * block_size

    # Reshape kv_cache to (2 * num_blocks * block_size, d_head) for linear indexing
    kv_cache_reshaped = kv_cache.reshape((2 * total_cache_size, d_head))

    key = key.reshape((num_tokens, d_head))
    value = value.reshape((num_tokens, d_head))

    # Assert that num_tokens is either divisible by 128 or less than 128
    assert num_tokens % MAX_TOKENS_PER_TILE == 0 or num_tokens < MAX_TOKENS_PER_TILE, (
        f"num_tokens ({num_tokens}) must be divisible"
        f" by {MAX_TOKENS_PER_TILE} or less than"
        f" {MAX_TOKENS_PER_TILE}"
    )

    num_token_tiles = cdiv(num_tokens, MAX_TOKENS_PER_TILE)

    # Process tokens in tiles
    slot_mapping_reshape = slot_mapping.reshape((num_tokens, 1))
    for i in nl.affine_range(num_token_tiles):
        # Determine current tile size
        if num_tokens <= MAX_TOKENS_PER_TILE:
            current_tile_size = num_tokens
            token_offset = 0
        else:
            current_tile_size = MAX_TOKENS_PER_TILE
            token_offset = i * MAX_TOKENS_PER_TILE

        # Load current tile data into SBUF
        i_token_in_tile = nl.arange(current_tile_size)[:, None]
        i_head = nl.arange(d_head)[None, :]

        slot_mapping_sbuf = nl.load(
            slot_mapping_reshape[token_offset : token_offset + current_tile_size],
        )
        key_sbuf = nl.load(key[token_offset : token_offset + current_tile_size])
        value_sbuf = nl.load(value[token_offset : token_offset + current_tile_size])

        # Calculate write offsets for keys (slot_mapping directly)
        key_write_offsets = slot_mapping_sbuf
        value_write_offsets = nisa.tensor_scalar(
            slot_mapping_sbuf, op0=np.add, operand0=total_cache_size
        )

        # Write keys to cache using computed indices
        nl.store(
            dst=kv_cache_reshaped[key_write_offsets[i_token_in_tile, 0], i_head],
            value=key_sbuf[i_token_in_tile, i_head],
        )

        # Write values to cache using computed indices
        nl.store(
            dst=kv_cache_reshaped[value_write_offsets[i_token_in_tile, 0], i_head],
            value=value_sbuf[i_token_in_tile, i_head],
        )

    return kv_cache
