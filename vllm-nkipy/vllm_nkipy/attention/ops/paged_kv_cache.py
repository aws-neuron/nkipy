# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch


def reshape_and_cache_kv1(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (torch.Tensor): Key tensor with shape
            (num_tokens, n_kv_head, d_head)
        value (torch.Tensor): Value tensor with shape
            (num_tokens, n_kv_head, d_head)
        kv_cache (torch.Tensor): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (torch.Tensor): Mapping tensor indicating cache positions
            with shape (num_tokens)

    Returns:
        None: Updates the kv_cache tensor in-place
    """
    _, NB, n_kv_head, BS, d_head = kv_cache.shape
    num_tokens, _, _ = key.shape
    assert n_kv_head == 1

    # key_cache = tmp.select(0, 0).flatten(0, 1)
    # value_cache = tmp.select(0, 1).flatten(0, 1)

    kv_cache_reshape = kv_cache.reshape(2, -1, d_head)
    key_reshape = key.reshape(num_tokens, -1)
    value_reshape = value.reshape(num_tokens, -1)
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([0], device=key.device), slot_mapping,),
    #     key,
    # )
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([1], device=key.device), slot_mapping,),
    #     value,
    # )

    # kv_cache_reshape[0, slot_mapping, :] = key_reshape
    # kv_cache_reshape[1, slot_mapping, :] = value_reshape

    kv_cache_reshape[0].index_put_((slot_mapping,), key_reshape)
    kv_cache_reshape[1].index_put_((slot_mapping,), value_reshape)


def reshape_and_cache_kv1_index0(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (torch.Tensor): Key tensor with shape
            (num_tokens, n_kv_head, d_head)
        value (torch.Tensor): Value tensor with shape
            (num_tokens, n_kv_head, d_head)
        kv_cache (torch.Tensor): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (torch.Tensor): Mapping tensor indicating cache positions
            with shape (num_tokens)

    Returns:
        None: Updates the kv_cache tensor in-place
    """
    _, NB, n_kv_head, BS, d_head = kv_cache.shape
    num_tokens, _, _ = key.shape
    assert n_kv_head == 1

    # key_cache = tmp.select(0, 0).flatten(0, 1)
    # value_cache = tmp.select(0, 1).flatten(0, 1)

    kv_cache_reshape = kv_cache.reshape(-1, d_head)
    key_reshape = key.reshape(num_tokens, -1)
    value_reshape = value.reshape(num_tokens, -1)
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([0], device=key.device), slot_mapping,),
    #     key,
    # )
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([1], device=key.device), slot_mapping,),
    #     value,
    # )

    # kv_cache_reshape[0, slot_mapping, :] = key_reshape
    # kv_cache_reshape[1, slot_mapping, :] = value_reshape
    k_slot_mapping = slot_mapping
    v_slot_mapping = slot_mapping + NB * BS

    kv_cache_reshape.index_put_((k_slot_mapping,), key_reshape)
    kv_cache_reshape.index_put_((v_slot_mapping,), value_reshape)


def reshape_and_cache_kv1_const(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (torch.Tensor): Key tensor with shape
            (num_tokens, n_kv_head, d_head)
        value (torch.Tensor): Value tensor with shape
            (num_tokens, n_kv_head, d_head)
        kv_cache (torch.Tensor): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (torch.Tensor): Mapping tensor indicating cache positions
            with shape (num_tokens)

    Returns:
        None: Updates the kv_cache tensor in-place
    """
    _, NB, n_kv_head, BS, d_head = kv_cache.shape
    num_tokens, _, _ = key.shape
    assert n_kv_head == 1

    # key_cache = tmp.select(0, 0).flatten(0, 1)
    # value_cache = tmp.select(0, 1).flatten(0, 1)

    kv_cache_reshape = kv_cache.reshape(2, -1, d_head)
    key_reshape = key.reshape(num_tokens, -1)
    value_reshape = value.reshape(num_tokens, -1)
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([0], device=key.device), slot_mapping,),
    #     key,
    # )
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([1], device=key.device), slot_mapping,),
    #     value,
    # )

    # kv_cache_reshape[0, slot_mapping, :] = key_reshape
    # kv_cache_reshape[1, slot_mapping, :] = value_reshape

    kv_cache_reshape.index_put_(
        (
            torch.tensor([0], device=key.device, dtype=torch.int32),
            slot_mapping,
        ),
        key_reshape,
    )
    kv_cache_reshape.index_put_(
        (
            torch.tensor([1], device=key.device, dtype=torch.int32),
            slot_mapping,
        ),
        value_reshape,
    )


def reshape_and_cache_scatter(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Writes key-value pairs to the KV cache at specified positions.

    Args:
        key (torch.Tensor): Key tensor with shape
            (num_tokens, n_kv_head, d_head)
        value (torch.Tensor): Value tensor with shape
            (num_tokens, n_kv_head, d_head)
        kv_cache (torch.Tensor): Key/value cache tensor with shape
            (2, num_blocks, n_kv_head, block_size, d_head)
        slot_mapping (torch.Tensor): Mapping tensor indicating cache positions
            with shape (num_tokens)

    Returns:
        None: Updates the kv_cache tensor in-place
    """
    _, NB, n_kv_head, BS, d_head = kv_cache.shape
    num_tokens, _, _ = key.shape
    assert n_kv_head == 1

    # key_cache = tmp.select(0, 0).flatten(0, 1)
    # value_cache = tmp.select(0, 1).flatten(0, 1)

    kv_cache_reshape = kv_cache.reshape(2, -1, d_head)
    key_reshape = key.reshape(num_tokens, -1)
    value_reshape = value.reshape(num_tokens, -1)
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([0], device=key.device), slot_mapping,),
    #     key,
    # )
    # kv_cache_reshape.index_put_(
    #     (torch.tensor([1], device=key.device), slot_mapping,),
    #     value,
    # )

    # kv_cache_reshape[0, slot_mapping, :] = key_reshape
    # kv_cache_reshape[1, slot_mapping, :] = value_reshape

    kv_cache_reshape.index_put_(
        (
            torch.tensor([0], device=key.device, dtype=torch.int32),
            slot_mapping,
        ),
        key_reshape,
    )
    kv_cache_reshape.index_put_(
        (
            torch.tensor([1], device=key.device, dtype=torch.int32),
            slot_mapping,
        ),
        value_reshape,
    )
