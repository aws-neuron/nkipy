# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding metadata tracking for multi-stage pipelines.

This module provides a weak-reference based registry to attach padding
metadata to tensors without preventing garbage collection. This enables
efficient multi-stage pipelines where padded tensors pass between stages
without redundant pad/unpad operations.
"""

import weakref
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class PaddingMetadata:
    """Tracks padding applied to a tensor or batch.

    Attributes:
        original_size: Original size before padding
        padded_size: Size after padding (= bucket_size)
        pad_dim: Dimension index that was padded
        arg_indices: Tuple of argument indices that were padded
    """

    original_size: int
    padded_size: int
    pad_dim: int
    arg_indices: Tuple[int, ...]


# Weak-reference based registry for tensor metadata
# Key: id(tensor), Value: PaddingMetadata
_metadata_registry: Dict[int, PaddingMetadata] = {}

# Track weak refs to enable cleanup when tensors are garbage collected
_weak_refs: Dict[int, weakref.ref] = {}


def _make_cleanup_callback(tensor_id: int):
    """Create a cleanup callback for when a tensor is garbage collected."""

    def callback(ref):
        _metadata_registry.pop(tensor_id, None)
        _weak_refs.pop(tensor_id, None)

    return callback


def attach_metadata(tensor, metadata: PaddingMetadata) -> None:
    """Attach padding metadata to a tensor.

    The metadata is stored in a weak-reference based registry, so it will
    be automatically cleaned up when the tensor is garbage collected.

    Args:
        tensor: The tensor to attach metadata to
        metadata: PaddingMetadata describing the padding applied
    """
    tensor_id = id(tensor)
    _metadata_registry[tensor_id] = metadata
    _weak_refs[tensor_id] = weakref.ref(tensor, _make_cleanup_callback(tensor_id))


def get_metadata(tensor) -> Optional[PaddingMetadata]:
    """Get padding metadata attached to a tensor.

    Args:
        tensor: The tensor to query

    Returns:
        PaddingMetadata if attached, None otherwise
    """
    return _metadata_registry.get(id(tensor))


def has_metadata(tensor) -> bool:
    """Check if a tensor has padding metadata attached.

    Args:
        tensor: The tensor to check

    Returns:
        True if metadata is attached, False otherwise
    """
    return id(tensor) in _metadata_registry


def clear_metadata(tensor) -> Optional[PaddingMetadata]:
    """Remove and return padding metadata from a tensor.

    Args:
        tensor: The tensor to clear metadata from

    Returns:
        The removed PaddingMetadata, or None if none was attached
    """
    tensor_id = id(tensor)
    _weak_refs.pop(tensor_id, None)
    return _metadata_registry.pop(tensor_id, None)


def is_compatible_padded(
    tensor,
    required_bucket: int,
    pad_dim: int,
) -> bool:
    """Check if a tensor is already padded to a compatible bucket size.

    Args:
        tensor: The tensor to check
        required_bucket: The required bucket size
        pad_dim: The dimension that should be padded

    Returns:
        True if tensor is already compatibly padded, False otherwise
    """
    meta = get_metadata(tensor)
    if meta is None:
        return False

    return (
        meta.pad_dim == pad_dim
        and meta.padded_size == required_bucket
        and tensor.shape[pad_dim] == meta.padded_size
    )
