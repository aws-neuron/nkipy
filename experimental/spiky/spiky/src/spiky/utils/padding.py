# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side padding and unpadding utilities for dynamic shape handling."""

from typing import Dict, List, Tuple

import numpy as np

from .dynamic_shapes import DynamicSpec
from .tensor_metadata import PaddingMetadata


def pad_inputs(
    inputs: List[np.ndarray],
    dynamic_specs: Dict[int, DynamicSpec],
    bucket_size: int,
) -> Tuple[List[np.ndarray], PaddingMetadata]:
    """Pad inputs to bucket size along dynamic dimensions.

    Args:
        inputs: List of numpy arrays to pad
        dynamic_specs: Dictionary mapping arg_idx to DynamicSpec
        bucket_size: Target bucket size to pad to

    Returns:
        Tuple of (padded_inputs, metadata) where metadata tracks the padding
    """
    padded = []
    original_sizes: Dict[int, int] = {}

    for i, inp in enumerate(inputs):
        if i in dynamic_specs:
            spec = dynamic_specs[i]
            original_size = inp.shape[spec.dim_idx]
            original_sizes[i] = original_size

            if original_size < bucket_size:
                # Create padding configuration
                pad_width = [(0, 0)] * inp.ndim
                pad_width[spec.dim_idx] = (0, bucket_size - original_size)
                padded.append(
                    np.pad(inp, pad_width, mode="constant", constant_values=0)
                )
            else:
                padded.append(inp)
        else:
            padded.append(inp)

    # Create metadata from primary spec
    if dynamic_specs:
        primary_spec = list(dynamic_specs.values())[0]
        primary_original = original_sizes.get(primary_spec.arg_idx, bucket_size)
    else:
        primary_spec = None
        primary_original = bucket_size

    metadata = PaddingMetadata(
        original_size=primary_original,
        padded_size=bucket_size,
        pad_dim=primary_spec.dim_idx if primary_spec else -1,
        arg_indices=tuple(dynamic_specs.keys()),
    )

    return padded, metadata


def unpad_outputs(
    outputs: List[np.ndarray],
    metadata: PaddingMetadata,
) -> List[np.ndarray]:
    """Remove padding from outputs based on metadata.

    Args:
        outputs: List of padded numpy arrays
        metadata: PaddingMetadata describing the padding

    Returns:
        List of unpadded numpy arrays
    """
    if metadata.pad_dim < 0:
        return outputs

    unpadded = []
    for out in outputs:
        if out.ndim > metadata.pad_dim:
            slices = [slice(None)] * out.ndim
            slices[metadata.pad_dim] = slice(0, metadata.original_size)
            unpadded.append(out[tuple(slices)])
        else:
            unpadded.append(out)
    return unpadded


def compute_pad_ratio(actual_len: int, bucket_size: int) -> float:
    """Compute the padding waste ratio.

    Args:
        actual_len: Actual sequence length
        bucket_size: Bucket size used for padding

    Returns:
        Ratio of wasted padding (0.0 = no waste, 1.0 = all padding)
    """
    if bucket_size <= 0:
        return 0.0
    return (bucket_size - actual_len) / bucket_size
