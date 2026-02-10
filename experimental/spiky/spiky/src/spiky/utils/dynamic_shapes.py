# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shape discovery and bucket inference utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class DynamicSpec:
    """Specification for a dynamic dimension.

    Attributes:
        arg_idx: Index of the input argument with dynamic dimension
        dim_idx: Index of the dynamic dimension within the tensor
        min_size: Minimum expected size (default: 1)
        max_size: Maximum expected size (default: 2048)
    """

    arg_idx: int
    dim_idx: int
    min_size: int = 1
    max_size: int = 2048


def discover_dynamic_specs(
    gm: Optional[Any],  # torch.fx.GraphModule, optional for flexibility
    example_inputs: Sequence,
) -> Dict[int, DynamicSpec]:
    """Discover dynamic dimensions from SymInt markers or dynamo attributes.

    Checks for:
    - SymInt in tensor shapes (from torch._dynamo.maybe_mark_dynamic)
    - tensor._dynamo_dynamic_indices attribute
    - tensor._dynamo_weak_dynamic_indices attribute

    Args:
        gm: FX GraphModule (currently unused, for future symbolic analysis)
        example_inputs: Sequence of example input tensors

    Returns:
        Dictionary mapping arg_idx to DynamicSpec for each dynamic input
    """
    specs: Dict[int, DynamicSpec] = {}

    for arg_idx, inp in enumerate(example_inputs):
        if not hasattr(inp, "shape"):
            continue

        # Check for SymInt in tensor shape
        for dim_idx, dim in enumerate(inp.shape):
            if isinstance(dim, torch.SymInt):
                concrete_size = int(dim)
                specs[arg_idx] = DynamicSpec(
                    arg_idx=arg_idx,
                    dim_idx=dim_idx,
                    min_size=1,
                    max_size=concrete_size * 4,  # Allow 4x growth
                )
                break  # One dynamic dim per input

        # Check for dynamo dynamic indices attribute
        if arg_idx not in specs:
            dynamic_indices = getattr(inp, "_dynamo_dynamic_indices", None)
            if dynamic_indices:
                dim_idx = next(iter(dynamic_indices))
                specs[arg_idx] = DynamicSpec(
                    arg_idx=arg_idx,
                    dim_idx=dim_idx,
                    min_size=1,
                    max_size=inp.shape[dim_idx] * 4,
                )

        # Check for weak dynamic indices
        if arg_idx not in specs:
            weak_indices = getattr(inp, "_dynamo_weak_dynamic_indices", None)
            if weak_indices:
                dim_idx = next(iter(weak_indices))
                specs[arg_idx] = DynamicSpec(
                    arg_idx=arg_idx,
                    dim_idx=dim_idx,
                    min_size=1,
                    max_size=inp.shape[dim_idx] * 4,
                )

    return specs


def infer_buckets(
    dynamic_specs: Dict[int, DynamicSpec],
    min_size: int = 32,
    max_size: int = 2048,
    strategy: str = "power_of_2",
) -> List[int]:
    """Generate bucket sizes based on dynamic specs and strategy.

    Args:
        dynamic_specs: Dictionary of dynamic specifications
        min_size: Minimum bucket size
        max_size: Maximum bucket size
        strategy: Bucket generation strategy ("power_of_2" or "linear")

    Returns:
        Sorted list of bucket sizes
    """
    # Adjust max_size based on dynamic specs if available
    if dynamic_specs:
        spec_max = max(spec.max_size for spec in dynamic_specs.values())
        max_size = min(max_size, spec_max)

    if strategy == "power_of_2":
        buckets = []
        size = min_size
        while size <= max_size:
            buckets.append(size)
            size *= 2
        # Ensure we have at least one bucket
        if not buckets:
            buckets = [min_size]
        return buckets
    elif strategy == "linear":
        step = 64
        return list(range(min_size, max_size + 1, step))
    else:
        raise ValueError(f"Unknown bucket strategy: {strategy}")


def select_bucket(actual_len: int, buckets: List[int]) -> int:
    """Select the smallest bucket that fits the actual length.

    Args:
        actual_len: Actual sequence/dimension length
        buckets: List of available bucket sizes

    Returns:
        Selected bucket size (smallest bucket >= actual_len)

    Raises:
        ValueError: If actual_len exceeds all bucket sizes and no fallback
    """
    sorted_buckets = sorted(buckets)
    for bucket in sorted_buckets:
        if bucket >= actual_len:
            return bucket

    # Fallback to largest bucket (may need JIT compilation)
    return sorted_buckets[-1]
