# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""Simplified distributed state for vLLM-NKIPy.

This module provides a minimal implementation of distributed state management,
focusing only on world group initialization and management.
"""

from typing import Optional

from vllm.distributed import GroupCoordinator
from vllm.logger import init_logger

logger = init_logger(__name__)

# Global world group
_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    """Get the world group coordinator.

    Returns:
        The world group coordinator.

    Raises:
        AssertionError: If the world group is not initialized.
    """
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: list[int],
    local_rank: int,
    backend: str,
    use_device_communicator: bool = True,
) -> GroupCoordinator:
    """Initialize the world group.

    Args:
        ranks: List of global ranks in the world group.
        local_rank: Local rank of the current process.
        backend: PyTorch distributed backend to use.
        use_device_communicator: Whether to use device communicator.

    Returns:
        The initialized world group coordinator.
    """
    global _WORLD

    if _WORLD is not None:
        logger.warning("World group already initialized, returning existing group")
        return _WORLD

    _WORLD = GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        group_name="world",
    )

    logger.info(
        f"Initialized world group: rank={_WORLD.rank}, "
        f"world_size={_WORLD.world_size}, "
        f"local_rank={local_rank}, "
        f"use_device_communicator={use_device_communicator}"
    )

    return _WORLD


def destroy_world_group() -> None:
    """Destroy the world group."""
    global _WORLD
    _WORLD = None
    logger.info("Destroyed world group")
