# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics API for spiky runtime.

Provides a Pythonic interface to the C++ RuntimeStats collected by the
spiky execution engine.
"""

from typing import TYPE_CHECKING

# Import core bindings
try:
    import spiky._spiky as _core

    _SPIKY_AVAILABLE = True
except ImportError:
    _core = None
    _SPIKY_AVAILABLE = False

if TYPE_CHECKING:
    from typing import Any


def get_stats() -> "Any":
    """Get current runtime statistics.

    Returns:
        RuntimeStats object with execution metrics including:
        - total_executions: Total number of bundle executions
        - bucket_hits: Number of times existing buckets were reused
        - total_execution_time_ms: Cumulative execution time
    """
    if not _SPIKY_AVAILABLE:
        return None
    return _core.get_stats()


def reset_stats() -> None:
    """Reset all runtime statistics.

    Note: This requires the reset_stats binding in the C++ layer.
    If not available, this is a no-op.
    """
    if not _SPIKY_AVAILABLE:
        return
    if hasattr(_core, "reset_stats"):
        _core.reset_stats()


def empty_cache() -> None:
    """Clear the device memory cache.

    Releases all cached memory blocks back to the system.
    Use this to free memory when transitioning between workloads.
    """
    if not _SPIKY_AVAILABLE:
        return
    _core.empty_cache()


def get_cached_blocks() -> int:
    """Get number of cached memory blocks.

    Returns:
        Number of memory blocks currently held in the cache
    """
    if not _SPIKY_AVAILABLE:
        return 0
    return _core.get_cached_blocks()


def get_memory_stats() -> dict:
    """Get device memory pool statistics.

    Returns:
        Dictionary with memory pool statistics including:
        - used_bytes: Currently allocated memory
        - cached_bytes: Memory held in cache
        - total_bytes: Total device memory
        - allocation_count: Number of allocations
        - reuse_count: Number of cache reuses
        - cache_hit_count: Cache hits
        - cache_miss_count: Cache misses
    """
    if not _SPIKY_AVAILABLE:
        return {
            "used_bytes": 0, "cached_bytes": 0, "total_bytes": 0,
            "allocation_count": 0, "reuse_count": 0,
            "cache_hit_count": 0, "cache_miss_count": 0,
        }
    if hasattr(_core, "get_memory_stats"):
        return _core.get_memory_stats()
    # Fallback: construct from available APIs
    return {
        "used_bytes": 0, "cached_bytes": 0, "total_bytes": 0,
        "allocation_count": 0, "reuse_count": 0,
        "cache_hit_count": 0, "cache_miss_count": 0,
    }


def clear_memory_pool() -> None:
    """Free all cached memory blocks."""
    if not _SPIKY_AVAILABLE:
        return
    if hasattr(_core, "clear_memory_pool"):
        _core.clear_memory_pool()
    else:
        _core.empty_cache()  # Fallback to existing API


def trim_memory_pool(target_bytes: int) -> None:
    """Trim cached memory to target size in bytes.

    Args:
        target_bytes: Target cache size in bytes. Pass 0 to free all cached memory.
    """
    if not _SPIKY_AVAILABLE:
        return
    if hasattr(_core, "trim_memory_pool"):
        _core.trim_memory_pool(target_bytes)
    elif target_bytes == 0:
        empty_cache()  # Fallback


# Re-export RuntimeStats type for type hints
if _SPIKY_AVAILABLE:
    RuntimeStats = _core.RuntimeStats if hasattr(_core, "RuntimeStats") else type(None)
else:
    RuntimeStats = type(None)
