# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Core C++ bindings - optional, may not be available in all environments
_SPIKY_CORE_AVAILABLE = False
try:
    from ._spiky import (  # type: ignore
        Bundle,
        DeviceTensor,
        close,
        device_count,
        empty_cache,
        execute_bundle,
        execute_pipelined,
        flush_pipeline,
        get_cached_blocks,
        get_stats,
        init,
        is_initialized,
        register_bundle,
        select_bucket,
        unregister_bundle,
    )

    _SPIKY_CORE_AVAILABLE = True
except ImportError:
    # C++ module not available - provide stubs for Python-only usage
    Bundle = None
    DeviceTensor = None

    def close():
        return None

    def device_count():
        return 0

    def empty_cache():
        return None

    def flush_pipeline(bundle_id):
        return None

    def get_cached_blocks():
        return 0

    def get_stats():
        return None

    def init(device_id=0):
        return None

    def is_initialized():
        return False

    def register_bundle(
        bucket_to_neff,
        dynamic_specs,
        cc_enabled=False,
        rank_id=0,
        world_size=1,
    ):
        return -1

    def select_bucket(bundle_id, actual_len):
        return actual_len

    def unregister_bundle(bundle_id):
        return None

    def execute_bundle(**kwargs):
        return []

    def execute_pipelined(**kwargs):
        return []


# Utility modules (always available, pure Python)
from . import utils  # noqa: E402

# High-level Python APIs (may depend on C++ for full functionality)
from .callable import CallableConfig, NKIPyCallable  # noqa: E402
from .compile import compile  # noqa: E402
from .stats import (  # noqa: E402
    RuntimeStats,
    clear_memory_pool,
    get_memory_stats,
    reset_stats,
    trim_memory_pool,
)
from .utils.tensor_metadata import (  # noqa: E402
    PaddingMetadata,
    attach_metadata,
    get_metadata,
)

# Note: The following submodules are available but not imported by default
# to avoid circular import issues:
#   - spiky.torch: PyTorch integration (import spiky.torch to register backend)
#   - spiky.runtime: Compilation and execution utilities
#   - spiky.backend: Runtime backend abstraction
#   - spiky.device: Device management

__all__ = [
    # Core C++ APIs
    "Bundle",
    "DeviceTensor",
    "init",
    "close",
    "is_initialized",
    "device_count",
    "register_bundle",
    "unregister_bundle",
    "select_bucket",
    "execute_bundle",
    "execute_pipelined",
    "flush_pipeline",
    "get_stats",
    "empty_cache",
    "get_cached_blocks",
    # High-level Python APIs
    "NKIPyCallable",
    "CallableConfig",
    "reset_stats",
    "RuntimeStats",
    # Memory pool management
    "get_memory_stats",
    "clear_memory_pool",
    "trim_memory_pool",
    # Tensor metadata
    "PaddingMetadata",
    "attach_metadata",
    "get_metadata",
    # Compile API
    "compile",
    # Submodules
    "utils",
    # Availability flag
    "_SPIKY_CORE_AVAILABLE",
]
