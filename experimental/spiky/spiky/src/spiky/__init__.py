# Core C++ bindings - optional, may not be available in all environments
_SPIKY_CORE_AVAILABLE = False
try:
    from ._spiky import (  # type: ignore
        Bundle,
        DeviceTensor,
        close,
        device_count,
        empty_cache,
        flush_pipeline,
        get_cached_blocks,
        get_stats,
        init,
        is_initialized,
        register_bundle,
        select_bucket,
        unregister_bundle,
        execute_bundle,
        execute_pipelined,
    )
    _SPIKY_CORE_AVAILABLE = True
except ImportError:
    # C++ module not available - provide stubs for Python-only usage
    Bundle = None
    DeviceTensor = None
    close = lambda: None
    device_count = lambda: 0
    empty_cache = lambda: None
    flush_pipeline = lambda bundle_id: None
    get_cached_blocks = lambda: 0
    get_stats = lambda: None
    init = lambda device_id=0: None
    is_initialized = lambda: False
    register_bundle = lambda bucket_to_neff, dynamic_specs, cc_enabled=False, rank_id=0, world_size=1: -1
    select_bucket = lambda bundle_id, actual_len: actual_len
    unregister_bundle = lambda bundle_id: None
    execute_bundle = lambda **kwargs: []
    execute_pipelined = lambda **kwargs: []

# Utility modules (always available, pure Python)
from . import utils

# High-level Python APIs (may depend on C++ for full functionality)
from .callable import NKIPyCallable, CallableConfig
from .stats import (
    reset_stats, RuntimeStats,
    get_memory_stats, clear_memory_pool, trim_memory_pool,
)
from .utils.tensor_metadata import PaddingMetadata, attach_metadata, get_metadata
from .compile import compile

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

