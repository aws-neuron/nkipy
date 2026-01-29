# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""spike-torch: PyTorch device integration for spike runtime on AWS Neuron.

Note on shutdown: An atexit handler is registered to clear allocator state
before the NRT runtime closes. This prevents errors when tensors are garbage
collected after nrt_close() has been called.

spike-torch provides PyTorch device registration and tensor allocation for
the ``nkipy`` (PrivateUse1) backend.

Important Limitations
---------------------
- All operations are synchronous (no async/streams)
- ``synchronize()`` is a no-op
- ``non_blocking`` parameter is accepted but ignored
- Copy operations require contiguous tensors
- Device-to-device copy only works on same Neuron core

Model loading/execution and other runtime utilities should be used directly
from the ``spike`` Python package.
"""

import atexit

import torch

from spike import configure as spike_configure
from spike import get_spike_singleton
from spike import reset as spike_reset

from spike_torch import _C
from spike_torch.device import device_module

_initialized = False


def _ensure_initialized() -> None:
    """Ensure the spike runtime and torch device are initialized."""
    global _initialized
    get_spike_singleton()
    # Mark allocator ready after NRT is initialized (resets shutdown flag)
    _C._mark_allocator_ready()
    if _initialized:
        return

    # Register 'nkipy' as the name for PrivateUse1 device type (C++ side)
    _C._register_spike_device()

    # Register the PrivateUse1HooksInterface for autograd support
    _C._register_spike_hooks()

    # Register the device module - this makes torch.nkipy available
    torch._register_device_module("nkipy", device_module)

    # Generate standard methods for the nkipy backend
    # This creates methods like tensor.spike(), module.spike(), etc.
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=True
    )

    _initialized = True


def device_count() -> int:
    """Return the number of available spike devices."""
    _ensure_initialized()
    return _C._device_count()


def current_device() -> int:
    """Return the current spike device index."""
    _ensure_initialized()
    return _C._current_device()


def set_device(device: int) -> None:
    """Set the current spike device."""
    _ensure_initialized()
    _C._set_device(device)


def is_available() -> bool:
    """Return True if spike devices are available."""
    _ensure_initialized()
    return _C._is_available()


def empty_cache() -> None:
    """Release all unused cached memory from the spike memory pool."""
    _ensure_initialized()
    _C._empty_cache()


def get_cached_blocks() -> int:
    """Return the number of cached memory blocks."""
    _ensure_initialized()
    return _C._get_cached_blocks()


def get_tensor_info(data_ptr: int) -> tuple[int, int, int]:
    """Get (nrt_ptr, size, core_id) for a PyTorch tensor's data pointer.

    Args:
        data_ptr: The data pointer from tensor.data_ptr()

    Returns:
        A tuple of (nrt_tensor_ptr, size, core_id) for the tensor

    Raises:
        RuntimeError: If no tensor is found for the data pointer
    """
    _ensure_initialized()
    return _C._get_tensor_info(data_ptr)


# Re-export spike's configure and reset functions
configure = spike_configure


def reset() -> None:
    """Reset the spike runtime.

    This closes the spike runtime and resets spike-torch state.
    Call configure() afterwards if you need to change visible cores
    before the next spike operation.

    Warning: All existing tensors on spike device become invalid
    after this call. Any operations on them will fail.
    """
    # Clear cached allocations before closing the runtime so we best-effort
    # release device memory and drop stale pointers.
    _C._clear_allocator_state()
    spike_reset()


__all__ = [
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "empty_cache",
    "get_cached_blocks",
    "get_tensor_info",
    "configure",
    "reset",
    "_ensure_initialized",
]

# Auto-configure for distributed mode if LOCAL_RANK is set
# This must happen BEFORE _ensure_initialized() to avoid core conflicts
# when multiple ranks try to claim all cores simultaneously
import os as _os
if "LOCAL_RANK" in _os.environ and "NEURON_RT_VISIBLE_CORES" not in _os.environ:
    _local_rank = int(_os.environ["LOCAL_RANK"])
    spike_configure(visible_cores=[_local_rank])

# Import side effect: register the nkipy device with torch.
_ensure_initialized()


def _cleanup_at_exit() -> None:
    """Clear allocator state before NRT closes at program exit.

    This handler runs BEFORE spike's atexit handler (LIFO order) because
    spike_torch imports spike, so spike's handler was registered first.
    Clearing allocator state ensures tensors being garbage collected don't
    try to call nrt_tensor_free() after nrt_close().
    """
    _C._clear_allocator_state()


atexit.register(_cleanup_at_exit)
