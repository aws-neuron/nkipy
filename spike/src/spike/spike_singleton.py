"""Spike runtime singleton management.

This module provides the singleton pattern for the Spike runtime, ensuring
only one NRT runtime instance exists at a time.

Thread Safety:
    - `get_spike_singleton()` and `reset()` are thread-safe (protected by lock)
    - `configure()` checks runtime state under lock, but env var setting is not locked
      (if multiple threads race to configure, last one wins - user's responsibility)
"""

import atexit
import os
import threading
import warnings
from typing import Iterable, Optional

from ._spike import Spike as _Spike
from .logger import get_logger

logger = get_logger()

_runtime: Optional[_Spike] = None
_debug_client = None
_lock = threading.Lock()


def _set_visible_cores(visible_cores: Iterable[int]) -> None:
    """Validate and set NEURON_RT_VISIBLE_CORES environment variable.

    Args:
        visible_cores: Iterable of NeuronCore IDs.

    Raises:
        TypeError: If visible_cores is not an iterable of integers.
        ValueError: If any core ID is negative.
    """
    if isinstance(visible_cores, int):
        raise TypeError(
            f"visible_cores must be an iterable, not int. "
            f"Use [{visible_cores}] or ({visible_cores},)"
        )

    cores_list = list(visible_cores)
    for core in cores_list:
        if not isinstance(core, int):
            raise TypeError(
                f"visible_cores must contain integers, got {type(core).__name__}: {core}"
            )
        if core < 0:
            raise ValueError(f"Core ID must be non-negative, got {core}")

    if cores_list:
        os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join(str(c) for c in cores_list)
        logger.info(
            f"Set NEURON_RT_VISIBLE_CORES to {os.environ['NEURON_RT_VISIBLE_CORES']}"
        )


def configure(visible_cores: Optional[Iterable[int]] = None) -> None:
    """Configure the Spike runtime before initialization.

    Must be called BEFORE any spike operations (SpikeTensor, SpikeModel, etc.),
    or after reset().

    Args:
        visible_cores: Iterable of NeuronCore IDs, e.g. [0, 1, 2], range(4)

    Raises:
        RuntimeError: If the runtime is already active.
        TypeError: If visible_cores is not an iterable of integers.
        ValueError: If any core ID is negative.

    Example:
        >>> import spike
        >>> spike.configure(visible_cores=[0, 1])
        >>> tensor = SpikeTensor(...)  # Uses cores 0 and 1
    """
    global _runtime

    # Check runtime state under lock
    with _lock:
        if _runtime is not None:
            raise RuntimeError(
                "Cannot configure: Spike runtime is already active. "
                "Call spike.reset() first to close the runtime, then configure()."
            )

    # Set env var outside lock (if users race, last one wins)
    if visible_cores is not None:
        _set_visible_cores(visible_cores)


def reset() -> None:
    """Reset the Spike runtime.

    This closes the current runtime (if any). Call configure() afterwards
    if you need to change visible cores before the next spike operation.

    Warning: All existing SpikeTensor and SpikeModel objects become invalid
    after this call. Any operations on them will fail.

    Example:
        >>> tensor = SpikeTensor(...)  # Uses default cores
        >>> spike.reset()
        >>> spike.configure(visible_cores=[2, 3])  # Optional: change cores
        >>> tensor2 = SpikeTensor(...)  # Uses cores 2, 3
    """
    global _runtime, _debug_client

    with _lock:
        if _runtime is not None:
            warnings.warn(
                "spike.reset() called. All existing SpikeTensor and SpikeModel "
                "objects are now invalid and should not be used.",
                UserWarning,
                stacklevel=2,
            )
            _stop_debug_client()
            _runtime.close()
            _runtime = None
            logger.info("Spike Runtime closed")


def get_spike_singleton() -> _Spike:
    """Get the Spike runtime singleton, initializing lazily if needed.

    This function is thread-safe using double-checked locking.

    Returns:
        The Spike runtime instance.

    Note:
        This function is primarily for internal use. Users should typically
        use SpikeTensor and SpikeModel directly, which call this internally.
    """
    global _runtime

    # Fast path: if already initialized, return without lock
    if _runtime is not None:
        return _runtime

    # Slow path: acquire lock and double-check
    with _lock:
        if _runtime is None:
            logger.info("Initializing Spike Runtime")
            _runtime = _Spike()  # Constructor does nrt_init via RAII
            logger.info("Spike Runtime initialized")
            _start_debug_client_if_configured()
        return _runtime


def _start_debug_client_if_configured() -> None:
    """Start the debug handler if NEURON_RT_DEBUG_OUTPUT_DIR is set.

    Must be called under _lock after NRT is initialized.
    """
    global _debug_client
    debug_dir = os.environ.get("NEURON_RT_DEBUG_OUTPUT_DIR")
    if not debug_dir:
        return
    try:
        from ._spike import DebugHandler

        _debug_client = DebugHandler(debug_dir)
        _debug_client.connect()
        _debug_client.start()
        logger.info("Debug handler started, output dir: %s", debug_dir)
    except Exception:
        logger.warning("Failed to start debug handler", exc_info=True)
        _debug_client = None


def _stop_debug_client() -> None:
    """Stop the debug client if running. Must be called under _lock."""
    global _debug_client
    if _debug_client is not None:
        try:
            _debug_client.stop()
        except Exception:
            logger.warning("Error stopping debug client", exc_info=True)
        _debug_client = None


def _cleanup() -> None:
    """Cleanup function called at program exit."""
    global _runtime
    with _lock:
        _stop_debug_client()
        # Note: Spike object from nanobind is RAII and Python GC managed, so no need
        # to call `.close()` explicitly. However, we do want to set the global to
        # None to make sure it does before nanobind ref leak check.
        _runtime = None


atexit.register(_cleanup)
