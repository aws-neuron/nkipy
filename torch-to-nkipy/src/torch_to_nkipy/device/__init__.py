# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device management for NKIPy on AWS Neuron hardware.

Note: Device registration (register_torch_device) is deferred to nkipy_init().
Eagerly importing spike_torch here would trigger NRT initialization before
workers have configured their visible cores, causing core allocation failures
in multi-worker setups (e.g., vLLM tensor parallelism).
"""

# Import submodules for side effects (distributed backend registration).
# These do NOT trigger NRT initialization.
from torch_to_nkipy.device import device_module, distributed_backend  # noqa: F401

# Re-export from submodules
from torch_to_nkipy.device.execution import (
    nkipy_execute_model,
    spike_execute,
)
from torch_to_nkipy.device.initialization import (
    is_nkipy_device_initialized,
    nkipy_close,
    nkipy_init,
)
from torch_to_nkipy.device.loader import (
    load_spike_model,
    nkipy_load_model,
)
from torch_to_nkipy.device.profiling import nkipy_profile

__all__ = [
    # Initialization
    "nkipy_init",
    "nkipy_close",
    "is_nkipy_device_initialized",
    # Model loading
    "load_spike_model",
    "nkipy_load_model",  # backward compat
    # Execution
    "spike_execute",
    "nkipy_execute_model",  # backward compat
    # Profiling
    "nkipy_profile",
]
