# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

from nkipy.core.logger import get_logger

# Disable the NRT per-execution collectives barrier by default: it adds fixed
# host-side dispatch latency (~1.8x on batch=1 decode) and is a no-op for
# non-collective kernels. Safe for single-rank and SPMD (nkipy's default); the
# load-time barrier is kept. Only MPMD-with-collectives needs it on, to detect
# ranks running mismatched graphs — see the guard in compile_and_load.
# Must be set here at import: NRT reads it at nrt_init() (the first device op),
# so setting it later is ignored. setdefault lets an explicit user override win.
_BARRIER_ENV = "NEURON_RT_DISABLE_EXECUTION_BARRIER"
if _BARRIER_ENV not in os.environ:
    os.environ[_BARRIER_ENV] = "1"
    get_logger().warning(
        "nkipy set %s=1 (skips NRT per-execution collectives barrier for lower "
        "dispatch latency; safe for single-rank/SPMD). For MPMD-with-collectives, "
        "set it =0 before importing nkipy.",
        _BARRIER_ENV,
    )

# Imports below intentionally follow the env setup above (must run before NRT init).
from .decorators import baremetal_jit  # noqa: E402
from .execute import baremetal_run_traced_kernel  # noqa: E402
from .utils import is_neuron_compatible  # noqa: E402

try:
    from .baremetal_executor import BaremetalExecutor, CompiledKernel
    from .device_kernel import DeviceKernel
    from .device_tensor import DeviceTensor
except ImportError:
    print("Runtime import failed. Is Spike installed?")
    pass

__all__ = [
    "BaremetalExecutor",
    "CompiledKernel",
    "DeviceKernel",
    "DeviceTensor",
    "baremetal_jit",
    "baremetal_run_traced_kernel",
    "is_neuron_compatible",
]
