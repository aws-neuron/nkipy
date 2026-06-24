# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config for nkigen-lite tests.

Provides Neuron core isolation under pytest-xdist so the HW tests
(``@pytest.mark.hw``) don't contend for the same core when run with
``-n auto``.  Mirrors the top-level ``tests/conftest.py``.
"""

import glob
import os


# NeuronCores per Neuron device. trn1/trn2 expose 2 cores per device; this is
# only used to bound the xdist worker count, so a conservative value is fine.
_CORES_PER_DEVICE = 2


def _num_visible_core():
    """Count NeuronCores without importing ``spike``.

    IMPORTANT: do NOT ``import spike`` here.  ``spike._spike`` and
    ``nki.runtime._spike`` are separate compiled extension modules that
    collide in CPython's loader — whichever is imported second resolves to
    the first, raising ``ImportError: cannot import name 'ModelTensorInfo'``.
    The nkigen-lite HW tests run through ``nki.runtime`` (compile_and_execute),
    so importing ``spike`` first in a worker would break them.  Enumerate the
    /dev/neuron* device nodes instead.
    """
    if os.environ.get("NEURON_RT_VISIBLE_CORES"):
        # Already pinned (e.g. by an outer harness): treat as a single core.
        return 1
    n_devices = len(glob.glob("/dev/neuron*"))
    return n_devices * _CORES_PER_DEVICE


def pytest_configure(config):
    # Register the ``hw`` marker so ``-m hw`` / ``-m "not hw"`` work and the
    # PytestUnknownMarkWarning goes away.
    config.addinivalue_line(
        "markers", "hw: test requires Neuron hardware (compiles/executes a kernel)"
    )

    # Isolate each xdist worker onto its own Neuron core. Worker IDs are
    # 'gw0', 'gw1', ...; the number selects the core index. Without this,
    # every worker targets the same core and nrt_init() fails under -n auto.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return

    num_visible_core = _num_visible_core()
    # No visible core (CPU-only host): nothing to isolate.
    if num_visible_core == 0:
        return

    core_idx = int(worker_id.replace("gw", ""))
    if num_visible_core <= core_idx:
        raise RuntimeError(
            f"Not enough visible cores ({num_visible_core}) for worker {worker_id}"
        )

    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_idx)


def pytest_xdist_auto_num_workers(config):
    """Cap xdist's ``-n auto`` worker count to the number of visible cores.

    More workers than cores causes core-allocation failures in the HW tests.
    (This hook is only consulted when the xdist plugin is active; run serial
    suites with ``-n0`` rather than ``-p no:xdist`` so the plugin stays
    loaded and this hook remains valid.)
    """
    num_visible_core = _num_visible_core()
    return num_visible_core if num_visible_core > 0 else None
