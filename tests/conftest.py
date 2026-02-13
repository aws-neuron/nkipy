# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _num_visible_core():
    try:
        from spike._spike import Spike

        return Spike.get_visible_neuron_core_count()
    except Exception:
        return 0


def pytest_xdist_auto_num_workers(config):
    """Cap xdist auto worker count to the number of visible Neuron cores.

    Using more workers than cores causes allocation failures.
    """

    num_visible_core = _num_visible_core()

    if num_visible_core == 0:
        return None
    else:
        return num_visible_core


def pytest_configure(config):
    # set global random seed
    np.random.seed(42)

    # Set NEURON_RT_VISIBLE_CORES based on xdist worker ID.
    # Worker IDs are like 'gw0', 'gw1', etc. We extract the number
    # and use it as the core index for Neuron hardware isolation.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")

    if worker_id is not None:
        num_visible_core = _num_visible_core()
        # if no visible core -- on non Neuron device or user configure it as one
        # don't bother, test can continue in CPU mode
        if num_visible_core == 0:
            return

        # if there is visible core but worker id is more than that, there's a problem
        core_idx = int(worker_id.replace("gw", ""))
        if num_visible_core <= core_idx:
            raise RuntimeError(
                f"Not enough visible cores ({num_visible_core}) for worker {worker_id}"
            )

        # acquire one Neuron core to do the test
        os.environ["NEURON_RT_NUM_CORES"] = "1"
