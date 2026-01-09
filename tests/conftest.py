# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def pytest_configure(config):
    """Set NEURON_RT_VISIBLE_CORES based on xdist worker ID.

    Worker IDs are like 'gw0', 'gw1', etc. We extract the number
    and use it as the core index for Neuron hardware isolation.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        # worker_id is like "gw0", "gw1", etc.
        core_idx = int(worker_id.replace("gw", ""))
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_idx)
