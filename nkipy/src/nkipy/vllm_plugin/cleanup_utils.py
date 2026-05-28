# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for cleaning up Neuron and RDMA resources."""

import logging

logger = logging.getLogger(__name__)


def release_neuron_cores_and_rdma():
    """Best-effort release of Neuron cores and NIXL RDMA resources."""
    try:
        from relay import endpoint
        if endpoint.registered:
            endpoint.destroy()
    except Exception as e:
        logger.warning("Failed to clean up RDMA endpoint: %s", e)

    try:
        from multiprocessing import resource_tracker
        if hasattr(resource_tracker, '_resource_tracker'):
            tracker = resource_tracker._resource_tracker
            if tracker is not None and hasattr(tracker, '_cache'):
                tracker._cache.clear()
    except Exception:
        pass

    try:
        from spike import reset as spike_reset
        spike_reset()
    except Exception:
        pass

    try:
        from nkipy.runtime.nrt import nrt_close
        nrt_close()
    except Exception:
        pass
