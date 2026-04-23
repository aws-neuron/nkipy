# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for cleaning up Neuron and RDMA resources."""

import logging

logger = logging.getLogger(__name__)


def release_neuron_cores_and_rdma():
    """Best-effort release of Neuron cores and P2P RDMA resources.

    This function should be called during shutdown, signal handling (Ctrl+C),
    or before spike_reset() to ensure proper cleanup of:
    - P2P RDMA memory registrations (MRs)
    - Shared memory objects
    - Neuron runtime resources
    """
    # Clean up P2P RDMA resources first
    try:
        from relay import rank_endpoint
        if rank_endpoint.ep is not None:
            logger.info("Cleaning up P2P endpoint and MRs...")
            # Synchronously wait for any pending dereg
            if rank_endpoint._dereg_thread and rank_endpoint._dereg_thread.is_alive():
                logger.info("Waiting for background MR deregistration to complete...")
                rank_endpoint.wait()
            # Clear endpoint (will deregister any remaining MRs synchronously)
            rank_endpoint.ep = None
            rank_endpoint.xfer_descs = []
            rank_endpoint.buf_info = []
            logger.info("P2P cleanup complete")
    except Exception as e:
        logger.warning("Failed to clean up P2P endpoint: %s", e)

    # Clean up shared memory objects from multiprocessing
    try:
        from multiprocessing import resource_tracker
        # Unregister all shared memory to prevent warnings
        if hasattr(resource_tracker, '_resource_tracker'):
            tracker = resource_tracker._resource_tracker
            if tracker is not None and hasattr(tracker, '_cache'):
                tracker._cache.clear()
    except Exception:
        pass

    # Release Neuron cores via spike.reset() + nrt_close
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
