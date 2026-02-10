# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model loading utilities for Neuron device."""

from torch_to_nkipy.device.runtime_backend import LoadedModel, get_backend


def load_spike_model(
    neff_file: str,
    cc_enabled: bool,
    device_id: int,
    device_count: int,
) -> LoadedModel:
    """Load NEFF and return a LoadedModel instance.

    Args:
        neff_file: Path to the NEFF file
        cc_enabled: Whether collective communication is enabled
        device_id: Device/rank ID
        device_count: Total number of devices/world size

    Returns:
        LoadedModel instance ready for execution
    """
    return get_backend().load_model(
        neff_file=neff_file,
        cc_enabled=cc_enabled,
        device_id=device_id,
        device_count=device_count,
    )


# Backward compatibility alias
nkipy_load_model = load_spike_model
