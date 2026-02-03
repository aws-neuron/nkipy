# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model loading utilities for Neuron device."""

import spike_torch
from spike import SpikeModel


def load_spike_model(
    neff_file: str,
    cc_enabled: bool,
    device_id: int,
    device_count: int,
) -> SpikeModel:
    """Load NEFF and return a SpikeModel instance.

    Args:
        neff_file: Path to the NEFF file
        cc_enabled: Whether collective communication is enabled
        device_id: Device/rank ID
        device_count: Total number of devices/world size

    Returns:
        SpikeModel instance ready for execution
    """
    return SpikeModel.load_from_neff(
        neff_path=neff_file,
        core_id=spike_torch.current_device(),
        cc_enabled=cc_enabled,
        rank_id=device_id,
        world_size=device_count,
    )


# Backward compatibility alias
nkipy_load_model = load_spike_model
