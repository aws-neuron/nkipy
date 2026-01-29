# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import spike_torch


def current_device() -> int:
    """Return the current Neuron device index."""
    return spike_torch.current_device()


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index to set as current
    """
    spike_torch.set_device(device)


def device_count() -> int:
    """Return the number of Neuron devices available."""
    return spike_torch.device_count()


def get_amp_supported_dtype():
    """Return list of dtypes supported by Neuron for automatic mixed precision."""
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]
