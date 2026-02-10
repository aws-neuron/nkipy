# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device utilities for Neuron hardware."""

from spiky.backend import get_backend


def current_device() -> int:
    """Return the current Neuron device index."""
    return get_backend().current_device()


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index to set as current
    """
    get_backend().set_device(device)


def device_count() -> int:
    """Return the number of Neuron devices available."""
    return get_backend().device_count()


def get_amp_supported_dtype():
    """Return list of dtypes supported by Neuron for automatic mixed precision."""
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]
