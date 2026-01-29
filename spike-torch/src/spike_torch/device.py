# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device module for nkipy - provides torch.nkipy.* functions."""

import torch


class DeviceModule:
    """Device module that gets registered as torch.nkipy.

    This class provides the standard device functions that PyTorch expects
    for custom backends, similar to torch.cuda.
    """

    @staticmethod
    def is_available() -> bool:
        """Return True if nkipy devices are available."""
        from spike_torch import _C

        return _C._is_available()

    @staticmethod
    def device_count() -> int:
        """Return the number of available nkipy devices."""
        from spike_torch import _C

        return _C._device_count()

    @staticmethod
    def current_device() -> int:
        """Return the current nkipy device index."""
        from spike_torch import _C

        return _C._current_device()

    @staticmethod
    def set_device(device: int) -> None:
        """Set the current nkipy device.

        Args:
            device: Device index to set as current
        """
        from spike_torch import _C

        _C._set_device(device)

    @staticmethod
    def synchronize(device: int | None = None) -> None:
        """Synchronize nkipy device.

        Note:
            This is a **no-op** on Neuron devices. Unlike CUDA, all Neuron
            operations through spike-torch complete synchronously before
            returning. This method exists only for API compatibility with
            torch.cuda.synchronize().

        Args:
            device: Device index (unused, for API compatibility)
        """
        pass

    @staticmethod
    def empty_cache() -> None:
        """Release all unused cached memory from the nkipy memory pool."""
        from spike_torch import _C

        _C._empty_cache()

    @staticmethod
    def get_device_name(device: int | None = None) -> str:
        """Get the name of the nkipy device.

        Args:
            device: Device index (optional)

        Returns:
            Device name string
        """
        if device is None:
            from spike_torch import _C

            device = _C._current_device()
        return f"AWS Neuron Core {device}"

    @staticmethod
    def get_device_capability(device: int | None = None) -> tuple[int, int]:
        """Get the capability of the nkipy device.

        Note: This returns a fixed version for Neuron devices.

        Args:
            device: Device index (optional)

        Returns:
            Tuple of (major, minor) version numbers
        """
        return (1, 0)

    @staticmethod
    def get_device_properties(device: int | None = None) -> dict:
        """Get the properties of the nkipy device.

        Args:
            device: Device index (optional)

        Returns:
            Dictionary of device properties
        """
        if device is None:
            from spike_torch import _C

            device = _C._current_device()
        return {
            "name": f"AWS Neuron Core {device}",
            "device_index": device,
        }

    @staticmethod
    def get_amp_supported_dtype() -> list[torch.dtype]:
        """Get the supported dtypes for automatic mixed precision.

        Returns:
            List of supported dtypes for AMP on Neuron devices
        """
        return [torch.float32, torch.float16, torch.bfloat16]

    @staticmethod
    def is_initialized() -> bool:
        """Check if the nkipy device is initialized.

        Returns:
            True if device is available and initialized
        """
        from spike_torch import _C

        return _C._is_available()

    @staticmethod
    def init() -> None:
        """Initialize the nkipy device.

        This is a no-op as nkipy initializes automatically on first use.
        """
        pass


# Create the device module instance
device_module = DeviceModule()
