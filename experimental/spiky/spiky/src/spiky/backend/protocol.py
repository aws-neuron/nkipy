# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol definitions for runtime backends."""

from typing import Protocol


class RuntimeBackend(Protocol):
    """Protocol defining the interface for the spiky runtime backend."""

    def init(self, visible_core: int) -> None:
        """Initialize the runtime for the given core.

        Args:
            visible_core: The NeuronCore index to use
        """
        ...

    def close(self) -> None:
        """Close the runtime and release resources."""
        ...

    def is_initialized(self) -> bool:
        """Check if runtime is initialized.

        Returns:
            True if the runtime is initialized, False otherwise
        """
        ...

    def current_device(self) -> int:
        """Return current device index.

        Returns:
            The current NeuronCore device index
        """
        ...

    def set_device(self, device: int) -> None:
        """Set current device.

        Args:
            device: The device index to set as current
        """
        ...

    def device_count(self) -> int:
        """Return number of available devices.

        Returns:
            The number of available NeuronCore devices
        """
        ...

    def register_torch_device(self) -> None:
        """Register the 'nkipy' PyTorch device.

        This imports the appropriate torch extension module to register
        PrivateUse1 as 'nkipy' device.
        """
        ...
