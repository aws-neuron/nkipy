# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol definitions for runtime backends."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch


@dataclass
class LoadedModel:
    """Abstraction for a loaded model (SpikeModel or spiky bundle_id).

    Attributes:
        model_ref: The underlying model reference (SpikeModel for spike,
                   bundle_id int for spiky)
        neff_path: Path to the NEFF file
    """

    model_ref: Any
    neff_path: str


class RuntimeBackend(Protocol):
    """Protocol defining the interface for spike/spiky backends.

    This protocol allows switching between different runtime implementations
    (spike vs spiky) while maintaining a consistent API.
    """

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

    def load_model(
        self,
        neff_file: str,
        cc_enabled: bool,
        device_id: int,
        device_count: int,
    ) -> LoadedModel:
        """Load a NEFF model and return a LoadedModel wrapper.

        Args:
            neff_file: Path to the NEFF file
            cc_enabled: Whether collective communication is enabled
            device_id: Device/rank ID for collective ops
            device_count: Total number of devices/world size

        Returns:
            LoadedModel instance ready for execution
        """
        ...

    def execute(
        self,
        model: LoadedModel,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        save_trace: bool = False,
        ntff_name: Optional[str] = None,
    ) -> None:
        """Execute the model with PyTorch tensors.

        Args:
            model: LoadedModel instance from load_model()
            inputs: Dict mapping input names to PyTorch tensors on device
            outputs: Dict mapping output names to PyTorch tensors on device
            save_trace: Whether to save execution trace
            ntff_name: Optional name for the trace file
        """
        ...

    def register_torch_device(self) -> None:
        """Register the 'nkipy' PyTorch device.

        This imports the appropriate torch extension module to register
        PrivateUse1 as 'nkipy' device.
        """
        ...
