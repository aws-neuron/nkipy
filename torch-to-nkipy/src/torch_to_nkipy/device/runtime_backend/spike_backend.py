# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spike runtime backend implementation."""

import os
import warnings
from typing import Dict, Optional

import torch

from .protocol import LoadedModel, RuntimeBackend


class SpikeBackend(RuntimeBackend):
    """Runtime backend implementation using spike."""

    def __init__(self):
        self._initialized = False

    def register_torch_device(self) -> None:
        """Import spike_torch to register nkipy device."""
        import spike_torch  # noqa: F401

    def init(self, visible_core: int) -> None:
        """Initialize spike runtime for the given core."""
        import spike_torch
        from spike import get_spike_singleton

        # Set root comm ID for collectives
        if os.environ.get("NEURON_RT_ROOT_COMM_ID", None) is None:
            root_addr = os.environ.get("MASTER_ADDR", "localhost")
            root_port = os.environ.get("NEURON_RT_PORT", "61234")
            os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

        # Reset and reconfigure spike with visible cores for this core.
        # spike_torch auto-initializes on import with default cores, so we need
        # to reset and reconfigure for the specific core.
        # Suppress the expected warning about invalidated objects since this is
        # initialization and no user objects exist yet.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="spike.reset\\(\\) called")
            spike_torch.reset()
        spike_torch.configure(visible_cores=[visible_core])

        # Re-initialize the spike singleton with the new configuration.
        # This is needed because spike_torch.reset() closed the previous runtime.
        get_spike_singleton()
        self._initialized = True

    def close(self) -> None:
        """Close spike runtime."""
        import spike_torch

        spike_torch.reset()
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if spike runtime is initialized."""
        return self._initialized

    def current_device(self) -> int:
        """Return current device index."""
        import spike_torch

        return spike_torch.current_device()

    def set_device(self, device: int) -> None:
        """Set current device."""
        import spike_torch

        spike_torch.set_device(device)

    def device_count(self) -> int:
        """Return number of available devices."""
        import spike_torch

        return spike_torch.device_count()

    def load_model(
        self,
        neff_file: str,
        cc_enabled: bool,
        device_id: int,
        device_count: int,
    ) -> LoadedModel:
        """Load NEFF and return a LoadedModel instance."""
        import spike_torch
        from spike import SpikeModel

        spike_model = SpikeModel.load_from_neff(
            neff_path=neff_file,
            core_id=spike_torch.current_device(),
            cc_enabled=cc_enabled,
            rank_id=device_id,
            world_size=device_count,
        )
        return LoadedModel(
            model_ref=spike_model,
            neff_path=neff_file,
        )

    def execute(
        self,
        model: LoadedModel,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        save_trace: bool = False,
        ntff_name: Optional[str] = None,
    ) -> None:
        """Execute model with PyTorch tensors on nkipy device."""
        import spike_torch
        from spike import get_spike_singleton
        from spike._spike import NrtTensor

        def torch_to_nrt_tensor(tensor: torch.Tensor, name: str) -> NrtTensor:
            """Convert PyTorch tensor (on nkipy device) to non-owning NrtTensor."""
            nrt_ptr, size, core_id = spike_torch.get_tensor_info(tensor.data_ptr())
            return NrtTensor.wrap(nrt_ptr, core_id, size, name)

        # Convert to NrtTensor (non-owning wrappers)
        input_tensors = {
            name: torch_to_nrt_tensor(t, name) for name, t in inputs.items()
        }
        output_tensors = {
            name: torch_to_nrt_tensor(t, name) for name, t in outputs.items()
        }

        # Execute using spike's existing API
        # model.model_ref is a SpikeModel, and SpikeModel.model_ref is the NrtModel
        spike_model = model.model_ref
        get_spike_singleton().execute(
            spike_model.model_ref,
            inputs=input_tensors,
            outputs=output_tensors,
            save_trace=save_trace,
            ntff_name=ntff_name,
        )
