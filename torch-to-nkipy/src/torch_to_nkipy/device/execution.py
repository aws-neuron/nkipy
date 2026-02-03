# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model execution on Neuron device."""

import spike_torch
import torch
from spike import SpikeModel, get_spike_singleton
from spike._spike import NrtTensor


def _torch_to_nkipy_tensor(tensor: torch.Tensor, name: str) -> NrtTensor:
    """Convert PyTorch tensor (on nkipy device) to non-owning NrtTensor."""
    nrt_ptr, size, core_id = spike_torch.get_tensor_info(tensor.data_ptr())
    return NrtTensor.wrap(nrt_ptr, core_id, size, name)


def spike_execute(
    model: SpikeModel,
    inputs: dict,
    outputs: dict,
    save_trace: bool = False,
    ntff_name: str = None,
):
    """Execute model with PyTorch tensors on nkipy device.

    Args:
        model: SpikeModel instance from load_spike_model
        inputs: Dict mapping input names to PyTorch tensors on nkipy device
        outputs: Dict mapping output names to PyTorch tensors on nkipy device
        save_trace: Whether to save execution trace
        ntff_name: Optional name for the trace file
    """
    # Convert to NrtTensor (non-owning wrappers)
    input_tensors = {
        name: _torch_to_nkipy_tensor(t, name) for name, t in inputs.items()
    }
    output_tensors = {
        name: _torch_to_nkipy_tensor(t, name) for name, t in outputs.items()
    }

    # Execute using spike's existing API
    get_spike_singleton().execute(
        model.model_ref,
        inputs=input_tensors,
        outputs=output_tensors,
        save_trace=save_trace,
        ntff_name=ntff_name,
    )


# Backward compatibility alias
nkipy_execute_model = spike_execute
