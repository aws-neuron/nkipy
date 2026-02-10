# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model execution on Neuron device."""

from typing import Dict, Optional

import torch

from torch_to_nkipy.device.runtime_backend import LoadedModel, get_backend


def spike_execute(
    model: LoadedModel,
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    save_trace: bool = False,
    ntff_name: Optional[str] = None,
):
    """Execute model with PyTorch tensors on nkipy device.

    Args:
        model: LoadedModel instance from load_spike_model
        inputs: Dict mapping input names to PyTorch tensors on nkipy device
        outputs: Dict mapping output names to PyTorch tensors on nkipy device
        save_trace: Whether to save execution trace
        ntff_name: Optional name for the trace file
    """
    get_backend().execute(
        model=model,
        inputs=inputs,
        outputs=outputs,
        save_trace=save_trace,
        ntff_name=ntff_name,
    )


# Backward compatibility alias
nkipy_execute_model = spike_execute
