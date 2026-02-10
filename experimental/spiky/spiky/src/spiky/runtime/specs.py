# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor specifications for Neuron execution."""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TensorSpec:
    """
    Metadata specification for a tensor without storing actual data.

    This class captures the essential properties needed to allocate and
    manage tensors during model execution, including name (for NEFF binding),
    shape, and data type.
    """

    name: str
    shape: tuple  # Tuple of dimensions (d1, d2, ..., dn)
    dtype: torch.dtype  # PyTorch data type


@dataclass
class IOSpecs:
    """
    Container for input and output tensor specifications of a compiled model.

    This class maintains the structural information about a model's expected
    inputs and outputs, ensuring proper tensor allocation and binding during
    execution without requiring recompilation or inference of shapes.
    """

    input_specs: List[TensorSpec]  # Specifications for all input tensors
    output_specs: List[TensorSpec]  # Specifications for all output tensors
