# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
NKI-specific utilities and configuration for MoE.
"""

from dataclasses import dataclass

import numpy as np
from neuronxcc.nki.language import bfloat16

# Import shared constants from parent module
from vllm_nkipy.ops.moe.constants import BLOCK_SIZE, ControlType

# Re-export for backwards compatibility
__all__ = ["ControlType", "BLOCK_SIZE", "Config"]


@dataclass
class Config:
    """NKI-specific configuration for MoE computation."""

    dtype: np.dtype = bfloat16  # we mainly live in numpy space
