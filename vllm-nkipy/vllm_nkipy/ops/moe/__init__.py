# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
MoE (Mixture of Experts) module for vLLM-NKIPy.

This module provides MoE implementations optimized for NKIPy:
- reference_impl: CPU/debug reference implementation (sequential processing)
- nki_impl: NKI-based optimized implementation (blockwise processing)

The implementation is selected based on VLLM_NKIPY_MOE_USE_NKI environment variable.
"""

from vllm_nkipy.ops.moe.common import custom_router, swiglu
from vllm_nkipy.ops.moe.constants import BLOCK_SIZE, ControlType

__all__ = [
    # Constants
    "BLOCK_SIZE",
    "ControlType",
    # Common functions
    "swiglu",
    "custom_router",
]
