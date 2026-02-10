# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
MoE constants and enums.

This module centralizes all constants used across the MoE implementation.
"""

from enum import Enum


class ControlType(Enum):
    """Control types for blockwise MoE token/block processing."""

    SKIP_DMA = -1  # Skip DMA transfer for this token
    SKIP_BLOCK = -2  # Skip entire block processing


# Block size for MoE computation (must match NKI kernel expectations)
BLOCK_SIZE = 128
