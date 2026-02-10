# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
MoE common functions shared across implementations.

This module contains shared utility functions used by both reference
and NKI implementations of MoE.
"""

import torch
from torch import nn


def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """
    SwiGLU activation function.

    This is a variant of the GLU activation that uses SiLU (Swish) gating.

    Args:
        x: Input tensor with last dimension split into gate and linear parts
        alpha: Scaling factor for sigmoid (default: 1.702)
        limit: Clamping limit for numerical stability (default: 7.0)

    Returns:
        Activated tensor with half the last dimension size
    """
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values for numerical stability
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note: we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def custom_router(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom router implementation for MoE.

    Computes routing scores and indices for top-k experts.

    Args:
        router_logits: Router output logits of shape (seq_len, num_experts)
        top_k: Number of experts to select per token

    Returns:
        Tuple of (router_scores, router_indices) where:
        - router_scores: Normalized routing weights of shape (seq_len, top_k)
        - router_indices: Selected expert indices of shape (seq_len, top_k)
    """
    # Apply softmax to normalize routing weights for all tokens at once
    router_probs = nn.functional.softmax(
        router_logits, dim=1, dtype=router_logits.dtype
    )

    # Get top-k experts for all tokens at once
    router_top_value, router_indices = torch.topk(
        router_probs, top_k, dim=-1
    )  # (seq_len, top_k)

    # Normalize the top-k probabilities
    router_scores = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    return router_scores, router_indices
