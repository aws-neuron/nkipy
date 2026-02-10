# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from torch_to_nkipy.utils.ops import mark_subgraph_identity
except ImportError as e:
    logger.error("Failed to import torch_to_nkipy: %s", e)
    raise


def _wrap_output_with_barrier(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return mark_subgraph_identity(output)
    if isinstance(output, (tuple, list)):
        out = [
            (mark_subgraph_identity(o) if isinstance(o, torch.Tensor) else o)
            for o in output
        ]
        return type(output)(out)
    if isinstance(output, dict):
        return {
            k: (mark_subgraph_identity(v) if isinstance(v, torch.Tensor) else v)
            for k, v in output.items()
        }
    return output


def install_layer_barriers(layers: torch.nn.ModuleList, group_size: int) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    # Avoid double installation (idempotent)
    if getattr(layers, "_group_barriers_installed", False):
        return
    layers._group_barriers_installed = True

    L = len(layers)
    for end_idx in range(0, L - 1, group_size):

        def _hook(_mod, _inputs, output):
            return _wrap_output_with_barrier(output)

        layers[end_idx].register_forward_hook(_hook)
