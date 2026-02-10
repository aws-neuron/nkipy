# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
from vllm.distributed import get_tp_group


def calc_ep():
    """
    Calculate Expert Parallel (EP) rank and size.

    In a 2D parallelism setup with TP and EP:
    - TP groups are formed within each EP group
    - EP groups are formed across TP groups

    Example: world_size=128, tp_size=32
    - EP size = 128 / 32 = 4 (4 EP groups)
    - Ranks 0-31: EP group 0, TP ranks 0-31
    - Ranks 32-63: EP group 1, TP ranks 0-31
    - Ranks 64-95: EP group 2, TP ranks 0-31
    - Ranks 96-127: EP group 3, TP ranks 0-31

    Returns:
        ep_rank: Which EP group this rank belongs to (0 to ep_size-1)
        ep_size: Total number of EP groups
    """
    tp_group = get_tp_group()
    tp_size = tp_group.world_size  # Size of each TP group

    # Get global rank
    world_size = torch.distributed.get_world_size()
    current_rank = torch.distributed.get_rank()

    # Calculate EP size (number of TP groups)
    ep_size = world_size // tp_size

    # Calculate which EP group this rank belongs to
    # EP rank is determined by which TP group the rank is in
    ep_rank = current_rank // tp_size

    return ep_rank, ep_size
