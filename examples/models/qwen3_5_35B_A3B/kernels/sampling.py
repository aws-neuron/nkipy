import numpy as np
import torch.distributed as dist
import nkipy.distributed.collectives as cc
from nkipy.core import tensor_apis

from config import Config

from .rmsnorm import rmsnorm_kernel


def greedy_sampling(h, norm_weight, lm_head_weight, configs: Config):
    """On-device greedy sampling: RMSNorm -> lm_head matmul -> all_gather -> topk.

    Returns a (B, 1) uint32 tensor of global token IDs.

    Works around the neuronx-cc bug where `all_gather` in the same kernel as
    a prior `topk` corrupts the topk output (see bug_topk_allgather_dynamic_index.md).
    The fix is to all_gather the full logits tensor first, then topk once on
    the gathered result -- topk is never upstream of all_gather in the graph,
    and no dynamic indexing is needed because topk over full-vocab logits
    directly returns the global winner ID.
    """
    h = rmsnorm_kernel(h, norm_weight, configs.norm_eps)
    logits = h[:, -1, :] @ lm_head_weight  # (B, vocab_per_device)
    logits = logits.astype(np.float32)

    if dist.get_world_size() > 1:
        logits = cc.all_gather(
            logits,
            all_gather_dim=1,
            replica_groups=[list(range(dist.get_world_size()))],
        )  # (B, vocab_total)

    _, next_id = tensor_apis.topk(logits, k=1, axis=1)  # (B, 1) uint32 global
    return next_id
