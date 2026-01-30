import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

# Import config from parent directory
from config import Config
from nkipy.core import tensor_apis

# Import kernels from the kernels directory
from .rmsnorm import rmsnorm_kernel


def greedy_sampling(h, norm_weight, lm_head_weight, configs: Config):
    """Greedy sampling kernel for token generation."""
    h = rmsnorm_kernel(h, norm_weight, configs.norm_eps)
    logits = h[:, h.shape[1] - 1, :] @ lm_head_weight

    logits, next_id = tensor_apis.topk(logits, k=1, axis=1)
    logits_all = cc.all_gather(
        logits, all_gather_dim=1, replica_groups=[list(range(dist.get_world_size()))]
    )
    next_id_all = cc.all_gather(
        next_id, all_gather_dim=1, replica_groups=[list(range(dist.get_world_size()))]
    )

    _, top_index = tensor_apis.topk(logits_all, k=1, axis=1)
    final_next_id = np.empty_like(next_id)

    vocab_per_device = lm_head_weight.shape[1]
    for b in range(configs.max_batch_size):
        device_idx = np.copy(top_index[b])
        local_idx = next_id_all[b, device_idx]
        global_idx = device_idx * vocab_per_device + local_idx
        final_next_id[b] = global_idx

    return final_next_id
