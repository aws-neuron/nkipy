"""Verification sampling for speculative decoding.

After the target model runs the K+1 candidate tokens through its decoder stack
(reusing the base ``transformer_layer`` in verify mode), this kernel turns the
final hidden states into one greedy target token per position.

Position i predicts the token that *follows* candidate i. With greedy
verification, draft token d_{i+1} is accepted iff it equals the target argmax at
position i; the first mismatch's target argmax becomes the bonus/correction
token. The caller does the accept/reject comparison on host.
"""

import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis

from .rmsnorm import rmsnorm_kernel


def verify_argmax(h, norm_weight, lm_head_weight, configs):
    """Return the target greedy token id at each of the S positions.

    Args:
        h: (B, S, H) final hidden states for the S=K+1 candidate positions.
    Returns:
        (B, S) int array of target argmax token ids (global vocab).
    """
    B, S, H = h.shape
    world = dist.get_world_size()
    h = rmsnorm_kernel(h, norm_weight, configs.norm_eps)
    vocab_per_device = lm_head_weight.shape[1]

    # All S positions at once: logits (B, S, vocab_local).
    logits = np.matmul(h, lm_head_weight)
    local_val, local_idx = tensor_apis.topk(logits, k=1, axis=-1)  # (B, S, 1)
    local_val = local_val[:, :, 0]  # (B, S)
    local_idx = local_idx[:, :, 0]

    # Gather every rank's local winner along a new leading axis, then pick the
    # winning rank per (B, S) and map back to the global vocab id. All ops stay on
    # traced tensors (no Python-scalar assignment into a numpy buffer).
    val_all = cc.all_gather(
        local_val, all_gather_dim=0, replica_groups=[list(range(world))]
    )  # (world*B, S)
    idx_all = cc.all_gather(
        local_idx, all_gather_dim=0, replica_groups=[list(range(world))]
    )
    val_all = val_all.reshape(world, B, S)
    idx_all = idx_all.reshape(world, B, S)

    # Winning rank per (B, S): argmax over the world axis.
    best_rank = np.argmax(val_all, axis=0)  # (B, S)
    # Gather the local index chosen by the winning rank.
    chosen_local = np.take_along_axis(idx_all, np.expand_dims(best_rank, 0), axis=0)[
        0
    ]  # (B, S)
    global_id = best_rank * vocab_per_device + chosen_local
    return global_id.astype(np.int32)
