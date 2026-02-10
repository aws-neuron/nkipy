# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
import torch.nn.functional as F


def router_tokengen(
    router_logits_sharded,
    top_k,
):
    # Get top-k values and indices
    top_k_logits_sharded, top_k_indices_sharded = torch.topk(
        router_logits_sharded, k=top_k, dim=1
    )

    # Apply softmax to top-k logits
    top_k_logits_sharded = F.softmax(top_k_logits_sharded.to(torch.float32), dim=1)
    top_k_logits_sharded = top_k_logits_sharded.to(router_logits_sharded.dtype)

    top_k_logits = top_k_logits_sharded  # (n_tokens, k)
    top_k_indices = top_k_indices_sharded.to(torch.int32)  # (n_tokens, k)

    # Create expert affinities matrix filled with zeros
    # expert_affinities_masked = torch.zeros(
    #     (n_tokens, n_experts),
    #     dtype=top_k_logits.dtype,
    #     device=top_k_logits.device,
    # )
    expert_affinities_masked = torch.zeros_like(
        router_logits_sharded,
    )

    batch_size, num_experts = expert_affinities_masked.shape
    row_indices = (
        torch.arange(
            batch_size, device=top_k_indices.device, dtype=top_k_indices.dtype
        )[:, None]
        * num_experts
    )
    row_indices = row_indices.to(dtype=top_k_indices.dtype)

    # Flatten, assign, and reshape back
    expert_affinities_flat = expert_affinities_masked.reshape(-1)
    flat_indices = (row_indices + top_k_indices).reshape(-1)
    expert_affinities_flat[flat_indices] = top_k_logits.reshape(-1)
    expert_affinities_masked = expert_affinities_flat.reshape(batch_size, num_experts)

    # expert_affinities_masked = torch.scatter(
    #     expert_affinities_masked,
    #     dim=1,
    #     index=top_k_indices,
    #     src=top_k_logits
    # )
    # Scatter the top-k logits into the expert affinities matrix
    # expert_affinities_masked.scatter_(
    #     dim=1,
    #     index=top_k_indices,
    #     src=top_k_logits
    # )
    return expert_affinities_masked


def expert_affinities_slice(
    expert_affinities_masked_all_experts,
    ep_size,
    ep_rank,
):
    n_experts = expert_affinities_masked_all_experts.shape[1]
    n_experts_per_ep = n_experts // ep_size
    expert_affinities_masked_all_experts = expert_affinities_masked_all_experts.reshape(
        -1, ep_size, n_experts_per_ep
    )
    expert_affinities_masked = expert_affinities_masked_all_experts[:, ep_rank].reshape(
        -1, n_experts_per_ep
    )
    return expert_affinities_masked


def router_prefill(
    router_logits_sharded,
    top_k: int,
):
    _, top_k_indices_sharded = torch.topk(router_logits_sharded, k=top_k, dim=1)
    top_k_indices_sharded = top_k_indices_sharded.to(torch.int32)

    # build mask by equality against arange (keep original arithmetic masking logic)
    n_tokens, n_experts = router_logits_sharded.shape
    expert_mask_sharded = torch.zeros_like(router_logits_sharded, dtype=torch.float32)

    expert_arrange = torch.arange(
        n_experts, device=router_logits_sharded.device, dtype=torch.float32
    )
    for k in range(top_k):
        # equality in integer domain, then accumulate as float
        eq = (
            top_k_indices_sharded[:, k : k + 1].to(torch.float32) == expert_arrange
        ).to(torch.float32)
        expert_mask_sharded = expert_mask_sharded + eq

    # masked full-softmax over E using large negative outside top-k
    masked_logits = (
        expert_mask_sharded * router_logits_sharded
        + (1.0 - expert_mask_sharded) * -100000.0
    )
    expert_affinities_masked_sharded = F.softmax(masked_logits, dim=1).to(
        router_logits_sharded.dtype
    )

    top_k_indices = top_k_indices_sharded

    return top_k_indices, expert_affinities_masked_sharded
