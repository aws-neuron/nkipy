import nki.isa as nisa
import nki.language as nl
import numpy as np
from collective import all_gather
import nki
from kernels.rmsnorm import rmsnorm
from kernels.softmax import softmax
from nkipy.core import tensor_apis
from nkipy.core.nki_op import wrap_nki_kernel
import parallel_state

TILE_SIZE = 128  # partition dim (nl.tile_size.pmax)


def div_ceil(n, d):
    return (n + d - 1) // d


def rmsnorm_router(
    hidden_states_sharded,
    post_attention_weight,
    router_weight,
    router_bias,
    norm_eps,
    top_k,
    is_nkipy,
):
    hidden_size = hidden_states_sharded.shape[-1]
    # reshape because when run in cpu mode, hidden_states_sharded shape is (B, L, D)
    hidden_states_sharded = hidden_states_sharded.reshape(-1, hidden_size)
    normed_hidden_states_sharded = rmsnorm(
        hidden_states_sharded,
        post_attention_weight,
        norm_eps,
        is_nkipy=is_nkipy,
    )
    top_k_indices, expert_affinities_masked_sharded = router(
        hidden_states_sharded=normed_hidden_states_sharded,
        router_weight=router_weight,
        router_bias=router_bias,
        top_k=top_k,
        is_prefill=True,
        is_nkipy=is_nkipy,
    )
    return top_k_indices, expert_affinities_masked_sharded, normed_hidden_states_sharded


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

@nki.jit
def transpose_2d(in_tensor):
    n_tokens, hidden_size = in_tensor.shape
    out_tensor = nl.ndarray(
        (hidden_size, n_tokens), dtype=in_tensor.dtype, buffer=nl.shared_hbm
    )

    hidden_n_tiles = div_ceil(hidden_size, TILE_SIZE)

    for i_hidden in nl.affine_range(hidden_n_tiles):
        h_start = i_hidden * TILE_SIZE
        h_size = min(TILE_SIZE, hidden_size - h_start)

        # Transpose a (n_tokens, h_size) tile to (h_size, n_tokens) in SBUF.
        out_sbuf = nl.ndarray((TILE_SIZE, n_tokens), dtype=in_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=out_sbuf[0:h_size, 0:n_tokens],
            src=in_tensor[0:n_tokens, h_start : h_start + h_size],
        )
        nisa.dma_copy(
            dst=out_tensor[h_start : h_start + h_size, 0:n_tokens],
            src=out_sbuf[0:h_size, 0:n_tokens],
        )

    return out_tensor

def router(
    hidden_states_sharded,
    router_weight,
    router_bias,
    top_k,
    is_prefill,
    is_nkipy,
):
    if is_nkipy:
        # manually insert transpose to avoid bad transpose
        transpose_nki = wrap_nki_kernel(
            transpose_2d,
            [hidden_states_sharded],
            is_nki_beta_3_version=True,
        )
        hidden_states_sharded_T = transpose_nki(hidden_states_sharded)
        hidden_states_sharded = np.transpose(hidden_states_sharded_T, (1, 0))
    router_logits_sharded = (hidden_states_sharded @ router_weight).astype(
        hidden_states_sharded.dtype
    )
    router_logits_sharded += router_bias
    if is_nkipy:
        _, top_k_indices_sharded = tensor_apis.topk(router_logits_sharded, k=top_k, axis=1)
        top_k_indices_sharded = top_k_indices_sharded.astype(np.int8)
    else:
        # numpy does not have top_k api
        top_k_indices_sharded = np.argsort(router_logits_sharded, axis=1)[:, -top_k:][:, ::-1].astype(np.int8)
    # calculate mask using vector engine
    expert_mask_sharded = np.zeros_like(router_logits_sharded).astype(np.float32)
    n_experts = router_logits_sharded.shape[1]
    if is_nkipy:
        expert_arrange = np.arange(n_experts, dtype=np.float32) + tensor_apis.zeros(n_experts, dtype=np.float32)
    else:
        expert_arrange = np.arange(n_experts, dtype=np.float32)
    for k in range(top_k):
        expert_mask_sharded += np.equal(top_k_indices_sharded[:, k:k+1].astype(np.float32), expert_arrange)
    expert_affinities_masked_sharded = softmax(
        (
            expert_mask_sharded * router_logits_sharded
            + (1 - expert_mask_sharded) * -100000
        ).astype(router_logits_sharded.dtype),
        is_nkipy=is_nkipy,
    )
    if is_prefill:
        top_k_indices = all_gather(
            data=top_k_indices_sharded,
            all_gather_dim=0,
            replica_groups=parallel_state.get_prefill_ep_world_group(),
            is_nkipy=is_nkipy,
        )
    else:
        # tkg tokens are replicated
        top_k_indices = top_k_indices_sharded
    return top_k_indices, expert_affinities_masked_sharded

def router_tokengen(
    hidden_states_sharded,
    router_weight,
    router_bias,
    top_k,
    is_nkipy,
):
    router_logits_sharded = (hidden_states_sharded @ router_weight).astype(hidden_states_sharded.dtype)
    router_logits_sharded += router_bias
    if is_nkipy:
        top_k_logits_sharded, top_k_indices_sharded = tensor_apis.topk(router_logits_sharded, k=top_k, axis=1)
    else:
        # numpy does not have top_k api
        top_k_indices_sharded = np.argsort(router_logits_sharded, axis=1)[:, -top_k:][:, ::-1].astype(np.int8)
        top_k_logits_sharded = np.take_along_axis(router_logits_sharded, top_k_indices_sharded, axis=1)
    top_k_logits_sharded = softmax(top_k_logits_sharded, is_nkipy=is_nkipy)

    top_k_indices = top_k_indices_sharded
    top_k_logits = top_k_logits_sharded
    n_tokens = top_k_logits.shape[0]
    n_experts = router_logits_sharded.shape[1]
    if is_nkipy:
        expert_affinities_masked = tensor_apis.zeros(
            (n_tokens, n_experts), dtype=top_k_logits.dtype
        )
    else:
        expert_affinities_masked = np.zeros((n_tokens, n_experts), dtype=top_k_logits_sharded.dtype)

    batch_size, num_experts = expert_affinities_masked.shape
    row_indices = np.arange(batch_size)[:, np.newaxis] * num_experts
    row_indices = row_indices.astype(top_k_indices.dtype)

    # Flatten, assign, and reshape back
    expert_affinities_flat = expert_affinities_masked.reshape(-1)
    flat_indices = (row_indices + top_k_indices).reshape(-1)
    expert_affinities_flat[flat_indices] = top_k_logits.reshape(-1)
    expert_affinities_masked = expert_affinities_flat.reshape(batch_size, num_experts)

    # FIXME: put along axis has minor performance issue
    # np.put_along_axis(expert_affinities_masked, top_k_indices, top_k_logits, axis=1)

    return expert_affinities_masked