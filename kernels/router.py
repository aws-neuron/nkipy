import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from collective import all_gather
from neuronxcc import nki
from kernels.rmsnorm import rmsnorm
from kernels.softmax import softmax
from nkipy.core import tensor_apis
from nkipy.core.nki_op import wrap_nki_kernel
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import (
    TILE_SIZE,
)
from neuronxcc.starfish.penguin.common import div_ceil
import parallel_state


def rmsnorm_router(
    hidden_states_sharded,
    post_attention_weight,
    router_weight,
    router_bias,
    norm_eps,
    top_k,
    is_neuronpy,
):
    hidden_size = hidden_states_sharded.shape[-1]
    # reshape because when run in cpu mode, hidden_states_sharded shape is (B, L, D)
    hidden_states_sharded = hidden_states_sharded.reshape(-1, hidden_size)
    normed_hidden_states_sharded = rmsnorm(
        hidden_states_sharded,
        post_attention_weight,
        norm_eps,
        is_neuronpy=is_neuronpy,
    )
    top_k_indices, expert_affinities_masked_sharded = router(
        hidden_states_sharded=normed_hidden_states_sharded,
        router_weight=router_weight,
        router_bias=router_bias,
        top_k=top_k,
        is_prefill=True,
        is_neuronpy=is_neuronpy,
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

@nki.compiler.skip_middle_end_transformations
@nki.jit(
    debug_kernel=True,
    show_compiler_tb=True,
)
def transpose_2d(in_tensor):
    n_tokens, hidden_size = in_tensor.shape
    out_tensor = nl.ndarray((hidden_size, n_tokens), dtype=in_tensor.dtype, buffer=nl.hbm)

    hidden_n_tiles = div_ceil(hidden_size, TILE_SIZE)

    for i_hidden in nl.affine_range(hidden_n_tiles):
        # out_sbuf = nl.ndarray((TILE_SIZE, n_tokens), dtype=in_tensor.dtype, buffer=nl.sbuf)
        in_p, in_f = nl.mgrid[0:n_tokens, 0:TILE_SIZE]
        out_sbuf = nisa.dma_transpose(
            in_tensor[in_p, i_hidden * TILE_SIZE + in_f],
            axes=(1, 0),
            mask=i_hidden * TILE_SIZE + in_f < hidden_size,
        )
        out_p, out_f = nl.mgrid[0:TILE_SIZE, 0:n_tokens]
        nisa.dma_copy(
            dst=out_tensor[i_hidden * TILE_SIZE + out_p, out_f],
            src=out_sbuf,
            mask=((i_hidden * TILE_SIZE + out_p < hidden_size)),
        )

    return out_tensor

def router(
    hidden_states_sharded,
    router_weight,
    router_bias,
    top_k,
    is_prefill,
    is_neuronpy,
):
    if is_neuronpy:
        # manually insert transpose to avoid bad transpose
        transpose_nki = wrap_nki_kernel(
            transpose_2d,
            [hidden_states_sharded],
        )
        hidden_states_sharded_T = transpose_nki(hidden_states_sharded)
        hidden_states_sharded = np.transpose(hidden_states_sharded_T, (1, 0))
    router_logits_sharded = (hidden_states_sharded @ router_weight).astype(
        hidden_states_sharded.dtype
    )
    router_logits_sharded += router_bias
    if is_neuronpy:
        _, top_k_indices_sharded = tensor_apis.topk(router_logits_sharded, k=top_k, axis=1)
        top_k_indices_sharded = top_k_indices_sharded.astype(np.int8)
    else:
        # numpy does not have top_k api
        top_k_indices_sharded = np.argsort(router_logits_sharded, axis=1)[:, -top_k:][:, ::-1].astype(np.int8)
    # calculate mask using vector engine
    expert_mask_sharded = np.zeros_like(router_logits_sharded).astype(np.float32)
    n_experts = router_logits_sharded.shape[1]
    if is_neuronpy:
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
        is_neuronpy=is_neuronpy,
    )
    if is_prefill:
        top_k_indices = all_gather(
            data=top_k_indices_sharded,
            all_gather_dim=0,
            replica_groups=parallel_state.get_prefill_ep_world_group(),
            is_neuronpy=is_neuronpy,
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
    is_neuronpy,
):
    router_logits_sharded = (hidden_states_sharded @ router_weight).astype(hidden_states_sharded.dtype)
    router_logits_sharded += router_bias
    if is_neuronpy:
        top_k_logits_sharded, top_k_indices_sharded = tensor_apis.topk(router_logits_sharded, k=top_k, axis=1)
    else:
        # numpy does not have top_k api
        top_k_indices_sharded = np.argsort(router_logits_sharded, axis=1)[:, -top_k:][:, ::-1].astype(np.int8)
        top_k_logits_sharded = np.take_along_axis(router_logits_sharded, top_k_indices_sharded, axis=1)
    top_k_logits_sharded = softmax(top_k_logits_sharded, is_neuronpy=is_neuronpy)

    top_k_indices = top_k_indices_sharded
    top_k_logits = top_k_logits_sharded
    n_tokens = top_k_logits.shape[0]
    n_experts = router_logits_sharded.shape[1]
    if is_neuronpy:
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