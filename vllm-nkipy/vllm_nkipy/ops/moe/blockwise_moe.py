# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Blockwise MoE implementation (optimized NKI processing).

This module provides the optimized NKI-based implementation of MoE using
blockwise processing for efficient computation on NKIPy.

Used when VLLM_NKIPY_MOE_USE_NKI=1.
"""

from typing import Callable, Optional

import torch
from vllm.logger import init_logger

from vllm_nkipy import envs
from vllm_nkipy.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod,
)
from vllm_nkipy.ops.moe import utils
from vllm_nkipy.ops.moe.blockwise_index import (
    get_blockwise_expert_and_token_mapping,
    get_n_blocks,
)
from vllm_nkipy.ops.moe.constants import BLOCK_SIZE, ControlType
from vllm_nkipy.ops.moe.nki.blockwise_nki_wrapper import (
    blockwise_nki_static_custom_op,
    blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op_compiled,
)
from vllm_nkipy.ops.moe.nki.blockwise_np import blockwise_np_compile
from vllm_nkipy.ops.moe.router import (
    expert_affinities_slice,
    router_prefill,
    router_tokengen,
)

logger = init_logger(__name__)


# Use centralized environment variables from envs module
logger.info(f"VLLM_NKIPY_MOE_2D={envs.VLLM_NKIPY_MOE_2D}")
logger.info(f"VLLM_NKIPY_MOE_BLOCKWISE_IMPL={envs.VLLM_NKIPY_MOE_BLOCKWISE_IMPL}")


def moe_tokengen(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
):
    # print(
    #     f"[DEBUG] {x.shape=}, {use_grouped_topk=}, "
    #     f"{top_k=}, {router_logits.shape=}, "
    #     f"{renormalize=}, {topk_group=}, "
    #     f"{num_expert_group=}, {expert_map=}, "
    #     f"{custom_routing_function=}, {scoring_func=}, "
    #     f"{e_score_correction_bias=}, "
    #     f"{apply_router_weight_on_input=}, "
    #     f"{activation=}, {enable_eplb=}, "
    # )

    # print(
    #     f"[DEBUG] {layer.w13_weight.shape=}, {layer.w2.shape=},"
    # )

    assert not use_grouped_topk
    assert num_expert_group is None
    assert topk_group is None
    assert custom_routing_function is None
    assert apply_router_weight_on_input is False

    expert_affinities_masked = router_tokengen(
        router_logits, top_k
    )  # (n_tokens, n_experts)

    from vllm.config import get_current_vllm_config
    from vllm.distributed import parallel_state

    vllm_config = get_current_vllm_config()
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size > 1:
        x = parallel_state.get_dp_group().all_gather(x, dim=0)
        expert_affinities_masked = parallel_state.get_dp_group().all_gather(
            expert_affinities_masked, dim=0
        )

    if envs.VLLM_NKIPY_MOE_2D:
        ep_rank, ep_size = utils.calc_ep()
        expert_affinities_masked = expert_affinities_slice(
            expert_affinities_masked_all_experts=expert_affinities_masked,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
    else:
        ep_rank, ep_size = 0, 1

    output = torch.zeros_like(x)
    n_experts_per_ep = layer.w13_weight.shape[0]
    batch_size = x.shape[0]

    # Create token position to ID mapping
    token_position_to_id = torch.full(
        (1, BLOCK_SIZE), ControlType.SKIP_DMA.value, dtype=torch.int32, device=x.device
    )
    token_position_to_id[0, :batch_size] = torch.arange(
        batch_size, dtype=torch.int32, device=x.device
    ) + torch.zeros((batch_size,), dtype=torch.int32, device=x.device)

    # Create block to expert mapping
    block_to_expert = torch.arange(
        n_experts_per_ep, dtype=torch.int8, device=x.device
    ) + torch.zeros((n_experts_per_ep,), dtype=torch.int8, device=x.device)

    # Broadcast token_position_to_id
    token_position_to_id = token_position_to_id.expand(n_experts_per_ep, BLOCK_SIZE)

    # Determine which implementation to use
    use_torch_impl = (x.device.type == "cpu") or (
        envs.VLLM_NKIPY_MOE_BLOCKWISE_IMPL == "torch"
    )
    # use_torch_impl = False

    # Call the custom operation
    if use_torch_impl:
        # output = blockwise_np(
        #     hidden_states=x,
        #     expert_affinities_masked=expert_affinities_masked,
        #     gate_up_proj_weight=layer.w13_weight,
        #     gate_up_bias_plus1_T=layer.w13_bias,
        #     down_proj_weight=layer.w2_weight,
        #     down_bias_broadcasted=layer.w2_bias,
        #     token_position_to_id=token_position_to_id,
        #     block_to_expert=block_to_expert,
        # )
        output = blockwise_np_compile(
            hidden_states=x,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=layer.w13_weight,
            gate_up_bias_plus1_T=layer.w13_bias,
            down_proj_weight=layer.w2_weight,
            down_bias_broadcasted=layer.w2_bias,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
        )
    else:
        # Transpose expert affinities for the custom op
        expert_affinities_masked_T = expert_affinities_masked.transpose(0, 1)
        # output = blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op(
        output = (
            blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op_compiled(
                x,
                expert_affinities_masked_T,
                layer.w13_weight,
                layer.w13_bias,  # gate_up_bias_plus1_T
                layer.w2_weight,
                layer.w2_bias,  #  down_bias_broadcasted,
                token_position_to_id,
                block_to_expert,
            )
        )

    return output


@torch.compiler.disable
def run_cpu_part(top_k_indices, expert_affinities_masked, n_experts_per_ep, layer):
    if envs.VLLM_NKIPY_MOE_2D:
        ep_rank, ep_size = utils.calc_ep()
    else:
        ep_rank, ep_size = 0, 1  # noqa

    top_k_indices = top_k_indices.cpu()

    # top_k_indices = top_k_indices.copy()

    # current EP rank expert index starts from 0
    smaller_indices = top_k_indices < n_experts_per_ep * ep_rank
    larger_indices = top_k_indices >= n_experts_per_ep * (ep_rank + 1)
    top_k_indices[smaller_indices | larger_indices] = ControlType.SKIP_DMA.value
    top_k_indices[~(smaller_indices | larger_indices)] -= n_experts_per_ep * ep_rank

    # TODO: access real seq_len here?
    # FIXME: assume reqs' length is all identical
    # for e in range(parallel_state.get_prefill_ep_size()):
    #     for b in range(self.config.max_batch_size_per_dp):
    #         seq_offset = self.config.max_model_len * (
    #             e * self.config.max_batch_size_per_dp + b
    #         )
    #         top_k_indices[
    #             seq_offset + seq_len[b] : seq_offset + self.config.max_model_len
    #         ] = ControlType.SKIP_DMA.value

    num_blocks, num_static_blocks = get_n_blocks(
        top_k_indices.shape[0],
        4,  # config.num_experts_per_tok
        128,  # config.num_experts
    )
    _, block_to_expert, token_position_to_id = get_blockwise_expert_and_token_mapping(
        top_k_indices=top_k_indices,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_experts=n_experts_per_ep,
        num_static_blocks=num_static_blocks,
    )

    token_position_to_id = token_position_to_id.to("nkipy")
    block_to_expert = block_to_expert.to("nkipy")
    return block_to_expert, token_position_to_id, num_static_blocks


def moe_prefill(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
):
    from vllm.config import get_current_vllm_config
    from vllm.distributed import parallel_state

    vllm_config = get_current_vllm_config()
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size > 1:
        x = parallel_state.get_dp_group().all_gather(x, dim=0)
        router_logits = parallel_state.get_dp_group().all_gather(router_logits, dim=0)

    top_k_indices, expert_affinities_masked = router_prefill(router_logits, top_k)

    if envs.VLLM_NKIPY_MOE_2D:
        ep_rank, ep_size = utils.calc_ep()
        expert_affinities_masked = expert_affinities_slice(
            expert_affinities_masked_all_experts=expert_affinities_masked,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
    else:
        ep_rank, ep_size = 0, 1

    # output = torch.zeros_like(x)
    n_experts_per_ep = layer.w13_weight.shape[0]

    block_to_expert, token_position_to_id, num_static_blocks = run_cpu_part(
        top_k_indices, expert_affinities_masked, n_experts_per_ep, layer
    )

    output = blockwise_nki_static_custom_op(
        x,
        # output,
        expert_affinities_masked,
        layer.w13_weight,
        layer.w13_bias,
        layer.w2_weight,
        layer.w2_bias,
        token_position_to_id,
        block_to_expert,
        num_static_blocks,
    )

    return output


def custom_unquantized_fused_moe_method(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
    logical_replica_count: Optional[torch.Tensor] = None,
):
    # print(
    #     f"[DEBUG] {x.shape=}, {use_grouped_topk=}, "
    #     f"{top_k=}, {router_logits.shape=}, "
    #     f"{renormalize=}, {topk_group=}, "
    #     f"{num_expert_group=}, {expert_map=}, "
    #     f"{custom_routing_function=}, {scoring_func=}, "
    #     f"{e_score_correction_bias=}, "
    #     f"{apply_router_weight_on_input=}, "
    #     f"{activation=}, {enable_eplb=}, "
    # )

    # print(
    #     f"[DEBUG] {layer.w13_weight.shape=}, {layer.w2.shape=},"
    # )

    assert not use_grouped_topk
    assert num_expert_group is None
    assert topk_group is None
    assert custom_routing_function is None
    assert apply_router_weight_on_input is False

    if x.shape[0] < 128:
        output = moe_tokengen(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
            enable_eplb,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )
    else:
        output = moe_prefill(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
            enable_eplb,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    return output


if envs.VLLM_NKIPY_MOE_USE_NKI:
    UnquantizedFusedMoEMethod.forward_oot = custom_unquantized_fused_moe_method
