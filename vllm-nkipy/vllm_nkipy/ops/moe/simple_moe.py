# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Simple MoE implementation (CPU/debug fallback).

This is a simple sequential implementation of MoE for debugging and CPU execution.
For optimized NKI implementation, see blockwise_moe.py.

Used when VLLM_NKIPY_MOE_USE_NKI=0 (default).
"""

from typing import Callable, Optional

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod

from vllm_nkipy import envs
from vllm_nkipy.ops.moe.common import custom_router, swiglu

logger = init_logger(__name__)


# Use centralized environment variables from envs module
logger.info(f"VLLM_NKIPY_MOE_TRUNC={envs.VLLM_NKIPY_MOE_TRUNC}")
logger.info(f"VLLM_NKIPY_MOE_MM_MODE={envs.VLLM_NKIPY_MOE_MM_MODE}")
logger.info(f"VLLM_NKIPY_MOE_TRANSPOSE={envs.VLLM_NKIPY_MOE_TRANSPOSE}")


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

    router_scores, router_indices = custom_router(router_logits, top_k)

    output = torch.empty_like(x)  # [S, H]

    S, H = x.shape
    for s in range(S):
        if envs.VLLM_NKIPY_MOE_TRUNC > 0 and s >= envs.VLLM_NKIPY_MOE_TRUNC:
            break
        # Get token input [1, H]
        token_input = x[s : s + 1, :]

        # Get pre-computed top-k experts and weights for this token
        token_indices = router_indices[s]  # [top_k]
        token_scores = router_scores[s]  # [top_k]

        if envs.VLLM_NKIPY_MOE_MM_MODE == "sequential":
            # Process through each selected expert
            token_output = torch.zeros((1, H), dtype=x.dtype, device=x.device)
            for e in range(top_k):
                # Get expert index and weight for this token
                expert_idx = token_indices[e]
                weight = token_scores[e]

                # MLP
                gate_up_proj_weight = torch.index_select(
                    layer.w13_weight, dim=0, index=expert_idx
                ).squeeze(0)

                down_proj_weight = torch.index_select(
                    layer.w2_weight, dim=0, index=expert_idx
                ).squeeze(0)
                gate_up_proj_bias = (
                    torch.index_select(layer.w13_bias, dim=0, index=expert_idx).squeeze(
                        0
                    )
                    if self.has_bias
                    else None
                )
                down_proj_bias = (
                    torch.index_select(layer.w2_bias, dim=0, index=expert_idx).squeeze(
                        0
                    )
                    if self.has_bias
                    else None
                )

                if not envs.VLLM_NKIPY_MOE_TRANSPOSE:
                    gate_up = nn.functional.linear(
                        token_input, gate_up_proj_weight, gate_up_proj_bias
                    )
                else:
                    gate_up = torch.addmm(
                        gate_up_proj_bias, token_input, gate_up_proj_weight
                    )
                t = swiglu(gate_up)
                if not envs.VLLM_NKIPY_MOE_TRANSPOSE:
                    expert_output = nn.functional.linear(
                        t, down_proj_weight, down_proj_bias
                    )
                else:
                    expert_output = torch.addmm(down_proj_bias, t, down_proj_weight)
                token_output = token_output + weight * expert_output
        elif envs.VLLM_NKIPY_MOE_MM_MODE == "batch":
            assert envs.VLLM_NKIPY_MOE_TRANSPOSE == 0
            # MLP
            gate_up_proj_weight = torch.index_select(
                layer.w13_weight, dim=0, index=token_indices
            )  # (k, out_feature * 2, in_feature)

            down_proj_weight = torch.index_select(
                layer.w2_weight, dim=0, index=token_indices
            )
            gate_up_proj_bias = (
                torch.index_select(
                    layer.w13_bias, dim=0, index=token_indices
                ).unsqueeze(1)
                if self.has_bias
                else None
            )
            down_proj_bias = (
                torch.index_select(layer.w2_bias, dim=0, index=token_indices).unsqueeze(
                    1
                )
                if self.has_bias
                else None
            )

            gate_up = (
                torch.matmul(token_input, gate_up_proj_weight.permute(0, 2, 1))
                + gate_up_proj_bias
            )
            # gate_up = nn.functional.linear(
            #     token_input, gate_up_proj_weight,
            #     gate_up_proj_bias,
            # )
            t = swiglu(gate_up)
            expert_output = (
                torch.matmul(t, down_proj_weight.permute(0, 2, 1)) + down_proj_bias
            )
            # expert_output = nn.functional.linear(t, down_proj_weight, down_proj_bias)

            token_output = token_scores.reshape(-1, 1, 1) * expert_output
            token_output = token_output.sum(dim=0)
        else:
            assert False

        # Store the result
        output[s] = token_output[0]
    return output


if not envs.VLLM_NKIPY_MOE_USE_NKI:
    UnquantizedFusedMoEMethod.forward_oot = custom_unquantized_fused_moe_method
