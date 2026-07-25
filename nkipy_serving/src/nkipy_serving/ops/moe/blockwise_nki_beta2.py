"""Beta2 NKI prefill blockwise MoE kernels for DSV4.

The ``neuronxcc.nki`` prefill variant accepts the right shapes but its dynamic
token gather/scatter drops most non-zero rows on device. This module uses the
Beta2 access-pattern API for per-token vector offsets, matching the pattern
used by the local nkilib MoE kernels.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import sendrecv
from nki.isa.constants import oob_mode
from nkilib.core.moe.moe_cte.moe_cte_utils import (
    PSUM_SIZE,
    TILE_SIZE,
    Configs,
    SkipMode,
    calculate_expert_affinities,
    div_ceil,
    load_block_expert,
    load_token_indices,
)
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.experimental.moe.forward.bwmm_shard_on_H import (
    DimensionSizes,
    compute_block_output_shard,
    create_block_hidden_states,
    load_down_proj_weight_shard,
    load_gate_up_proj_weights_shard,
    load_hidden_states_shard_with_scale,
    load_old_block_shard,
    output_initialization_shard,
    store_block_output_shard,
    transpose_hidden_states,
)


def _load_gate_up_bias_tile(gate_up_bias_plus1_T_hbm, block_expert, i_tile_idx, I_TP):
    num_i = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
    bias = nl.ndarray((TILE_SIZE, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias[0:num_i, 0:2],
        src=gate_up_bias_plus1_T_hbm.ap(
            pattern=[[2, num_i], [1, 2]],
            offset=i_tile_idx * TILE_SIZE * 2,
            scalar_offset=block_expert,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )
    return bias


def _apply_dsv4_bias_and_clamps(
    gate_and_up_proj_states,
    gate_up_bias_plus1_T_hbm,
    block_expert,
    dims,
    *,
    gate_clamp_upper,
    gate_clamp_lower,
    up_clamp_upper,
    up_clamp_lower,
):
    n_psum_tile = div_ceil(dims.B, PSUM_SIZE)
    gup_n_tile = div_ceil(dims.I_TP, TILE_SIZE)
    free_size = gate_and_up_proj_states[0][0][0].shape[-1]

    for i_tile_idx in range(gup_n_tile):
        num_i = min(TILE_SIZE, dims.I_TP - TILE_SIZE * i_tile_idx)
        bias = _load_gate_up_bias_tile(
            gate_up_bias_plus1_T_hbm,
            block_expert,
            i_tile_idx,
            dims.I_TP,
        )
        for b_psum_idx in range(n_psum_tile):
            gate = gate_and_up_proj_states[0][b_psum_idx][i_tile_idx]
            up = gate_and_up_proj_states[1][b_psum_idx][i_tile_idx]
            nisa.tensor_scalar(
                dst=gate[0:num_i, 0:free_size],
                data=gate[0:num_i, 0:free_size],
                op0=nl.add,
                operand0=bias[0:num_i, 0:1],
                op1=nl.minimum,
                operand1=gate_clamp_upper,
            )
            if gate_clamp_lower is not None:
                nisa.tensor_scalar(
                    dst=gate[0:num_i, 0:free_size],
                    data=gate[0:num_i, 0:free_size],
                    op0=nl.maximum,
                    operand0=gate_clamp_lower,
                )
            nisa.tensor_scalar(
                dst=up[0:num_i, 0:free_size],
                data=up[0:num_i, 0:free_size],
                op0=nl.add,
                operand0=bias[0:num_i, 1:2],
                op1=nl.minimum,
                operand1=up_clamp_upper,
            )
            nisa.tensor_scalar(
                dst=up[0:num_i, 0:free_size],
                data=up[0:num_i, 0:free_size],
                op0=nl.maximum,
                operand0=up_clamp_lower,
            )


def compute_gate_and_up_projections_shard_dsv4(
    block_hidden_states_T,
    gup_weights,
    gate_up_bias_plus1_T_hbm,
    block_expert,
    dims,
    shard_id,
    *,
    gate_clamp_upper,
    gate_clamp_lower,
    up_clamp_upper,
    up_clamp_lower,
):
    """Gate/up projection with optional LNC all-reduce and DSV4 clamps."""

    n_psum_tile = div_ceil(dims.B, PSUM_SIZE)
    gup_n_tile = div_ceil(dims.I_TP, TILE_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    free_size = block_hidden_states_T[0][0].shape[-1]
    linearized_tripcount = div_ceil(dims.H_per_shard, TILE_SIZE)
    h_outer_tripcount = div_ceil(dims.H_per_shard, PSUM_SIZE)

    gate_and_up_proj_states = []
    for _ in range(2):
        outer_list = []
        for _ in range(n_psum_tile):
            inner_list = []
            for _ in range(gup_n_tile):
                tile = nl.ndarray(
                    (TILE_SIZE, free_size),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                inner_list.append(tile)
            outer_list.append(inner_list)
        gate_and_up_proj_states.append(outer_list)

    for gate_or_up in range(2):
        for b_tile_idx in range(n_psum_tile):
            for i_tile_idx in range(gup_n_tile):
                num_i = min(TILE_SIZE, dims.I_TP - TILE_SIZE * i_tile_idx)
                psum_acc = nl.ndarray(
                    (num_i, free_size),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for h_outer_idx in range(h_outer_tripcount):
                    for h_inner_idx in range(h_inner_tripcount):
                        h_lin_idx = h_outer_idx * h_inner_tripcount + h_inner_idx
                        if h_lin_idx < linearized_tripcount:
                            h_offset = PSUM_SIZE * h_outer_idx + TILE_SIZE * h_inner_idx
                            num_h = min(TILE_SIZE, dims.H_per_shard - h_offset)
                            nisa.nc_matmul(
                                dst=psum_acc[0:num_i, 0:free_size],
                                stationary=gup_weights[h_outer_idx][h_inner_idx][
                                    0:num_h,
                                    gate_or_up,
                                    i_tile_idx * TILE_SIZE : i_tile_idx * TILE_SIZE
                                    + num_i,
                                ],
                                moving=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                    0:num_h,
                                    b_tile_idx,
                                    0:free_size,
                                ],
                            )

                local = nl.ndarray((num_i, free_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=local, src=psum_acc)
                dst = gate_and_up_proj_states[gate_or_up][b_tile_idx][i_tile_idx]
                if dims.NUM_SHARDS == 2:
                    recv = nl.ndarray(
                        (num_i, free_size), dtype=nl.float32, buffer=nl.sbuf
                    )
                    sendrecv(
                        src=local[0:num_i, 0:free_size],
                        dst=recv[0:num_i, 0:free_size],
                        send_to_rank=(1 - shard_id),
                        recv_from_rank=(1 - shard_id),
                        pipe_id=0,
                    )
                    nisa.tensor_tensor(
                        dst=dst[0:num_i, 0:free_size],
                        data1=local,
                        op=nl.add,
                        data2=recv,
                    )
                else:
                    nisa.tensor_copy(dst=dst[0:num_i, 0:free_size], src=local)

    _apply_dsv4_bias_and_clamps(
        gate_and_up_proj_states,
        gate_up_bias_plus1_T_hbm,
        block_expert,
        dims,
        gate_clamp_upper=gate_clamp_upper,
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=up_clamp_upper,
        up_clamp_lower=up_clamp_lower,
    )
    return gate_and_up_proj_states


def compute_intermediate_states_dsv4(gate_and_up_proj_states, B, I_TP, dtype):
    n_psum_tile = div_ceil(B, PSUM_SIZE)
    gup_n_tile = div_ceil(I_TP, TILE_SIZE)
    free_size = gate_and_up_proj_states[0][0][0].shape[-1]

    intermediate_states = []
    tmp_states = []
    for _ in range(gup_n_tile):
        intermediate_states.append(
            nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf)
        )
        tmp_states.append(nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf))

    for i_tile_idx in range(gup_n_tile):
        num_i = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
        for b_psum_idx in range(n_psum_tile):
            start = b_psum_idx * PSUM_SIZE
            end = start + free_size
            nisa.activation(
                dst=tmp_states[i_tile_idx][0:num_i, start:end],
                op=nl.silu,
                data=gate_and_up_proj_states[0][b_psum_idx][i_tile_idx][
                    0:num_i,
                    0:free_size,
                ],
                scale=1.0,
            )
            nisa.tensor_tensor(
                dst=intermediate_states[i_tile_idx][0:num_i, start:end],
                data1=tmp_states[i_tile_idx][0:num_i, start:end],
                op=nl.multiply,
                data2=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][
                    0:num_i,
                    0:free_size,
                ],
            )
    return intermediate_states


@nki.jit
def blockwise_nki_prefill_dsv4_beta2(
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    gate_up_bias_plus1_T_hbm: nl.ndarray,
    down_proj_weight: nl.ndarray,
    token_position_to_id_flat: nl.ndarray,
    block_to_expert: nl.ndarray,
    block_size: int = TILE_SIZE,
    compute_dtype: nki.dtype = nl.bfloat16,
    gate_clamp_upper: float = 10.0,
    gate_clamp_lower: float | None = None,
    up_clamp_upper: float = 10.0,
    up_clamp_lower: float = -10.0,
) -> nl.ndarray:
    """DSV4 prefill MoE with AP vector-offset token gather/scatter.

    This correctness-first kernel implements the real DSV4 no-down-bias path:
    post-scale expert affinities, gate/up bias, DSV4 SwiGLU clamps, and output
    accumulation for top-k routing.
    """

    T, H = hidden_states.shape
    B = block_size
    _, _, _, I_TP = gate_up_proj_weight.shape
    E, I_TP_padded, _ = down_proj_weight.shape
    N = token_position_to_id_flat.shape[0] // B

    if I_TP_padded % 16 != 0:
        raise AssertionError("down_proj_weight I_TP must be divisible by 16")

    shard_id = nl.program_id(axis=0)
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=I_TP)
    dims.derive_all_dims()

    if H % dims.NUM_SHARDS != 0:
        raise AssertionError("hidden dim must be divisible by LNC shard count")

    cfg = Configs(
        skip_dma=SkipMode(skip_token=True, skip_weight=False),
        compute_dtype=compute_dtype,
        scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=True,
        use_dynamic_while=False,
        linear_bias=False,
        activation_function=ActFnType.SiLU,
        is_quant=False,
        fuse_gate_and_up_load=False,
    )

    output = nl.ndarray(
        hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm
    )
    output_initialization_shard(output, dims, shard_id)

    for block_idx in nl.sequential_range(N):
        token_indices = load_token_indices(
            token_position_to_id_flat, block_idx, B, dims.NUM_TILES
        )
        block_expert = load_block_expert(block_to_expert, block_idx)

        block_hidden_states = create_block_hidden_states(
            dims.H_per_shard,
            dims.NUM_TILES,
            compute_dtype,
        )
        load_hidden_states_shard_with_scale(
            hidden_states,
            block_hidden_states,
            token_indices,
            None,
            dims,
            cfg,
            shard_id,
        )
        block_hidden_states_T = transpose_hidden_states(
            block_hidden_states,
            dims,
            compute_dtype,
        )

        gup_weights = load_gate_up_proj_weights_shard(
            gate_up_proj_weight,
            block_expert,
            cfg,
            dims,
            shard_id,
        )
        gate_and_up_proj_states = compute_gate_and_up_projections_shard_dsv4(
            block_hidden_states_T,
            gup_weights,
            gate_up_bias_plus1_T_hbm,
            block_expert,
            dims,
            shard_id,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
        )
        intermediate_states = compute_intermediate_states_dsv4(
            gate_and_up_proj_states,
            B,
            I_TP,
            compute_dtype,
        )

        block_old = load_old_block_shard(output, token_indices, dims, cfg, shard_id)
        dp_weights = load_down_proj_weight_shard(
            down_proj_weight,
            block_expert,
            cfg,
            dims,
            shard_id,
        )
        expert_affinity = calculate_expert_affinities(
            expert_affinities_masked,
            token_indices,
            block_expert,
            E,
            dims.NUM_TILES,
            nl.float32,
            cfg.skip_dma,
        )

        block_new = compute_block_output_shard(
            intermediate_states,
            dp_weights,
            expert_affinity,
            block_old,
            None,
            block_idx,
            dims,
            cfg,
            shard_id,
        )

        store_block_output_shard(
            output, block_new, token_indices, dims, shard_id, cfg.skip_dma
        )
        if dims.NUM_SHARDS == 2:
            nisa.core_barrier(output, (0, 1))

    return output
