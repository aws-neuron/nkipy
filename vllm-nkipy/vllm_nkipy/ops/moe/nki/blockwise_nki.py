# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# ruff: noqa: E741
"""
kernels - Builtin high performance blockwise matmul kernels
"""

import math

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import (
    TILE_SIZE,
    output_initialization,
)
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
from neuronxcc.nki._pre_prod_kernels.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from neuronxcc.nki.isa.constants import oob_mode

# from config import Config
# nkipy import
# from kernels.blockwise_index import ControlType
from vllm_nkipy.ops.moe.nki.utils import Config, ControlType


def load_token_indices(buffer_idx, token_indices, token_position_to_id, block_idx):
    nisa.dma_copy(
        dst=token_indices[:, buffer_idx],
        src=token_position_to_id[block_idx, nl.arange(TILE_SIZE)[:, None]],
    )
    # return token_indices


def compute_intermediate_states_T(
    gate_and_up_proj_state_T,
    I_TP,
    dtype,
    activation_function: ActFnType,
    mask=None,
    gup_scale=None,
):
    i_n_tile = math.ceil(I_TP / TILE_SIZE)
    # [I, B]
    intermediate_states_T = nl.ndarray(
        (nl.par_dim(TILE_SIZE), i_n_tile, TILE_SIZE), dtype=dtype, buffer=nl.sbuf
    )
    # Note: Avoid compiler created bias create unnecessary aliasing
    bias = nl.zeros((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
    for i_i in nl.affine_range(i_n_tile):
        mask = nl.arange(TILE_SIZE)[:, None] + TILE_SIZE * i_i < I_TP
        if activation_function == ActFnType.Swish:
            intermediate_states_T[:, i_i] = nisa.activation(
                op=nl.gelu_apprx_sigmoid,
                data=gate_and_up_proj_state_T[0, i_i],
                mask=mask,
                bias=bias,
            )
            intermediate_states_T[:, i_i] = nl.multiply(
                intermediate_states_T[:, i_i],
                gate_and_up_proj_state_T[1, i_i],
                mask=mask,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(
                f"Activation function {activation_function} not implemented"
            )
    return intermediate_states_T


def load_gate_up_proj_weights(
    buffer_idx, gate_up_proj_weight, gup_weights_sbuf, block_expert
):
    E, H, _, I = gate_up_proj_weight.shape
    h_n_tile = math.ceil(H / TILE_SIZE)

    load_p, load_p_offset, load_f = nl.mgrid[0:TILE_SIZE, 0:h_n_tile, 0 : 2 * I]
    nisa.dma_copy(
        dst=gup_weights_sbuf[load_p, buffer_idx, load_p_offset, load_f],
        src=gate_up_proj_weight.reshape((E, H, 2 * I))[
            block_expert[0, buffer_idx, 0],
            load_p + load_p_offset * TILE_SIZE,
            load_f,
        ],
        mask=load_p + load_p_offset * TILE_SIZE < H,
        oob_mode=oob_mode.skip,
    )


def load_down_proj_weights(
    buffer_idx,
    down_proj_weight,
    block_expert,
    down_weights_sbuf,
):
    _, I, H = down_proj_weight.shape
    i_n_tile = int(np.ceil(I / TILE_SIZE))

    load_p, load_p_offset, load_f = nl.mgrid[0:TILE_SIZE, 0:i_n_tile, 0:H]
    nisa.dma_copy(
        dst=down_weights_sbuf[load_p, buffer_idx, load_p_offset, load_f],
        src=down_proj_weight[
            block_expert[0, buffer_idx, 0],
            load_p + load_p_offset * TILE_SIZE,
            load_f,
        ],
        mask=load_p + load_p_offset * TILE_SIZE < I,
        oob_mode=oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,  # FIXME: fix issue with ToT compiler
    )


def load_gate_up_bias_T(
    buffer_idx,
    gate_up_bias,
    gate_up_bias_hbm,
    expert,
    I,
):
    i_n_tile = math.ceil(I / TILE_SIZE)
    load_p, load_f0, load_f1 = nl.mgrid[0:TILE_SIZE, 0:i_n_tile, 0:2]
    nisa.dma_copy(
        dst=gate_up_bias[:, buffer_idx],
        src=gate_up_bias_hbm[
            expert[0, buffer_idx, 0],
            load_f0 * TILE_SIZE + load_p,
            load_f1,
        ],
        mask=load_f0 * TILE_SIZE + load_p < I,
        oob_mode=oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,  # FIXME: fix issue with ToT compiler
    )


def load_block_hidden_states(
    buffer_idx, block_hidden_states, hidden_states, token_indices, compute_dtype
):
    H = hidden_states.shape[-1]
    _, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nisa.dma_copy(
        dst=block_hidden_states[:, buffer_idx],
        src=hidden_states[
            # FIXME: omit index since select entire tile
            token_indices[
                nl.arange(TILE_SIZE)[:, None], buffer_idx + nl.arange(1)[None, :]
            ],
            load_f,
        ],
        oob_mode=oob_mode.skip,
    )
    # return block_hidden_states


def store_block_hidden_states(buffer_idx, output, block_new, token_indices):
    H = output.shape[-1]
    _, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nisa.dma_copy(
        dst=output[
            # FIXME: omit index since select entire tile
            token_indices[
                nl.arange(TILE_SIZE)[:, None], buffer_idx + nl.arange(1)[None, :]
            ],
            load_f,
        ],
        src=block_new,
        oob_mode=oob_mode.skip,
    )


def transpose_block_hidden_states(
    buffer_idx, block_hidden_states_T, block_hidden_states, H, compute_dtype
):
    h_n_tiles = math.ceil(H / TILE_SIZE)
    for h_i in nl.affine_range(h_n_tiles):
        tmp = nisa.nc_transpose(
            block_hidden_states[
                nl.arange(TILE_SIZE)[:, None],
                buffer_idx,
                h_i * TILE_SIZE + nl.arange(TILE_SIZE)[None, :],
            ],
            mask=h_i * TILE_SIZE + nl.arange(TILE_SIZE)[None, :] < H,
        )
        block_hidden_states_T[
            nl.arange(TILE_SIZE)[:, None],
            buffer_idx,
            h_i,
            nl.arange(TILE_SIZE)[None, :],
        ] = nisa.tensor_copy(
            src=tmp, mask=h_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < H
        )
    # return block_hidden_states_T


def load_expert_affinities(
    buffer_idx,
    expert_affinities_masked,
    expert_affinities_masked_hbm,
    token_indices,
    expert,
    compute_dtype,
):
    T, E = expert_affinities_masked_hbm.shape
    expert_boardcasted = nl.ndarray(
        (nl.par_dim(TILE_SIZE), 1), dtype=expert.dtype, buffer=nl.sbuf
    )
    stream_shuffle_broadcast(expert, expert_boardcasted)
    # tensor_scalar requires fp32 input
    expert_boardcasted = nisa.tensor_copy(expert_boardcasted, dtype=np.float32)
    # When token skip, index still negative
    indices_1d = nisa.tensor_scalar(
        token_indices[:, buffer_idx],
        op0=np.multiply,
        operand0=E,
        op1=nl.add,
        operand1=expert_boardcasted,
        dtype=np.int32,
    )
    nisa.dma_copy(
        dst=expert_affinities_masked[
            nl.arange(TILE_SIZE)[:, None], buffer_idx + nl.arange(1)[None, :]
        ],
        # reshape because not support 2d indirect indexing
        src=expert_affinities_masked_hbm.reshape((T * E, 1))[indices_1d],
        oob_mode=oob_mode.skip,
    )
    # return expert_affinities_masked


def load_expert_affinities_transposed(
    buffer_idx,
    expert_affinities_masked,
    expert_affinities_masked_transposed_hbm,
    token_indices,
    expert,
    compute_dtype,
):
    # FIXME: a hack because we are only reading one block
    E, T = expert_affinities_masked_transposed_hbm.shape
    nisa.dma_copy(
        dst=expert_affinities_masked[
            nl.arange(T)[:, None], buffer_idx + nl.arange(1)[None, :]
        ],
        # reshape because not support 2d indirect indexing
        src=expert_affinities_masked_transposed_hbm[expert[0, 0], :],
        oob_mode=oob_mode.skip,
    )
    # return expert_affinities_masked


def compute_gate_and_up_projections(
    weight_buffer_idx,
    hidden_buffer_idx,
    block_hidden_states_T,
    gup_weights_sbuf,
    gate_up_bias_plus1,
    H,
    I,
    dtype,
):
    """Compute gate and up projections."""
    i_n_tile = math.ceil(I / TILE_SIZE)
    h_n_tiles = math.ceil(H / TILE_SIZE)
    # [2, I, B]
    gate_and_up_proj_state_T = nl.ndarray(
        (
            2,
            i_n_tile,
            nl.par_dim(TILE_SIZE),
            TILE_SIZE,
        ),
        dtype=np.float32,
        lazy_initialization=True,
        buffer=nl.psum,
    )
    p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    for gate_or_up in nl.affine_range(2):
        for i_i in nl.affine_range(i_n_tile):
            for h_i in nl.affine_range(h_n_tiles):
                gate_and_up_proj_state_T[gate_or_up, i_i] += nisa.nc_matmul(
                    gup_weights_sbuf[
                        p_dim,
                        weight_buffer_idx,
                        h_i,
                        gate_or_up * I + i_i * TILE_SIZE + f_dim,
                    ][(h_i * TILE_SIZE + p_dim < H) & (i_i * TILE_SIZE + f_dim < I)],
                    block_hidden_states_T[:, hidden_buffer_idx, h_i][
                        h_i * TILE_SIZE + p_dim < H
                    ],
                )
    if gate_up_bias_plus1 is not None:
        # gate
        for i_i in nl.affine_range(i_n_tile):
            gate_and_up_proj_state_T[0, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[0, i_i],
                op0=nl.add,
                operand0=gate_up_bias_plus1[:, weight_buffer_idx, i_i, 0],
                op1=nl.minimum,
                operand1=7.0,
                mask=p_dim + TILE_SIZE * i_i < I,
                dtype=dtype,
            )
        # up
        for i_i in nl.affine_range(i_n_tile):
            gate_and_up_proj_state_T[1, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[1, i_i],
                op0=nl.add,
                operand0=gate_up_bias_plus1[:, weight_buffer_idx, i_i, 1],
                mask=p_dim + TILE_SIZE * i_i < I,
                dtype=dtype,
            )
            gate_and_up_proj_state_T[1, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[1, i_i],
                op0=nl.minimum,
                operand0=8.0,
                op1=nl.maximum,
                operand1=-6.0,
                mask=p_dim + TILE_SIZE * i_i < I,
                dtype=dtype,
            )
    return gate_and_up_proj_state_T


def compute_block_output(
    buffer_idx,
    intermediate_states_T,
    down_weights_sbuf,
    expert_affinities_masked,
    block_old,
    down_bias_broadcasted,
    compute_dtype,
    H,
    I,
):
    block_new = nl.ndarray(
        (nl.par_dim(TILE_SIZE), H), dtype=compute_dtype, buffer=nl.sbuf
    )
    i_n_tile = math.ceil(I / TILE_SIZE)
    h_tile_size = nl.tile_size.gemm_moving_fmax
    h_n_tile = math.ceil(H / h_tile_size)

    for h_i in nl.affine_range(h_n_tile):
        down_proj_psum = nl.zeros(
            (nl.par_dim(TILE_SIZE), h_tile_size),
            dtype=np.float32,
            lazy_initialization=True,
            buffer=nl.psum,
        )
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:h_tile_size]
        h_mask = h_i * h_tile_size + f_dim < H
        for i_i in nl.affine_range(i_n_tile):
            down_proj_psum += nisa.nc_matmul(
                intermediate_states_T[:, i_i][
                    i_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < I
                ],
                down_weights_sbuf[
                    p_dim,
                    buffer_idx,
                    i_i,
                    h_i * h_tile_size + f_dim,
                ][(i_i * TILE_SIZE + p_dim < I) & h_mask],
            )
        if down_bias_broadcasted is not None:
            down_proj_psum[...] = nisa.tensor_tensor(
                down_proj_psum[...],
                down_bias_broadcasted[p_dim, buffer_idx, h_i * h_tile_size + f_dim],
                op=nl.add,
                mask=h_mask,
            )
        block_new[
            p_dim,
            h_i * h_tile_size + f_dim,
        ] = nisa.scalar_tensor_tensor(
            data=down_proj_psum,
            op0=nl.multiply,
            operand0=expert_affinities_masked[:, buffer_idx],
            op1=nl.add,
            operand1=block_old[
                p_dim,
                buffer_idx,
                h_i * h_tile_size + f_dim,
            ],
            mask=h_mask,
            dtype=compute_dtype,
        )
    return block_new


def compute_block_output_in_place(
    buffer_idx,
    intermediate_states_T,
    down_weights_sbuf,
    expert_affinities_masked,
    block,
    down_bias_broadcasted,
    compute_dtype,
    H,
    I,
):
    i_n_tile = math.ceil(I / TILE_SIZE)
    h_tile_size = nl.tile_size.gemm_moving_fmax
    h_n_tile = math.ceil(H / h_tile_size)

    for h_i in nl.affine_range(h_n_tile):
        down_proj_psum = nl.zeros(
            (nl.par_dim(TILE_SIZE), h_tile_size),
            dtype=np.float32,
            lazy_initialization=True,
            buffer=nl.psum,
        )
        p_dim, f_dim = nl.mgrid[0:TILE_SIZE, 0:h_tile_size]
        h_mask = h_i * h_tile_size + f_dim < H
        for i_i in nl.affine_range(i_n_tile):
            down_proj_psum += nisa.nc_matmul(
                intermediate_states_T[:, i_i][
                    i_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < I
                ],
                down_weights_sbuf[
                    p_dim,
                    buffer_idx,
                    i_i,
                    h_i * h_tile_size + f_dim,
                ][(i_i * TILE_SIZE + p_dim < I) & h_mask],
            )
        if down_bias_broadcasted is not None:
            down_proj_psum[...] = nisa.tensor_tensor(
                down_proj_psum[...],
                down_bias_broadcasted[p_dim, buffer_idx, h_i * h_tile_size + f_dim],
                op=nl.add,
                mask=h_mask,
            )
        block[
            p_dim,
            h_i * h_tile_size + f_dim,
        ] = nisa.scalar_tensor_tensor(
            data=down_proj_psum,
            op0=nl.multiply,
            operand0=expert_affinities_masked[:, buffer_idx],
            op1=nl.add,
            operand1=block[
                p_dim,
                h_i * h_tile_size + f_dim,
            ],
            mask=h_mask,
            dtype=compute_dtype,
        )
    # return block_new


@nki.jit(
    debug_kernel=True,
    show_compiler_tb=True,
)
def output_init(output: nt.tensor[nt.mutable]):
    output_initialization(output)
    return output


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    debug_kernel=True,
    show_compiler_tb=True,
)
def blockwise_nki_static(
    hidden_states: nt.tensor,
    # output: nt.tensor[nt.mutable],
    expert_affinities_masked_hbm: nt.tensor,  # TODO: only need (T, TOP_K)
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    activation_function: ActFnType = ActFnType.Swish,
    compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=1,
):
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.hbm)
    output_initialization(output)

    assert is_tensor_update_accumulating
    E, I, H = down_proj_weight.shape
    assert len(hidden_states.shape) == 2
    assert len(output.shape) == 2
    T, _ = hidden_states.shape
    n_blocks = block_to_expert.shape[0]
    assert 0 < num_static_blocks <= n_blocks

    assert gate_up_bias_plus1_T_hbm is not None
    assert down_bias_broadcasted_hbm is not None

    assert gate_up_proj_weight.shape == (E, H, 2, I)

    token_indices = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE),
        dtype=np.int32,
        buffer=nl.sbuf,
    )
    block_hidden_states = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    block_output = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tiles = math.ceil(H / TILE_SIZE)
    # [H, B]
    block_hidden_states_T = nl.ndarray(
        (
            nl.par_dim(TILE_SIZE),
            BUFFER_DEGREE,
            h_n_tiles,
            TILE_SIZE,
        ),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE), dtype=compute_dtype, buffer=nl.sbuf
    )

    # TODO: overlap with compute
    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, h_n_tile, 2 * I),
        # dtype=compute_dtype,
        dtype=gate_up_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(I / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, i_n_tile, H),
        # dtype=compute_dtype,
        dtype=down_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    current_expert_real = nl.zeros(
        (1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf
    )
    current_expert_may_skip = nl.zeros(
        (1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf
    )

    if gate_up_bias_plus1_T_hbm is not None:
        assert gate_up_bias_plus1_T_hbm.shape == (E, I, 2)
        gate_up_bias_plus1_T = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
            dtype=nl.float32,  # later tensor_scalar input dtype must be fp32
            buffer=nl.sbuf,
        )
    else:
        gate_up_bias_plus1_T = None

    if down_bias_broadcasted_hbm is not None:
        assert down_bias_broadcasted_hbm.shape == (E, TILE_SIZE, H)
        down_bias_broadcasted = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
        )
    else:
        down_bias_broadcasted = None

    # with ncc.no_reorder(): # place holder for manual scheduling later
    if True:
        for block_idx in nl.sequential_range(num_static_blocks):
            # A

            buffer_idx_prev = (block_idx - 1) % BUFFER_DEGREE
            buffer_idx_now = block_idx % BUFFER_DEGREE
            # buffer_idx_next = (block_idx + 1) % BUFFER_DEGREE # for future experiments

            load_token_indices(
                buffer_idx_now, token_indices, token_position_to_id, block_idx
            )

            # B
            current_expert_may_skip[:, buffer_idx_now] = nl.load(
                block_to_expert[block_idx], dtype=np.int32
            )
            nisa.tensor_copy_predicated(
                src=current_expert_may_skip[:, buffer_idx_now, :],
                dst=current_expert_real[:, buffer_idx_now, :],
                predicate=nl.not_equal(
                    current_expert_may_skip[0, buffer_idx_now, 0],
                    ControlType.SKIP_DMA.value,
                ),
            )

            # Copy from previous real if skipped
            nisa.tensor_copy_predicated(
                src=current_expert_real[:, buffer_idx_prev, :],
                dst=current_expert_real[:, buffer_idx_now, :],
                predicate=nl.equal(
                    current_expert_may_skip[0, buffer_idx_now, 0],
                    ControlType.SKIP_DMA.value,
                ),
            )

            # C
            load_block_hidden_states(
                buffer_idx=buffer_idx_now,
                block_hidden_states=block_hidden_states,
                hidden_states=hidden_states,
                token_indices=token_indices,
                compute_dtype=compute_dtype,
            )

            # D
            # FIXME:
            #   This N buffering as-is won't work with DMA
            #   skipping. Multiple copies of weights.
            #   Additional logic is required to copy from previous block if skipped
            #
            load_gate_up_proj_weights(
                buffer_idx=buffer_idx_now,
                gate_up_proj_weight=gate_up_proj_weight,
                gup_weights_sbuf=gup_weights_sbuf,
                block_expert=current_expert_may_skip,
            )

            # E
            transpose_block_hidden_states(
                buffer_idx_now,
                block_hidden_states_T,
                block_hidden_states,
                H,
                compute_dtype,
            )

            # F
            # FIXME:
            #   This N buffering as-is won't work with DMA
            #   skipping. Multiple copies of weights.
            #   Additional logic is required to copy from previous block if skipped
            if gate_up_bias_plus1_T_hbm is not None:
                load_gate_up_bias_T(
                    buffer_idx=buffer_idx_now,
                    gate_up_bias=gate_up_bias_plus1_T,
                    gate_up_bias_hbm=gate_up_bias_plus1_T_hbm,
                    expert=current_expert_may_skip,
                    I=I,
                )

            # G
            load_expert_affinities(
                buffer_idx=buffer_idx_now,
                expert_affinities_masked=expert_affinities_masked,
                expert_affinities_masked_hbm=expert_affinities_masked_hbm,
                token_indices=token_indices,
                expert=current_expert_real[:, buffer_idx_now, :],
                compute_dtype=compute_dtype,
            )

            # H
            load_block_hidden_states(
                buffer_idx=buffer_idx_now,
                block_hidden_states=block_output,
                hidden_states=output,
                token_indices=token_indices,
                compute_dtype=compute_dtype,
            )

            # I
            # FIXME: for prefill, we need to copy from previous block if skipped
            load_down_proj_weights(
                buffer_idx=buffer_idx_now,
                down_proj_weight=down_proj_weight,
                block_expert=current_expert_may_skip,
                down_weights_sbuf=down_weights_sbuf,
            )

            # J
            gate_and_up_proj_state_T = compute_gate_and_up_projections(
                weight_buffer_idx=buffer_idx_now,
                hidden_buffer_idx=buffer_idx_now,
                block_hidden_states_T=block_hidden_states_T,
                gup_weights_sbuf=gup_weights_sbuf,
                gate_up_bias_plus1=gate_up_bias_plus1_T,
                H=H,
                I=I,
                dtype=compute_dtype,
            )

            # K
            intermediate_states_T = compute_intermediate_states_T(
                gate_and_up_proj_state_T=gate_and_up_proj_state_T,
                I_TP=I,
                dtype=compute_dtype,
                activation_function=activation_function,
            )

            # L
            if down_bias_broadcasted_hbm is not None:
                nisa.dma_copy(
                    dst=down_bias_broadcasted[:, buffer_idx_now],
                    src=down_bias_broadcasted_hbm[
                        current_expert_may_skip[0, buffer_idx_now, 0]
                    ],
                    oob_mode=oob_mode.skip,
                )

            # M
            block_new = compute_block_output(
                buffer_idx=buffer_idx_now,
                intermediate_states_T=intermediate_states_T,
                down_weights_sbuf=down_weights_sbuf,
                expert_affinities_masked=expert_affinities_masked,
                block_old=block_output,
                down_bias_broadcasted=down_bias_broadcasted,
                compute_dtype=compute_dtype,
                H=H,
                I=I,
            )

            # N
            store_block_hidden_states(buffer_idx_now, output, block_new, token_indices)

    return output


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    debug_kernel=True,
    show_compiler_tb=True,
)
def blockwise_nki_tokengen_one_tile_replicated_hidden_state(
    hidden_states: nt.tensor,
    expert_affinities_masked_transposed_hbm: nt.tensor,  # (E, T)
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    activation_function: ActFnType = ActFnType.Swish,
    compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=3,
):
    """
    This kernel assumes there is only one tile of hidden
    state and it is replicated across multiple experts.
    All tokens are computed against all experts, then
    they are masked off (introduce redundant compute).
    """
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.hbm)

    assert is_tensor_update_accumulating
    E, I, H = down_proj_weight.shape
    assert len(hidden_states.shape) == 2
    assert len(output.shape) == 2
    T, _ = hidden_states.shape
    n_blocks = block_to_expert.shape[0]

    assert gate_up_bias_plus1_T_hbm is not None
    assert down_bias_broadcasted_hbm is not None

    assert gate_up_proj_weight.shape == (E, H, 2, I)

    token_indices = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE),
        dtype=np.int32,
        buffer=nl.sbuf,
    )

    block_hidden_states = nl.ndarray(
        (nl.par_dim(TILE_SIZE), 1, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    block_output = nl.zeros(
        (nl.par_dim(TILE_SIZE), H), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tiles = math.ceil(H / TILE_SIZE)
    # [H, B]
    block_hidden_states_T = nl.ndarray(
        (
            nl.par_dim(TILE_SIZE),
            1,
            h_n_tiles,
            TILE_SIZE,
        ),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.zeros(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, h_n_tile, 2 * I),
        # dtype=compute_dtype,
        dtype=gate_up_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(I / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, i_n_tile, H),
        # dtype=compute_dtype,
        dtype=down_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    current_expert = nl.zeros((1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf)

    if gate_up_bias_plus1_T_hbm is not None:
        assert gate_up_bias_plus1_T_hbm.shape == (E, I, 2)
        gate_up_bias_plus1_T = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
            dtype=nl.float32,  # later tensor_scalar input dtype must be fp32
            buffer=nl.sbuf,
        )
    else:
        gate_up_bias_plus1_T = None

    if down_bias_broadcasted_hbm is not None:
        assert down_bias_broadcasted_hbm.shape == (E, TILE_SIZE, H)
        down_bias_broadcasted = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
        )
    else:
        down_bias_broadcasted = None

    # A
    load_token_indices(0, token_indices, token_position_to_id, 0)

    # C
    load_block_hidden_states(
        buffer_idx=0,
        block_hidden_states=block_hidden_states,
        hidden_states=hidden_states,
        token_indices=token_indices,
        compute_dtype=compute_dtype,
    )

    # E
    transpose_block_hidden_states(
        0, block_hidden_states_T, block_hidden_states, H, compute_dtype
    )

    # with ncc.no_reorder(): # place holder for manual scheduling later
    if True:
        for block_idx in nl.sequential_range(n_blocks):
            # A
            buffer_idx_prev = (block_idx - 1) % BUFFER_DEGREE  # noqa
            buffer_idx_now = block_idx % BUFFER_DEGREE
            # buffer_idx_next = (block_idx + 1) % BUFFER_DEGREE # for future experiments

            load_token_indices(
                buffer_idx_now, token_indices, token_position_to_id, block_idx
            )

            # B
            current_expert[:, buffer_idx_now] = nl.load(
                block_to_expert[block_idx], dtype=np.int32
            )

            # D
            load_gate_up_proj_weights(
                buffer_idx=buffer_idx_now,
                gate_up_proj_weight=gate_up_proj_weight,
                gup_weights_sbuf=gup_weights_sbuf,
                block_expert=current_expert,
            )

            # F
            if gate_up_bias_plus1_T_hbm is not None:
                load_gate_up_bias_T(
                    buffer_idx=buffer_idx_now,
                    gate_up_bias=gate_up_bias_plus1_T,
                    gate_up_bias_hbm=gate_up_bias_plus1_T_hbm,
                    expert=current_expert,
                    I=I,
                )

            # J
            gate_and_up_proj_state_T = compute_gate_and_up_projections(
                weight_buffer_idx=buffer_idx_now,
                hidden_buffer_idx=0,
                block_hidden_states_T=block_hidden_states_T,
                gup_weights_sbuf=gup_weights_sbuf,
                gate_up_bias_plus1=gate_up_bias_plus1_T,
                H=H,
                I=I,
                dtype=compute_dtype,
            )

            # I
            # FIXME: for prefill, we need to copy from previous block if skipped
            load_down_proj_weights(
                buffer_idx=buffer_idx_now,
                down_proj_weight=down_proj_weight,
                block_expert=current_expert,
                down_weights_sbuf=down_weights_sbuf,
            )

            # G
            load_expert_affinities_transposed(
                buffer_idx=buffer_idx_now,
                expert_affinities_masked=expert_affinities_masked,
                expert_affinities_masked_transposed_hbm=(
                    expert_affinities_masked_transposed_hbm
                ),
                token_indices=token_indices,
                expert=current_expert[:, buffer_idx_now, :],
                compute_dtype=compute_dtype,
            )

            # K
            intermediate_states_T = compute_intermediate_states_T(
                gate_and_up_proj_state_T=gate_and_up_proj_state_T,
                I_TP=I,
                dtype=compute_dtype,
                activation_function=activation_function,
            )

            # L
            if down_bias_broadcasted_hbm is not None:
                nisa.dma_copy(
                    dst=down_bias_broadcasted[:, buffer_idx_now],
                    src=down_bias_broadcasted_hbm[current_expert[0, buffer_idx_now, 0]],
                    oob_mode=oob_mode.skip,
                )

            # M
            compute_block_output_in_place(
                buffer_idx=buffer_idx_now,
                intermediate_states_T=intermediate_states_T,
                down_weights_sbuf=down_weights_sbuf,
                expert_affinities_masked=expert_affinities_masked,
                block=block_output,
                down_bias_broadcasted=down_bias_broadcasted,
                compute_dtype=compute_dtype,
                H=H,
                I=I,
            )

    # N
    store_block_hidden_states(0, output, block_output, token_indices)

    return output


def blockwise_add_residual(
    hidden_states: nt.tensor,
    residual_2d_shard: nt.tensor,
    output: nt.tensor,
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    is_nkipy: bool,
):
    """
    Fused kernel that performs blockwise MoE computation
    followed by reduce_scatter and residual addition.
    """
    import parallel_state
    from collective import reduce_scatter
    from kernels.blockwise_np import blockwise_np

    from nkipy.nki.nki_op import NKICustomOp

    if is_nkipy:
        # Use NKICustomOp to call the NKI kernel from within NKIPy
        nki_op = NKICustomOp(
            blockwise_nki_static,
            [
                hidden_states,
                output,
                expert_affinities_masked_hbm,
                gate_up_proj_weight,
                gate_up_bias_plus1_T_hbm,
                down_proj_weight,
                down_bias_broadcasted_hbm,
                token_position_to_id,
                block_to_expert,
                num_static_blocks,
            ],
        )
        output = nki_op(
            [
                hidden_states,
                output,
                expert_affinities_masked_hbm,
                gate_up_proj_weight,
                gate_up_bias_plus1_T_hbm,
                down_proj_weight,
                down_bias_broadcasted_hbm,
                token_position_to_id,
                block_to_expert,
            ]
        )
    else:
        # For non-NKIPy mode, just call the original kernel
        output = blockwise_np(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked_hbm,
            gate_up_proj_weight=gate_up_proj_weight,
            gate_up_bias_plus1_T=gate_up_bias_plus1_T_hbm,
            down_proj_weight=down_proj_weight,
            down_bias_broadcasted=down_bias_broadcasted_hbm,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
        )

    # Perform reduce_scatter and add residual
    hidden_states_shard = reduce_scatter(
        output,
        reduce_scatter_dim=0,
        replica_groups=parallel_state.get_prefill_ep_world_group(),
        is_nkipy=is_nkipy,
    )
    hidden_states_shard = residual_2d_shard + hidden_states_shard
    return hidden_states_shard
