"""
NKI blockwise MoE kernels (prefill + decode).

Adapted from NeuronPyExps' blockwise MoE implementation, kept Torch/JAX-free
so it can live in the nkipy-serving runtime path.

Two kernel paths:
  - **Prefill** (`blockwise_nki_static` / `blockwise_add_residual`):
    CPU builds dynamic (block_to_expert, token_position_to_id) per step.
    Supports arbitrary token counts.
  - **Decode** (`blockwise_nki_decode` / `blockwise_decode_add_residual`):
    Static block mappings baked at compile time (1 block per expert, all tokens
    replicated). No CPU scheduling. Requires token_bucket <= TILE_SIZE (128).

Both wrappers end with reduce-scatter across TP ranks and a residual add.
The expert dimension is kept intact; TP shards the intermediate dimension (I).
"""

import math

import ml_dtypes
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
from nkipy.core.nki_op import wrap_nki_kernel
from nkipy.core.typing import mutable_tensor

# Scheduling sentinel values (must match blockwise_index.ControlType).
_SKIP_DMA = -1


def load_token_indices(buffer_idx, token_indices, token_position_to_id, block_idx):
    nisa.dma_copy(
        dst=token_indices[:, buffer_idx],
        src=token_position_to_id[block_idx, nl.arange(TILE_SIZE)[:, None]],
    )


def compute_intermediate_states_T(
    gate_and_up_proj_state_T,
    I_TP,
    dtype,
    activation_function: ActFnType,
):
    i_n_tile = math.ceil(I_TP / TILE_SIZE)
    intermediate_states_T = nl.ndarray(
        (nl.par_dim(TILE_SIZE), i_n_tile, TILE_SIZE), dtype=dtype, buffer=nl.sbuf
    )
    # Avoid compiler-created bias aliasing.
    bias = nl.zeros((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
    for i_i in nl.affine_range(i_n_tile):
        mask = nl.arange(TILE_SIZE)[:, None] + TILE_SIZE * i_i < I_TP
        if activation_function == ActFnType.Swish:
            # GPT-OSS swiglu_oai uses GELU-approx (alpha=1.702).
            intermediate_states_T[:, i_i] = nisa.activation(
                op=nl.gelu_apprx_sigmoid,
                data=gate_and_up_proj_state_T[0, i_i],
                mask=mask,
                bias=bias,
            )
        elif activation_function == ActFnType.SiLU:
            # V4 swiglu_with_limit: true SiLU (x * sigmoid(x)).
            intermediate_states_T[:, i_i] = nisa.activation(
                op=nl.silu,
                data=gate_and_up_proj_state_T[0, i_i],
                mask=mask,
                bias=bias,
            )
        else:
            raise NotImplementedError(
                f"Activation function {activation_function} not implemented"
            )
        intermediate_states_T[:, i_i] = nl.multiply(
            intermediate_states_T[:, i_i],
            gate_and_up_proj_state_T[1, i_i],
            mask=mask,
            dtype=dtype,
        )
    return intermediate_states_T


def load_gate_up_proj_weights(
    buffer_idx,
    gate_up_proj_weight,
    gup_weights_sbuf,
    block_expert,
):
    E, H, _, intermediate = gate_up_proj_weight.shape
    h_n_tile = math.ceil(H / TILE_SIZE)

    load_p, load_p_offset, load_f = nl.mgrid[
        0:TILE_SIZE, 0:h_n_tile, 0 : 2 * intermediate
    ]
    nisa.dma_copy(
        dst=gup_weights_sbuf[load_p, buffer_idx, load_p_offset, load_f],
        src=gate_up_proj_weight.reshape((E, H, 2 * intermediate))[
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
    _, intermediate, H = down_proj_weight.shape
    i_n_tile = int(np.ceil(intermediate / TILE_SIZE))

    load_p, load_p_offset, load_f = nl.mgrid[0:TILE_SIZE, 0:i_n_tile, 0:H]
    nisa.dma_copy(
        dst=down_weights_sbuf[load_p, buffer_idx, load_p_offset, load_f],
        src=down_proj_weight[
            block_expert[0, buffer_idx, 0],
            load_p + load_p_offset * TILE_SIZE,
            load_f,
        ],
        mask=load_p + load_p_offset * TILE_SIZE < intermediate,
        oob_mode=oob_mode.skip,
        # Empirically helps ToT compiler stability.
        dge_mode=nisa.dge_mode.swdge,
    )


def load_gate_up_bias_T(
    buffer_idx,
    gate_up_bias,
    gate_up_bias_hbm,
    expert,
    intermediate,
):
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
    load_p, load_f0, load_f1 = nl.mgrid[0:TILE_SIZE, 0:i_n_tile, 0:2]
    nisa.dma_copy(
        dst=gate_up_bias[:, buffer_idx],
        src=gate_up_bias_hbm[
            expert[0, buffer_idx, 0],
            load_f0 * TILE_SIZE + load_p,
            load_f1,
        ],
        mask=load_f0 * TILE_SIZE + load_p < intermediate,
        oob_mode=oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
    )


def load_block_hidden_states(
    buffer_idx,
    block_hidden_states,
    hidden_states,
    token_indices,
    compute_dtype,
):
    H = hidden_states.shape[-1]
    _, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nisa.dma_copy(
        dst=block_hidden_states[:, buffer_idx],
        src=hidden_states[
            token_indices[
                nl.arange(TILE_SIZE)[:, None], buffer_idx + nl.arange(1)[None, :]
            ],
            load_f,
        ],
        oob_mode=oob_mode.skip,
    )


def store_block_hidden_states(
    buffer_idx,
    output,
    block_new,
    token_indices,
):
    H = output.shape[-1]
    _, load_f = nl.mgrid[0:TILE_SIZE, 0:H]
    nisa.dma_copy(
        dst=output[
            token_indices[
                nl.arange(TILE_SIZE)[:, None], buffer_idx + nl.arange(1)[None, :]
            ],
            load_f,
        ],
        src=block_new,
        oob_mode=oob_mode.skip,
    )


def transpose_block_hidden_states(
    buffer_idx,
    block_hidden_states_T,
    block_hidden_states,
    H,
    compute_dtype,
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
            src=tmp,
            mask=h_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < H,
        )


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
        src=expert_affinities_masked_hbm.reshape((T * E, 1))[indices_1d],
        oob_mode=oob_mode.skip,
    )


def compute_gate_and_up_projections(
    weight_buffer_idx,
    hidden_buffer_idx,
    block_hidden_states_T,
    gup_weights_sbuf,
    gate_up_bias_plus1,
    H,
    intermediate,
    dtype,
    gate_clamp_upper=7.0,
    gate_clamp_lower=None,
    up_clamp_upper=8.0,
    up_clamp_lower=-6.0,
):
    """Gate+up matmul with per-model clamps.

    GPT-OSS defaults: pre-shifts up-bias by +1 (caller's responsibility)
    and clamps ``gate <= 7, up in [-6, 8]``. V4 passes
    ``gate_clamp_upper=10, up_clamp_upper=10, up_clamp_lower=-10``.
    ``gate_clamp_lower`` is optional (V4's swiglu_with_limit has none
    on gate — only ``minimum(gate, limit)``).
    """
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
    h_n_tiles = math.ceil(H / TILE_SIZE)
    gate_and_up_proj_state_T = nl.ndarray(
        (2, i_n_tile, nl.par_dim(TILE_SIZE), TILE_SIZE),
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
                        gate_or_up * intermediate + i_i * TILE_SIZE + f_dim,
                    ][
                        (h_i * TILE_SIZE + p_dim < H)
                        & (i_i * TILE_SIZE + f_dim < intermediate)
                    ],
                    block_hidden_states_T[:, hidden_buffer_idx, h_i][
                        h_i * TILE_SIZE + p_dim < H
                    ],
                )
    if gate_up_bias_plus1 is not None:
        # gate: add bias and clamp to upper (and optionally lower).
        for i_i in nl.affine_range(i_n_tile):
            gate_and_up_proj_state_T[0, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[0, i_i],
                op0=nl.add,
                operand0=gate_up_bias_plus1[:, weight_buffer_idx, i_i, 0],
                op1=nl.minimum,
                operand1=float(gate_clamp_upper),
                mask=p_dim + TILE_SIZE * i_i < intermediate,
                dtype=dtype,
            )
            if gate_clamp_lower is not None:
                gate_and_up_proj_state_T[0, i_i] = nisa.tensor_scalar(
                    gate_and_up_proj_state_T[0, i_i],
                    op0=nl.maximum,
                    operand0=float(gate_clamp_lower),
                    mask=p_dim + TILE_SIZE * i_i < intermediate,
                    dtype=dtype,
                )
        # up: add bias and clamp to [lower, upper].
        for i_i in nl.affine_range(i_n_tile):
            gate_and_up_proj_state_T[1, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[1, i_i],
                op0=nl.add,
                operand0=gate_up_bias_plus1[:, weight_buffer_idx, i_i, 1],
                mask=p_dim + TILE_SIZE * i_i < intermediate,
                dtype=dtype,
            )
            gate_and_up_proj_state_T[1, i_i] = nisa.tensor_scalar(
                gate_and_up_proj_state_T[1, i_i],
                op0=nl.minimum,
                operand0=float(up_clamp_upper),
                op1=nl.maximum,
                operand1=float(up_clamp_lower),
                mask=p_dim + TILE_SIZE * i_i < intermediate,
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
    intermediate,
):
    block_new = nl.ndarray(
        (nl.par_dim(TILE_SIZE), H), dtype=compute_dtype, buffer=nl.sbuf
    )
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
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
                    i_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < intermediate
                ],
                down_weights_sbuf[p_dim, buffer_idx, i_i, h_i * h_tile_size + f_dim][
                    (i_i * TILE_SIZE + p_dim < intermediate) & h_mask
                ],
            )
        if down_bias_broadcasted is not None:
            down_proj_psum[...] = nisa.tensor_tensor(
                down_proj_psum[...],
                down_bias_broadcasted[p_dim, buffer_idx, h_i * h_tile_size + f_dim],
                op=nl.add,
                mask=h_mask,
            )
        block_new[p_dim, h_i * h_tile_size + f_dim] = nisa.scalar_tensor_tensor(
            data=down_proj_psum,
            op0=nl.multiply,
            operand0=expert_affinities_masked[:, buffer_idx],
            op1=nl.add,
            operand1=block_old[p_dim, buffer_idx, h_i * h_tile_size + f_dim],
            mask=h_mask,
            dtype=compute_dtype,
        )
    return block_new


def output_init_nki(output: nt.tensor[nt.mutable]):
    output_initialization(output)
    return output


def output_init(output: mutable_tensor):
    """Traceable output initialization wrapper (zeros output buffer)."""
    nki_op = wrap_nki_kernel(output_init_nki, [output])
    # Assign back so NKIPy tracing can detect input/output aliasing for the
    # mutable buffer (required to pass a preallocated output tensor at runtime).
    output = nki_op(output)
    return output


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    debug_kernel=False,
    show_compiler_tb=True,
)
def blockwise_nki_static(
    hidden_states: nt.tensor,
    output: nt.tensor[nt.mutable],
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    activation_function: ActFnType = ActFnType.Swish,
    compute_dtype: np.dtype = ml_dtypes.bfloat16,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 1,
    has_down_bias: bool = True,
    initialize_output: bool = False,
    gate_clamp_upper: float = 7.0,
    gate_clamp_lower: float | None = None,
    up_clamp_upper: float = 8.0,
    up_clamp_lower: float = -6.0,
):
    assert is_tensor_update_accumulating
    E, intermediate, H = down_proj_weight.shape
    assert gate_up_proj_weight.shape == (E, H, 2, intermediate)
    assert gate_up_bias_plus1_T_hbm.shape == (E, intermediate, 2)
    if has_down_bias:
        assert down_bias_broadcasted_hbm.shape == (E, TILE_SIZE, H)
    assert len(hidden_states.shape) == 2 and len(output.shape) == 2
    T, _ = hidden_states.shape
    n_blocks = block_to_expert.shape[0]
    assert 0 < num_static_blocks <= n_blocks

    token_indices = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE), dtype=np.int32, buffer=nl.sbuf
    )
    block_hidden_states = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )
    block_output = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tiles = math.ceil(H / TILE_SIZE)
    block_hidden_states_T = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, h_n_tiles, TILE_SIZE),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, h_n_tile, 2 * intermediate),
        dtype=gate_up_proj_weight.dtype,
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, i_n_tile, H),
        dtype=down_proj_weight.dtype,
        buffer=nl.sbuf,
    )
    current_expert_real = nl.zeros(
        (1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf
    )
    current_expert_may_skip = nl.zeros(
        (1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf
    )

    gate_up_bias_plus1_T = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    if has_down_bias:
        down_bias_broadcasted = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
        )

    if initialize_output:
        output_initialization(output)

    for block_idx in nl.sequential_range(num_static_blocks):
        buffer_idx_prev = (block_idx - 1) % BUFFER_DEGREE
        buffer_idx_now = block_idx % BUFFER_DEGREE

        load_token_indices(
            buffer_idx_now, token_indices, token_position_to_id, block_idx
        )

        current_expert_may_skip[:, buffer_idx_now] = nl.load(
            block_to_expert[block_idx], dtype=np.int32
        )
        nisa.tensor_copy_predicated(
            src=current_expert_may_skip[:, buffer_idx_now, :],
            dst=current_expert_real[:, buffer_idx_now, :],
            predicate=nl.not_equal(
                current_expert_may_skip[0, buffer_idx_now, 0], _SKIP_DMA
            ),
        )
        nisa.tensor_copy_predicated(
            src=current_expert_real[:, buffer_idx_prev, :],
            dst=current_expert_real[:, buffer_idx_now, :],
            predicate=nl.equal(
                current_expert_may_skip[0, buffer_idx_now, 0], _SKIP_DMA
            ),
        )

        load_block_hidden_states(
            buffer_idx=buffer_idx_now,
            block_hidden_states=block_hidden_states,
            hidden_states=hidden_states,
            token_indices=token_indices,
            compute_dtype=compute_dtype,
        )
        load_gate_up_proj_weights(
            buffer_idx=buffer_idx_now,
            gate_up_proj_weight=gate_up_proj_weight,
            gup_weights_sbuf=gup_weights_sbuf,
            block_expert=current_expert_may_skip,
        )
        transpose_block_hidden_states(
            buffer_idx_now, block_hidden_states_T, block_hidden_states, H, compute_dtype
        )
        load_gate_up_bias_T(
            buffer_idx=buffer_idx_now,
            gate_up_bias=gate_up_bias_plus1_T,
            gate_up_bias_hbm=gate_up_bias_plus1_T_hbm,
            expert=current_expert_may_skip,
            intermediate=intermediate,
        )
        load_expert_affinities(
            buffer_idx=buffer_idx_now,
            expert_affinities_masked=expert_affinities_masked,
            expert_affinities_masked_hbm=expert_affinities_masked_hbm,
            token_indices=token_indices,
            expert=current_expert_real[:, buffer_idx_now, :],
            compute_dtype=compute_dtype,
        )

        # Accumulate into output buffer.
        load_block_hidden_states(
            buffer_idx=buffer_idx_now,
            block_hidden_states=block_output,
            hidden_states=output,
            token_indices=token_indices,
            compute_dtype=compute_dtype,
        )
        load_down_proj_weights(
            buffer_idx=buffer_idx_now,
            down_proj_weight=down_proj_weight,
            block_expert=current_expert_may_skip,
            down_weights_sbuf=down_weights_sbuf,
        )

        gate_and_up_proj_state_T = compute_gate_and_up_projections(
            weight_buffer_idx=buffer_idx_now,
            hidden_buffer_idx=buffer_idx_now,
            block_hidden_states_T=block_hidden_states_T,
            gup_weights_sbuf=gup_weights_sbuf,
            gate_up_bias_plus1=gate_up_bias_plus1_T,
            H=H,
            intermediate=intermediate,
            dtype=compute_dtype,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
        )
        intermediate_states_T = compute_intermediate_states_T(
            gate_and_up_proj_state_T=gate_and_up_proj_state_T,
            I_TP=intermediate,
            dtype=compute_dtype,
            activation_function=activation_function,
        )

        if has_down_bias:
            nisa.dma_copy(
                dst=down_bias_broadcasted[:, buffer_idx_now],
                src=down_bias_broadcasted_hbm[
                    current_expert_may_skip[0, buffer_idx_now, 0]
                ],
                oob_mode=oob_mode.skip,
            )

        block_new = compute_block_output(
            buffer_idx=buffer_idx_now,
            intermediate_states_T=intermediate_states_T,
            down_weights_sbuf=down_weights_sbuf,
            expert_affinities_masked=expert_affinities_masked,
            block_old=block_output,
            down_bias_broadcasted=(down_bias_broadcasted if has_down_bias else None),
            compute_dtype=compute_dtype,
            H=H,
            intermediate=intermediate,
        )
        store_block_hidden_states(buffer_idx_now, output, block_new, token_indices)

    return output


def load_expert_affinities_transposed(
    buffer_idx,
    expert_affinities_masked,
    expert_affinities_masked_transposed_hbm,
    token_indices,
    expert,
    compute_dtype,
):
    """Load affinities from transposed [E, T] layout — one expert-row at a time."""
    E, T = expert_affinities_masked_transposed_hbm.shape
    nisa.dma_copy(
        dst=expert_affinities_masked[
            nl.arange(T)[:, None], buffer_idx + nl.arange(1)[None, :]
        ],
        src=expert_affinities_masked_transposed_hbm[expert[0, 0], :],
        oob_mode=oob_mode.skip,
    )


def compute_block_output_in_place(
    buffer_idx,
    intermediate_states_T,
    down_weights_sbuf,
    expert_affinities_masked,
    block,
    down_bias_broadcasted,
    compute_dtype,
    H,
    intermediate,
):
    """Accumulate down-projection into existing block buffer (in-place add)."""
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
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
                    i_i * TILE_SIZE + nl.arange(TILE_SIZE)[:, None] < intermediate
                ],
                down_weights_sbuf[
                    p_dim,
                    buffer_idx,
                    i_i,
                    h_i * h_tile_size + f_dim,
                ][(i_i * TILE_SIZE + p_dim < intermediate) & h_mask],
            )
        if down_bias_broadcasted is not None:
            down_proj_psum[...] = nisa.tensor_tensor(
                down_proj_psum[...],
                down_bias_broadcasted[
                    p_dim,
                    buffer_idx,
                    h_i * h_tile_size + f_dim,
                ],
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


@nki.compiler.skip_middle_end_transformations
@nki.jit(debug_kernel=False, show_compiler_tb=True)
def blockwise_nki_decode(
    hidden_states: nt.tensor,
    expert_affinities_masked_transposed_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    activation_function: ActFnType = ActFnType.Swish,
    compute_dtype: np.dtype = ml_dtypes.bfloat16,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 3,
    has_down_bias: bool = True,
    gate_clamp_upper: float = 7.0,
    gate_clamp_lower: float | None = None,
    up_clamp_upper: float = 8.0,
    up_clamp_lower: float = -6.0,
):
    """Decode MoE kernel: single tile of hidden states replicated across experts.

    Adapted from NeuronPyExps' blockwise_nki_tokengen_one_tile_replicated_hidden_state.
    Key differences from blockwise_nki_static (prefill):
      - Hidden states loaded once, reused for all expert blocks (hidden_buffer_idx=0)
      - Output accumulated in SBUF across all experts, stored once at the end
      - BUFFER_DEGREE=3 for better weight prefetching
      - Affinities in transposed [E, T] layout (reads one expert-row per block)
    """
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.hbm)

    assert is_tensor_update_accumulating
    E, intermediate, H = down_proj_weight.shape
    assert len(hidden_states.shape) == 2 and len(output.shape) == 2
    T, _ = hidden_states.shape
    n_blocks = block_to_expert.shape[0]

    assert gate_up_proj_weight.shape == (E, H, 2, intermediate)
    assert gate_up_bias_plus1_T_hbm.shape == (E, intermediate, 2)
    if has_down_bias:
        assert down_bias_broadcasted_hbm.shape == (E, TILE_SIZE, H)

    token_indices = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE),
        dtype=np.int32,
        buffer=nl.sbuf,
    )
    # Single tile of hidden states (not multi-buffered — reused across experts).
    block_hidden_states = nl.ndarray(
        (nl.par_dim(TILE_SIZE), 1, H),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )
    block_output = nl.zeros(
        (nl.par_dim(TILE_SIZE), H),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    h_n_tiles = math.ceil(H / TILE_SIZE)
    block_hidden_states_T = nl.ndarray(
        (nl.par_dim(TILE_SIZE), 1, h_n_tiles, TILE_SIZE),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.zeros(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, h_n_tile, 2 * intermediate),
        dtype=gate_up_proj_weight.dtype,
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(intermediate / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (nl.par_dim(TILE_SIZE), BUFFER_DEGREE, i_n_tile, H),
        dtype=down_proj_weight.dtype,
        buffer=nl.sbuf,
    )
    current_expert = nl.zeros((1, BUFFER_DEGREE, 1), dtype=np.int32, buffer=nl.sbuf)

    gate_up_bias_plus1_T = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    if has_down_bias:
        down_bias_broadcasted = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, H),
            dtype=compute_dtype,
            buffer=nl.sbuf,
        )

    # Pre-load hidden states once (buffer_idx=0, only copy used throughout).
    load_token_indices(0, token_indices, token_position_to_id, 0)
    load_block_hidden_states(
        buffer_idx=0,
        block_hidden_states=block_hidden_states,
        hidden_states=hidden_states,
        token_indices=token_indices,
        compute_dtype=compute_dtype,
    )
    transpose_block_hidden_states(
        0, block_hidden_states_T, block_hidden_states, H, compute_dtype
    )

    for block_idx in nl.sequential_range(n_blocks):
        buffer_idx_now = block_idx % BUFFER_DEGREE

        load_token_indices(
            buffer_idx_now, token_indices, token_position_to_id, block_idx
        )

        current_expert[:, buffer_idx_now] = nl.load(
            block_to_expert[block_idx], dtype=np.int32
        )

        load_gate_up_proj_weights(
            buffer_idx=buffer_idx_now,
            gate_up_proj_weight=gate_up_proj_weight,
            gup_weights_sbuf=gup_weights_sbuf,
            block_expert=current_expert,
        )

        load_gate_up_bias_T(
            buffer_idx=buffer_idx_now,
            gate_up_bias=gate_up_bias_plus1_T,
            gate_up_bias_hbm=gate_up_bias_plus1_T_hbm,
            expert=current_expert,
            intermediate=intermediate,
        )

        gate_and_up_proj_state_T = compute_gate_and_up_projections(
            weight_buffer_idx=buffer_idx_now,
            hidden_buffer_idx=0,  # Always 0 — single tile, replicated.
            block_hidden_states_T=block_hidden_states_T,
            gup_weights_sbuf=gup_weights_sbuf,
            gate_up_bias_plus1=gate_up_bias_plus1_T,
            H=H,
            intermediate=intermediate,
            dtype=compute_dtype,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
        )

        load_down_proj_weights(
            buffer_idx=buffer_idx_now,
            down_proj_weight=down_proj_weight,
            block_expert=current_expert,
            down_weights_sbuf=down_weights_sbuf,
        )

        load_expert_affinities_transposed(
            buffer_idx=buffer_idx_now,
            expert_affinities_masked=expert_affinities_masked,
            expert_affinities_masked_transposed_hbm=expert_affinities_masked_transposed_hbm,
            token_indices=token_indices,
            expert=current_expert[:, buffer_idx_now, :],
            compute_dtype=compute_dtype,
        )

        intermediate_states_T = compute_intermediate_states_T(
            gate_and_up_proj_state_T=gate_and_up_proj_state_T,
            I_TP=intermediate,
            dtype=compute_dtype,
            activation_function=activation_function,
        )

        if has_down_bias:
            nisa.dma_copy(
                dst=down_bias_broadcasted[:, buffer_idx_now],
                src=down_bias_broadcasted_hbm[current_expert[0, buffer_idx_now, 0]],
                oob_mode=oob_mode.skip,
            )

        compute_block_output_in_place(
            buffer_idx=buffer_idx_now,
            intermediate_states_T=intermediate_states_T,
            down_weights_sbuf=down_weights_sbuf,
            expert_affinities_masked=expert_affinities_masked,
            block=block_output,
            down_bias_broadcasted=(down_bias_broadcasted if has_down_bias else None),
            compute_dtype=compute_dtype,
            H=H,
            intermediate=intermediate,
        )

    # Store accumulated output once (all experts processed).
    store_block_hidden_states(0, output, block_output, token_indices)

    return output


def blockwise_decode_add_residual(
    hidden_states: nt.tensor,
    residual_2d_shard: nt.tensor,
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    tp_degree: int,
    num_experts: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
):
    """Traceable wrapper: decode MoE + reduce-scatter + residual add.

    Unlike the prefill wrapper, this does NOT require CPU scheduling.
    Block mappings are static: 1 block per expert, all tokens in every block.
    """
    import nkipy.distributed.collectives as cc
    from nkipy.core import tensor_apis

    T = hidden_states.shape[0]

    # Transpose affinities from [T, E] to [E, T] for the decode kernel.
    affinities_T = np.transpose(expert_affinities_masked_hbm)

    # Static block mappings (compile-time constants — no CPU scheduling).
    # token_position_to_id: [num_experts, BLOCK_SIZE] — all tokens in every expert block.
    token_position_to_id = tensor_apis.full(
        (1, TILE_SIZE),
        _SKIP_DMA,
        dtype=np.int32,
    )
    token_position_to_id[0, :T] = np.arange(T, dtype=np.int32) + tensor_apis.zeros(
        (T,), dtype=np.int32
    )
    token_position_to_id = np.broadcast_to(
        token_position_to_id,
        (num_experts, TILE_SIZE),
    )
    # block_to_expert: [num_experts] — expert i in block i.
    block_to_expert = np.arange(num_experts, dtype=np.int8) + tensor_apis.zeros(
        (num_experts,),
        dtype=np.int8,
    )

    nki_op = wrap_nki_kernel(
        blockwise_nki_decode,
        [
            hidden_states,
            affinities_T,
            gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm,
            down_proj_weight,
            down_bias_broadcasted_hbm,
            token_position_to_id,
            block_to_expert,
        ],
    )
    output = nki_op(
        hidden_states,
        affinities_T,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )
    # EP all-reduce: sum partial outputs from all EP partitions.
    if int(ep_degree) > 1:
        output = cc.all_reduce(output, replica_groups=list(ep_replica_groups))
    # TP reduce-scatter for seq-parallel.
    if tp_replica_groups:
        hidden_states_shard = cc.reduce_scatter(
            output,
            reduce_scatter_dim=0,
            replica_groups=list(tp_replica_groups),
        )
    else:
        hidden_states_shard = cc.reduce_scatter(
            output,
            reduce_scatter_dim=0,
            replica_groups=[list(range(int(tp_degree)))],
        )
    hidden_states_shard = residual_2d_shard + hidden_states_shard
    return hidden_states_shard


def blockwise_decode_all_reduce_add_residual(
    hidden_states: nt.tensor,
    residual_2d: nt.tensor,
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    tp_degree: int,
    num_experts: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
):
    """Traceable wrapper: decode MoE + TP all-reduce + residual add.

    This variant is intended for decode paths that keep the full token batch on
    every TP rank instead of sequence-parallel token sharding.
    """
    import nkipy.distributed.collectives as cc
    from nkipy.core import tensor_apis

    T = hidden_states.shape[0]
    affinities_T = np.transpose(expert_affinities_masked_hbm)
    token_position_to_id = tensor_apis.full(
        (1, TILE_SIZE),
        _SKIP_DMA,
        dtype=np.int32,
    )
    token_position_to_id[0, :T] = np.arange(T, dtype=np.int32) + tensor_apis.zeros(
        (T,), dtype=np.int32
    )
    token_position_to_id = np.broadcast_to(
        token_position_to_id,
        (num_experts, TILE_SIZE),
    )
    block_to_expert = np.arange(num_experts, dtype=np.int8) + tensor_apis.zeros(
        (num_experts,),
        dtype=np.int8,
    )

    nki_op = wrap_nki_kernel(
        blockwise_nki_decode,
        [
            hidden_states,
            affinities_T,
            gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm,
            down_proj_weight,
            down_bias_broadcasted_hbm,
            token_position_to_id,
            block_to_expert,
        ],
    )
    output = nki_op(
        hidden_states,
        affinities_T,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )
    if int(ep_degree) > 1:
        output = cc.all_reduce(output, replica_groups=list(ep_replica_groups))
    if int(tp_degree) > 1:
        _tp_groups = (
            list(tp_replica_groups)
            if tp_replica_groups
            else [list(range(int(tp_degree)))]
        )
        output = cc.all_reduce(output, replica_groups=_tp_groups, reduce_op=np.add)
    return residual_2d + output


def blockwise_add_residual(
    hidden_states: nt.tensor,
    residual_2d_shard: nt.tensor,
    output: nt.tensor[nt.mutable],
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
):
    """Traceable wrapper: blockwise MoE + reduce-scatter + residual add."""
    import nkipy.distributed.collectives as cc

    nki_op = wrap_nki_kernel(
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
        hidden_states,
        output,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )
    # EP all-reduce: sum partial outputs from all EP partitions.
    if int(ep_degree) > 1:
        output = cc.all_reduce(output, replica_groups=list(ep_replica_groups))
    # TP reduce-scatter for seq-parallel.
    if tp_replica_groups:
        hidden_states_shard = cc.reduce_scatter(
            output,
            reduce_scatter_dim=0,
            replica_groups=list(tp_replica_groups),
        )
    else:
        hidden_states_shard = cc.reduce_scatter(
            output,
            reduce_scatter_dim=0,
            replica_groups=[list(range(int(tp_degree)))],
        )
    hidden_states_shard = residual_2d_shard + hidden_states_shard
    return hidden_states_shard, output


def blockwise_prefill_all_reduce_add_residual(
    hidden_states: nt.tensor,
    residual_2d: nt.tensor,
    output: nt.tensor[nt.mutable],
    expert_affinities_masked_hbm: nt.tensor,
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm: nt.tensor,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm: nt.tensor,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
):
    """Traceable wrapper: prefill blockwise MoE + TP all-reduce + residual add (no-SP)."""
    import nkipy.distributed.collectives as cc

    nki_op = wrap_nki_kernel(
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
        hidden_states,
        output,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )
    if int(ep_degree) > 1:
        output = cc.all_reduce(output, replica_groups=list(ep_replica_groups))
    if int(tp_degree) > 1:
        _tp_groups = (
            list(tp_replica_groups)
            if tp_replica_groups
            else [list(range(int(tp_degree)))]
        )
        output = cc.all_reduce(output, replica_groups=_tp_groups, reduce_op=np.add)
    return residual_2d + output
