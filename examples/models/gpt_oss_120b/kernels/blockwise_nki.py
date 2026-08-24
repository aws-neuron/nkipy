"""
Copyright (c) 2024, Amazon.com. All Rights Reserved

kernels - Builtin high performance blockwise matmul kernels

Migrated from the legacy ``neuronxcc.nki`` API to standalone ``nki`` (beta-3).
The data-dependent DMA layer (token/expert gathers) is re-authored around the
newest idioms:
  * ``tensor.ap(pattern=[[stride, size], ...], offset=, scalar_offset=)`` for
    dynamic *expert*-indexed weight/bias loads (one runtime scalar base index).
  * ``tensor.vector_select(0, idx)`` for per-partition *token* gather/scatter
    (the old ``hidden_states[token_indices[...]]`` fancy indexing).
  * ``nisa`` ops are dst-first and return nothing; ``nc_matmul`` accumulates by
    targeting the same PSUM bank with ``accumulate=(i > 0)``; ``nc_transpose``
    writes to a PSUM tile of the same dtype using the Tensor engine.
``nl.mgrid`` / ``nl.arange`` / ``nl.par_dim`` are gone: partition dim is the
leading plain int of ``nl.ndarray`` and tiles are indexed with plain slices /
``nl.ds``.
"""

import math
from enum import Enum

import nki
import nki.isa as nisa
import nki.language as nl
import nkipy.core.typing as nt
from kernels.blockwise_index import ControlType
from nki.isa.constants import oob_mode
from nkipy.core.typing import mutable_tensor

TILE_SIZE = 128  # partition dim (nl.tile_size.pmax)

_SKIP_DMA = ControlType.SKIP_DMA.value  # -1


class ActFnType(Enum):
    """Activation function selector (vendored; matches nkilib common_types)."""

    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3


def stream_shuffle_broadcast(src, dst):
    """Broadcast src's first partition across all partitions of dst.

    Vendored from the standalone-nki 20b example (pre_prod helper is gone).
    src is [1, F], dst is [P, F] with P a multiple of 32.
    """
    dst_npar = dst.shape[0]
    free_dim = dst.shape[1]
    shuffle_mask = [0] * 32
    assert dst_npar % 32 == 0
    for i in range(dst_npar // 32):
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : (i + 1) * 32, 0:free_dim],
            shuffle_mask=shuffle_mask,
        )


def output_initialization(output, dims=None):
    """Zero-initialize an HBM output buffer for accumulation.

    Vendored (pre_prod helper is gone); mirrors the old neuronxcc behavior of
    zeroing an [T, H] tensor tile by tile.
    """
    T, H = output.shape
    n_tiles = math.ceil(T / TILE_SIZE)
    for tile_idx in range(n_tiles):
        t_start = tile_idx * TILE_SIZE
        t_size = min(TILE_SIZE, T - t_start)
        zeros = nl.zeros((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=output[t_start : t_start + t_size, 0:H],
            src=zeros[0:t_size, 0:H],
        )


def load_token_indices(buffer_idx, token_indices, token_position_to_id, block_idx):
    """Load the [TILE_SIZE] token id map for a (static) block onto the partition dim."""
    n_blocks, per_block = token_position_to_id.shape
    nisa.dma_copy(
        dst=token_indices[0:TILE_SIZE, buffer_idx : buffer_idx + 1],
        src=token_position_to_id.reshape((n_blocks * per_block, 1)).ap(
            pattern=[[1, TILE_SIZE], [1, 1]], offset=block_idx * per_block
        ),
    )


def compute_intermediate_states_T(
    gate_and_up_proj_state_T,
    I_TP,
    dtype,
    activation_function: ActFnType,
    mask=None,
    gup_scale=None,
):
    # gate_and_up_proj_state_T is a nested list [gate|up][i_i] of PSUM [I(par), B] tiles.
    i_n_tile = math.ceil(I_TP / TILE_SIZE)
    # [I, B]
    intermediate_states_T = nl.ndarray(
        (TILE_SIZE, i_n_tile, TILE_SIZE), dtype=dtype, buffer=nl.sbuf
    )
    # Note: Avoid compiler created bias create unnecessary aliasing
    bias = nl.zeros((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
    assert (
        activation_function == ActFnType.Swish
    ), f"Activation function {activation_function} not implemented"
    for i_i in range(i_n_tile):  # reads python PSUM-tile list
        num_i = min(TILE_SIZE, I_TP - i_i * TILE_SIZE)
        # swish/silu-gated: gelu_apprx_sigmoid(gate) * up
        nisa.activation(
            dst=intermediate_states_T[0:num_i, i_i],
            op=nl.gelu_apprx_sigmoid,
            data=gate_and_up_proj_state_T[0][i_i][0:num_i, 0:TILE_SIZE],
            bias=bias[0:num_i],
        )
        nisa.tensor_tensor(
            dst=intermediate_states_T[0:num_i, i_i],
            data1=intermediate_states_T[0:num_i, i_i],
            data2=gate_and_up_proj_state_T[1][i_i][0:num_i, 0:TILE_SIZE],
            op=nl.multiply,
        )
    return intermediate_states_T


def load_gate_up_proj_weights(buffer_idx, gate_up_proj_weight, gup_weights_sbuf, block_expert):
    """Gather the current expert's gate/up weights [H, 2I] into SBUF, tiled on H.

    ``block_expert`` is the (may-skip) expert index tile; a negative value drives
    the DMA out of bounds so ``oob_mode.skip`` keeps the previously-loaded weights.
    """
    E, H, _, I = gate_up_proj_weight.shape
    two_i = 2 * I
    h_n_tile = math.ceil(H / TILE_SIZE)
    n_full = H // TILE_SIZE  # full 128-row h-tiles
    rem = H - n_full * TILE_SIZE
    weight_2d = gate_up_proj_weight.reshape((E, H, two_i))
    expert = block_expert[0:1, buffer_idx, 0:1]
    # One bulk DMA for all full h-tiles (dst tile dim is the middle .ap axis) so the
    # whole weight lands in a single transfer that overlaps compute -- a per-h_i loop
    # instead floods the sync engine with hundreds of DMA_DIRECT2D/TENSOR_LOAD/ALU_OP
    # descriptor ops that serialize against the matmuls (halves tensor-engine MFU).
    if n_full > 0:
        nisa.dma_copy(
            dst=gup_weights_sbuf[0:TILE_SIZE, buffer_idx, 0:n_full, 0:two_i],
            # [expert, t*128 + p, f]: p ramps 128 rows (stride 2I), t ramps h-tiles
            # (stride 128*2I), f ramps 2I (stride 1); expert scaled by H*2I.
            src=weight_2d.ap(
                pattern=[[two_i, TILE_SIZE], [TILE_SIZE * two_i, n_full], [1, two_i]],
                offset=0,
                scalar_offset=expert,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if rem > 0:
        nisa.dma_copy(
            dst=gup_weights_sbuf[0:rem, buffer_idx, n_full, 0:two_i],
            src=weight_2d.ap(
                pattern=[[two_i, rem], [1, two_i]],
                offset=n_full * TILE_SIZE * two_i,
                scalar_offset=expert,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
        )


def load_down_proj_weights(buffer_idx, down_proj_weight, block_expert, down_weights_sbuf):
    """Gather the current expert's down weights [I, H] into SBUF, tiled on I."""
    _, I, H = down_proj_weight.shape
    i_n_tile = math.ceil(I / TILE_SIZE)
    n_full = I // TILE_SIZE
    rem = I - n_full * TILE_SIZE
    expert = block_expert[0:1, buffer_idx, 0:1]
    # Single bulk DMA for the full i-tiles (see load_gate_up_proj_weights rationale).
    if n_full > 0:
        nisa.dma_copy(
            dst=down_weights_sbuf[0:TILE_SIZE, buffer_idx, 0:n_full, 0:H],
            # [expert, t*128 + p, h]: p ramps 128 (stride H), t ramps i-tiles
            # (stride 128*H), h ramps H (stride 1); expert scaled by I*H.
            src=down_proj_weight.ap(
                pattern=[[H, TILE_SIZE], [TILE_SIZE * H, n_full], [1, H]],
                offset=0,
                scalar_offset=expert,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
        )
    if rem > 0:
        nisa.dma_copy(
            dst=down_weights_sbuf[0:rem, buffer_idx, n_full, 0:H],
            src=down_proj_weight.ap(
                pattern=[[H, rem], [1, H]],
                offset=n_full * TILE_SIZE * H,
                scalar_offset=expert,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
        )


def load_gate_up_bias_T(buffer_idx, gate_up_bias, gate_up_bias_hbm, expert, I):
    """Gather the current expert's gate/up bias [I, 2] into SBUF, tiled on I."""
    i_n_tile = math.ceil(I / TILE_SIZE)
    expert_11 = expert[0:1, buffer_idx, 0:1]
    for i_i in range(i_n_tile):
        i_start = i_i * TILE_SIZE
        num_i = min(TILE_SIZE, I - i_start)
        nisa.dma_copy(
            dst=gate_up_bias[0:num_i, buffer_idx, i_i, 0:2],
            # gate_up_bias_hbm: [E, I, 2] -> [expert, i_start + p, g]
            src=gate_up_bias_hbm.ap(
                pattern=[[2, num_i], [1, 2]],
                offset=i_start * 2,
                scalar_offset=expert_11,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,  # HW descriptor gen; see load_gate_up_proj_weights
        )


def load_block_hidden_states(buffer_idx, block_hidden_states, hidden_states, token_indices, compute_dtype):
    """Per-token gather of hidden rows: dst[p, :] = hidden_states[token_indices[p], :]."""
    H = hidden_states.shape[-1]
    token_idx = token_indices[0:TILE_SIZE, buffer_idx : buffer_idx + 1]
    nisa.dma_copy(
        dst=block_hidden_states[0:TILE_SIZE, buffer_idx, 0:H],
        src=hidden_states.vector_select(0, token_idx),
        oob_mode=oob_mode.skip,
    )


def store_block_hidden_states(buffer_idx, output, block_new, token_indices):
    """Per-token scatter: output[token_indices[p], :] = block_new[p, :]."""
    H = output.shape[-1]
    token_idx = token_indices[0:TILE_SIZE, buffer_idx : buffer_idx + 1]
    nisa.dma_copy(
        dst=output.vector_select(0, token_idx),
        src=block_new[0:TILE_SIZE, 0:H],
        oob_mode=oob_mode.skip,
    )


def transpose_block_hidden_states(buffer_idx, block_hidden_states_T, block_hidden_states, H, compute_dtype):
    """Transpose block hidden [B(tokens), H] -> [H(tiled), B] so H sits on partition."""
    h_n_tiles = math.ceil(H / TILE_SIZE)
    for h_i in nl.affine_range(h_n_tiles):
        h_start = h_i * TILE_SIZE
        h_size = min(TILE_SIZE, H - h_start)
        # Tensor-engine transpose lands in PSUM with dtype == input dtype.
        t_psum = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=compute_dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=t_psum[0:h_size, 0:TILE_SIZE],
            data=block_hidden_states[0:TILE_SIZE, buffer_idx, h_start : h_start + h_size],
            engine=nisa.engine.tensor,
        )
        nisa.tensor_copy(
            dst=block_hidden_states_T[0:h_size, buffer_idx, h_i, 0:TILE_SIZE],
            src=t_psum[0:h_size, 0:TILE_SIZE],
        )


def load_expert_affinities(
    buffer_idx,
    expert_affinities_masked,
    expert_affinities_masked_hbm,
    token_indices,
    expert,
    compute_dtype,
):
    """Gather per-token expert affinity: eaff[p] = eaff_hbm.flat[token_indices[p]*E + expert]."""
    T, E = expert_affinities_masked_hbm.shape
    # Broadcast the (real) expert id across all partitions.
    v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    stream_shuffle_broadcast(expert, v_expert)
    # addr = token_indices * E + expert; skipped tokens (-1) stay negative -> OOB skip.
    addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=addr,
        data=token_indices[0:TILE_SIZE, buffer_idx : buffer_idx + 1],
        op0=nl.multiply,
        operand0=E,
    )
    addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)
    nisa.dma_copy(
        dst=expert_affinities_masked[0:TILE_SIZE, buffer_idx : buffer_idx + 1],
        # reshape because 2d indirect indexing is unsupported
        src=expert_affinities_masked_hbm.reshape((T * E, 1)).vector_select(0, addr_fin),
        oob_mode=oob_mode.skip,
    )


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
    expert_11 = expert[0:1, 0:1]
    nisa.dma_copy(
        dst=expert_affinities_masked[0:T, buffer_idx : buffer_idx + 1],
        # eaff_T_hbm: [E, T] -> row `expert`, T tokens onto the partition dim.
        src=expert_affinities_masked_transposed_hbm.ap(
            pattern=[[1, T], [1, 1]],
            offset=0,
            scalar_offset=expert_11,
        ),
        oob_mode=oob_mode.skip,
    )


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
    """Compute gate and up projections -> PSUM [I(par), 2, i_n_tile, B].

    One contiguous partition-leading PSUM buffer (not a python list of separate
    tiles). This lets the compiler pipeline the independent (gate|up, i_i) matmuls
    across PSUM banks — the python-list-of-tiles form serialized them and halved
    tensor-engine MFU (45%->21%). Only the h reduction (accumulate=) is sequential.
    """
    i_n_tile = math.ceil(I / TILE_SIZE)
    h_n_tiles = math.ceil(H / TILE_SIZE)
    # One PSUM tile per (gate|up, i_i), held in a python list. A single contiguous
    # (128, 2, i_n_tile, 128) PSUM ndarray caused NRT_EXEC_HW_ERR_DMA_ABORT on the
    # tokengen variant (matmul dst slice straddling PSUM-bank boundaries); separate
    # tiles keep each matmul output within one bank.
    gate_and_up_proj_state_T = []
    for _gate_or_up in range(2):
        _tiles = []
        for _ in range(i_n_tile):
            _tiles.append(nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=nl.float32, buffer=nl.psum))
        gate_and_up_proj_state_T.append(_tiles)
    for gate_or_up in range(2):
        for i_i in range(i_n_tile):
            i_start = i_i * TILE_SIZE
            num_i = min(TILE_SIZE, I - i_start)
            f_start = gate_or_up * I + i_start
            psum_tile = gate_and_up_proj_state_T[gate_or_up][i_i]
            # h reduction accumulates into one PSUM bank -> plain sequential range.
            for h_i in range(h_n_tiles):
                h_start = h_i * TILE_SIZE
                h_size = min(TILE_SIZE, H - h_start)
                # stationary [H(contract), I], moving [H(contract), B] -> [I, B]
                nisa.nc_matmul(
                    dst=psum_tile[0:num_i, 0:TILE_SIZE],
                    stationary=gup_weights_sbuf[
                        0:h_size, weight_buffer_idx, h_i, f_start : f_start + num_i
                    ],
                    moving=block_hidden_states_T[0:h_size, hidden_buffer_idx, h_i, 0:TILE_SIZE],
                    accumulate=(h_i > 0),
                )
    if gate_up_bias_plus1 is not None:
        for i_i in range(i_n_tile):
            i_start = i_i * TILE_SIZE
            num_i = min(TILE_SIZE, I - i_start)
            gate_tile = gate_and_up_proj_state_T[0][i_i]
            up_tile = gate_and_up_proj_state_T[1][i_i]
            # gate: + bias, then min(_, 7.0)
            nisa.tensor_scalar(
                dst=gate_tile[0:num_i, 0:TILE_SIZE],
                data=gate_tile[0:num_i, 0:TILE_SIZE],
                op0=nl.add,
                operand0=gate_up_bias_plus1[0:num_i, weight_buffer_idx, i_i, 0:1],
                op1=nl.minimum,
                operand1=7.0,
            )
            # up: + bias, then clip to [-6.0, 8.0]
            nisa.tensor_scalar(
                dst=up_tile[0:num_i, 0:TILE_SIZE],
                data=up_tile[0:num_i, 0:TILE_SIZE],
                op0=nl.add,
                operand0=gate_up_bias_plus1[0:num_i, weight_buffer_idx, i_i, 1:2],
            )
            nisa.tensor_scalar(
                dst=up_tile[0:num_i, 0:TILE_SIZE],
                data=up_tile[0:num_i, 0:TILE_SIZE],
                op0=nl.minimum,
                operand0=8.0,
                op1=nl.maximum,
                operand1=-6.0,
            )
    return gate_and_up_proj_state_T


def _down_projection(
    buffer_idx,
    intermediate_states_T,
    down_weights_sbuf,
    expert_affinities_masked,
    block_old,
    block_dst,
    down_bias_broadcasted,
    compute_dtype,
    H,
    I,
):
    """Shared down-projection + affinity scale + residual add.

    ``block_dst[p, h] = down(intermediate)[p, h] * affinity[p] + block_old[p, h]``.
    """
    i_n_tile = math.ceil(I / TILE_SIZE)
    h_tile_size = nl.tile_size.gemm_moving_fmax
    h_n_tile = math.ceil(H / h_tile_size)

    for h_i in nl.affine_range(h_n_tile):
        h_start = h_i * h_tile_size
        h_size = min(h_tile_size, H - h_start)
        down_proj_psum = nl.ndarray((TILE_SIZE, h_tile_size), dtype=nl.float32, buffer=nl.psum)
        # i reduction accumulates into one PSUM bank -> plain sequential range.
        for i_i in range(i_n_tile):
            i_start = i_i * TILE_SIZE
            num_i = min(TILE_SIZE, I - i_start)
            # stationary [I(contract), B], moving [I(contract), H] -> [B, H]
            nisa.nc_matmul(
                dst=down_proj_psum[0:TILE_SIZE, 0:h_size],
                stationary=intermediate_states_T[0:num_i, i_i, 0:TILE_SIZE],
                moving=down_weights_sbuf[0:num_i, buffer_idx, i_i, h_start : h_start + h_size],
                accumulate=(i_i > 0),
            )
        if down_bias_broadcasted is not None:
            nisa.tensor_tensor(
                dst=down_proj_psum[0:TILE_SIZE, 0:h_size],
                data1=down_proj_psum[0:TILE_SIZE, 0:h_size],
                data2=down_bias_broadcasted[0:TILE_SIZE, buffer_idx, h_start : h_start + h_size],
                op=nl.add,
            )
        nisa.scalar_tensor_tensor(
            dst=block_dst[0:TILE_SIZE, h_start : h_start + h_size],
            data=down_proj_psum[0:TILE_SIZE, 0:h_size],
            op0=nl.multiply,
            operand0=expert_affinities_masked[0:TILE_SIZE, buffer_idx : buffer_idx + 1],
            op1=nl.add,
            operand1=block_old[0:TILE_SIZE, h_start : h_start + h_size],
        )


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
    block_new = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
    _down_projection(
        buffer_idx=buffer_idx,
        intermediate_states_T=intermediate_states_T,
        down_weights_sbuf=down_weights_sbuf,
        expert_affinities_masked=expert_affinities_masked,
        block_old=block_old[0:TILE_SIZE, buffer_idx],
        block_dst=block_new,
        down_bias_broadcasted=down_bias_broadcasted,
        compute_dtype=compute_dtype,
        H=H,
        I=I,
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
    _down_projection(
        buffer_idx=buffer_idx,
        intermediate_states_T=intermediate_states_T,
        down_weights_sbuf=down_weights_sbuf,
        expert_affinities_masked=expert_affinities_masked,
        block_old=block,
        block_dst=block,
        down_bias_broadcasted=down_bias_broadcasted,
        compute_dtype=compute_dtype,
        H=H,
        I=I,
    )


def _load_block_expert(dst, buffer_idx, block_to_expert, block_idx):
    """Load block_to_expert[block_idx] (a static int index) into dst[0,buf,0] as int32."""
    n_blocks = block_to_expert.shape[0]
    raw = nl.ndarray((1, 1), dtype=block_to_expert.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=raw,
        src=block_to_expert.reshape((n_blocks, 1)).ap(pattern=[[1, 1], [1, 1]], offset=block_idx),
    )
    nisa.tensor_copy(dst=dst[0:1, buffer_idx, 0:1], src=raw)  # cast to int32


@nki.jit
def output_init_nki(output: nt.mutable_tensor):
    output_initialization(output)
    return output


def output_init(output: mutable_tensor):
    from nkipy.core.nki_op import wrap_nki_kernel

    nki_op = wrap_nki_kernel(
        output_init_nki,
        [output],
        is_nki_beta_3_version=True,
    )
    output = nki_op(output)
    return output


@nki.jit
def blockwise_nki_static(
    hidden_states: nt.tensor,
    output: nt.mutable_tensor,
    expert_affinities_masked_hbm: nt.tensor,  # TODO: only need (T, TOP_K)
    gate_up_proj_weight: nt.tensor,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight: nt.tensor,
    down_bias_broadcasted_hbm,
    token_position_to_id: nt.tensor,
    block_to_expert: nt.tensor,
    num_static_blocks: int,
    activation_function: ActFnType = ActFnType.Swish,
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=1,
):
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
        (TILE_SIZE, BUFFER_DEGREE),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )
    block_hidden_states = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    block_output = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tiles = math.ceil(H / TILE_SIZE)
    # [H, B]
    block_hidden_states_T = nl.ndarray(
        (
            TILE_SIZE,
            BUFFER_DEGREE,
            h_n_tiles,
            TILE_SIZE,
        ),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE), dtype=compute_dtype, buffer=nl.sbuf
    )

    # TODO: overlap with compute
    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, h_n_tile, 2 * I),
        dtype=gate_up_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(I / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, i_n_tile, H),
        dtype=down_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    current_expert_real = nl.zeros((1, BUFFER_DEGREE, 1), dtype=nl.int32, buffer=nl.sbuf)
    current_expert_may_skip = nl.zeros((1, BUFFER_DEGREE, 1), dtype=nl.int32, buffer=nl.sbuf)

    if gate_up_bias_plus1_T_hbm is not None:
        assert gate_up_bias_plus1_T_hbm.shape == (E, I, 2)
        gate_up_bias_plus1_T = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
            dtype=nl.float32,  # later tensor_scalar operand dtype must be fp32
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

    # predicate scratch for expert skip handling
    pred = nl.ndarray((1, 1), dtype=nl.uint8, buffer=nl.sbuf)

    for block_idx in range(num_static_blocks):
        # A
        buffer_idx_prev = (block_idx - 1) % BUFFER_DEGREE
        buffer_idx_now = block_idx % BUFFER_DEGREE

        load_token_indices(buffer_idx_now, token_indices, token_position_to_id, block_idx)

        # B
        _load_block_expert(current_expert_may_skip, buffer_idx_now, block_to_expert, block_idx)
        # real = may_skip where block is not a skip-continuation
        nisa.tensor_scalar(
            dst=pred,
            data=current_expert_may_skip[0:1, buffer_idx_now, 0:1],
            op0=nl.not_equal,
            operand0=_SKIP_DMA,
        )
        nisa.tensor_copy_predicated(
            dst=current_expert_real[0:1, buffer_idx_now, 0:1],
            src=current_expert_may_skip[0:1, buffer_idx_now, 0:1],
            predicate=pred,
        )
        # Copy from previous real if skipped
        nisa.tensor_scalar(
            dst=pred,
            data=current_expert_may_skip[0:1, buffer_idx_now, 0:1],
            op0=nl.equal,
            operand0=_SKIP_DMA,
        )
        nisa.tensor_copy_predicated(
            dst=current_expert_real[0:1, buffer_idx_now, 0:1],
            src=current_expert_real[0:1, buffer_idx_prev, 0:1],
            predicate=pred,
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
        #   This N buffering as-is won't work with DMA skipping. Because there are multiple copies of weights.
        #   Additional logic is required to copy from previous block if skipped
        load_gate_up_proj_weights(
            buffer_idx=buffer_idx_now,
            gate_up_proj_weight=gate_up_proj_weight,
            gup_weights_sbuf=gup_weights_sbuf,
            block_expert=current_expert_may_skip,
        )

        # E
        transpose_block_hidden_states(
            buffer_idx_now, block_hidden_states_T, block_hidden_states, H, compute_dtype
        )

        # F
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
                dst=down_bias_broadcasted[0:TILE_SIZE, buffer_idx_now, 0:H],
                src=down_bias_broadcasted_hbm.ap(
                    pattern=[[H, TILE_SIZE], [1, H]],
                    offset=0,
                    scalar_offset=current_expert_may_skip[0:1, buffer_idx_now, 0:1],
                ),
                oob_mode=oob_mode.skip,
                dge_mode=nisa.dge_mode.hwdge,
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


@nki.jit
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
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=3,
):
    """
    This kernel assumes there is only one tile of hidden state and it is replicated across multiple experts
    All tokens are computed against all experts, then they are masked off (introduce redundant compute)
    """
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)

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
        (TILE_SIZE, BUFFER_DEGREE),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )

    block_hidden_states = nl.ndarray(
        (TILE_SIZE, 1, H), dtype=compute_dtype, buffer=nl.sbuf
    )

    block_output = nl.zeros((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)

    h_n_tiles = math.ceil(H / TILE_SIZE)
    # [H, B]
    block_hidden_states_T = nl.ndarray(
        (
            TILE_SIZE,
            1,
            h_n_tiles,
            TILE_SIZE,
        ),
        dtype=compute_dtype,
        buffer=nl.sbuf,
    )

    expert_affinities_masked = nl.zeros(
        (TILE_SIZE, BUFFER_DEGREE), dtype=compute_dtype, buffer=nl.sbuf
    )

    h_n_tile = math.ceil(H / TILE_SIZE)
    gup_weights_sbuf = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, h_n_tile, 2 * I),
        dtype=gate_up_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    i_n_tile = math.ceil(I / TILE_SIZE)
    down_weights_sbuf = nl.ndarray(
        (TILE_SIZE, BUFFER_DEGREE, i_n_tile, H),
        dtype=down_proj_weight.dtype,  # keep original dtype
        buffer=nl.sbuf,
    )
    current_expert = nl.zeros((1, BUFFER_DEGREE, 1), dtype=nl.int32, buffer=nl.sbuf)

    if gate_up_bias_plus1_T_hbm is not None:
        assert gate_up_bias_plus1_T_hbm.shape == (E, I, 2)
        gate_up_bias_plus1_T = nl.ndarray(
            (TILE_SIZE, BUFFER_DEGREE, i_n_tile, 2),
            dtype=nl.float32,  # later tensor_scalar operand dtype must be fp32
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
    transpose_block_hidden_states(0, block_hidden_states_T, block_hidden_states, H, compute_dtype)

    for block_idx in range(n_blocks):
        # A
        buffer_idx_now = block_idx % BUFFER_DEGREE

        load_token_indices(buffer_idx_now, token_indices, token_position_to_id, block_idx)

        # B
        _load_block_expert(current_expert, buffer_idx_now, block_to_expert, block_idx)

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
            expert_affinities_masked_transposed_hbm=expert_affinities_masked_transposed_hbm,
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
                dst=down_bias_broadcasted[0:TILE_SIZE, buffer_idx_now, 0:H],
                src=down_bias_broadcasted_hbm.ap(
                    pattern=[[H, TILE_SIZE], [1, H]],
                    offset=0,
                    scalar_offset=current_expert[0:1, buffer_idx_now, 0:1],
                ),
                oob_mode=oob_mode.skip,
                dge_mode=nisa.dge_mode.hwdge,
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
    output: nt.mutable_tensor,
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
    Fused kernel that performs blockwise MoE computation followed by reduce_scatter and residual addition.
    """
    import parallel_state
    from collective import reduce_scatter
    from kernels.blockwise_np import blockwise_np
    from nkipy.core.nki_op import wrap_nki_kernel

    if is_nkipy:
        # Use wrap_nki_kernel to call the NKI kernel from within NKIPy
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
            ],
            is_nki_beta_3_version=True,
            kernel_kwargs={"num_static_blocks": num_static_blocks},
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
    return hidden_states_shard, output
