# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
kernels - Builtin high performance attention kernels
"""
# ruff: noqa

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.language import par_dim


def ceil_div(a, b):
    return (a + b - 1) // b


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def is_multiple_of(number, divisor):
    """
    Check if number is a multiple of divisor
    Returns True if number is a multiple of divisor, False otherwise
    """
    if divisor == 0:
        return False  # Handle division by zero
    return number % divisor == 0


def load_indices(indices_hbm):
    """
    Load a 2D indices array of shape [num_tiles, num_indices] from HBM to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically partitions num_tiles
    with partition_size set to min(num_tiles, B_P_SIZE=128)

    Output SBUF tensor shape:
      [par_dim(partition_size), ceil_div(num_tiles, partition_size), num_indices]
    """
    B_P_SIZE = 128
    num_tiles, num_indices = indices_hbm.shape
    partition_size = min(B_P_SIZE, num_tiles)
    num_partitions = ceil_div(num_tiles, partition_size)
    indices_sbuf = nl.zeros(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    for i in nl.affine_range(num_partitions):
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[i_p + i * partition_size, i_f],
            mask=(i_p + i * partition_size < num_tiles),
        )
    return indices_sbuf


def load_indices_for_loop_step(indices_hbm, loop_index, step_size, partition_size=None):
    """
    Load a 2D indices array with dim 0 range [loop_index * step_size, (loop_index + 1) * size) from
    HBM with start offset to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically partitions num_tiles
    with partition_size set to min(step_size, B_P_SIZE=128)

    Output SBUF tensor shape: [par_dim(partition_size), ceil_div(size, partition_size), num_indices]
    """

    B_P_SIZE = 128
    _, num_indices = indices_hbm.shape
    if partition_size is None:
        partition_size = min(B_P_SIZE, step_size)
    else:
        assert partition_size <= B_P_SIZE, f"Expect {partition_size=} <= {B_P_SIZE=}"
    assert step_size % partition_size == 0, (
        f"Expect {step_size=} % {partition_size=} == 0"
    )
    num_partitions = step_size // partition_size

    assert loop_index.shape == (1, 1)
    base_addr = nl.ndarray((par_dim(partition_size), 1), dtype=nl.int32)
    broadcast_partition_with_PE(
        src=nl.multiply(loop_index, step_size, dtype=nl.uint32),
        out=base_addr,
        out_in_psum=False,
    )
    indices_sbuf = nl.ndarray(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    for i in nl.affine_range(num_partitions):
        offset = nisa.iota(
            nl.arange(partition_size)[None, :] + i * partition_size, dtype=nl.int32
        )
        offset_transposed = nl.ndarray((partition_size, 1), dtype=nl.int32)
        PF_transpose_with_PE_int4byte(src=offset, out=offset_transposed)
        start_offsets = nisa.tensor_tensor(base_addr, offset_transposed, op=nl.add)
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[start_offsets[i_p, 0], i_f],
        )
    return indices_sbuf


def transform_to_vector_dge_layout(
    indices_in, indices_out, partition_size=None, identity_for_transpose=None
):
    """
    Transpose an tile of shape [tile_size, num_indices] so that num_indices is mapped to partition
    dimension and perform partition with partition_size=min(num_indices, B_P_SIZE=128)

    indices_in:
      [par_dim(tile_size), num_indices]
    indices_out:
      [par_dim(partition_size), ceil_div(num_indices, partition_size), tile_size]
    """
    B_P_SIZE = 128
    tile_size, num_indices = indices_in.shape
    if partition_size is None:
        partition_size = min(num_indices, B_P_SIZE)
    else:
        assert partition_size <= B_P_SIZE, f"Expect {partition_size=} <= {B_P_SIZE=}"
    num_partitions = ceil_div(num_indices, partition_size)
    assert indices_out.shape == (
        partition_size,
        num_partitions,
        tile_size,
    )
    for i in nl.affine_range(num_partitions):
        PF_transpose_with_PE(
            indices_in[:, nl.ds(i * partition_size, partition_size)],
            indices_out[:, i, :],
            identity_for_transpose=identity_for_transpose,
        )


def PF_transpose_with_PE_integer(src, out):
    assert nisa.get_nc_version() == nisa.nc_version.gen2
    # lower as 1/2/4 uint8 matmul
    assert src.dtype == out.dtype
    assert src.dtype == nl.int32 or src.dtype == nl.uint32
    p, f = src.shape
    nbytes = src.itemsize
    if nbytes > 1:
        src_copy = nl.copy(src)
    else:
        src_copy = src
    src_reinterpreted = src_copy.view(nl.uint8)
    out_reinterpreted = out.view(nl.uint8)
    for i in nl.affine_range(nbytes):
        out_psum = nl.ndarray((par_dim(f), p), dtype=nl.int32, buffer=nl.psum)
        i_p = nl.arange(p)[:, None]
        i_f = nl.arange(f)[None, :] * nbytes + i
        out_psum[:, :] = nisa.nc_transpose(
            src_reinterpreted[i_p, i_f],
            engine=nisa.tensor_engine,
        )
        i_p = nl.arange(f)[:, None]
        i_f = nl.arange(p)[None, :]
        out_reinterpreted[i_p, i_f * nbytes + i] = nl.copy(
            out_psum[i_p, i_f], dtype=nl.uint8
        )


def get_move_dtype(src):
    itemsize = src.itemsize
    assert itemsize <= 4, f"{src.dtype=} has itemsize > 4"
    if itemsize == 1:
        return nl.uint8
    elif itemsize == 2:
        return nl.bfloat16
    else:
        return nl.float32


def create_identity_for_transpose(src, size):
    identity_dtype = get_move_dtype(src)
    identity_for_transpose_hbm = nl.shared_constant(
        np.identity(n=size, dtype=np.uint8),
        dtype=identity_dtype,
    )
    identity_for_transpose = nl.ndarray(
        (size, size), buffer=nl.sbuf, dtype=identity_dtype
    )
    nisa.dma_copy(
        dst=identity_for_transpose,
        src=identity_for_transpose_hbm,
        dge_mode=nisa.dge_mode.swdge,
    )
    return identity_for_transpose


def PF_transpose_with_PE(src, out, identity_for_transpose=None, out_in_psum=False):
    B_P_SIZE = nl.tile_size.pmax
    p, f = src.shape
    assert p <= B_P_SIZE and f <= B_P_SIZE
    assert out.shape == (f, p), f"{src.shape=} {out.shape=}"
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    if out_in_psum:
        if is_nc_gen2 and src.dtype == nl.float32:
            # XXX: work around an accuracy issue on Trn1
            # When src and out dtype is float32, using nc_transpose
            # leads to result mismatch.
            # Not sure why nc_matmul does not have this issue on Trn1.
            assert out.dtype == nl.float32
            assert (
                identity_for_transpose is not None
                and identity_for_transpose.dtype == nl.float32
            )
            out[...] = nisa.nc_matmul(
                src,
                identity_for_transpose,
                is_moving_onezero=True,
                is_transpose=True,
            )
        else:
            psum_dtype = nl.float32 if is_nc_gen2 else src.dtype
            assert psum_dtype == out.dtype
            out[...] = nisa.nc_transpose(src, engine=nisa.tensor_engine)
    else:
        if src.dtype in (nl.int32, nl.uint32, nl.uint8):
            assert src.dtype == out.dtype
            if is_nc_gen2:
                PF_transpose_with_PE_integer(src, out)
            else:
                move_dtype = get_move_dtype(src)
                src_reinterpreted = src.view(move_dtype)
                out_reinterpreted = out.view(move_dtype)
                out_psum = nl.ndarray(out.shape, dtype=move_dtype, buffer=nl.psum)
                out_psum[...] = nisa.nc_transpose(
                    src_reinterpreted,
                    engine=nisa.tensor_engine,
                )
                out_reinterpreted[...] = nl.copy(out_psum)
        elif src.dtype == out.dtype == nl.float32:
            out_psum = nl.ndarray(out.shape, dtype=nl.float32, buffer=nl.psum)
            if is_nc_gen2:
                # XXX: work around an accuracy issue on Trn1
                # When src and out dtype is float32, using nc_transpose
                # leads to result mismatch.
                # Not sure why nc_matmul does not have this issue on Trn1.
                assert (
                    identity_for_transpose is not None
                    and identity_for_transpose.dtype == nl.float32
                )
                out_psum[...] = nisa.nc_matmul(
                    src,
                    identity_for_transpose,
                    is_moving_onezero=True,
                    is_transpose=True,
                )
            else:
                out_psum[...] = nisa.nc_transpose(src, engine=nisa.tensor_engine)
            out[...] = nl.copy(out_psum, dtype=out.dtype)
        else:
            assert src.dtype in (nl.bfloat16, nl.float16, nl.float32), src.dtype
            if is_nc_gen2:
                out_psum = nl.ndarray(out.shape, dtype=nl.float32, buffer=nl.psum)
            else:
                out_psum = nl.ndarray(out.shape, dtype=src.dtype, buffer=nl.psum)
            out_psum[...] = nisa.nc_transpose(src, engine=nisa.tensor_engine)
            out[...] = nl.copy(out_psum, dtype=out.dtype)


def broadcast_partition_with_PE(src, out, src_one_zero=False, out_in_psum=False):
    assert src.dtype != nl.int32, (
        f"{src.dtype=} may produce wrong results if input has negative values"
    )
    assert (
        src.dtype not in (nl.uint32, nl.uint8)
        or nisa.get_nc_version() == nisa.nc_version.gen2
    )
    out_shape = out.shape
    assert len(src.shape) == 2 and len(out_shape) == 2
    assert src.shape[0] == 1 and src.shape[1] == out_shape[1]
    move_dtype = get_move_dtype(src)

    src_reinterpreted = src.view(move_dtype)
    ones = nl.ones((1, out_shape[0]), dtype=move_dtype)
    if out_in_psum:
        out_psum = out
    else:
        psum_dtype = nl.int32 if move_dtype == nl.uint8 else nl.float32
        out_psum = nl.ndarray(out_shape, dtype=psum_dtype, buffer=nl.psum)
    out_psum[:, :] = nisa.nc_matmul(
        ones,
        src_reinterpreted,
        is_stationary_onezero=True,
        is_moving_onezero=src_one_zero,
    )
    if out_in_psum:
        assert out.dtype == nl.float32
    elif src.dtype == move_dtype:
        out[...] = nl.copy(out_psum, dtype=out.dtype)
    else:
        if src.dtype == out.dtype:
            out_reinterpreted = out.view(move_dtype)
            out_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
        else:
            out_tmp = nl.ndarray(out.shape, dtype=src.dtype)
            out_tmp_reinterpreted = out_tmp.view(move_dtype)
            out_tmp_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
            out[...] = nl.copy(out_tmp, dtype=out.dtype)


def load_permutation_matrix(permute_matrix_hbm, dtype):
    K, N = permute_matrix_hbm.shape
    # Define max tile sizes according to hardware constraints
    TILE_SIZE_K = min(K, nl.tile_size.pmax)  # 128
    TILE_SIZE_N = min(N, nl.tile_size.gemm_moving_fmax)  # 512

    # Calculate number of tiles needed using ceiling division
    num_k_tiles = ceil_div(K, TILE_SIZE_K)
    num_n_tiles = ceil_div(N, TILE_SIZE_N)

    # Allocate space for all permute_matrix_hbm tiles in SBUF
    permute_matrix_sbuf = nl.ndarray(
        (par_dim(TILE_SIZE_K), num_n_tiles, num_k_tiles, TILE_SIZE_N),
        dtype=dtype,
    )

    for n_idx in nl.affine_range(num_n_tiles):
        n_offset = n_idx * TILE_SIZE_N
        n_oob = n_offset + TILE_SIZE_N > N

        for k_idx in nl.affine_range(num_k_tiles):
            k_offset = k_idx * TILE_SIZE_K
            k_oob = k_offset + TILE_SIZE_K > K

            # Create indices for masking
            i_k_local = nl.arange(TILE_SIZE_K)[:, None]
            i_k_global = k_offset + i_k_local
            i_n_local = nl.arange(TILE_SIZE_N)[None, :]
            i_n_global = n_offset + i_n_local

            # Load permute_matrix_hbm tile with masking for boundaries
            load_buffer = nl.ndarray(
                (par_dim(TILE_SIZE_K), TILE_SIZE_N),
                dtype=permute_matrix_hbm.dtype,
            )
            if n_oob or k_oob:
                # Last tile may be smaller than TILE_SIZE_K and TILE_SIZE_N
                load_buffer[...] = 0
            load_buffer[i_k_local, i_n_local] = nl.load(
                permute_matrix_hbm[i_k_global, i_n_global],
                mask=((i_k_global < K) & (i_n_global < N)),
            )
            permute_matrix_sbuf[:, n_idx, k_idx] = nl.copy(load_buffer, dtype=dtype)

    return permute_matrix_sbuf


def select_dynamic_column(src_sbuf, permute_matrix, dst_sbuf):
    """NKI kernel to compute matrix multiplication dst_sbuf = src_sbuf @ permute_matrix

    Handles cases where dimensions are not multiples of tile sizes using masking.

    Args:
        src_sbuf: Input tensor of shape [M, K] where M ≤ 128
        permute_matrix: Input tensor of shape [K, N]
        dst_sbuf: Output tensor of shape [M, N], the result of src_sbuf @ permute_matrix
    """
    TILE_SIZE_K, num_n_tiles, num_k_tiles, TILE_SIZE_N = permute_matrix.shape
    M, K = src_sbuf.shape
    M_, N = dst_sbuf.shape
    assert M == M_, f"{src_sbuf.shape=} and {dst_sbuf.shape=} are not compatible"
    assert M <= nl.tile_size.pmax, (
        f"M dimension {M=} must not exceed {nl.tile_size.pmax=}"
    )
    assert num_k_tiles * TILE_SIZE_K >= K and num_n_tiles * TILE_SIZE_N >= N

    # 1. Transpose src tiles
    # ------------------------------------------
    # Allocate space for all transposed A tiles in SBUF
    src_tiles_transposed = nl.ndarray(
        (par_dim(TILE_SIZE_K), num_k_tiles, M),
        dtype=src_sbuf.dtype,
    )
    identity_for_transpose = create_identity_for_transpose(src_sbuf, M)

    for k_idx in nl.affine_range(K // TILE_SIZE_K):
        src_transposed_psum = nl.ndarray(
            (TILE_SIZE_K, M), dtype=nl.float32, buffer=nl.psum
        )
        src_transposed_psum[:, :] = nisa.nc_matmul(
            src_sbuf[:, nl.ds(k_idx * TILE_SIZE_K, TILE_SIZE_K)],
            identity_for_transpose,
            is_moving_onezero=True,
            is_transpose=True,
        )
        src_tiles_transposed[:, k_idx, :] = nl.copy(
            src_transposed_psum,
            dtype=src_sbuf.dtype,
        )

    if K % TILE_SIZE_K != 0:
        TILE_SIZE_K_LAST = K % TILE_SIZE_K
        src_transposed_psum = nl.ndarray(
            (TILE_SIZE_K_LAST, M), dtype=nl.float32, buffer=nl.psum
        )
        src_transposed_psum[:, :] = nisa.nc_matmul(
            src_sbuf[:, nl.ds((num_k_tiles - 1) * TILE_SIZE_K, TILE_SIZE_K_LAST)],
            identity_for_transpose,
            is_moving_onezero=True,
            is_transpose=True,
        )
        src_tiles_transposed[:, num_k_tiles - 1, :] = 0
        src_tiles_transposed[nl.ds(0, TILE_SIZE_K_LAST), num_k_tiles - 1, :] = nl.copy(
            src_transposed_psum,
            dtype=src_sbuf.dtype,
        )

    # 2. Compute matrix multiplication using preloaded data
    # ------------------------------------------
    for n_idx in nl.affine_range(num_n_tiles):
        n_offset = n_idx * TILE_SIZE_N

        # Create indices for output masking
        i_m_out = nl.arange(M)[:, None]
        i_n_local = nl.arange(TILE_SIZE_N)[None, :]
        i_n_global = n_offset + i_n_local

        # Create PSUM buffer for accumulating results for this N tile
        result_psum = nl.zeros((M, TILE_SIZE_N), dtype=nl.float32, buffer=nl.psum)

        for k_idx in nl.affine_range(num_k_tiles):
            # No need to load anything - use preloaded tiles
            src_tile = src_tiles_transposed[:, k_idx, :]
            permute_matrix_tile = permute_matrix[:, n_idx, k_idx, :]

            # For the last tile in K dimension, we need masking in the matmul
            # Full tile computation
            result_psum += nisa.nc_matmul(
                src_tile,
                permute_matrix_tile,
                is_moving_onezero=True,
            )

        # Copy result from PSUM to SBUF and cast to output data type
        dst_sbuf[i_m_out, i_n_global] = nl.copy(
            result_psum[i_m_out, i_n_local],
            dtype=dst_sbuf.dtype,
            mask=(i_n_global < N),
        )
