# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# ruff: noqa
import math

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import torch
from neuronxcc.nki.isa.constants import oob_mode
from torch_to_nkipy.utils.nki import NKIOpRegistry


def cdiv(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return cdiv(a, b) * b


def _cumsum(
    x, y=None, axis=None, p_size=None, f_size=None, acc_dtype=None, inp_sbuf=False
):
    """
    Compute cumulative sum along the last dim: (axis = -1)

    Equivalent to np.cumsum(x, axis=axis, dtype=acc_dtype)

    Args:
        x (nt.tensor): Input tensor to compute cumulative sum over
        y (nt.tensor, optional): Output tensor. If None and inp_sbuf=True, creates SBUF output.
                                 Must have same shape as x when provided.
        axis (int, optional): Axis along which to compute cumsum. Only supports last axis (-1).
                             Defaults to -1.
        p_size (int, optional): Partition dimension tile size. Defaults to nl.tile_size.pmax.
        f_size (int, optional): Free dimension tile size. Defaults to 2048.
        acc_dtype (dtype, optional): Accumulation data type. Defaults to x.dtype.
        inp_sbuf (bool): If True, input x is in SBUF and output will be in SBUF.
                        If False, input is in HBM and output goes to y in HBM.

    Returns:
        nt.tensor: Cumulative sum result. Only returned when inp_sbuf=True, otherwise
                  stores result in provided y tensor.

    Raises:
        AssertionError: If axis is not the last dimension or if tensor shapes don't match.

    Notes:
        - Only supports cumulative sum along the last dimension (axis=-1)
        - Uses tiled processing with configurable tile sizes for memory efficiency
        - Maintains cumulative state across tiles using sequential processing
        - Input tensor is internally reshaped to 2D for processing

    Example:
        # HBM to HBM
        result = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.hbm)
        cumsum(x, y=result, axis=-1)

        # SBUF processing
        result = cumsum(x_sbuf, inp_sbuf=True)
    """

    assert isinstance(axis, int) or axis is None
    if axis is None:
        axis = -1

    rank = x.ndim

    axis = axis % rank
    assert axis == rank - 1, "Only support cusum over last dim"

    x_shape = x.shape
    shape_2d = (math.prod(x_shape[:-1]), x_shape[-1])

    if inp_sbuf:
        assert x.shape == shape_2d, (
            f"If input is in sbuf, it must have a shape that is 2D, but is {x.shape}"
        )

    # Reshape if needed for HBM
    else:
        x = x.reshape(shape_2d)
        assert y.shape == x_shape, "Expect x and y has the same shape!"

    pmax = nl.tile_size.pmax if p_size is None else p_size
    f_tile_size = 2048 if f_size is None else f_size

    pi, fi = nl.mgrid[0:pmax, 0:f_tile_size]

    acc_dtype = acc_dtype or x.dtype

    ones = nl.ones((pmax, f_tile_size), dtype=acc_dtype)
    # init = nl.zeros((pmax, 1), dtype=x.dtype)

    if inp_sbuf:
        output = nl.ndarray(shape_2d, dtype=acc_dtype, buffer=nl.sbuf)

    for i in nl.affine_range(cdiv(shape_2d[0], pmax)):
        n_f_tiles = cdiv(shape_2d[1], f_tile_size)
        init = nl.zeros((pmax, 1), dtype=acc_dtype)

        for j in nl.sequential_range(n_f_tiles):
            mask = (i * pmax + pi < shape_2d[0]) & (j * f_tile_size + fi < shape_2d[1])
            if inp_sbuf:
                data = x
            else:
                data = nl.load(x[i * pmax + pi, j * f_tile_size + fi], mask=mask)

            result = nisa.tensor_tensor_scan(
                data0=ones,
                data1=data,
                initial=init,
                op0=np.multiply,
                op1=np.add,
                dtype=acc_dtype,
                mask=mask,
            )

            if inp_sbuf:
                output[i * pmax + pi, j * f_tile_size + fi] = result
            else:
                nl.store(y[i * pmax + pi, j * f_tile_size + fi], result, mask=mask)

            # update init for the next iteration
            init[:, :] = nl.copy(result[:, f_tile_size - 1], mask=j + 1 < n_f_tiles)

    if inp_sbuf:
        return output


@NKIOpRegistry.register("mylib::cumsum")
def cumsum(x):
    output = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.hbm)
    _cumsum(x, output)
    return output


@torch.library.custom_op("mylib::cumsum", mutates_args=())
def custom_cumsum(x: torch.Tensor) -> torch.Tensor:
    return cumsum(x)


@custom_cumsum.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@NKIOpRegistry.register("mylib::argmax")
def nki_argmax(v1: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = v1.shape
    tile_size = 1024 * 16  # up to 16k
    n_tiles = cdiv(vocab_size, tile_size)
    n_unroll_steps = 1
    dtype = v1.dtype
    print(
        f"sampler.py: {n_tiles=}, {v1.shape=}, {tile_size=}, {batch_size=}, {vocab_size=}, {dtype=}"
    )

    v2 = nl.ndarray((batch_size, 1), dtype=np.int32, buffer=nl.shared_hbm)

    v3 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v3", buffer=nl.sbuf
    )
    v4 = nl.ndarray(
        (nl.par_dim(batch_size), n_unroll_steps, tile_size),
        dtype=dtype,
        name="v4",
        buffer=nl.sbuf,
    )
    v5 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="v5", buffer=nl.sbuf
    )
    v6 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="v6", buffer=nl.sbuf
    )
    v7 = nl.ndarray(
        (nl.par_dim(batch_size), n_unroll_steps, tile_size),
        dtype=np.int32,
        name="v7",
        buffer=nl.sbuf,
    )
    v8 = nl.ndarray(
        (nl.par_dim(batch_size), n_unroll_steps, tile_size),
        dtype=np.int32,
        name="v8",
        buffer=nl.sbuf,
    )
    v9 = nl.ndarray(
        (nl.par_dim(batch_size), n_unroll_steps, tile_size),
        dtype=np.uint8,
        name="v9",
        buffer=nl.sbuf,
    )
    v10 = nl.ndarray(
        (nl.par_dim(batch_size), n_unroll_steps, tile_size),
        dtype=np.int32,
        name="v10",
        buffer=nl.sbuf,
    )
    v11 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v11", buffer=nl.sbuf
    )
    v12 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v12", buffer=nl.sbuf
    )
    v13 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v13", buffer=nl.sbuf
    )
    v14 = nl.zeros(
        (nl.par_dim(batch_size), 1, n_tiles), dtype=dtype, name="v14", buffer=nl.sbuf
    )
    v15 = nl.ndarray(
        (nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v15", buffer=nl.sbuf
    )

    # v3_hbm = nl.ndarray((batch_size, n_tiles), dtype=np.int32, name="v3_hbm", buffer=nl.shared_hbm)
    # v12_hbm = nl.ndarray((batch_size,), dtype=np.int32, name="v12_hbm", buffer=nl.shared_hbm)
    # v14_hbm = nl.ndarray((batch_size, n_tiles), dtype=dtype, name="v14_hbm", buffer=nl.shared_hbm)
    v15_hbm = nl.ndarray(
        (batch_size, n_tiles), dtype=np.int32, name="v15_hbm", buffer=nl.shared_hbm
    )

    i_p = nl.arange(batch_size)[:, None, None]
    i_f = nl.arange(tile_size)[None, None, :]
    iota = nisa.iota(i_f, dtype=np.int32, mask=None)

    for tile_idx in nl.sequential_range(n_tiles):
        local_tile_idx = 0
        mask = (tile_idx * tile_size + i_f) < vocab_size
        v4[i_p, local_tile_idx, i_f] = nisa.memset(
            shape=(batch_size, 1, tile_size), value=-np.inf, dtype=dtype
        )
        v4[i_p, local_tile_idx, i_f] = nl.load(
            v1[i_p, tile_idx * tile_size + i_f], dtype=dtype, mask=mask
        )
        # nl.device_print("sampler.py: v4[i_p, 0, 0]=", v4[i_p, 0, 0])
        v5[i_p, tile_idx, 0] = nisa.tensor_reduce(
            nl.maximum,
            data=v4[i_p, local_tile_idx, i_f],
            mask=None,
            axis=[2],
            dtype=dtype,
            negate=False,
        )
        v6[i_p, tile_idx, 0] = nisa.tensor_scalar(
            data=v5[i_p, tile_idx, 0],
            op0=nl.maximum,
            operand0=-np.inf,
            reverse0=False,
            dtype=dtype,
            mask=None,
        )
        v7[i_p, local_tile_idx, i_f] = nl.broadcast_to(
            iota, shape=(batch_size, 1, tile_size)
        )
        v8[i_p, local_tile_idx, i_f] = nisa.tensor_scalar(
            data=v7[i_p, local_tile_idx, i_f],
            op0=nl.subtract,
            operand0=tile_size,
            reverse0=True,
            dtype=np.int32,
            mask=None,
        )
        v9[i_p, local_tile_idx, i_f] = nisa.tensor_tensor(
            data1=v4[i_p, local_tile_idx, i_f], data2=v6[i_p, tile_idx, 0], op=nl.equal
        )
        v10[i_p, local_tile_idx, i_f] = nl.multiply(
            v8[i_p, local_tile_idx, i_f],
            v9[i_p, local_tile_idx, i_f],
            mask=None,
            dtype=np.int32,
        )
        v11[i_p, tile_idx, 0] = nisa.tensor_reduce(
            nl.maximum,
            data=v10[i_p, local_tile_idx, i_f],
            mask=None,
            axis=[2],
            dtype=np.int32,
            negate=False,
        )
        v12[i_p, tile_idx, 0] = nisa.tensor_scalar(
            data=v11[i_p, tile_idx, 0],
            op0=nl.maximum,
            operand0=-np.inf,
            reverse0=False,
            op1=nl.subtract,
            operand1=tile_size,
            reverse1=True,
            dtype=np.int32,
            mask=None,
        )
        v3[i_p, tile_idx, 0] = nl.copy(v12[i_p, tile_idx, 0], dtype=np.int32, mask=None)
        # nl.store(v3_hbm[i_p, tile_idx], value=v3[i_p, tile_idx, 0], mask=None)
        v13[i_p, tile_idx, 0] = nisa.tensor_scalar(
            data=v7[i_p, 0, tile_idx],
            op0=nl.multiply,
            operand0=tile_size,
            dtype=np.int32,
            reverse0=False,
        )
        v15[i_p, tile_idx, 0] = nisa.tensor_tensor(
            data1=v12[i_p, tile_idx, 0], data2=v13[i_p, tile_idx, 0], op=nl.add
        )
        # nl.device_print("sampler.py: v15[i_p, tile_idx, 0]=", v15[i_p, tile_idx, 0])
        nl.store(v15_hbm[i_p, tile_idx], value=v15[i_p, tile_idx, 0], mask=None)

    # nl.device_print("sampler.py: v13[i_p, 0, 0]=", v13[i_p, 0, 0])
    for batch_idx in nl.affine_range(batch_size):
        for tile_idx in nl.affine_range(n_tiles):
            v14[batch_idx, 0, tile_idx] = nl.load(
                v1[batch_idx, v15[batch_idx, tile_idx, 0]],
                dtype=dtype,
                mask=None,
                mode=oob_mode.skip,
            )  ### HACK, HACK, HACK, HACK, HACK, DO-NOT-ENABLE OOB-SKIP!!!!!
    # nl.device_print("sampler.py: v14[i_p, 0, 0]=", v14[i_p, 0, 0])

    # nl.device_print("sampler.py: v13[i_p, 0, 0]=", v13[i_p, 0, 0])
    i_b = nl.arange(n_tiles)[None, :, None]
    i_f = nl.arange(n_tiles)[None, None, :]
    # nl.store(v14_hbm[i_p, i_f], value=v14[i_p, 0, i_f], mask=None)
    v5[i_p, 0, 0] = nisa.tensor_reduce(
        nl.maximum,
        data=v14[i_p, 0, i_f],
        mask=None,
        axis=[2],
        dtype=dtype,
        negate=False,
    )
    v6[i_p, 0, 0] = nisa.tensor_scalar(
        data=v5[i_p, 0, 0],
        op0=nl.maximum,
        operand0=-np.inf,
        reverse0=False,
        dtype=dtype,
        mask=None,
    )
    v7[i_p, 0, i_f] = nl.broadcast_to(
        nisa.iota(i_f, dtype=np.int32, mask=None), shape=(batch_size, 1, n_tiles)
    )
    v8[i_p, 0, i_f] = nisa.tensor_scalar(
        data=v7[i_p, 0, i_f],
        op0=nl.subtract,
        operand0=n_tiles,
        reverse0=True,
        dtype=np.int32,
        mask=None,
    )
    v9[i_p, 0, i_f] = nisa.tensor_tensor(
        data1=v14[i_p, 0, i_f], data2=v6[i_p, 0, 0], op=nl.equal
    )
    v10[i_p, 0, i_f] = nisa.tensor_tensor(
        v8[i_p, 0, i_f], v9[i_p, 0, i_f], nl.multiply, mask=None, dtype=np.int32
    )
    v11[i_p, 0, 0] = nisa.tensor_reduce(
        nl.maximum,
        data=v10[i_p, 0, i_f],
        mask=None,
        axis=[2],
        dtype=np.int32,
        negate=False,
    )
    v12[i_p, 0, 0] = nisa.tensor_scalar(
        data=v11[i_p, 0, 0],
        op0=nl.maximum,
        operand0=-np.inf,
        reverse0=False,
        op1=nl.subtract,
        operand1=n_tiles,
        reverse1=True,
        dtype=np.int32,
        mask=None,
    )
    # nl.device_print("sampler.py: v12[i_p, 0, 0]=", v12[i_p, 0, 0])
    # nl.store(v12_hbm[i_p], value=v12[i_p, 0, 0], mask=None)

    for batch_idx in nl.affine_range(batch_size):
        nl.store(v2[batch_idx, 0], nl.load(v15_hbm[batch_idx, v12[batch_idx, 0, 0]]))

    # return v2, v3_hbm, v12_hbm, v14_hbm, v15_hbm
    return v2


@torch.library.custom_op("mylib::argmax", mutates_args=())
def custom_argmax(x: torch.Tensor) -> torch.Tensor:
    return nki_argmax(x)


@custom_argmax.register_fake
def _(
    x: torch.Tensor,
) -> torch.Tensor:
    batch_size, vocab_size = x.shape
    return torch.empty((batch_size, 1), dtype=torch.int32)
