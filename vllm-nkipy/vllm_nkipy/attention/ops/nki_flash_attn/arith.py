# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# ruff: noqa

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim


def _remainder(
    v1,
    v2,
    v3,
):
    psize, tile_size = v1.shape
    ix = nl.arange(psize)[:, None]
    iy = nl.arange(tile_size)[None, :]

    v4 = nl.ndarray(
        (nl.par_dim(psize), 1), dtype=np.float32, name="memset.45", buffer=nl.sbuf
    )
    v5 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="remainder.6.53",
        buffer=nl.sbuf,
    )
    v6 = nl.ndarray((nl.par_dim(1), 1), dtype=np.float32, name="", buffer=nl.sbuf)
    v7 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.55",
        buffer=nl.sbuf,
    )
    v8 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.47",
        buffer=nl.sbuf,
    )
    v9 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.59",
        buffer=nl.sbuf,
    )
    v10 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.57",
        buffer=nl.sbuf,
    )
    v11 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.61",
        buffer=nl.sbuf,
    )
    v12 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.49",
        buffer=nl.sbuf,
    )
    v13 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.uint8,
        name="broadcast.5.65",
        buffer=nl.sbuf,
    )
    v14 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.uint8,
        name="broadcast.5.63",
        buffer=nl.sbuf,
    )
    v15 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.69",
        buffer=nl.sbuf,
    )
    v16 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.67",
        buffer=nl.sbuf,
    )
    v17 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.73",
        buffer=nl.sbuf,
    )
    v18 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.71",
        buffer=nl.sbuf,
    )
    v19 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.77",
        buffer=nl.sbuf,
    )
    v20 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.75",
        buffer=nl.sbuf,
    )
    v21 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="broadcast.5.51",
        buffer=nl.sbuf,
    )
    v22 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="broadcast.5.79",
        buffer=nl.sbuf,
    )

    v4[ix, 0] = nisa.memset(
        shape=(psize, 1),
        value=np.dtype(np.uint16).type(0),
        dtype=np.float32,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v5[ix, iy] = nl.copy(v1[ix, iy], dtype=np.int32, mask=None)
    v6[0, 0] = nl.copy(v2[0, 0], dtype=np.float32, mask=None)
    v7[ix, iy] = nisa.reciprocal(data=v6[0, 0], mask=None, dtype=np.float32)
    v8[ix, iy] = nl.multiply(v5[ix, iy], v7[ix, iy], mask=None, dtype=np.float32)
    v9[ix, iy] = nisa.activation(
        op=nl.sign,
        data=v8[ix, iy],
        bias=v4[ix, 0],
        scale=1.0,
        mask=None,
        dtype=np.float32,
    )
    v10[ix, iy] = nl.multiply(v8[ix, iy], v9[ix, iy], mask=None, dtype=np.float32)
    v11[ix, iy] = nl.copy(v10[ix, iy], dtype=np.int32, mask=None)
    v12[ix, iy] = nl.copy(v11[ix, iy], dtype=np.float32, mask=None)
    v13[ix, iy] = nl.greater(v12[ix, iy], v10[ix, iy], mask=None, dtype=np.uint8)
    v14[ix, iy] = nisa.tensor_scalar(
        data=v13[ix, iy],
        op0=nl.logical_xor,
        operand0=np.dtype(np.uint8).type(1),
        reverse0=False,
        dtype=np.uint8,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v15[ix, iy] = nisa.tensor_scalar(
        data=v11[ix, iy],
        op0=nl.multiply,
        operand0=np.dtype(np.int32).type(1),
        reverse0=False,
        op1=nl.add,
        operand1=np.dtype(np.int32).type(-1),
        reverse1=False,
        dtype=np.int32,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v16[ix, iy] = nl.multiply(v13[ix, iy], v15[ix, iy], mask=None, dtype=np.int32)
    v17[ix, iy] = nl.multiply(v14[ix, iy], v11[ix, iy], mask=None, dtype=np.int32)
    v18[ix, iy] = nl.add(v16[ix, iy], v17[ix, iy], mask=None, dtype=np.int32)
    v19[ix, iy] = nl.copy(v9[ix, iy], dtype=np.int32, mask=None)
    v20[ix, iy] = nl.multiply(v18[ix, iy], v19[ix, iy], mask=None, dtype=np.int32)
    v21[ix, iy] = nl.multiply(v20[ix, iy], v6[0, 0], mask=None, dtype=np.float32)
    v22[ix, iy] = nl.subtract(v5[ix, iy], v21[ix, iy], mask=None, dtype=np.int32)
    v3[ix, iy] = nl.copy(v22[ix, iy], mask=None)


@nki.jit
def remainder(x, y):
    tile_size = x.shape[-1]
    ix, iy = nl.mgrid[0:1, 0:tile_size]
    x_sbuf = nl.load(x[iy])
    y_sbuf = nl.load(y)
    output = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.sbuf)
    output_hbm = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.hbm)
    _remainder(x_sbuf, y_sbuf, output)
    nl.store(output_hbm, output)
    return output_hbm


def _floor_divide(
    v1,
    v2,
    v3,
):
    psize, tile_size = v1.shape
    ix = nl.arange(psize)[:, None]
    iy = nl.arange(tile_size)[None, :]

    v4 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="divide.6.49",
        buffer=nl.sbuf,
    )
    v5 = nl.ndarray(
        (nl.par_dim(1), 1), dtype=np.float32, name="broadcast.5.63", buffer=nl.sbuf
    )
    v6 = nl.ndarray(
        (nl.par_dim(psize), 1), dtype=np.float32, name="divide.6.51", buffer=nl.sbuf
    )
    v7 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="divide.6.47",
        buffer=nl.sbuf,
    )
    v8 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="floor.7.53",
        buffer=nl.sbuf,
    )
    v9 = nl.zeros(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="floor.7.43",
        buffer=nl.sbuf,
    )
    v10 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.uint8,
        name="floor.7.57",
        buffer=nl.sbuf,
    )
    v11 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.uint8,
        name="floor.7.55",
        buffer=nl.sbuf,
    )
    v12 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.int32,
        name="floor.7.61",
        buffer=nl.sbuf,
    )
    v13 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="floor.7.59",
        buffer=nl.sbuf,
    )
    v14 = nl.zeros(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="floor.7.45",
        buffer=nl.sbuf,
    )
    v15 = nl.ndarray(
        (nl.par_dim(psize), tile_size),
        dtype=np.float32,
        name="floor.7.65",
        buffer=nl.sbuf,
    )

    v4[ix, iy] = nl.copy(v1[ix, iy], dtype=np.float32, mask=None)
    v5[0, 0] = nl.copy(v2[0, 0], dtype=np.float32, mask=None)

    v6[ix, 0] = nisa.reciprocal(
        data=v5[0, 0], mask=None, dtype=np.float32
    ).broadcast_to([psize, 1])
    v7[ix, iy] = nisa.tensor_scalar(
        data=v4[ix, iy],
        op0=nl.multiply,
        operand0=v6[ix, 0],
        reverse0=False,
        dtype=np.float32,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v8[ix, iy] = nl.copy(v7[ix, iy], dtype=np.int32, mask=None)
    v9[ix, iy] = nl.copy(v8[ix, iy], dtype=np.float32, mask=None)
    v10[ix, iy] = nl.greater(v9[ix, iy], v7[ix, iy], mask=None, dtype=np.uint8)
    v11[ix, iy] = nisa.tensor_scalar(
        data=v10[ix, iy],
        op0=nl.logical_xor,
        operand0=np.dtype(np.uint8).type(1),
        reverse0=False,
        dtype=np.uint8,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v12[ix, iy] = nisa.tensor_scalar(
        data=v8[ix, iy],
        op0=nl.multiply,
        operand0=np.dtype(np.int32).type(1),
        reverse0=False,
        op1=nl.add,
        operand1=np.dtype(np.int32).type(-1),
        reverse1=False,
        dtype=np.int32,
        mask=None,
        engine=nki.isa.unknown_engine,
    )
    v13[ix, iy] = nisa.tensor_tensor(
        v10[ix, iy], v12[ix, iy], nl.multiply, mask=None, dtype=np.float32
    )
    v14[ix, iy] = nisa.tensor_tensor(
        v11[ix, iy], v8[ix, iy], nl.multiply, mask=None, dtype=np.float32
    )
    v15[ix, iy] = nl.add(v13[ix, iy], v14[ix, iy], mask=None, dtype=np.float32)

    v3[ix, iy] = nl.copy(v15[ix, iy], mask=None)


@nki.jit
def floor_divide(x, y):
    tile_size = x.shape[-1]
    ix, iy = nl.mgrid[0:1, 0:tile_size]
    x_sbuf = nl.load(x[iy])
    y_sbuf = nl.load(y)
    output = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.sbuf)
    output_hbm = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.hbm)
    _floor_divide(x_sbuf, y_sbuf, output)
    nl.store(output_hbm, output)
    return output_hbm


def _cumsum(
    v1,
    # v2,
):
    tile_size = v1.shape[-1]
    window_size = tile_size - 1
    ix = nl.arange(1)[:, None]
    iy = nl.arange(tile_size)[None, :]
    ones = nl.full((1, tile_size), fill_value=1, dtype=nl.float32)
    mask = iy < tile_size
    result = nisa.tensor_tensor_scan(
        data0=ones,
        data1=nl.copy(v1, dtype=nl.float32),
        initial=0,
        op0=np.multiply,
        op1=np.add,
        dtype=nl.float32,
        mask=mask,
    )
    # v2[...] = nl.copy(result, dtype=v2.dtype)
    return result


@nki.jit
def cumsum(x):
    tile_size = x.shape[-1]
    ix, iy = nl.mgrid[0:1, 0:tile_size]
    # i_p = nl.arange(tile_size)[:, None]
    x_sbuf = nl.load(x[iy])
    output = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.sbuf)
    output_hbm = nl.ndarray(x_sbuf.shape, dtype=x_sbuf.dtype, buffer=nl.hbm)
    output[...] = _cumsum(x_sbuf)
    nl.store(output_hbm, output)
    return output_hbm
