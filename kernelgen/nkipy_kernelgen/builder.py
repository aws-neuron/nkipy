# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builder API for nkipy kernelgen backend.

Provides an opaque IR construction interface so that nkipy never imports
``mlir`` directly.  All MLIR types (``ir.Value``, ``ir.Type``, dialect ops)
are encapsulated behind :class:`TensorHandle`, :class:`LoopIndexHandle`,
and :class:`IRBuilder`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np
from mlir import ir, passmanager
from mlir.dialects import arith, func, linalg, scf, tensor
from mlir.dialects import math as mlir_math

from nkipy_kernelgen._mlir.dialects import nkipy as nkipy_d
from nkipy_kernelgen.mlir_utils import (
    const_scalar,
    make_empty,
    make_filled,
    make_zeros,
    ranked_tensor_of,
    to_mlir_type,
)

Scalar = Union[int, float]

_MEM_SPACE_CONSTANT = 4

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

_MLIR_TO_NP: dict[str, np.dtype] = {
    "f16": np.dtype("float16"),
    "bf16": np.dtype("bfloat16"),
    "f32": np.dtype("float32"),
    "f64": np.dtype("float64"),
    "i32": np.dtype("int32"),
    "i64": np.dtype("int64"),
}


def _mlir_type_to_np(mlir_ty: ir.Type) -> np.dtype:
    s = str(mlir_ty)
    if s in _MLIR_TO_NP:
        return _MLIR_TO_NP[s]
    raise KeyError(f"Cannot convert MLIR type {mlir_ty} to numpy dtype")


def _np_to_mlir(dtype) -> ir.Type:
    if isinstance(dtype, ir.Type):
        return dtype
    return to_mlir_type(dtype)


def _is_float(elem_ty: ir.Type) -> bool:
    return isinstance(elem_ty, ir.FloatType)


def _is_int(elem_ty: ir.Type) -> bool:
    return isinstance(elem_ty, ir.IntegerType)


# ---------------------------------------------------------------------------
# TensorHandle — opaque tensor value
# ---------------------------------------------------------------------------


class TensorHandle:
    """Opaque handle to a traced tensor value.

    Users see ``.shape`` and ``.dtype`` only.  The internal ``_value``
    (an ``ir.Value``) is never accessed from nkipy.
    """

    __slots__ = ("_value", "shape", "dtype", "_elem_ty")

    def __init__(
        self, value: ir.Value, shape: tuple, dtype: np.dtype, elem_ty: ir.Type
    ):
        self._value = value
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        self._elem_ty = elem_ty


def _make_handle(value, shape, elem_ty) -> TensorHandle:
    return TensorHandle(value, shape, _mlir_type_to_np(elem_ty), elem_ty)


def _loc() -> ir.Location:
    return ir.Location.unknown()


# ---------------------------------------------------------------------------
# LoopIndexHandle — opaque loop induction variable
# ---------------------------------------------------------------------------


class LoopIndexHandle:
    """Opaque handle to a loop induction variable with factor/offset tracking."""

    __slots__ = ("_value", "mul_factor", "add_offset")

    def __init__(self, value: ir.Value, mul_factor: int = 1, add_offset: int = 0):
        self._value = value
        self.mul_factor = mul_factor
        self.add_offset = add_offset


# ---------------------------------------------------------------------------
# IRBuilder — lifecycle management
# ---------------------------------------------------------------------------


class IRBuilder:
    """Manages MLIR context, module, and function lifecycle."""

    def __init__(self, source_file: str = "nkipy_kernel"):
        self._ctx = ir.Context()
        nkipy_d.register_dialect(self._ctx)
        self._ctx.__enter__()

        self._file_loc = ir.Location.file(source_file, 0, 0, context=self._ctx)
        self._file_loc.__enter__()

        self._module = ir.Module.create()
        self._parameters: list[TensorHandle] = []
        self._func_op = None
        self._entry_block = None
        self._ip = None

    @property
    def context(self):
        return self._ctx

    @property
    def module(self):
        return self._module

    def begin_function(
        self,
        name: str,
        arg_shapes: list[tuple],
        arg_dtypes: list,
    ) -> list[TensorHandle]:
        arg_types = [
            ranked_tensor_of(shape, _np_to_mlir(dtype))
            for shape, dtype in zip(arg_shapes, arg_dtypes)
        ]
        fn_type = ir.FunctionType.get(arg_types, [arg_types[0]])
        with ir.InsertionPoint(self._module.body):
            self._func_op = func.FuncOp(name=name, type=fn_type, loc=self._file_loc)
            self._entry_block = self._func_op.add_entry_block()
        self._ip = ir.InsertionPoint(self._entry_block)
        self._ip.__enter__()

        handles: list[TensorHandle] = []
        for arg, (shape, dtype) in zip(
            self._entry_block.arguments, zip(arg_shapes, arg_dtypes)
        ):
            elem_ty = _np_to_mlir(dtype)
            h = TensorHandle(arg, shape, np.dtype(dtype) if not isinstance(dtype, str) else _mlir_type_to_np(elem_ty), elem_ty)
            self._parameters.append(h)
            handles.append(h)
        return handles

    def finish_function(self, result_handles: list[TensorHandle]):
        values = [h._value for h in result_handles]
        func.ReturnOp(values, loc=self._file_loc)
        self._ip.__exit__(None, None, None)
        self._ip = None

        arg_types = [ranked_tensor_of(p.shape, p._elem_ty) for p in self._parameters]
        res_types = [v.type for v in values]
        self._func_op.attributes["function_type"] = ir.TypeAttr.get(
            ir.FunctionType.get(arg_types, res_types)
        )

    def emit_custom_op_declarations(self, custom_ops: list):
        """Emit func.func private declarations and stash NISA bodies for custom ops."""
        if not custom_ops:
            return
        with ir.InsertionPoint(self._module.body):
            for custom in custom_ops:
                input_types = [
                    ranked_tensor_of(s, _np_to_mlir(d))
                    for s, d in zip(custom.input_shapes, custom.input_dtypes)
                ]
                result_types = [
                    ranked_tensor_of(s, _np_to_mlir(d))
                    for s, d in zip(custom.output_shapes, custom.output_dtypes)
                ]
                fn_type = ir.FunctionType.get(input_types, result_types)
                fn = func.FuncOp(name=custom.func_name, type=fn_type)
                fn.attributes["sym_visibility"] = ir.StringAttr.get("private")
                fn.attributes["nkipy.custom_op"] = ir.UnitAttr.get()

        bodies = {custom.func_name: custom.nisa_mlir for custom in custom_ops}
        self._module.operation.attributes["nkipy.custom_op_bodies"] = (
            ir.DictAttr.get({k: ir.StringAttr.get(v) for k, v in bodies.items()})
        )

    def run_canonicalize(self):
        pm = passmanager.PassManager.parse("builtin.module(func.func(canonicalize))")
        pm.run(self._module.operation)

    def get_ir_text(self) -> str:
        return str(self._module)

    def cleanup(self):
        if self._ip is not None:
            self._ip.__exit__(None, None, None)
            self._ip = None
        self._file_loc.__exit__(None, None, None)
        self._ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Broadcasting helpers (internal)
# ---------------------------------------------------------------------------


def _broadcast_shape(sa: tuple, sb: tuple) -> tuple:
    mr = max(len(sa), len(sb))
    pa = (1,) * (mr - len(sa)) + tuple(sa)
    pb = (1,) * (mr - len(sb)) + tuple(sb)
    result = []
    for da, db in zip(pa, pb):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {sa} vs {sb}")
    return tuple(result)


def _broadcast_indexing_map(in_shape: tuple, out_shape: tuple) -> ir.AffineMap:
    rd = len(out_shape) - len(in_shape)
    exprs = []
    for oi in range(len(out_shape)):
        ii = oi - rd
        if ii < 0:
            continue
        if in_shape[ii] == 1 and out_shape[oi] > 1:
            exprs.append(ir.AffineConstantExpr.get(0))
        else:
            exprs.append(ir.AffineDimExpr.get(oi))
    return ir.AffineMap.get(len(out_shape), 0, exprs)


# ---------------------------------------------------------------------------
# Cast helpers (internal)
# ---------------------------------------------------------------------------


def _cast_to_float(val, shape, elem_ty, loc):
    out_elem = ir.F32Type.get()
    result_type = ranked_tensor_of(shape, out_elem)
    out = make_empty(loc, shape, out_elem)
    nd = len(shape)
    imap = ir.AffineMap.get_identity(nd)
    g = linalg.GenericOp(
        [result_type],
        [val],
        [out],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(imap)] * 2),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(elem_ty, out_elem)
    with ir.InsertionPoint(blk):
        r = arith.SIToFPOp(out_elem, blk.arguments[0], loc=loc).result
        linalg.YieldOp([r], loc=loc)
    return g.results[0], shape, out_elem


# ---------------------------------------------------------------------------
# Generic element-wise helpers (internal)
# ---------------------------------------------------------------------------


def _scalar_binary(
    tensor_val,
    tensor_shape,
    tensor_elem,
    scalar_val,
    arith_fn,
    loc,
    scalar_is_lhs=False,
):
    rt = ranked_tensor_of(tensor_shape, tensor_elem)
    out = make_empty(loc, tensor_shape, tensor_elem)
    nd = len(tensor_shape)
    imap = ir.AffineMap.get_identity(nd)
    g = linalg.GenericOp(
        [rt],
        [tensor_val],
        [out],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(imap)] * 2),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(tensor_elem, tensor_elem)
    with ir.InsertionPoint(blk):
        cst = const_scalar(scalar_val, tensor_elem, loc)
        if scalar_is_lhs:
            r = arith_fn(cst, blk.arguments[0])
        else:
            r = arith_fn(blk.arguments[0], cst)
        linalg.YieldOp([r], loc=loc)
    return g.results[0], tensor_shape, tensor_elem


def _broadcast_binary(a_val, a_shape, a_elem, b_val, b_shape, b_elem, body_fn, loc):
    if str(a_elem) != str(b_elem):
        raise TypeError(f"Element type mismatch: {a_elem} vs {b_elem}")
    elem = a_elem
    out_shape = _broadcast_shape(a_shape, b_shape)
    rt = ranked_tensor_of(out_shape, elem)
    out = make_empty(loc, out_shape, elem)
    ma = _broadcast_indexing_map(a_shape, out_shape)
    mb = _broadcast_indexing_map(b_shape, out_shape)
    mo = ir.AffineMap.get_identity(len(out_shape))
    g = linalg.GenericOp(
        [rt],
        [a_val, b_val],
        [out],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(ma),
                ir.AffineMapAttr.get(mb),
                ir.AffineMapAttr.get(mo),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(out_shape)
        ),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(elem, elem, elem)
    with ir.InsertionPoint(blk):
        r = body_fn(blk.arguments[0], blk.arguments[1])
        linalg.YieldOp([r], loc=loc)
    return g.results[0], out_shape, elem


# ---------------------------------------------------------------------------
# Internal: unary generic
# ---------------------------------------------------------------------------


def _unary_named(x: TensorHandle, named_cls, body_fn, loc) -> TensorHandle:
    val, shape, elem = x._value, x.shape, x._elem_ty
    rt = ranked_tensor_of(shape, elem)
    out = make_empty(loc, shape, elem)
    op = named_cls([rt], [val], [out], loc=loc)
    blk = op.regions[0].blocks.append(elem, elem)
    with ir.InsertionPoint(blk):
        r = body_fn(blk.arguments[0], elem, loc)
        linalg.YieldOp([r], loc=loc)
    return _make_handle(op.results[0], shape, elem)


def _unary_generic(x: TensorHandle, body_fn, loc) -> TensorHandle:
    val, shape, elem = x._value, x.shape, x._elem_ty
    rt = ranked_tensor_of(shape, elem)
    out = make_empty(loc, shape, elem)
    nd = len(shape)
    imap = ir.AffineMap.get_identity(nd)
    g = linalg.GenericOp(
        [rt],
        [val],
        [out],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(imap)] * 2),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(elem, elem)
    with ir.InsertionPoint(blk):
        r = body_fn(blk.arguments[0], elem, loc)
        linalg.YieldOp([r], loc=loc)
    return _make_handle(g.results[0], shape, elem)


# ---------------------------------------------------------------------------
# Internal: binary dispatch (tensor-tensor, tensor-scalar, scalar-tensor)
# ---------------------------------------------------------------------------


def _binary_dispatch(x, y, float_cls, int_cls, named_cls, loc):
    x_is_t = isinstance(x, TensorHandle)
    y_is_t = isinstance(y, TensorHandle)

    if not x_is_t and not y_is_t:
        raise TypeError("At least one operand must be a TensorHandle")

    float_op = lambda a, b: float_cls(a, b, loc=loc).result
    int_op = lambda a, b: int_cls(a, b, loc=loc).result

    if not x_is_t or not y_is_t:
        if x_is_t:
            tv, ts, te = x._value, x.shape, x._elem_ty
            sv, slhs = y, False
        else:
            tv, ts, te = y._value, y.shape, y._elem_ty
            sv, slhs = x, True
        fn = float_op if _is_float(te) else int_op
        rv, rs, re = _scalar_binary(tv, ts, te, sv, fn, loc, slhs)
        return _make_handle(rv, rs, re)

    xv, xs, xe = x._value, x.shape, x._elem_ty
    yv, ys, ye = y._value, y.shape, y._elem_ty
    if _is_float(xe) and not _is_float(ye):
        yv, ys, ye = _cast_to_float(yv, ys, ye, loc)
    elif _is_float(ye) and not _is_float(xe):
        xv, xs, xe = _cast_to_float(xv, xs, xe, loc)

    if xs != ys or named_cls is None:
        fn = float_op if _is_float(xe) else int_op
        rv, rs, re = _broadcast_binary(xv, xs, xe, yv, ys, ye, fn, loc)
        return _make_handle(rv, rs, re)

    elem = xe
    rt = ranked_tensor_of(xs, elem)
    out = make_empty(loc, xs, elem)
    nop = named_cls([rt], [xv, yv], [out], loc=loc)
    blk = nop.regions[0].blocks.append(elem, elem, elem)
    with ir.InsertionPoint(blk):
        fn = float_op if _is_float(elem) else int_op
        r = fn(blk.arguments[0], blk.arguments[1])
        linalg.YieldOp([r], loc=loc)
    return _make_handle(nop.results[0], xs, elem)


# ---------------------------------------------------------------------------
# Internal: reshape
# ---------------------------------------------------------------------------


def _emit_reshape(loc, value, old_shape, new_shape, elem_ty):
    dst_ty = ranked_tensor_of(tuple(new_shape), elem_ty)
    if tuple(old_shape) == tuple(new_shape):
        return value

    def _reassoc(from_shape, to_shape):
        reassoc = []
        to_idx = 0
        to_rank = len(to_shape)
        for from_dim in from_shape:
            group = []
            product = 1
            while to_idx < to_rank and product < from_dim:
                product *= to_shape[to_idx]
                group.append(to_idx)
                to_idx += 1
            if from_dim == 1 and not group:
                if to_idx < to_rank and to_shape[to_idx] == 1:
                    group.append(to_idx)
                    to_idx += 1
                    product = 1
                else:
                    return None
            if product != from_dim or not group:
                return None
            reassoc.append(group)
        while to_idx < to_rank and to_shape[to_idx] == 1:
            reassoc[-1].append(to_idx)
            to_idx += 1
        return reassoc if to_idx == to_rank else None

    if len(old_shape) >= len(new_shape):
        r = _reassoc(new_shape, old_shape)
        if r is not None:
            return tensor.CollapseShapeOp(dst_ty, value, r, loc=loc).result
    if len(old_shape) <= len(new_shape):
        r = _reassoc(old_shape, new_shape)
        if r is not None:
            return tensor.ExpandShapeOp(
                dst_ty,
                value,
                r,
                output_shape=[],
                static_output_shape=list(new_shape),
                loc=loc,
            ).result

    idx_ty = ir.IndexType.get()
    shape_ty = ir.RankedTensorType.get([len(new_shape)], idx_ty)
    shape_vals = [
        arith.ConstantOp(idx_ty, ir.IntegerAttr.get(idx_ty, d), loc=loc).result
        for d in new_shape
    ]
    fe = tensor.FromElementsOp(shape_ty, shape_vals, loc=loc)
    return tensor.ReshapeOp(dst_ty, value, fe.result, loc=loc).result


# ---------------------------------------------------------------------------
# Internal: matmul body helper
# ---------------------------------------------------------------------------


def _matmul_body(op, elem_ty, loc):
    blk = op.regions[0].blocks.append(elem_ty, elem_ty, elem_ty)
    with ir.InsertionPoint(blk):
        a, b, c = blk.arguments
        if _is_float(elem_ty):
            p = arith.MulFOp(a, b, loc=loc).result
            r = arith.AddFOp(c, p, loc=loc).result
        else:
            p = arith.MulIOp(a, b, loc=loc).result
            r = arith.AddIOp(c, p, loc=loc).result
        linalg.YieldOp([r], loc=loc)


# ===================================================================
# PUBLIC OP API
# ===================================================================

# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------


def add(x: Union[TensorHandle, Scalar], y: Union[TensorHandle, Scalar], loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.AddFOp, arith.AddIOp, linalg.AddOp, loc or _loc())


def subtract(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.SubFOp, arith.SubIOp, linalg.SubOp, loc or _loc())


def multiply(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.MulFOp, arith.MulIOp, linalg.MulOp, loc or _loc())


def divide(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.DivFOp, arith.DivSIOp, linalg.DivOp, loc or _loc())


def maximum(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.MaximumFOp, arith.MaxSIOp, linalg.MaxOp, loc or _loc())


def minimum(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.MinimumFOp, arith.MinSIOp, linalg.MinOp, loc or _loc())


def power(x, y, loc=None) -> TensorHandle:
    x_is_t = isinstance(x, TensorHandle)
    y_is_t = isinstance(y, TensorHandle)

    if x_is_t and not y_is_t:
        if isinstance(y, (int, float)):
            if y == 2:
                return multiply(x, x, loc=loc)
            if y == 0.5:
                return sqrt(x, loc=loc)
        log_x = log(x, loc=loc)
        scaled = multiply(log_x, float(y), loc=loc)
        return exp(scaled, loc=loc)
    elif not x_is_t and y_is_t:
        log_scalar = math.log(float(x))
        scaled = multiply(y, log_scalar, loc=loc)
        return exp(scaled, loc=loc)
    else:
        loc = loc or _loc()
        xv, xs, xe = x._value, x.shape, x._elem_ty
        yv, ys, ye = y._value, y.shape, y._elem_ty
        pow_fn = lambda a, b: mlir_math.PowFOp(a, b, loc=loc).result
        rv, rs, re = _broadcast_binary(xv, xs, xe, yv, ys, ye, pow_fn, loc)
        return _make_handle(rv, rs, re)


# ---------------------------------------------------------------------------
# Unary math
# ---------------------------------------------------------------------------


def exp(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.ExpOp, lambda v, _, l: mlir_math.ExpOp(v, loc=l).result, loc or _loc()
    )


def log(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.LogOp, lambda v, _, l: mlir_math.LogOp(v, loc=l).result, loc or _loc()
    )


def sqrt(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.SqrtOp, lambda v, _, l: mlir_math.SqrtOp(v, loc=l).result, loc or _loc()
    )


def tanh(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.TanhOp, lambda v, _, l: mlir_math.TanhOp(v, loc=l).result, loc or _loc()
    )


def abs_(x: TensorHandle, loc=None) -> TensorHandle:
    def _body(v, elem, l):
        if _is_float(elem):
            return mlir_math.AbsFOp(v, loc=l).result
        return mlir_math.AbsIOp(v, loc=l).result

    return _unary_named(x, linalg.AbsOp, _body, loc or _loc())


def ceil_(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.CeilOp, lambda v, _, l: mlir_math.CeilOp(v, loc=l).result, loc or _loc()
    )


def floor_(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(
        x, linalg.FloorOp, lambda v, _, l: mlir_math.FloorOp(v, loc=l).result, loc or _loc()
    )


def sin(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_generic(
        x, lambda v, _, l: mlir_math.SinOp(v, loc=l).result, loc or _loc()
    )


def cos(x: TensorHandle, loc=None) -> TensorHandle:
    shifted = add(x, math.pi / 2, loc=loc)
    return sin(shifted, loc=loc)


def sign(x: TensorHandle, loc=None) -> TensorHandle:
    def _body(v, elem, l):
        one = arith.ConstantOp(elem, ir.FloatAttr.get(elem, 1.0), loc=l).result
        return mlir_math.CopySignOp(one, v, loc=l).result

    return _unary_generic(x, _body, loc or _loc())


def square(x: TensorHandle, loc=None) -> TensorHandle:
    def _body(v, elem, l):
        if _is_float(elem):
            return arith.MulFOp(v, v, loc=l).result
        return arith.MulIOp(v, v, loc=l).result

    return _unary_named(x, linalg.SquareOp, _body, loc or _loc())


def reciprocal(x: TensorHandle, loc=None) -> TensorHandle:
    def _body(v, elem, l):
        one = arith.ConstantOp(elem, ir.FloatAttr.get(elem, 1.0), loc=l).result
        return arith.DivFOp(one, v, loc=l).result

    return _unary_named(x, linalg.ReciprocalOp, _body, loc or _loc())


def negative(x: TensorHandle, loc=None) -> TensorHandle:
    return multiply(x, -1.0, loc=loc)


def copy_(x: TensorHandle, loc=None) -> TensorHandle:
    return _unary_named(x, linalg.CopyOp, lambda v, _, __: v, loc or _loc())


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _comparison(x, y, pred_name, loc) -> TensorHandle:
    loc = loc or _loc()
    x_is_t = isinstance(x, TensorHandle)
    y_is_t = isinstance(y, TensorHandle)

    pred = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64),
        arith.CmpFPredicate.__members__[pred_name].value,
    ).value

    def cmp_fn(lhs, rhs):
        c = arith.CmpFOp(pred, lhs, rhs, loc=loc).result
        return arith.UIToFPOp(lhs.type, c, loc=loc).result

    if not x_is_t or not y_is_t:
        if x_is_t:
            tv, ts, te = x._value, x.shape, x._elem_ty
            sv, slhs = y, False
        else:
            tv, ts, te = y._value, y.shape, y._elem_ty
            sv, slhs = x, True
        if _is_int(te):
            tv, ts, te = _cast_to_float(tv, ts, te, loc)
        rv, rs, re = _scalar_binary(tv, ts, te, sv, cmp_fn, loc, slhs)
        return _make_handle(rv, rs, re)

    xv, xs, xe = x._value, x.shape, x._elem_ty
    yv, ys, ye = y._value, y.shape, y._elem_ty
    if _is_int(xe):
        xv, xs, xe = _cast_to_float(xv, xs, xe, loc)
    if _is_int(ye):
        yv, ys, ye = _cast_to_float(yv, ys, ye, loc)

    rv, rs, re = _broadcast_binary(xv, xs, xe, yv, ys, ye, cmp_fn, loc)
    return _make_handle(rv, rs, re)


def equal(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "OEQ", loc)


def not_equal(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "UNE", loc)


def greater(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "OGT", loc)


def greater_equal(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "OGE", loc)


def less(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "OLT", loc)


def less_equal(x, y, loc=None) -> TensorHandle:
    return _comparison(x, y, "OLE", loc)


# ---------------------------------------------------------------------------
# Bitwise / logical
# ---------------------------------------------------------------------------


def _logical_binary(x, y, int_cls, loc) -> TensorHandle:
    loc = loc or _loc()
    x_is_t = isinstance(x, TensorHandle)
    y_is_t = isinstance(y, TensorHandle)

    ref = x if x_is_t else y
    elem = ref._elem_ty
    is_fp = _is_float(elem)

    if is_fp:
        i1 = ir.IntegerType.get_signless(1)

        def body_fn(lhs, rhs):
            l1 = arith.FPToUIOp(i1, lhs, loc=loc).result
            r1 = arith.FPToUIOp(i1, rhs, loc=loc).result
            ri = int_cls(l1, r1, loc=loc).result
            return arith.UIToFPOp(lhs.type, ri, loc=loc).result
    else:

        def body_fn(lhs, rhs):
            return int_cls(lhs, rhs, loc=loc).result

    if not x_is_t or not y_is_t:
        if x_is_t:
            bt = x
            sv, slhs = y, False
        else:
            bt = y
            sv, slhs = x, True
        rv, rs, re = _scalar_binary(
            bt._value, bt.shape, bt._elem_ty, sv, body_fn, loc, slhs
        )
        return _make_handle(rv, rs, re)

    rv, rs, re = _broadcast_binary(
        x._value, x.shape, x._elem_ty,
        y._value, y.shape, y._elem_ty,
        body_fn, loc,
    )
    return _make_handle(rv, rs, re)


def bitwise_and(x, y, loc=None) -> TensorHandle:
    return _logical_binary(x, y, arith.AndIOp, loc)


def bitwise_or(x, y, loc=None) -> TensorHandle:
    return _logical_binary(x, y, arith.OrIOp, loc)


def bitwise_xor(x, y, loc=None) -> TensorHandle:
    return _logical_binary(x, y, arith.XOrIOp, loc)


def logical_not(x, loc=None) -> TensorHandle:
    return subtract(1, x, loc=loc)


def mod(x, y, loc=None) -> TensorHandle:
    return _binary_dispatch(x, y, arith.RemFOp, arith.RemSIOp, None, loc or _loc())


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------


def matmul(x: TensorHandle, y: TensorHandle, loc=None) -> TensorHandle:
    loc = loc or _loc()
    xv, xs, xe = x._value, x.shape, x._elem_ty
    yv, ys, ye = y._value, y.shape, y._elem_ty

    m, k = xs[-2], xs[-1]
    k2, n = ys[-2], ys[-1]
    if k != k2:
        raise ValueError(f"Incompatible shapes for matmul: {xs} @ {ys}")

    if len(xs) == 2 and len(ys) == 2:
        out_shape = (m, n)
        out_val = make_zeros(loc, out_shape, xe)
        rt = ranked_tensor_of(out_shape, xe)
        mm = linalg.MatmulOp([rt], [xv, yv], [out_val], loc=loc)
        _matmul_body(mm, xe, loc)
        return _make_handle(mm.results[0], out_shape, xe)

    ba = xs[:-2]
    bb = ys[:-2]
    ml = max(len(ba), len(bb))
    pa = (1,) * (ml - len(ba)) + ba
    pb = (1,) * (ml - len(bb)) + bb
    bs = tuple(max(da, db) for da, db in zip(pa, pb))

    bsz = 1
    for d in bs:
        bsz *= d

    a3 = (bsz, m, k)
    b3 = (bsz, k, n)
    o3 = (bsz, m, n)

    av = xv if xs == a3 else _emit_reshape(loc, xv, xs, a3, xe)
    bv = yv if ys == b3 else _emit_reshape(loc, yv, ys, b3, ye)

    out_val = make_zeros(loc, o3, xe)
    rt = ranked_tensor_of(o3, xe)
    mm = linalg.BatchMatmulOp([rt], [av, bv], [out_val], loc=loc)
    _matmul_body(mm, xe, loc)

    final = bs + (m, n)
    if final == o3:
        rv = mm.results[0]
    else:
        rv = _emit_reshape(loc, mm.results[0], o3, final, xe)
    return _make_handle(rv, final, xe)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def transpose(x: TensorHandle, axes=None, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, shape, elem = x._value, x.shape, x._elem_ty
    rank = len(shape)
    if axes is None:
        perm = list(range(rank - 1, -1, -1))
    else:
        perm = [ax if ax >= 0 else ax + rank for ax in axes]
    new_shape = tuple(shape[p] for p in perm)
    out = make_empty(loc, new_shape, elem)
    rt = ranked_tensor_of(new_shape, elem)
    top = linalg.TransposeOp([rt], val, out, perm, loc=loc)
    blk = top.regions[0].blocks.append(elem, elem)
    with ir.InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]], loc=loc)
    return _make_handle(top.results[0], new_shape, elem)


# ---------------------------------------------------------------------------
# Reshape / expand_dims
# ---------------------------------------------------------------------------


def reshape(x: TensorHandle, newshape, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, shape, elem = x._value, x.shape, x._elem_ty
    newshape = list(newshape)
    if -1 in newshape:
        if newshape.count(-1) > 1:
            raise ValueError("can only specify one unknown dimension (-1)")
        total = 1
        for d in shape:
            total *= d
        known = 1
        neg = -1
        for i, d in enumerate(newshape):
            if d == -1:
                neg = i
            else:
                known *= d
        newshape[neg] = total // known
    ns = tuple(newshape)
    rv = _emit_reshape(loc, val, shape, ns, elem)
    return _make_handle(rv, ns, elem)


def expand_dims(x: TensorHandle, axis, loc=None) -> TensorHandle:
    shape = x.shape
    if isinstance(axis, int):
        axis = (axis,)
    ns = list(shape)
    for ax in sorted(axis):
        if ax < 0:
            ax = len(ns) + ax + 1
        ns.insert(ax, 1)
    return reshape(x, tuple(ns), loc=loc)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _normalize_axis(axis, rank):
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = [axis]
    return sorted([ax % rank for ax in axis])


def _reduce(x: TensorHandle, axis, keepdims, init_fn, body_fn, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, shape, elem = x._value, x.shape, x._elem_ty
    rank = len(shape)
    if axis is None:
        axis = list(range(rank))

    if keepdims:
        out_shape = tuple(1 if i in axis else shape[i] for i in range(rank))
        out_exprs = [
            ir.AffineConstantExpr.get(0) if i in axis else ir.AffineDimExpr.get(i)
            for i in range(rank)
        ]
    else:
        out_shape = tuple(shape[i] for i in range(rank) if i not in axis)
        out_exprs = [ir.AffineDimExpr.get(i) for i in range(rank) if i not in axis]

    rt = ranked_tensor_of(out_shape, elem)
    init = init_fn(loc, out_shape, elem)
    imap = ir.AffineMap.get_identity(rank)
    omap = ir.AffineMap.get(rank, 0, out_exprs)

    g = linalg.GenericOp(
        [rt],
        [val],
        [init],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(imap), ir.AffineMapAttr.get(omap)]),
        ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<reduction>")
                if i in axis
                else ir.Attribute.parse("#linalg.iterator_type<parallel>")
                for i in range(rank)
            ]
        ),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(elem, elem)
    with ir.InsertionPoint(blk):
        r = body_fn(blk.arguments[0], blk.arguments[1], elem, loc)
        linalg.YieldOp([r], loc=loc)
    return _make_handle(g.results[0], out_shape, elem)


def reduce_sum(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    na = _normalize_axis(axis, len(x.shape))

    def body(inp, acc, elem, l):
        if _is_float(elem):
            return arith.AddFOp(acc, inp, loc=l).result
        return arith.AddIOp(acc, inp, loc=l).result

    return _reduce(x, na, keepdims, make_zeros, body, loc)


def reduce_prod(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    na = _normalize_axis(axis, len(x.shape))
    init_fn = lambda loc, shape, elem: make_filled(loc, shape, elem, 1.0)

    def body(inp, acc, elem, l):
        if _is_float(elem):
            return arith.MulFOp(acc, inp, loc=l).result
        return arith.MulIOp(acc, inp, loc=l).result

    return _reduce(x, na, keepdims, init_fn, body, loc)


def reduce_max(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    na = _normalize_axis(axis, len(x.shape))
    init_fn = lambda loc, shape, elem: make_filled(loc, shape, elem, float("-inf"))

    def body(inp, acc, elem, l):
        if _is_float(elem):
            return arith.MaximumFOp(acc, inp, loc=l).result
        return arith.MaxSIOp(acc, inp, loc=l).result

    return _reduce(x, na, keepdims, init_fn, body, loc)


def reduce_min(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    na = _normalize_axis(axis, len(x.shape))
    init_fn = lambda loc, shape, elem: make_filled(loc, shape, elem, float("inf"))

    def body(inp, acc, elem, l):
        if _is_float(elem):
            return arith.MinimumFOp(acc, inp, loc=l).result
        return arith.MinSIOp(acc, inp, loc=l).result

    return _reduce(x, na, keepdims, init_fn, body, loc)


def reduce_mean(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    shape = x.shape
    s = reduce_sum(x, axis=axis, keepdims=keepdims, loc=loc)
    if axis is None:
        count = int(np.prod(shape))
    else:
        axes = [axis] if isinstance(axis, int) else list(axis)
        count = int(np.prod([shape[i] for i in axes]))
    return divide(s, float(count), loc=loc)


def reduce_std(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    mean_val = reduce_mean(x, axis=axis, keepdims=True, loc=loc)
    diff = subtract(x, mean_val, loc=loc)
    sq = multiply(diff, diff, loc=loc)
    variance = reduce_mean(sq, axis=axis, keepdims=keepdims, loc=loc)
    return sqrt(variance, loc=loc)


def reduce_var(x: TensorHandle, axis=None, keepdims=False, loc=None) -> TensorHandle:
    mean_val = reduce_mean(x, axis=axis, keepdims=True, loc=loc)
    diff = subtract(x, mean_val, loc=loc)
    sq = multiply(diff, diff, loc=loc)
    return reduce_mean(sq, axis=axis, keepdims=keepdims, loc=loc)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def zeros(shape: tuple, dtype, loc=None) -> TensorHandle:
    loc = loc or _loc()
    elem = _np_to_mlir(dtype)
    v = make_filled(loc, tuple(shape), elem, 0.0)
    return _make_handle(v, tuple(shape), elem)


def full(shape: tuple, fill_value, dtype, loc=None) -> TensorHandle:
    loc = loc or _loc()
    elem = _np_to_mlir(dtype)
    v = make_filled(loc, tuple(shape), elem, fill_value)
    return _make_handle(v, tuple(shape), elem)


def empty(shape: tuple, dtype, loc=None) -> TensorHandle:
    loc = loc or _loc()
    elem = _np_to_mlir(dtype)
    v = make_empty(loc, tuple(shape), elem)
    return _make_handle(v, tuple(shape), elem)


def constant_tensor(val: Scalar, shape: tuple, elem_ty, loc=None) -> TensorHandle:
    """Create a filled tensor annotated with CONSTANT memory space."""
    loc = loc or _loc()
    if not isinstance(elem_ty, ir.Type):
        elem_ty = _np_to_mlir(elem_ty)
    v = make_filled(loc, tuple(shape), elem_ty, val)
    ms_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), _MEM_SPACE_CONSTANT)
    nkipy_d.AnnotateOp(
        target=v,
        mem_space=ms_attr,
        partition_dim=None,
        tile_size=None,
        reduction_tile=None,
        loc=loc,
    )
    return _make_handle(v, tuple(shape), elem_ty)


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------


def concatenate(arrays: list[TensorHandle], axis: int = 0, loc=None) -> TensorHandle:
    loc = loc or _loc()
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    if len(arrays) == 1:
        return arrays[0]

    first = arrays[0]
    elem = first._elem_ty
    out_shape = list(first.shape)
    out_shape[axis] = sum(a.shape[axis] for a in arrays)
    out_shape = tuple(out_shape)

    output = make_empty(loc, out_shape, elem)
    offset = 0
    for a in arrays:
        offsets = [0] * len(out_shape)
        offsets[axis] = offset
        sizes = list(a.shape)
        strides = [1] * len(out_shape)
        output = tensor.InsertSliceOp(
            a._value,
            output,
            [],
            [],
            [],
            offsets,
            sizes,
            strides,
            loc=loc,
        ).result
        offset += a.shape[axis]

    return _make_handle(output, out_shape, elem)


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


def broadcast_to(x: TensorHandle, shape: tuple, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, xs, elem = x._value, x.shape, x._elem_ty
    ts = tuple(shape)
    if xs == ts:
        return x
    rt = ranked_tensor_of(ts, elem)
    out = make_empty(loc, ts, elem)
    im = _broadcast_indexing_map(xs, ts)
    om = ir.AffineMap.get_identity(len(ts))
    g = linalg.GenericOp(
        [rt],
        [val],
        [out],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(im), ir.AffineMapAttr.get(om)]),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(ts)
        ),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(elem, elem)
    with ir.InsertionPoint(blk):
        linalg.YieldOp([blk.arguments[0]], loc=loc)
    return _make_handle(g.results[0], ts, elem)


# ---------------------------------------------------------------------------
# Where (conditional select)
# ---------------------------------------------------------------------------


def where(condition, x, y, loc=None) -> TensorHandle:
    cond_is_t = isinstance(condition, TensorHandle)
    x_is_t = isinstance(x, TensorHandle)
    y_is_t = isinstance(y, TensorHandle)

    ref = x if x_is_t else y if y_is_t else None
    if ref is None:
        raise TypeError("where requires at least one tensor for x or y")
    out_elem = ref._elem_ty

    needs_cast = cond_is_t and _is_int(condition._elem_ty) and _is_float(out_elem)

    # cond * x + (1 - cond) * y — works when condition values are 0/1.
    if not needs_cast:
        cx = multiply(condition, x, loc=loc)
        inv = subtract(1, condition, loc=loc)
        iy = multiply(inv, y, loc=loc)
        return add(cx, iy, loc=loc)

    # Integer condition with float output: fuse cast + select into one generic.
    loc = loc or _loc()

    shapes = [condition.shape]
    if x_is_t:
        shapes.append(x.shape)
    if y_is_t:
        shapes.append(y.shape)
    out_shape = shapes[0]
    for s in shapes[1:]:
        out_shape = _broadcast_shape(out_shape, s)

    rt = ranked_tensor_of(out_shape, out_elem)
    out = make_empty(loc, out_shape, out_elem)

    inputs = [condition._value]
    in_maps = [_broadcast_indexing_map(condition.shape, out_shape)]
    in_elem_tys = [condition._elem_ty]

    if x_is_t:
        inputs.append(x._value)
        in_maps.append(_broadcast_indexing_map(x.shape, out_shape))
        in_elem_tys.append(x._elem_ty)
    if y_is_t:
        inputs.append(y._value)
        in_maps.append(_broadcast_indexing_map(y.shape, out_shape))
        in_elem_tys.append(y._elem_ty)

    om = ir.AffineMap.get_identity(len(out_shape))
    all_maps = [ir.AffineMapAttr.get(m) for m in in_maps]
    all_maps.append(ir.AffineMapAttr.get(om))

    g = linalg.GenericOp(
        [rt],
        inputs,
        [out],
        ir.ArrayAttr.get(all_maps),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(out_shape)
        ),
        loc=loc,
    )

    blk_types = in_elem_tys + [out_elem]
    blk = g.regions[0].blocks.append(*blk_types)
    with ir.InsertionPoint(blk):
        tensor_args = iter(blk.arguments[1:])
        ci = blk.arguments[0]
        cf = arith.SIToFPOp(out_elem, ci, loc=loc).result
        xv = next(tensor_args) if x_is_t else const_scalar(x, out_elem, loc)
        yv = next(tensor_args) if y_is_t else const_scalar(y, out_elem, loc)
        one = const_scalar(1.0, out_elem, loc)
        inv = arith.SubFOp(one, cf, loc=loc).result
        t1 = arith.MulFOp(cf, xv, loc=loc).result
        t2 = arith.MulFOp(inv, yv, loc=loc).result
        r = arith.AddFOp(t1, t2, loc=loc).result
        linalg.YieldOp([r], loc=loc)

    return _make_handle(g.results[0], tuple(out_shape), out_elem)


# ---------------------------------------------------------------------------
# Take (gather)
# ---------------------------------------------------------------------------


def take(a: TensorHandle, indices: TensorHandle, axis: int = 0, loc=None) -> TensorHandle:
    loc = loc or _loc()
    av, a_shape, a_elem = a._value, a.shape, a._elem_ty
    iv, i_shape, i_elem = indices._value, indices.shape, indices._elem_ty

    if axis != 0:
        raise NotImplementedError("Only axis=0 gather is currently supported")

    out_shape = i_shape + a_shape[1:]
    rt = ranked_tensor_of(out_shape, a_elem)
    output = make_empty(loc, out_shape, a_elem)

    gather = nkipy_d.GatherOp(rt, av, iv, output, loc=loc)

    src_type = av.type
    idx_type = iv.type
    # Linalg fallback for LLVM JIT; NISA path lowers GatherOp directly to DMA.
    region = gather.reference_impl
    blk = region.blocks.append(src_type, idx_type)
    with ir.InsertionPoint(blk):
        src_arg, idx_arg = blk.arguments
        rank = len(out_shape)
        irank = len(i_shape)
        out2 = make_empty(loc, out_shape, a_elem)

        ie = [ir.AffineDimExpr.get(i) for i in range(irank)]
        im = ir.AffineMap.get(rank, 0, ie)
        om = ir.AffineMap.get_identity(rank)

        g = linalg.GenericOp(
            [rt],
            [idx_arg],
            [out2],
            ir.ArrayAttr.get([ir.AffineMapAttr.get(im), ir.AffineMapAttr.get(om)]),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
            ),
            loc=loc,
        )
        with ir.InsertionPoint(g.regions[0].blocks.append(i_elem, a_elem)):
            index_val = g.regions[0].blocks[0].arguments[0]
            if str(i_elem) != "index":
                index_val = arith.IndexCastOp(
                    ir.IndexType.get(), index_val, loc=loc
                ).result
            ext_idx = [index_val]
            for di in range(1, len(a_shape)):
                ext_idx.append(linalg.IndexOp(irank + di - 1, loc=loc).result)
            extracted = tensor.ExtractOp(src_arg, ext_idx, loc=loc).result
            linalg.YieldOp([extracted], loc=loc)

        nkipy_d.YieldOp(values=[g.results[0]], loc=loc)

    return _make_handle(gather.result, tuple(out_shape), a_elem)


# ---------------------------------------------------------------------------
# Astype (cast)
# ---------------------------------------------------------------------------


def astype(x: TensorHandle, dtype, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, shape, src_elem = x._value, x.shape, x._elem_ty
    dst_elem = _np_to_mlir(dtype)

    if str(src_elem) == str(dst_elem):
        return x

    rt = ranked_tensor_of(shape, dst_elem)
    out = make_empty(loc, shape, dst_elem)
    nd = len(shape)
    imap = ir.AffineMap.get_identity(nd)
    g = linalg.GenericOp(
        [rt],
        [val],
        [out],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(imap)] * 2),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd),
        loc=loc,
    )
    blk = g.regions[0].blocks.append(src_elem, dst_elem)
    with ir.InsertionPoint(blk):
        in_e = blk.arguments[0]
        sf = _is_float(src_elem)
        df = _is_float(dst_elem)
        if sf and df:
            if dst_elem.width > src_elem.width:
                oe = arith.ExtFOp(dst_elem, in_e, loc=loc).result
            elif dst_elem.width < src_elem.width:
                oe = arith.TruncFOp(dst_elem, in_e, loc=loc).result
            else:
                oe = in_e
        elif sf and not df:
            oe = arith.FPToSIOp(dst_elem, in_e, loc=loc).result
        elif not sf and df:
            oe = arith.SIToFPOp(dst_elem, in_e, loc=loc).result
        else:
            if dst_elem.width > src_elem.width:
                oe = arith.ExtSIOp(dst_elem, in_e, loc=loc).result
            elif dst_elem.width < src_elem.width:
                oe = arith.TruncIOp(dst_elem, in_e, loc=loc).result
            else:
                oe = in_e
        linalg.YieldOp([oe], loc=loc)
    return _make_handle(g.results[0], shape, dst_elem)


# ---------------------------------------------------------------------------
# Static / dynamic slicing
# ---------------------------------------------------------------------------


def static_slice(
    x: TensorHandle,
    start_indices,
    limit_indices,
    strides,
    squeeze_dims,
    loc=None,
) -> TensorHandle:
    loc = loc or _loc()
    val, shape, elem = x._value, x.shape, x._elem_ty

    slice_shape = []
    for s, l, st in zip(start_indices, limit_indices, strides):
        slice_shape.append((l - s + st - 1) // st)

    rt = ranked_tensor_of(tuple(slice_shape), elem)
    sliced = tensor.ExtractSliceOp(
        rt,
        val,
        [],
        [],
        [],
        start_indices,
        slice_shape,
        strides,
        loc=loc,
    ).result

    if squeeze_dims:
        out_shape = tuple(s for i, s in enumerate(slice_shape) if i not in squeeze_dims)
        if out_shape != tuple(slice_shape):
            sliced = _emit_reshape(loc, sliced, tuple(slice_shape), out_shape, elem)
            slice_shape = list(out_shape)

    return _make_handle(sliced, tuple(slice_shape), elem)


def _parse_dynamic_indices(indices, shape, loc):
    """Parse a tuple of indices (LoopIndexHandle, slice, int) into static/dynamic components.

    Returns (static_offsets, static_sizes, static_strides, dynamic_offsets, full_indices).
    """
    if not isinstance(indices, tuple):
        indices = (indices,)

    full_indices = list(indices) + [slice(None)] * (len(shape) - len(indices))

    static_offsets = []
    static_sizes = []
    static_strides = []
    dynamic_offsets = []

    DYNAMIC = ir.ShapedType.get_dynamic_size()

    for idx, dim_size in zip(full_indices, shape):
        if isinstance(idx, LoopIndexHandle):
            ov = arith.IndexCastOp(ir.IndexType.get(), idx._value, loc=loc).result
            dynamic_offsets.append(ov)
            static_offsets.append(DYNAMIC)
            static_sizes.append(1)
            static_strides.append(1)
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else dim_size
            step = idx.step if idx.step is not None else 1

            start_is_li = isinstance(start, LoopIndexHandle)
            stop_is_li = isinstance(stop, LoopIndexHandle)

            if start_is_li:
                ov = arith.IndexCastOp(ir.IndexType.get(), start._value, loc=loc).result
                dynamic_offsets.append(ov)
                static_offsets.append(DYNAMIC)
            else:
                static_offsets.append(int(start))

            if start_is_li and stop_is_li and start.mul_factor == stop.mul_factor:
                static_sizes.append(stop.add_offset - start.add_offset)
            elif not start_is_li and not stop_is_li:
                static_sizes.append(int(stop) - int(start))
            else:
                raise NotImplementedError(
                    "Mixed static/dynamic slice sizes not yet supported"
                )

            static_strides.append(int(step))
        elif isinstance(idx, int):
            static_offsets.append(int(idx))
            static_sizes.append(1)
            static_strides.append(1)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    return static_offsets, static_sizes, static_strides, dynamic_offsets, full_indices


def dynamic_slice(x: TensorHandle, indices, loc=None) -> TensorHandle:
    loc = loc or _loc()
    val, shape, elem = x._value, x.shape, x._elem_ty

    static_offsets, static_sizes, static_strides, dynamic_offsets, full_indices = \
        _parse_dynamic_indices(indices, shape, loc)

    DYNAMIC = ir.ShapedType.get_dynamic_size()
    result_shape = []
    for idx, size, stride in zip(full_indices, static_sizes, static_strides):
        if isinstance(idx, LoopIndexHandle) or isinstance(idx, int):
            continue
        if size != DYNAMIC:
            result_shape.append(size // stride if stride > 1 else size)

    rt = ranked_tensor_of(tuple(result_shape), elem)
    extract = tensor.ExtractSliceOp(
        rt,
        val,
        dynamic_offsets,
        [],
        [],
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
    )
    return _make_handle(extract.result, tuple(result_shape), elem)


# ---------------------------------------------------------------------------
# Insert slice (setitem)
# ---------------------------------------------------------------------------


def static_insert_slice(
    dest: TensorHandle,
    src: TensorHandle,
    offsets: list[int],
    sizes: list[int],
    strides: list[int],
    loc=None,
) -> TensorHandle:
    loc = loc or _loc()
    new_tensor = tensor.InsertSliceOp(
        src._value,
        dest._value,
        [],
        [],
        [],
        offsets,
        sizes,
        strides,
        loc=loc,
    ).result
    return _make_handle(new_tensor, dest.shape, dest._elem_ty)


def dynamic_insert_slice(
    dest: TensorHandle,
    src: TensorHandle,
    indices,
    loc=None,
) -> TensorHandle:
    loc = loc or _loc()

    static_offsets, static_sizes, static_strides, dynamic_offsets, _ = \
        _parse_dynamic_indices(indices, dest.shape, loc)

    new_tensor = tensor.InsertSliceOp(
        src._value,
        dest._value,
        dynamic_offsets,
        [],
        [],
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
    ).result
    return _make_handle(new_tensor, dest.shape, dest._elem_ty)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split(x: TensorHandle, sections: int, axis: int = 0, loc=None) -> list[TensorHandle]:
    shape = x.shape
    size = shape[axis]
    if size % sections != 0:
        raise ValueError("array split does not result in an equal division")
    section_size = size // sections
    results = []
    for i in range(sections):
        start_indices = [0] * len(shape)
        start_indices[axis] = i * section_size
        limit_indices = list(shape)
        limit_indices[axis] = (i + 1) * section_size
        strides = [1] * len(shape)
        results.append(static_slice(x, start_indices, limit_indices, strides, [], loc=loc))
    return results


# ---------------------------------------------------------------------------
# Annotations (knob)
# ---------------------------------------------------------------------------


def annotate(
    x: TensorHandle,
    *,
    partition_dim: Optional[int] = None,
    mem_space: Optional[str] = None,
    tile_size: Optional[list[int]] = None,
    reduction_tile: Optional[list[int]] = None,
) -> TensorHandle:
    value = x._value
    defining_op = value.owner
    if defining_op is None:
        return x

    loc = _loc()

    if isinstance(tile_size, int):
        tile_size = [tile_size]
    if isinstance(reduction_tile, int):
        reduction_tile = [reduction_tile]

    valid = {"Hbm", "Psum", "Sbuf", "SharedHbm"}
    if mem_space is not None and mem_space not in valid:
        raise ValueError(f"Invalid mem_space '{mem_space}'. Must be one of: {valid}")

    ms_attr = None
    if mem_space is not None:
        ms_map = {"Hbm": 1, "Psum": 2, "Sbuf": 3, "SharedHbm": 4}
        ms_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), ms_map[mem_space])

    pd_attr = None
    if partition_dim is not None:
        pd_attr = ir.IntegerAttr.get(ir.IntegerType.get_unsigned(32), partition_dim)

    ts_attr = None
    if tile_size is not None:
        ts_attr = ir.DenseI64ArrayAttr.get(tile_size)

    rt_attr = None
    if reduction_tile is not None:
        rt_attr = ir.DenseI64ArrayAttr.get(reduction_tile)

    nkipy_d.AnnotateOp(
        target=value,
        mem_space=ms_attr,
        partition_dim=pd_attr,
        tile_size=ts_attr,
        reduction_tile=rt_attr,
        loc=loc,
    )
    return x


# ---------------------------------------------------------------------------
# Control flow: fori_loop
# ---------------------------------------------------------------------------


def fori_loop(
    lower: int,
    upper: int,
    body_fn: Callable,
    init_handles: list[TensorHandle],
) -> list[TensorHandle]:
    """Build an ``scf.for`` loop with loop-carried tensor accumulators.

    Args:
        lower: inclusive lower bound
        upper: exclusive upper bound
        body_fn: ``(LoopIndexHandle, list[TensorHandle]) -> list[TensorHandle]``
        init_handles: initial accumulator values

    Returns:
        Final accumulator values after all iterations.
    """
    loc = _loc()
    i32 = ir.IntegerType.get_signless(32)
    lb = arith.ConstantOp(i32, lower, loc=loc)
    ub = arith.ConstantOp(i32, upper, loc=loc)
    step = arith.ConstantOp(i32, 1, loc=loc)

    loop_op = scf.ForOp(
        lb.result,
        ub.result,
        step.result,
        [h._value for h in init_handles],
        loc=loc,
    )

    loop_block = loop_op.body
    loop_idx_value = loop_block.arguments[0]
    loop_acc_values = loop_block.arguments[1:]

    loop_idx = LoopIndexHandle(loop_idx_value)

    acc_handles = [
        TensorHandle(av, ih.shape, ih.dtype, ih._elem_ty)
        for av, ih in zip(loop_acc_values, init_handles)
    ]

    with ir.InsertionPoint(loop_block):
        results = body_fn(loop_idx, acc_handles)
        result_values = [r._value for r in results]

        # Rewire linalg ops to use loop accumulators as their output operand.
        # Without this, bufferization can't see the loop-carried dependence
        # and may allocate a fresh buffer instead of updating in place.
        for rv, ia in zip(result_values, loop_acc_values):
            producer = rv.owner
            if not producer.name.startswith("linalg."):
                continue
            if len(list(producer.results)) != 1:
                continue
            operands = list(producer.operands)
            if not operands or operands[-1] == ia:
                continue
            if len(producer.regions) == 1 and len(producer.regions[0].blocks) == 1:
                bb = producer.regions[0].blocks[0]
                ba = list(bb.arguments)
                if ba:
                    oe = ba[-1]
                    used = any(
                        op_arg == oe for op in bb.operations for op_arg in op.operands
                    )
                    if used:
                        continue
            producer.operands[-1] = ia

        scf.YieldOp(result_values, loc=loc)

    return [
        TensorHandle(
            loop_op.results[i],
            init_handles[i].shape,
            init_handles[i].dtype,
            init_handles[i]._elem_ty,
        )
        for i in range(len(init_handles))
    ]


def lift_scalar_to_tensor(val, dtype_hint: str = "float") -> TensorHandle:
    """Lift a Python scalar to a 0-d tensor handle."""
    loc = _loc()
    if dtype_hint == "float" or isinstance(val, float):
        elem = ir.F32Type.get()
    else:
        elem = ir.IntegerType.get_signless(32)
    v = make_filled(loc, (), elem, val)
    return _make_handle(v, (), elem)


# ---------------------------------------------------------------------------
# LoopIndex arithmetic
# ---------------------------------------------------------------------------


def loop_index_mul(idx: LoopIndexHandle, factor: int) -> LoopIndexHandle:
    loc = _loc()
    i32 = ir.IntegerType.get_signless(32)
    cst = arith.ConstantOp(i32, factor, loc=loc).result
    rv = arith.MulIOp(idx._value, cst, loc=loc).result
    return LoopIndexHandle(rv, idx.mul_factor * factor, idx.add_offset * factor)


def loop_index_add(idx: LoopIndexHandle, offset: int) -> LoopIndexHandle:
    loc = _loc()
    i32 = ir.IntegerType.get_signless(32)
    cst = arith.ConstantOp(i32, offset, loc=loc).result
    rv = arith.AddIOp(idx._value, cst, loc=loc).result
    return LoopIndexHandle(rv, idx.mul_factor, idx.add_offset + offset)


def loop_index_add_loop_index(
    a: LoopIndexHandle, b: LoopIndexHandle
) -> LoopIndexHandle:
    loc = _loc()
    rv = arith.AddIOp(a._value, b._value, loc=loc).result
    return LoopIndexHandle(rv, a.mul_factor + b.mul_factor, a.add_offset + b.add_offset)


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


def apply_custom_op(kernel_builder, reference_fn, input_specs, output_specs, args):
    """Compile a kernel_builder function and call it during tracing.

    Handles the nki.compiler.kernel_builder spec translation that was
    previously in nkipy's KernelGenTraceContext.

    Args:
        kernel_builder: NKI kernel_builder function.
        reference_fn: NumPy reference (for fallback).
        input_specs: List of (shape, dtype_str) tuples.
        output_specs: List of (shape, dtype_str) tuples.
        args: Traced tensor arguments to pass to the custom op.

    Returns:
        Result from the custom op call.
    """
    import nki.compiler.kernel_builder as nb
    from nkipy_kernelgen.custom_op import CustomOp

    _dtype_map = {"f32": nb.float32, "f16": nb.float16, "bf16": nb.bfloat16}
    nb_input_specs = {
        f"input_{i}": nb.Tensor(shape, _dtype_map[dtype], nb.shared_hbm)
        for i, (shape, dtype) in enumerate(input_specs)
    }
    nb_output_specs = {
        f"output_{i}": nb.Tensor(shape, _dtype_map[dtype], nb.shared_hbm)
        for i, (shape, dtype) in enumerate(output_specs)
    }
    internal = CustomOp.from_kernel_builder(
        kernel_func=kernel_builder,
        input_specs=nb_input_specs,
        output_specs=nb_output_specs,
        reference_fn=reference_fn,
    )
    return internal(*args)
