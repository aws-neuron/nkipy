"""Emit nki_ir graphs to NKI Kernel Builder.

Walks an nki_ir graph and directly invokes Kernel Builder API calls
inside a KB tracing context, producing NISA MLIR.

The main entry point is ``build_kb_kernel(graph)`` which returns a
kernel function suitable for ``nb.build_kernel()`` or
``nb.compile_and_execute()``.

Example usage:
    import nki.compiler.kernel_builder as nb
    from nki.compiler.kernel_builder import Tensor
    from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel
    from nkigen_lite.nki_ir.examples import lower_softmax

    graph = lower_softmax(256, 512)
    kernel_fn = build_kb_kernel(graph)

    module = nb.build_kernel(
        kernel_fn,
        input_specs={"x": Tensor((256, 512), nb.float32)},
        output_specs={"y": Tensor((256, 512), nb.float32)},
        target="trn2",
    )
    print(module)
"""

from __future__ import annotations

import numpy as np

from nkigen_lite.core import DType, Op, Value
from nkigen_lite.nki_ir.ir import (
    Graph,
    MemorySpace,
    NisaActivationOp,
    NisaArithOp,
    NisaBitvecOp,
    NisaRangeSelectCmp,
    NisaReduceOp,
)

import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor, isa as nisa


# ===========================
# nki_ir → KB mapping tables
# ===========================

_DTYPE_TO_KB = {
    DType.F32: nb.float32,
    DType.F16: nb.float16,
    DType.BF16: nb.bfloat16,
    DType.TF32: nb.tfloat32,
    DType.FP8_E4M3: nb.float8_e4m3fn,
    DType.FP8_E4M3_IEEE: nb.float8_e4m3,
    DType.FP8_E5M2: nb.float8_e5m2,
    DType.FP8_E3M4: nb.float8_e3m4,
    DType.I32: nb.int32,
    DType.I16: nb.int16,
    DType.I8: nb.int8,
    DType.U32: nb.uint32,
    DType.U16: nb.uint16,
    DType.U8: nb.uint8,
    DType.BOOL: nb.uint8,
}

_MEMSPACE_TO_KB = {
    MemorySpace.SBUF: nb.sbuf,
    MemorySpace.PSUM: nb.psum,
    MemorySpace.HBM: nb.hbm,
}

_ACTIVATION_TO_KB = {
    NisaActivationOp.EXP: nisa.activation_function.exp,
    NisaActivationOp.LOG: nisa.activation_function.log,
    NisaActivationOp.SQRT: nisa.activation_function.sqrt,
    NisaActivationOp.RSQRT: nisa.activation_function.rsqrt,
    NisaActivationOp.TANH: nisa.activation_function.tanh,
    NisaActivationOp.SIGMOID: nisa.activation_function.sigmoid,
    NisaActivationOp.RELU: nisa.activation_function.relu,
    NisaActivationOp.GELU: nisa.activation_function.gelu,
    NisaActivationOp.SILU: nisa.activation_function.silu,
    NisaActivationOp.SIN: nisa.activation_function.sin,
    NisaActivationOp.RECIPROCAL: nisa.activation_function.reciprocal,
    NisaActivationOp.ABS: nisa.activation_function.abs,
    NisaActivationOp.SQUARE: nisa.activation_function.square,
    NisaActivationOp.SIGN: nisa.activation_function.sign,
    NisaActivationOp.COPY: nisa.activation_function.copy,
    NisaActivationOp.ARCTAN: nisa.activation_function.arctan,
    NisaActivationOp.ERF: nisa.activation_function.erf,
    NisaActivationOp.SOFTPLUS: nisa.activation_function.softplus,
    NisaActivationOp.MISH: nisa.activation_function.mish,
}

_ARITH_TO_KB = {
    NisaArithOp.ADD: nisa.arith_op.Add,
    NisaArithOp.SUBTRACT: nisa.arith_op.Subtract,
    NisaArithOp.MULTIPLY: nisa.arith_op.Multiply,
    NisaArithOp.MAXIMUM: nisa.arith_op.Max,
    NisaArithOp.MINIMUM: nisa.arith_op.Min,
    NisaArithOp.POW: nisa.arith_op.Pow,
    NisaArithOp.IS_GT: nisa.arith_op.IsGT,
    NisaArithOp.IS_GE: nisa.arith_op.IsGE,
    NisaArithOp.IS_LT: nisa.arith_op.IsLT,
    NisaArithOp.IS_LE: nisa.arith_op.IsLE,
    NisaArithOp.IS_EQ: nisa.arith_op.IsEQ,
    NisaArithOp.IS_NE: nisa.arith_op.IsNE,
    NisaArithOp.LOGICAL_XOR: nisa.arith_op.LogicalXor,
    NisaArithOp.LOGICAL_AND: nisa.arith_op.LogicalAnd,
    NisaArithOp.LOGICAL_OR: nisa.arith_op.LogicalOr,
}

_REDUCE_TO_KB = {
    NisaReduceOp.ADD: nisa.arith_op.Add,
    NisaReduceOp.MAX: nisa.arith_op.Max,
    NisaReduceOp.MIN: nisa.arith_op.Min,
}

_ACTIVATION_REDUCE_TO_KB = {
    NisaReduceOp.ADD: nisa.activation_reduce_op.Add,
    NisaReduceOp.MAX: nisa.activation_reduce_op.Max,
    NisaReduceOp.MIN: nisa.activation_reduce_op.Min,
}

_PARTITION_REDUCE_TO_KB = {
    NisaReduceOp.ADD: nisa.cross_lane_reduce_arith_op.Add,
    NisaReduceOp.MAX: nisa.cross_lane_reduce_arith_op.Max,
}

_RANGE_CMP_TO_KB = {
    NisaRangeSelectCmp.IS_EQ: nisa.range_select_cmp.IsEq,
    NisaRangeSelectCmp.IS_GT: nisa.range_select_cmp.IsGt,
    NisaRangeSelectCmp.IS_GE: nisa.range_select_cmp.IsGe,
    NisaRangeSelectCmp.IS_LE: nisa.range_select_cmp.IsLe,
    NisaRangeSelectCmp.IS_LT: nisa.range_select_cmp.IsLt,
}

_BITVEC_TO_KB = {
    NisaBitvecOp.AND: nisa.bitvec_op.BitwiseAnd,
    NisaBitvecOp.OR: nisa.bitvec_op.BitwiseOr,
    NisaBitvecOp.XOR: nisa.bitvec_op.BitwiseXor,
    NisaBitvecOp.NOT: nisa.bitvec_op.BitwiseNot,
}


# ===========================
# Graph walker
# ===========================

def _emit_graph(graph: Graph, tiles: dict[str, object]) -> None:
    """Walk all ops in the graph and emit KB API calls.

    ``tiles`` maps nki_ir Value names to KB TileView objects.
    HBM inputs must be pre-populated by the caller.
    """
    for op in graph.ops:
        _emit_op(op, tiles)


def _emit_op(op: Op, tiles: dict[str, object]) -> None:
    """Emit KB API calls for a single nki_ir op."""

    def _get(v: Value):
        return tiles[v.name]

    def _alloc(v: Value, num_buffers: int = 1):
        tt = v.type
        t = nb.compiler.alloc(
            tt.shape, _DTYPE_TO_KB[tt.dtype], space=_MEMSPACE_TO_KB[tt.memory],
            num_buffers=num_buffers,
        )
        tiles[v.name] = t
        return t

    if op.opcode == "scalar_const":
        value = op.attrs["value"]
        tile = nb.compiler.alloc((1, 1), nb.int32, space=nb.sbuf)
        nisa.memset(dst=tile, value=float(value))
        reg = nisa.load_register(tile[0:1, 0])
        tiles[op.result.name] = reg

    elif op.opcode == "affine":
        scale = op.attrs["scale"]
        base = op.attrs["base"]
        idx = _get(op.inputs[0])
        tiles[op.result.name] = base + idx * scale

    elif op.opcode == "scalar_add":
        a = _get(op.inputs[0])
        b = _get(op.inputs[1])
        tiles[op.result.name] = a + b

    elif op.opcode == "dma_copy":
        direction = op.attrs["direction"]
        strides = op.attrs.get("strides")
        sizes = op.attrs.get("sizes")
        if direction == "load":
            dst = _get(op.inputs[0])
            src_hbm = _get(op.inputs[1])
            tile_shape = op.result.type.shape
            hbm_rank = op.inputs[1].type.rank
        else:
            src = _get(op.inputs[0])
            dst_hbm = _get(op.inputs[1])
            tile_shape = op.inputs[0].type.shape
            hbm_rank = op.inputs[1].type.rank
        if op.attrs.get("dynamic_offsets"):
            offsets = [_get(v) for v in op.inputs[2:]]
        else:
            offsets = list(op.attrs["offsets"])

        if strides and any(s != 1 for s in strides):
            # Strided DMA: use coords-based affine indexing (only
            # works when slice_sizes match the strided rank exactly).
            slice_sizes = list(sizes) if sizes is not None else list(tile_shape)
            stride_used = list(strides)[-len(slice_sizes):]
            off_used = list(offsets)[-len(slice_sizes):]
            coords = nb.coords(*slice_sizes)
            index_exprs = tuple(
                off + c * s for off, c, s in zip(off_used, coords, stride_used)
            )
            if direction == "load":
                nisa.dma_copy(dst=dst, src=src_hbm[index_exprs])
                tiles[op.result.name] = dst
            else:
                nisa.dma_copy(dst=dst_hbm[index_exprs], src=src)
            return

        slice_expr = _build_kb_slices(
            sizes, offsets, strides, tile_shape, hbm_rank,
        )
        if direction == "load":
            nisa.dma_copy(dst=dst, src=src_hbm[slice_expr])
            tiles[op.result.name] = dst
        else:
            nisa.dma_copy(dst=dst_hbm[slice_expr], src=src)

    elif op.opcode == "access_pattern":
        src = _get(op.inputs[0])
        pattern = op.attrs["pattern"]

        # Resolve offset (static int or dynamic Reg)
        input_idx = 1
        if op.attrs.get("dynamic_offset"):
            offset = _get(op.inputs[input_idx])
            input_idx += 1
        else:
            offset = op.attrs.get("offset", 0)

        # Resolve register_offsets
        register_offsets = None
        reg_mask = op.attrs.get("register_offsets")
        if reg_mask is not None:
            register_offsets = []
            for has_reg in reg_mask:
                if has_reg:
                    register_offsets.append(_get(op.inputs[input_idx]))
                    input_idx += 1
                else:
                    register_offsets.append(None)
            register_offsets = tuple(register_offsets)

        # Resolve vector_offset
        vector_offset = None
        if op.attrs.get("vector_offset"):
            vector_offset = _get(op.inputs[input_idx])

        tiles[op.result.name] = src.ap(
            pattern, offset=offset,
            register_offsets=register_offsets,
            vector_offset=vector_offset,
        )

    elif op.opcode == "tensor_copy":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        nisa.tensor_copy(dst=dst, src=src)
        tiles[op.result.name] = dst

    elif op.opcode == "alloc":
        _alloc(op.result, num_buffers=op.attrs.get("num_buffers", 1) if op.attrs else 1)

    elif op.opcode == "rotate":
        src = _get(op.inputs[0])
        tiles[op.result.name] = nb.compiler.rotate(src)

    elif op.opcode == "dealloc":
        nb.compiler.release(_get(op.inputs[0]))

    elif op.opcode == "constant":
        dst = _alloc(op.result)
        nisa.memset(dst=dst, value=op.attrs["value"])

    elif op.opcode == "matmul":
        dst = _get(op.inputs[0])
        stat = _get(op.inputs[1])
        mov = _get(op.inputs[2])
        accum = bool(op.attrs.get("accumulate", False))
        is_transpose = bool(op.attrs.get("is_transpose", False))
        nisa.matmul(dst=dst, stationary=stat, moving=mov, accum=accum,
                    is_transpose=is_transpose)
        tiles[op.result.name] = dst

    elif op.opcode == "activation":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        act = _ACTIVATION_TO_KB[op.attrs["op"]]
        scale = op.attrs.get("scale", 1.0)
        has_reduce = "reduce_op" in op.attrs
        if has_reduce:
            reduce_dst = _get(op.inputs[-1])
            num_extra = len(op.inputs) - 3
            bias = _get(op.inputs[2]) if num_extra > 0 else 0.0
            nisa.activation(
                dst=dst, src=x, bias=bias, scale=scale, op=act,
                reduce_res=reduce_dst,
                reduce_op=_ACTIVATION_REDUCE_TO_KB[op.attrs["reduce_op"]],
                reduce_cmd=nisa.reduce_cmd.ResetReduce,
            )
            tiles[op.result.name] = dst
        else:
            bias = _get(op.inputs[2]) if len(op.inputs) > 2 else 0.0
            nisa.activation(dst=dst, src=x, bias=bias, scale=scale, op=act)
            tiles[op.result.name] = dst

    elif op.opcode == "activation_reduce":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        act = _ACTIVATION_TO_KB[op.attrs["act_op"]]
        reduce_op = _ACTIVATION_REDUCE_TO_KB[op.attrs["reduce_op"]]
        nisa.activation(
            dst=dst, src=x, bias=0.0, scale=1.0, op=act,
            reduce_op=reduce_op, reduce_res=dst,
            reduce_cmd=nisa.reduce_cmd.ResetReduce,
        )
        tiles[op.result.name] = dst

    elif op.opcode == "tensor_tensor_arith":
        dst = _get(op.inputs[0])
        a = _get(op.inputs[1])
        b = _get(op.inputs[2])
        nisa.tensor_tensor_arith(
            dst=dst, lhs=a, rhs=b, op=_ARITH_TO_KB[op.attrs["op"]],
        )
        tiles[op.result.name] = dst

    elif op.opcode == "tensor_tensor_bitvec":
        dst = _get(op.inputs[0])
        a = _get(op.inputs[1])
        b = _get(op.inputs[2])
        nisa.tensor_tensor_bitvec(
            dst=dst, lhs=a, rhs=b, op=_BITVEC_TO_KB[op.attrs["op"]],
        )
        tiles[op.result.name] = dst

    elif op.opcode == "tensor_scalar_bitvec":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        operand0 = _get(op.inputs[2])
        op0 = _BITVEC_TO_KB[op.attrs["op0"]]
        nisa.tensor_scalar_bitvec(dst=dst, src=x, operand0=operand0, op0=op0)
        tiles[op.result.name] = dst

    elif op.opcode == "tensor_scalar_arith":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        operand0 = _get(op.inputs[2])
        op0 = _ARITH_TO_KB[op.attrs.get("op0") or op.attrs.get("op")]
        kwargs = {}
        if "op1" in op.attrs and len(op.inputs) > 3:
            kwargs["operand1"] = _get(op.inputs[3])
            kwargs["op1"] = _ARITH_TO_KB[op.attrs["op1"]]
        if op.attrs.get("reverse_operands"):
            kwargs["reverse_operands"] = nisa.tens_scalar_rev_ops.None_
        # tensor_scalar_arith requires f32; upcast if needed
        needs_cast = (op.inputs[1].type.dtype != DType.F32)
        if needs_cast:
            x_f32 = nb.compiler.alloc(
                op.inputs[1].type.shape, nb.float32, space=nb.sbuf)
            nisa.tensor_copy(dst=x_f32, src=x)
            op0_f32 = nb.compiler.alloc(
                op.inputs[2].type.shape, nb.float32, space=nb.sbuf)
            nisa.tensor_copy(dst=op0_f32, src=operand0)
            if "operand1" in kwargs:
                op1_orig = kwargs["operand1"]
                op1_f32 = nb.compiler.alloc(
                    op.inputs[3].type.shape, nb.float32, space=nb.sbuf)
                nisa.tensor_copy(dst=op1_f32, src=op1_orig)
                kwargs["operand1"] = op1_f32
            dst_f32 = nb.compiler.alloc(
                op.inputs[0].type.shape, nb.float32, space=nb.sbuf)
            nisa.tensor_scalar_arith(
                dst=dst_f32, src=x_f32, operand0=op0_f32, op0=op0, **kwargs,
            )
            nisa.tensor_copy(dst=dst, src=dst_f32)
        else:
            nisa.tensor_scalar_arith(
                dst=dst, src=x, operand0=operand0, op0=op0, **kwargs,
            )
        tiles[op.result.name] = dst

    elif op.opcode == "scalar_tensor_tensor_arith":
        dst = _get(op.inputs[0])
        src0 = _get(op.inputs[1])
        src1 = _get(op.inputs[2])
        imm0 = _get(op.inputs[3])
        nisa.scalar_tensor_tensor_arith(
            dst=dst, src0=src0, src1=src1, imm0=imm0,
            op0=_ARITH_TO_KB[op.attrs["op0"]],
            op1=_ARITH_TO_KB[op.attrs["op1"]],
        )
        tiles[op.result.name] = dst

    elif op.opcode == "tensor_reduce_arith":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        num_r_dim = op.attrs.get("num_r_dim") or sum(1 for a in op.attrs.get("axis", ()) if a >= 1)
        nisa.tensor_reduce_arith(
            dst=dst, src=x, op=_REDUCE_TO_KB[op.attrs["op"]],
            num_r_dim=num_r_dim,
        )
        tiles[op.result.name] = dst

    elif op.opcode == "cross_lane_reduce_arith":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        nisa.cross_lane_reduce_arith(
            dst=dst, src=x,
            reduce_op=_PARTITION_REDUCE_TO_KB[op.attrs["op"]],
            num_r_dim=0,
        )
        tiles[op.result.name] = dst

    elif op.opcode == "iota":
        dst = _get(op.inputs[0])
        pattern = op.attrs.get("pattern", [[1, op.result.type.shape[-1]]])
        offset = op.attrs.get("offset", 0)
        ch_mul = op.attrs.get("channel_multiplier", 0)
        nisa.iota(dst=dst, pattern=pattern, offset=offset, channel_multiplier=ch_mul)
        tiles[op.result.name] = dst

    elif op.opcode == "stream_shuffle":
        dst = _get(op.inputs[0])
        x = _get(op.inputs[1])
        nisa.stream_shuffle(dst=dst, src=x, shuffle_mask=op.attrs["shuffle_mask"])
        tiles[op.result.name] = dst

    elif op.opcode == "dma_transpose":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        perm = op.attrs["perm"]
        nisa.dma_transpose(dst=dst, src=src, permutation=list(perm))
        tiles[op.result.name] = dst

    elif op.opcode == "stream_transpose":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        nisa.stream_transpose(dst=dst, src=src)
        tiles[op.result.name] = dst

    elif op.opcode in ("broadcast", "reshape"):
        tiles[op.result.name] = _get(op.inputs[0])

    elif op.opcode == "view":
        src = _get(op.inputs[0])
        new_shape = op.attrs["shape"]
        out_dtype = op.attrs["dtype"]
        kb_dtype = _DTYPE_TO_KB.get(out_dtype)
        tiles[op.result.name] = src.view(new_shape, dtype=kb_dtype)

    elif op.opcode == "cast":
        src = _get(op.inputs[0])
        dst = _alloc(op.result)
        nisa.tensor_copy(dst=dst, src=src)

    elif op.opcode == "memset":
        tile = _get(op.inputs[0])
        nisa.memset(dst=tile, value=op.attrs["value"])
        tiles[op.result.name] = tile

    elif op.opcode in ("fori_loop", "tile_loop"):
        _emit_tile_loop(op, tiles)

    elif op.opcode == "if_else":
        cond = _get(op.inputs[0])
        then_body = op.attrs["then_body"]
        else_body = op.attrs.get("else_body")

        def then_fn():
            inner = dict(tiles)
            for body_op in then_body.ops:
                _emit_op(body_op, inner)
            tiles.update(inner)

        if else_body is not None:
            def else_fn():
                inner = dict(tiles)
                for body_op in else_body.ops:
                    _emit_op(body_op, inner)
                tiles.update(inner)
            nb.if_else(cond, then_fn, else_fn)
        else:
            nb.if_else(cond, then_fn)

    elif op.opcode == "while_loop":
        cond_body = op.attrs["cond_body"]
        body_body = op.attrs["body_body"]
        init_val = _get(op.inputs[0])

        carry_state = [init_val]

        def cond_fn(r):
            inner = dict(tiles)
            inner[cond_body.inputs[0].name] = r
            for body_op in cond_body.ops:
                _emit_op(body_op, inner)
            cond_val = inner[cond_body.output_values[0].name]
            out_val = inner[cond_body.output_values[1].name]
            return cond_val, out_val

        def body_fn(r):
            inner = dict(tiles)
            inner[body_body.inputs[0].name] = r
            for body_op in body_body.ops:
                _emit_op(body_op, inner)
            return inner[body_body.output_values[0].name]

        result = nb.while_loop(init_val, cond_fn, body_fn)
        tiles[op.result.name] = result

    elif op.opcode == "reg_compare":
        a = _get(op.inputs[0])
        b = _get(op.inputs[1])
        cmp = op.attrs["op"]
        if cmp == "<":
            tiles[op.result.name] = a < b
        elif cmp == "<=":
            tiles[op.result.name] = a <= b
        elif cmp == ">":
            tiles[op.result.name] = a > b
        elif cmp == ">=":
            tiles[op.result.name] = a >= b
        elif cmp == "!=":
            tiles[op.result.name] = a != b

    elif op.opcode == "load_register":
        src = _get(op.inputs[0])
        tiles[op.result.name] = nisa.load_register(src[0])

    elif op.opcode == "store_register":
        dst = _get(op.inputs[0])
        reg = _get(op.inputs[1])
        nisa.store_register(dst[0], reg)
        tiles[op.result.name] = dst

    elif op.opcode == "affine_select":
        dst = _get(op.inputs[0])
        pred = _get(op.inputs[1])
        on_true = _get(op.inputs[2])
        on_false = _get(op.inputs[3])
        nisa.tensor_copy(dst=dst, src=on_false)
        pred_type = op.inputs[1].type
        if pred_type.dtype not in (DType.U8, DType.U16, DType.U32):
            pred_u8 = nb.compiler.alloc(pred_type.shape, nb.uint8, space=nb.sbuf)
            nisa.tensor_copy(dst=pred_u8, src=pred)
            nisa.copy_predicated(dst=dst, pred_mask=pred_u8, src=on_true)
        else:
            nisa.copy_predicated(dst=dst, pred_mask=pred, src=on_true)
        tiles[op.result.name] = dst

    elif op.opcode == "dma_copy_indirect":
        direction = op.attrs["direction"]
        if direction == "load":
            dst = _get(op.inputs[0])
            src = _get(op.inputs[1])
            index = _get(op.inputs[2])
            nisa.dma_copy_indirect(dst=dst, src=src, src_index=index)
            tiles[op.result.name] = dst
        else:
            src = _get(op.inputs[0])
            dst = _get(op.inputs[1])
            index = _get(op.inputs[2])
            nisa.dma_copy_indirect(dst=dst, src=src, dst_index=index)

    elif op.opcode == "tensor_tensor_scan":
        dst = _get(op.inputs[0])
        data0 = _get(op.inputs[1])
        data1 = _get(op.inputs[2])
        initial = _get(op.inputs[3])
        nisa.tensor_tensor_scan(dst=dst, src0=data0, src1=data1,
                                imm0=initial,
                                op0=_ARITH_TO_KB[op.attrs["op0"]],
                                op1=_ARITH_TO_KB[op.attrs["op1"]])
        tiles[op.result.name] = dst

    elif op.opcode == "sequence_bounds":
        dst = _get(op.inputs[0])
        segment_ids = _get(op.inputs[1])
        nisa.sequence_bounds(dst=dst, src=segment_ids)
        tiles[op.result.name] = dst

    elif op.opcode == "dma_gather_transpose":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        index = _get(op.inputs[2])
        nisa.dma_gather_transpose(dst=dst, src=src, gather_index=index)
        tiles[op.result.name] = dst

    elif op.opcode == "copy_predicated":
        dst = _get(op.inputs[0])
        pred = _get(op.inputs[1])
        src = _get(op.inputs[2])
        nisa.copy_predicated(dst=dst, pred_mask=pred, src=src)
        tiles[op.result.name] = dst

    elif op.opcode == "gather":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        indices = _get(op.inputs[2])
        nisa.gather(dst=dst, src=src, indices=indices)
        tiles[op.result.name] = dst

    elif op.opcode == "exponential":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        kwargs = {}
        if len(op.inputs) > 2:
            kwargs["max_value"] = _get(op.inputs[2])
        nisa.exponential(dst=dst, src=src, **kwargs)
        tiles[op.result.name] = dst

    elif op.opcode == "range_select":
        dst = _get(op.inputs[0])
        src = _get(op.inputs[1])
        bound0 = _get(op.inputs[2])
        bound1 = _get(op.inputs[3])
        nisa.range_select(
            dst=dst, src=src, bound0=bound0, bound1=bound1,
            fill_value=op.attrs["fill_value"],
            comp_op0=_RANGE_CMP_TO_KB[op.attrs["comp_op0"]],
            comp_op1=_RANGE_CMP_TO_KB[op.attrs["comp_op1"]],
        )
        tiles[op.result.name] = dst

    elif op.opcode == "select_reduce":
        dst = _get(op.inputs[0])
        pred = _get(op.inputs[1])
        on_true = _get(op.inputs[2])
        on_false_scalar = op.attrs.get("on_false_scalar")
        kwargs = {}
        if on_false_scalar is not None:
            kwargs["on_false"] = np.float32(on_false_scalar)
        if "reduce_op" in op.attrs:
            reduce_dst = _get(op.inputs[-1])
            kwargs["reduce_res"] = reduce_dst
            kwargs["reduce_cmd"] = nisa.reduce_cmd.ResetReduce
            kwargs["reduce_op"] = _REDUCE_TO_KB[op.attrs["reduce_op"]]
        nisa.select_reduce(dst=dst, predicate=pred, on_true=on_true, **kwargs)
        tiles[op.result.name] = dst

    elif op.opcode in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
        _emit_collective(op, tiles)

    else:
        raise NotImplementedError(f"Unhandled nki_ir opcode: {op.opcode!r}")


# Map nkigen_lite collective reduce-op names to KB dma_compute_reduce_op.
_COLLECTIVE_REDUCE_TO_KB = {
    "add": "Add",
    "max": "Max",
    "min": "Min",
    "multiply": "Multiply",
}


def _to_cc_dim(dim: int):
    """Convert an integer collective dim to the KB CollectiveDimension enum.

    The KB nisa collective APIs forward ``cc_dim`` to the native builder
    un-converted, which raises ``std::bad_cast`` on a bare int — the enum
    must be passed explicitly.
    """
    from nki.compiler._internal.dialects.nisa import CollectiveDimension

    mapping = {0: CollectiveDimension.DIM_0, 1: CollectiveDimension.DIM_1}
    if dim not in mapping:
        raise NotImplementedError(f"unsupported collective dim {dim}")
    return mapping[dim]


def _emit_collective(op: Op, tiles: dict[str, object]) -> None:
    """Emit a collective op (HBM->HBM) as a KB nisa collective call.

    inputs are [dst_hbm, src_hbm]; both are pre-allocated HBM TileViews.
    The replica group comes through verbatim from the tensor_ir op.
    """
    from nki.compiler._internal.dialects import nisa as nisa_dialect

    dst = tiles[op.inputs[0].name]
    src = tiles[op.inputs[1].name]
    replica_groups = [list(g) for g in op.attrs["replica_groups"]]
    replica_group_attr = nisa_dialect.ExplicitReplicaGroupAttr.get(replica_groups)

    def _reduce_op():
        name = _COLLECTIVE_REDUCE_TO_KB[op.attrs.get("reduce_op", "add")]
        return getattr(nisa.dma_compute_reduce_op, name)

    if op.opcode == "all_reduce":
        nisa.all_reduce(
            dsts=dst, srcs=src,
            reduce_op=_reduce_op(), replica_group=replica_group_attr,
        )
    elif op.opcode == "all_gather":
        nisa.all_gather(
            dsts=dst, srcs=src,
            replica_group=replica_group_attr,
            cc_dim=_to_cc_dim(op.attrs["all_gather_dim"]),
        )
    elif op.opcode == "reduce_scatter":
        nisa.reduce_scatter(
            dsts=dst, srcs=src,
            reduce_op=_reduce_op(), replica_group=replica_group_attr,
            cc_dim=_to_cc_dim(op.attrs["reduce_scatter_dim"]),
        )
    elif op.opcode == "all_to_all":
        nisa.all_to_all(
            dsts=dst, srcs=src,
            replica_group=replica_group_attr,
            cc_dim=_to_cc_dim(op.attrs["split_dimension"]),
        )


def _emit_tile_loop(op: Op, tiles: dict[str, object]) -> None:
    """Emit a loop as ``nb.fori_loop``.

    The body graph is walked inside the fori_loop callback, so KB
    traces the body ops into an ``scf.for`` MLIR region.

    For fori_loop: no carries. Body captures HBM from outer scope.
    Extent may be static (int) or dynamic (register Value).
    For tile_loop (legacy, from tiling pass): carries map to in-place
    mutation of on-chip tiles.
    """
    body_graph = op.attrs["body"]
    static_extent = op.attrs["extent"]

    if op.opcode == "fori_loop":
        if static_extent is not None:
            loop_bound = static_extent
        else:
            loop_bound = tiles[op.inputs[0].name]

        def body_fn(i_reg):
            inner = dict(tiles)
            inner[body_graph.inputs[0].name] = i_reg
            for body_op in body_graph.ops:
                _emit_op(body_op, inner)

        nb.fori_loop(loop_bound, body_fn)
    else:
        carried_init = [tiles[v.name] for v in op.inputs]
        carry_state = list(carried_init)

        def body_fn(i_reg):
            inner = dict(tiles)
            inner[body_graph.inputs[0].name] = i_reg
            for j, ph in enumerate(body_graph.inputs[1:]):
                inner[ph.name] = carry_state[j]
            for body_op in body_graph.ops:
                _emit_op(body_op, inner)
            for j, out_val in enumerate(body_graph.output_values):
                carry_state[j] = inner[out_val.name]

        nb.fori_loop(static_extent, body_fn)

        for j, result_val in enumerate(op.results):
            tiles[result_val.name] = carry_state[j]


# ===========================
# Public API
# ===========================

def _build_kb_slices(
    sizes_attr,
    offsets,
    strides,
    tile_shape,
    hbm_rank: int,
):
    """Build a kb slice expression for ``hbm[expr]`` matching the
    on-chip tile rank.

    For a rank-N HBM with a 2D on-chip tile, the leading
    ``(N - 2)`` dims of the slice should be **bare ints/Values**
    (single-element selection), and only the trailing 2 entries
    should be ``DynamicSlice`` objects describing the partition/free
    extents. This matches kb's rank-aware interpretation of the
    indexing expression.
    """
    on_chip_rank = len(tile_shape)
    if sizes_attr is None:
        # Legacy: zip-truncate against tile_shape.
        slices = tuple(
            nb.ds(off, ext) for off, ext in zip(offsets, tile_shape)
        )
        return slices

    sizes = list(sizes_attr)
    offs = list(offsets)
    if strides is None:
        strides = [1] * len(sizes)
    else:
        strides = list(strides)

    # Leading dims that are size 1: emit as bare offsets (kb selects a
    # single element). Trailing on_chip_rank dims: emit as DynamicSlice.
    expr: list = []
    n_lead = max(0, len(sizes) - on_chip_rank)
    for i in range(n_lead):
        if sizes[i] != 1:
            # Can't express larger-than-1 leading dims with bare int —
            # fall back to DynamicSlice (kb may handle it).
            expr.append(nb.ds(offs[i], sizes[i]))
        else:
            expr.append(offs[i])
    for i in range(n_lead, len(sizes)):
        if strides[i] != 1:
            # Strided trailing slice: caller will switch to nb.coords.
            return None
        expr.append(nb.ds(offs[i], sizes[i]))
    return tuple(expr)


def build_kb_kernel(graph: Graph):
    """Build a KB kernel function from an nki_ir graph.

    Returns a kernel function whose signature matches the graph's HBM
    inputs (annotated with ``: Tensor``). Pass it to ``nb.build_kernel``
    or ``nb.compile_and_execute``.

    The graph may contain fori_loop ops — these are lowered to
    ``nb.fori_loop`` (scf.for in MLIR).
    """
    hbm_inputs = list(graph.inputs)
    param_names = [v.name for v in hbm_inputs]

    def kernel_fn(**kwargs):
        tiles: dict[str, object] = {}
        for v in hbm_inputs:
            tiles[v.name] = kwargs[v.name]
        _emit_graph(graph, tiles)

    kernel_fn.__name__ = graph.name
    kernel_fn.__qualname__ = graph.name
    kernel_fn.__annotations__ = {name: Tensor for name in param_names}

    return kernel_fn
