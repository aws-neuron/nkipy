"""Numpy interpreter for nki_ir graphs.

Executes a tile-level NKI IR graph using numpy, providing a reference
implementation for correctness testing without hardware.
"""

from __future__ import annotations

import numpy as np

from nkigen_lite.core import (
    DType,
    Op,
    Value,
    to_np_dtype,
    eval_common_op,
)
from nkigen_lite.nki_ir.ir import (
    Graph,
    MemorySpace,
    NisaActivationOp,
    NisaArithOp,
    NisaBitvecOp,
    NisaRangeSelectCmp,
    NisaReduceOp,
    TileType,
)

# ===========================
# NISA interpreter dispatch
# ===========================

_NISA_ACTIVATION_NP = {
    NisaActivationOp.EXP: np.exp,
    NisaActivationOp.LOG: np.log,
    NisaActivationOp.SQRT: np.sqrt,
    NisaActivationOp.TANH: np.tanh,
    NisaActivationOp.SIN: np.sin,
    NisaActivationOp.ABS: np.abs,
    NisaActivationOp.RELU: lambda x: np.maximum(x, 0),
    NisaActivationOp.SQUARE: np.square,
    NisaActivationOp.SIGN: np.sign,
    NisaActivationOp.ARCTAN: np.arctan,
    NisaActivationOp.COPY: lambda x: x.copy(),
}

_NISA_ARITH_NP = {
    NisaArithOp.ADD: np.add,
    NisaArithOp.SUBTRACT: np.subtract,
    NisaArithOp.MULTIPLY: np.multiply,
    NisaArithOp.MAXIMUM: np.maximum,
    NisaArithOp.MINIMUM: np.minimum,
    NisaArithOp.POW: np.power,
}

_NISA_BITVEC_NP = {
    NisaBitvecOp.AND: np.bitwise_and,
    NisaBitvecOp.OR: np.bitwise_or,
    NisaBitvecOp.XOR: np.bitwise_xor,
}

_NISA_REDUCE_NP = {
    NisaReduceOp.ADD: np.sum,
    NisaReduceOp.MAX: np.max,
    NisaReduceOp.MIN: np.min,
}


def _eval_activation(act: NisaActivationOp, x: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Evaluate a single activation function with numpy."""
    if act in _NISA_ACTIVATION_NP:
        return _NISA_ACTIVATION_NP[act](x)
    elif act == NisaActivationOp.RSQRT:
        return (1.0 / np.sqrt(x)).astype(out_dtype)
    elif act == NisaActivationOp.SIGMOID:
        return (1.0 / (1.0 + np.exp(-x))).astype(out_dtype)
    elif act in (NisaActivationOp.GELU, NisaActivationOp.GELU_APPRX_TANH):
        xf = x.astype(np.float64)
        return (
            0.5 * xf * (1 + np.tanh(np.sqrt(2 / np.pi) * (xf + 0.044715 * xf**3)))
        ).astype(out_dtype)
    elif act == NisaActivationOp.RECIPROCAL:
        return (1.0 / x).astype(out_dtype)
    elif act == NisaActivationOp.SILU:
        return (x / (1.0 + np.exp(-x))).astype(out_dtype)
    elif act == NisaActivationOp.ERF:
        from scipy.special import erf as _erf
        return _erf(x).astype(out_dtype)
    elif act == NisaActivationOp.SOFTPLUS:
        return np.log1p(np.exp(x)).astype(out_dtype)
    elif act == NisaActivationOp.MISH:
        return (x * np.tanh(np.log1p(np.exp(x)))).astype(out_dtype)
    elif act == NisaActivationOp.GELU_APPRX_SIGMOID:
        return (x / (1.0 + np.exp(-1.702 * x))).astype(out_dtype)
    else:
        raise NotImplementedError(f"activation op {act!r}")


def _dma_load(
    src: np.ndarray,
    offsets: tuple[int, ...],
    tile_shape: tuple[int, ...],
    strides: tuple[int, ...] | None,
    sizes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Materialize a DMA load tile from src.

    The number of *offsets* / *strides* matches the source HBM rank.
    When src and tile have the same rank the slice extents come from
    *tile_shape* (with numpy's natural boundary clipping). When the
    ranks differ — the lowering may collapse a rank-N HBM tile into a
    2D SBUF tile — we slice src on its native rank using the full
    remaining extent on each axis and then reshape into *tile_shape*.

    Stride 0 on a source dim is a broadcast along that axis: a single
    element is read for every output position. This is how the
    lowering encodes partition-axis broadcasts.
    """
    src_rank = len(offsets)
    if strides is None:
        strides = (1,) * src_rank
    same_rank = src_rank == len(tile_shape)

    # Per-source-axis extent. Priority:
    #   1. Same rank: take from tile_shape — numpy slicing clips at the
    #      source boundary, so partial / remainder tiles work
    #      naturally.
    #   2. Explicit `sizes` attr (set when on-chip tile rank differs
    #      from HBM rank).
    #   3. Different rank without `sizes`: pad tile_shape with leading
    #      1s (kb-style "load (1, P, F) from rank-3 HBM" convention).
    if same_rank:
        per_axis_size = tile_shape
    elif sizes is not None:
        per_axis_size = tuple(sizes)
    else:
        rank_diff = src_rank - len(tile_shape)
        per_axis_size = (1,) * rank_diff + tuple(tile_shape)

    if all(s == 1 for s in strides):
        slices = tuple(
            slice(o, o + sz) for o, sz in zip(offsets, per_axis_size)
        )
        loaded = src[slices].copy()
    else:
        base_slices = []
        bcast_axes = []
        base_shape = []
        for i, (o, sz, st) in enumerate(zip(offsets, per_axis_size, strides)):
            if st == 0:
                base_slices.append(slice(o, o + 1))
                base_shape.append(1)
                bcast_axes.append(i)
            else:
                base_slices.append(slice(o, o + sz * st, st))
                base_shape.append(sz)
        base = src[tuple(base_slices)]
        if not bcast_axes:
            loaded = base.copy()
        else:
            target_nd = list(base_shape)
            for i in bcast_axes:
                target_nd[i] = per_axis_size[i]
            loaded = np.broadcast_to(base.reshape(base_shape), tuple(target_nd)).copy()

    # Reshape into tile_shape only when ranks differ — same-rank loads
    # may have shorter extent (boundary tiles) and shouldn't be
    # reshape-padded.
    if not same_rank and loaded.shape != tile_shape:
        if loaded.size == np.prod(tile_shape):
            loaded = loaded.reshape(tile_shape)
        else:
            # Boundary tile from rank-N HBM into 2D SBUF.  The N-D tile
            # shape from `sizes` tells us the planned per-axis extents;
            # use it to compute each leading-dim's stride in the 2D tile
            # so boundary data lands at the correct offset.
            f_dim = tile_shape[-1]
            if sizes is not None and loaded.ndim > 2:
                # sizes gives the planned N-D tile (e.g. (3, 42, 128)).
                # Each leading dim d occupies stride = prod(sizes[d+1:])
                # in the flattened 2D P-axis.
                nd_sizes = tuple(sizes)
                padded = np.zeros(tile_shape, dtype=loaded.dtype)
                # Iterate over all leading-dim indices of the loaded data
                # and place each innermost slice at the correct 2D offset.
                leading_shape = loaded.shape[:-1]
                for idx in np.ndindex(*leading_shape):
                    # 2D row offset for this N-D index using planned strides
                    row = 0
                    for d, i in enumerate(idx):
                        stride = int(np.prod(nd_sizes[d + 1:-1])) if d + 1 < len(nd_sizes) - 1 else 1
                        row += i * stride
                    src_row = loaded[idx]
                    if row < tile_shape[0]:
                        padded[row, :len(src_row)] = src_row
                loaded = padded
            else:
                actual_p = loaded.size // f_dim
                loaded = loaded.reshape(actual_p, f_dim)
    return loaded


def _has_explicit_dst(op: Op) -> bool:
    """Detect whether this op uses the nki_ir explicit-dst encoding.

    nki_ir ops have TileType (with .memory) on inputs[0]; tensor-level
    NISA ops (from legalize_to_nisa) have TensorType (no .memory).
    """
    return len(op.inputs) > 0 and isinstance(op.inputs[0].type, TileType)


def eval_nisa_op(op: Op, get: callable, env: dict[str, np.ndarray]) -> bool:
    """Try to evaluate a NISA opcode, storing into env. Returns True if handled."""
    # Offset: nki_ir ops have explicit dst at inputs[0], tensor-level ops don't.
    d = 1 if _has_explicit_dst(op) else 0

    if op.opcode == "activation":
        x = get(op.inputs[d])
        scale = op.attrs.get("scale", 1.0)
        has_reduce = "reduce_op" in op.attrs
        num_extra = len(op.inputs) - d - 1
        if has_reduce:
            num_extra -= 1
        if num_extra > 0:
            bias = get(op.inputs[d + 1])
            x = x * scale + bias
        elif scale != 1.0:
            x = x * scale
        out_dtype = to_np_dtype(op.result.type.dtype)
        activated = _eval_activation(op.attrs["op"], x, out_dtype)
        env[op.result.name] = activated
        if has_reduce:
            reduce_dst = op.inputs[-1]
            reduce_op = op.attrs["reduce_op"]
            if reduce_op not in _NISA_REDUCE_NP:
                raise NotImplementedError(f"activation fused reduce op {reduce_op!r}")
            rank = len(activated.shape)
            axes = tuple(range(1, rank))
            reduced = _NISA_REDUCE_NP[reduce_op](activated, axis=axes, keepdims=True)
            env[reduce_dst.name] = reduced
    elif op.opcode == "tensor_tensor_arith":
        a, b = get(op.inputs[d]), get(op.inputs[d + 1])
        arith = op.attrs["op"]
        if arith not in _NISA_ARITH_NP:
            raise NotImplementedError(f"tensor_tensor op {arith!r}")
        env[op.result.name] = _NISA_ARITH_NP[arith](a, b)
    elif op.opcode == "tensor_tensor_bitvec":
        a, b = get(op.inputs[d]), get(op.inputs[d + 1])
        bitvec = op.attrs["op"]
        if bitvec not in _NISA_BITVEC_NP:
            raise NotImplementedError(f"tensor_tensor_bitvec op {bitvec!r}")
        env[op.result.name] = _NISA_BITVEC_NP[bitvec](a, b)
    elif op.opcode == "tensor_scalar_arith":
        x = get(op.inputs[d])
        operand0 = get(op.inputs[d + 1])
        op0 = op.attrs.get("op0") or op.attrs.get("op")
        if op0 not in _NISA_ARITH_NP:
            raise NotImplementedError(f"tensor_scalar op0 {op0!r}")
        result = _NISA_ARITH_NP[op0](x, operand0)
        if "op1" in op.attrs and len(op.inputs) > d + 2:
            operand1 = get(op.inputs[d + 2])
            op1 = op.attrs["op1"]
            if op1 not in _NISA_ARITH_NP:
                raise NotImplementedError(f"tensor_scalar op1 {op1!r}")
            result = _NISA_ARITH_NP[op1](result, operand1)
        env[op.result.name] = result
    elif op.opcode == "scalar_tensor_tensor_arith":
        src0 = get(op.inputs[d])
        src1 = get(op.inputs[d + 1])
        imm0 = get(op.inputs[d + 2])
        op0 = op.attrs["op0"]
        op1 = op.attrs["op1"]
        if op0 not in _NISA_ARITH_NP or op1 not in _NISA_ARITH_NP:
            raise NotImplementedError(f"scalar_tensor_tensor ops {op0!r}, {op1!r}")
        intermediate = _NISA_ARITH_NP[op0](src0, imm0)
        env[op.result.name] = _NISA_ARITH_NP[op1](intermediate, src1)
    elif op.opcode == "tensor_reduce_arith":
        x = get(op.inputs[d])
        reduce_op = op.attrs["op"]
        if reduce_op not in _NISA_REDUCE_NP:
            raise NotImplementedError(f"tensor_reduce op {reduce_op!r}")
        if "num_r_dim" in op.attrs:
            rank = len(x.shape)
            num_r_dim = op.attrs["num_r_dim"]
            axes = tuple(range(rank - num_r_dim, rank))
        else:
            axes = op.attrs["axis"]
        env[op.result.name] = _NISA_REDUCE_NP[reduce_op](
            x, axis=axes, keepdims=op.attrs["keepdims"],
        )
    elif op.opcode == "activation_reduce":
        x = get(op.inputs[d])
        out_dtype = to_np_dtype(op.result.type.dtype)
        activated = _eval_activation(op.attrs["act_op"], x, out_dtype)
        reduce_op = op.attrs["reduce_op"]
        if reduce_op not in _NISA_REDUCE_NP:
            raise NotImplementedError(f"activation_reduce: reduce op {reduce_op!r}")
        if "num_r_dim" in op.attrs:
            rank = len(x.shape)
            num_r_dim = op.attrs["num_r_dim"]
            axes = tuple(range(rank - num_r_dim, rank))
        else:
            axes = op.attrs["axis"]
        env[op.result.name] = _NISA_REDUCE_NP[reduce_op](
            activated, axis=axes, keepdims=op.attrs["keepdims"],
        )
    elif op.opcode == "nisa_nc_matmul":
        stat = get(op.inputs[d]).astype(np.float32)
        mov = get(op.inputs[d + 1]).astype(np.float32)
        result = np.matmul(np.swapaxes(stat, -2, -1), mov)
        if op.attrs.get("accum"):
            if d > 0:
                result = result + get(op.inputs[0]).astype(np.float32)
            else:
                result = result + get(op.inputs[2]).astype(np.float32)
        env[op.result.name] = result.astype(np.float32)
    else:
        return False
    return True


# ===========================
# Numpy interpreter
# ===========================

def interpret(
    graph: Graph,
    inputs: dict[str, np.ndarray],
    outer_env: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Execute a NKI IR graph with numpy."""
    env: dict[str, np.ndarray] = {}
    if outer_env is not None:
        env.update(outer_env)

    for v in graph.inputs:
        if v.name not in inputs:
            raise ValueError(f"Missing input: {v.name}")
        env[v.name] = inputs[v.name]

    def _get(v: Value) -> np.ndarray:
        return env[v.name]

    for op in graph.ops:
        if op.opcode == "alloc":
            dtype = to_np_dtype(op.result.type.dtype)
            if np.issubdtype(dtype, np.floating):
                env[op.result.name] = np.full(op.result.type.shape, np.nan, dtype=dtype)
            else:
                env[op.result.name] = np.zeros(op.result.type.shape, dtype=dtype)

        elif op.opcode == "dealloc":
            pass

        elif op.opcode == "rotate":
            env[op.result.name] = _get(op.inputs[0])

        elif op.opcode == "scalar_const":
            env[op.result.name] = np.array(op.attrs["value"], dtype=np.int32)

        elif op.opcode == "affine":
            idx = int(_get(op.inputs[0]))
            env[op.result.name] = np.array(
                op.attrs["base"] + idx * op.attrs["scale"], dtype=np.int32,
            )

        elif op.opcode == "scalar_add":
            a = int(_get(op.inputs[0]))
            b = int(_get(op.inputs[1]))
            env[op.result.name] = np.array(a + b, dtype=np.int32)

        elif op.opcode == "dma_copy":
            direction = op.attrs["direction"]
            strides_attr = op.attrs.get("strides")
            sizes_attr = op.attrs.get("sizes")
            if direction == "load":
                # dst is inputs[0] (on-chip), src is inputs[1] (HBM)
                src = _get(op.inputs[1])
                if op.attrs.get("dynamic_offsets"):
                    offsets = tuple(int(_get(v)) for v in op.inputs[2:])
                else:
                    offsets = op.attrs["offsets"]
                tile_shape = op.result.type.shape
                loaded = _dma_load(
                    src, offsets, tile_shape, strides_attr, sizes_attr,
                ).astype(to_np_dtype(op.result.type.dtype))
                # Pad to full tile allocation shape for partial (boundary)
                # tiles.  On real HW the unused partitions contain garbage;
                # here we zero-pad so downstream ops execute at the static
                # tile shape without broadcast errors.
                if loaded.shape != tile_shape:
                    padded = np.zeros(tile_shape, dtype=loaded.dtype)
                    slices = tuple(slice(0, s) for s in loaded.shape)
                    padded[slices] = loaded
                    loaded = padded
                env[op.result.name] = loaded
            else:  # store
                src_tile = _get(op.inputs[0])
                dst_name = op.inputs[1].name
                dst_arr = env[dst_name]
                if op.attrs.get("dynamic_offsets"):
                    offsets = tuple(int(_get(v)) for v in op.inputs[2:])
                else:
                    offsets = op.attrs["offsets"]
                # Per-HBM-dim slice extent. Priority:
                #   1. Same rank as src: take from src.shape (handles
                #      boundary clipping for partial tiles).
                #   2. Explicit `sizes`: reshape src to those extents
                #      (typical 2D-src → rank-N HBM case).
                #   3. Different rank without `sizes`: pad src.shape
                #      with leading 1s.
                src_rank = len(offsets)
                if src_tile.ndim == src_rank:
                    per_axis_size = src_tile.shape
                    src_view = src_tile
                elif sizes_attr is not None:
                    per_axis_size = tuple(sizes_attr)
                    if src_tile.size == int(np.prod(per_axis_size)):
                        # Exact fit: reshape the 2D tile to the N-D extents.
                        src_view = src_tile.reshape(per_axis_size)
                    else:
                        # Boundary tile: the 2D source is the full (padded)
                        # allocation, larger than the clamped sizes. Map its
                        # leading axes (P-side) and trailing axis (F-side) to
                        # the N-D HBM layout, then clip to per_axis_size.
                        f_size = per_axis_size[-1]
                        p_size = src_tile.size // src_tile.shape[-1]
                        nd_p_shape = per_axis_size[:-1]  # leading P-dims
                        # Reshape 2D (P, F) into (P-dims..., F) using the full
                        # P extent split across nd_p_shape with row-major order,
                        # padding the P axis to the product of nd_p_shape.
                        full_p = int(np.prod(nd_p_shape)) if nd_p_shape else 1
                        flat = src_tile.reshape(src_tile.shape[0], src_tile.shape[-1])
                        nd = flat[:full_p, :f_size].reshape(nd_p_shape + (f_size,))
                        src_view = nd
                        per_axis_size = tuple(sizes_attr)
                else:
                    rank_diff = src_rank - src_tile.ndim
                    per_axis_size = (1,) * rank_diff + src_tile.shape
                    src_view = src_tile.reshape(per_axis_size)
                if strides_attr and any(s != 1 for s in strides_attr):
                    slices = tuple(
                        slice(o, o + sz * st, st)
                        for o, sz, st in zip(offsets, per_axis_size, strides_attr)
                    )
                else:
                    slices = tuple(
                        slice(o, o + sz) for o, sz in zip(offsets, per_axis_size)
                    )
                # Clip source to destination bounds (partial/boundary tiles
                # may have been zero-padded to full tile size on load).
                dst_region = dst_arr[slices]
                if dst_region.shape != src_view.shape:
                    src_slices = tuple(slice(0, s) for s in dst_region.shape)
                    src_view = src_view[src_slices]
                dst_arr[slices] = src_view.astype(dst_arr.dtype)

        elif op.opcode == "access_pattern":
            src = _get(op.inputs[0])
            pattern = op.attrs["pattern"]
            input_idx = 1
            if op.attrs.get("dynamic_offset"):
                offset = int(_get(op.inputs[input_idx]))
                input_idx += 1
            else:
                offset = op.attrs.get("offset", 0)
            flat = src.reshape(-1)
            out_shape = tuple(p[1] for p in pattern)
            result = np.empty(out_shape, dtype=src.dtype)
            for idx in np.ndindex(*out_shape):
                addr = offset
                for dim_idx, (stride, _count) in zip(idx, pattern):
                    addr += dim_idx * stride
                result[idx] = flat[addr]
            env[op.result.name] = result

        elif op.opcode == "tensor_copy":
            src_data = _get(op.inputs[1])
            dst_dtype = to_np_dtype(op.result.type.dtype)
            env[op.result.name] = src_data.astype(dst_dtype)

        elif op.opcode == "dma_transpose":
            src = _get(op.inputs[1])
            perm = op.attrs["perm"]
            env[op.result.name] = np.transpose(src, perm).copy()

        elif op.opcode == "stream_transpose":
            src = _get(op.inputs[1])
            env[op.result.name] = src.T.copy()

        elif op.opcode == "memset":
            env[op.result.name] = np.full_like(_get(op.inputs[0]), op.attrs["value"])

        elif op.opcode == "iota":
            shape = op.result.type.shape
            dtype = to_np_dtype(op.result.type.dtype)
            pattern = op.attrs.get("pattern", [[1, shape[-1]]])
            offset = op.attrs.get("offset", 0)
            ch_mul = op.attrs.get("channel_multiplier", 0)
            P = shape[0] if len(shape) >= 2 else 1
            F = shape[-1]
            result = np.empty(shape, dtype=dtype)
            for p in range(P):
                for f in range(F):
                    val = offset + p * ch_mul
                    rem = f
                    for step, count in reversed(pattern):
                        digit = rem % count
                        rem //= count
                        val += digit * step
                    if len(shape) >= 2:
                        result[p, f] = val
                    else:
                        result[f] = val
            env[op.result.name] = result

        elif op.opcode == "stream_shuffle":
            x = _get(op.inputs[1])
            mask = op.attrs["shuffle_mask"]
            env[op.result.name] = x[mask]

        elif op.opcode == "matmul":
            stat = _get(op.inputs[1]).astype(np.float32)
            mov = _get(op.inputs[2]).astype(np.float32)
            result = stat.T @ mov
            if op.attrs.get("accumulate"):
                result = result + _get(op.inputs[0]).astype(np.float32)
            env[op.result.name] = result

        elif op.opcode == "broadcast":
            env[op.result.name] = np.broadcast_to(
                _get(op.inputs[0]), op.attrs["shape"]
            )

        elif op.opcode == "view":
            x = _get(op.inputs[0])
            out_dtype = to_np_dtype(op.attrs["dtype"])
            env[op.result.name] = x.view(out_dtype).reshape(op.attrs["shape"])

        elif op.opcode == "cross_lane_reduce_arith":
            x = _get(op.inputs[1])
            reduce_op = op.attrs["op"]
            if reduce_op not in _NISA_REDUCE_NP:
                raise NotImplementedError(f"cross_lane_reduce_arith op {reduce_op!r}")
            env[op.result.name] = _NISA_REDUCE_NP[reduce_op](
                x, axis=0, keepdims=True,
            )

        elif op.opcode == "fori_loop":
            body = op.attrs["body"]
            static_extent = op.attrs["extent"]
            step = op.attrs["step"]
            if static_extent is not None:
                extent = static_extent
            else:
                extent = int(_get(op.inputs[0]))
            idx_name = body.inputs[0].name
            for i in range(0, extent, step):
                body_inputs = {idx_name: np.array(i, dtype=np.int32)}
                body_env = interpret(body, body_inputs, outer_env=env)
                env.update(body_env)

        elif op.opcode == "tile_loop":
            body = op.attrs["body"]
            extent = op.attrs["extent"]
            step = op.attrs["step"]
            carried = [_get(v) for v in op.inputs]
            for i in range(0, extent, step):
                body_inputs = {
                    body.inputs[0].name: np.array(i, dtype=np.int32),
                }
                for j, bv in enumerate(body.inputs[1:]):
                    body_inputs[bv.name] = carried[j]
                body_env = interpret(body, body_inputs, outer_env=env)
                carried = [
                    body_env[bv.name] for bv in body.output_values
                ]
            for j, rv in enumerate(op.results):
                env[rv.name] = carried[j]

        elif op.opcode == "affine_select":
            pred = _get(op.inputs[1]).astype(bool)
            on_true = _get(op.inputs[2])
            on_false = _get(op.inputs[3])
            env[op.result.name] = np.where(pred, on_true, on_false)

        elif op.opcode == "dma_copy_indirect":
            direction = op.attrs["direction"]
            if direction == "load":
                src = _get(op.inputs[1])
                index = _get(op.inputs[2]).astype(np.intp)
                env[op.result.name] = np.take(src.reshape(-1), index).reshape(
                    op.result.type.shape
                )
            else:
                src_tile = _get(op.inputs[0])
                dst_name = op.inputs[1].name
                index = _get(op.inputs[2]).astype(np.intp)
                flat = env[dst_name].reshape(-1)
                np.put(flat, index.reshape(-1), src_tile.reshape(-1))
                env[dst_name] = flat.reshape(env[dst_name].shape)

        elif op.opcode == "tensor_tensor_scan":
            data0 = _get(op.inputs[1])
            data1 = _get(op.inputs[2])
            initial = _get(op.inputs[3])
            np_op0 = _NISA_ARITH_NP[op.attrs["op0"]]
            np_op1 = _NISA_ARITH_NP[op.attrs["op1"]]
            result = np.empty_like(data0)
            if data0.ndim >= 2:
                for p in range(data0.shape[0]):
                    acc_init = initial.flat[p] if initial.size > 1 else initial.flat[0]
                    acc = np_op1(np_op0(data0[p, 0], acc_init), data1[p, 0])
                    result[p, 0] = acc
                    for f in range(1, data0.shape[1]):
                        acc = np_op1(np_op0(data0[p, f], acc), data1[p, f])
                        result[p, f] = acc
            else:
                acc_init = initial.flat[0]
                acc = np_op1(np_op0(data0[0], acc_init), data1[0])
                result[0] = acc
                for f in range(1, data0.shape[0]):
                    acc = np_op1(np_op0(data0[f], acc), data1[f])
                    result[f] = acc
            env[op.result.name] = result

        elif op.opcode == "sequence_bounds":
            segment_ids = _get(op.inputs[1])
            P = segment_ids.shape[0]
            F = segment_ids.shape[-1]
            out_shape = op.result.type.shape
            result = np.zeros(out_shape, dtype=to_np_dtype(op.result.type.dtype))
            for p in range(P):
                ids = segment_ids[p].flatten()
                for f in range(F):
                    sid = int(ids[f])
                    if sid == 0:
                        result[p, 0, f] = F
                        result[p, 1, f] = -1
                    else:
                        positions = np.where(ids == sid)[0]
                        result[p, 0, f] = int(positions[0])
                        result[p, 1, f] = int(positions[-1]) + 1
            env[op.result.name] = result

        elif op.opcode == "dma_gather_transpose":
            src = _get(op.inputs[1])
            index = _get(op.inputs[2]).astype(np.intp)
            gathered = np.take(src, index, axis=0)
            env[op.result.name] = gathered.T.copy() if gathered.ndim == 2 else gathered

        elif op.opcode == "copy_predicated":
            dst_arr = _get(op.inputs[0]).copy()
            pred = _get(op.inputs[1])
            src = _get(op.inputs[2])
            mask = pred > 0 if not np.issubdtype(pred.dtype, np.bool_) else pred
            dst_arr[mask] = src[mask]
            env[op.result.name] = dst_arr

        elif op.opcode == "gather":
            src = _get(op.inputs[1])
            indices = _get(op.inputs[2]).astype(np.intp)
            result = np.empty(op.result.type.shape, dtype=to_np_dtype(op.result.type.dtype))
            for p in range(src.shape[0]):
                result[p] = src[p][indices[p]]
            env[op.result.name] = result

        elif op.opcode == "exponential":
            src = _get(op.inputs[1])
            if len(op.inputs) > 2:
                max_val = _get(op.inputs[2])
                env[op.result.name] = np.exp(src - max_val)
            else:
                env[op.result.name] = np.exp(src)

        elif op.opcode == "range_select":
            src = _get(op.inputs[1])
            bound0 = _get(op.inputs[2])
            bound1 = _get(op.inputs[3])
            fill_value = np.float32(op.attrs["fill_value"])
            comp0 = op.attrs["comp_op0"]
            comp1 = op.attrs["comp_op1"]
            shape = src.shape
            idx = np.broadcast_to(np.arange(shape[-1], dtype=np.float32), shape)
            _CMP_FNS = {
                NisaRangeSelectCmp.IS_EQ: np.equal,
                NisaRangeSelectCmp.IS_GT: np.greater,
                NisaRangeSelectCmp.IS_GE: np.greater_equal,
                NisaRangeSelectCmp.IS_LE: np.less_equal,
                NisaRangeSelectCmp.IS_LT: np.less,
            }
            in_range = _CMP_FNS[comp0](idx, bound0) & _CMP_FNS[comp1](idx, bound1)
            env[op.result.name] = np.where(in_range, src, fill_value).astype(
                to_np_dtype(op.result.type.dtype)
            )

        elif op.opcode == "select_reduce":
            pred = _get(op.inputs[1])
            on_true = _get(op.inputs[2])
            on_false_scalar = op.attrs.get("on_false_scalar")
            if on_false_scalar is not None:
                on_false = np.float32(on_false_scalar)
            else:
                on_false = _get(op.inputs[3])
            mask = pred > 0 if not np.issubdtype(pred.dtype, np.bool_) else pred
            selected = np.where(mask, on_true, on_false)
            env[op.result.name] = selected.astype(to_np_dtype(op.result.type.dtype))
            if "reduce_op" in op.attrs:
                reduce_dst = op.inputs[-1]
                reduce_op_val = op.attrs["reduce_op"]
                axes = tuple(range(1, selected.ndim))
                env[reduce_dst.name] = _NISA_REDUCE_NP[reduce_op_val](
                    selected, axis=axes, keepdims=True,
                )

        elif op.opcode == "if_else":
            cond = _get(op.inputs[0])
            then_body = op.attrs["then_body"]
            else_body = op.attrs.get("else_body")
            if bool(cond):
                interpret(then_body, {}, outer_env=env)
            elif else_body is not None:
                interpret(else_body, {}, outer_env=env)

        elif op.opcode == "while_loop":
            cond_body = op.attrs["cond_body"]
            body_body = op.attrs["body_body"]
            carry = _get(op.inputs[0])
            for _ in range(10_000):
                cond_env = interpret(
                    cond_body,
                    {cond_body.inputs[0].name: carry},
                    outer_env=env,
                )
                cond_val = cond_env[cond_body.output_values[0].name]
                if not bool(cond_val):
                    break
                output_val = cond_env[cond_body.output_values[1].name]
                body_env = interpret(
                    body_body,
                    {body_body.inputs[0].name: output_val},
                    outer_env=env,
                )
                carry = body_env[body_body.output_values[0].name]
            env[op.result.name] = carry

        elif op.opcode == "reg_compare":
            a = _get(op.inputs[0])
            b = _get(op.inputs[1])
            cmp_op = op.attrs["op"]
            _CMP = {"<": np.less, "<=": np.less_equal, ">": np.greater,
                     ">=": np.greater_equal, "!=": np.not_equal}
            env[op.result.name] = _CMP[cmp_op](a, b)

        elif op.opcode == "load_register":
            tile = _get(op.inputs[0])
            env[op.result.name] = tile.flat[0]

        elif op.opcode == "store_register":
            dst = _get(op.inputs[0]).copy()
            reg = _get(op.inputs[1])
            dst.flat[0] = reg
            env[op.result.name] = dst

        elif eval_nisa_op(op, _get, env):
            pass

        # Fallback: tensor_ir-level ops emitted by tiling pass (add, mul, etc.)
        # These should eventually be replaced by NISA ops in the tiling pass.
        elif eval_common_op(op, _get, env):
            pass

        else:
            raise NotImplementedError(
                f"nki_ir interpret: unknown opcode {op.opcode!r}"
            )

        # Compute ops with explicit pre-allocated dst: alias the result
        # back to the dst name so fori_loop bodies see in-place mutations.
        _INPLACE_DST_OPS = {
            "matmul", "tensor_tensor_arith", "tensor_scalar_arith",
            "scalar_tensor_tensor_arith", "tensor_reduce_arith",
            "activation", "activation_reduce", "cross_lane_reduce_arith",
            "tensor_copy", "copy_predicated", "exponential",
            "range_select", "select_reduce", "gather",
            "affine_select", "tensor_tensor_scan", "sequence_bounds",
            "memset", "store_register",
        }
        if (op.opcode in _INPLACE_DST_OPS
                and op.results and op.inputs
                and op.inputs[0].name != op.results[0].name
                and op.results[0].name in env):
            env[op.inputs[0].name] = env[op.results[0].name]

    return env


def run(
    graph: Graph,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Execute and return named output arrays."""
    if not graph.outputs:
        raise ValueError("Graph has no outputs. Call builder.set_outputs().")
    env = interpret(graph, inputs)
    return {name: env[v.name] for name, v in graph.outputs.items()}
