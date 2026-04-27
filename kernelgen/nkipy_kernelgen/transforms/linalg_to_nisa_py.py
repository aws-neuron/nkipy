"""
Python replacement for the (deleted) C++ `linalg-to-nisa` pass.

Reads post-Phase-4 MLIR (linalg+memref+scf+arith+func, with integer-encoded
nkipy memory spaces) and emits NISA MLIR using the `nki` wheel's Python
bindings. Everything downstream (resolve-custom-ops, prepare-for-nki,
`nki-opt-pipeline`) then consumes standard NISA IR.

Design
------

This mirrors the (pre-open-source) C++ `LinalgToNisa.cpp` architecture:

1.  **Parse in upstream ctx, re-parse in NKI ctx.** Memref/scf/linalg exist
    only in upstream MLIR; NISA exists only in the NKI wheel's context. We
    print the module as generic IR in the upstream context, rewrite integer
    memspace markers to `#nisa.mem<...>`, and re-parse in the NKI context
    (with `allow_unregistered_dialects`) so unrecognised ops survive as
    opaque ops we can still walk.

2.  **Walk + rewrite.** For each supported op we compute a ``MemRefAccess``
    per operand by tracing back through subview/collapse_shape/expand_shape
    to the base memref (materialising `arith.constant` / `arith.addi` /
    `arith.muli` for the offset math along the way). We then build a plain
    ``AffineMap`` per operand following ``createStandardNisaMap`` (d0 at the
    first kept dim, d1 at the last kept dim, symbols everywhere else), flatten
    it via ``nisa.flatten_affine_map``, and hand the base+indices+map tuple
    straight to the Python ``nisa.<op>(...)`` builder. No ``prepare_operand``
    layer — that path inside ``_nki_irbuilder`` runs a linearisation that can
    merge per-dim expressions into a single multi-symbol flat_affine_expr,
    which the NISA verifier rejects.

3.  **Post-pass cleanup.** DCE the dead view ops, fold `reinterpret_cast`
    on fresh allocs, fold HBM reshape chains into the alloc type.

Validation
----------

Per the plan's 2026-04-20 decision: we do not byte-diff against pre-refactor
golden MLIR. Each e2e test is expected to simulate correctly through BIRSim
(and/or run on HW). The pass is a success when every fixture's numerical
output matches NumPy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from mlir import ir as up_ir  # type: ignore[import-not-found]

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal._mlir_libs import _nki  # type: ignore[import-not-found]
from nki.compiler._internal._mlir_libs import _nki_irbuilder  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]


# NKIPy emits memref memory-space annotations as `N : i32` integers (matching
# the `MemSpaceEnum` in `NkipyAttrs.td`). NKI's parser needs them as
# `#nisa.mem<...>` attribute syntax. The enum values start at 1 (not 0) — see
# the comment in NkipyAttrs.td for why 0 cannot be used.
_NKIPY_TO_NISA_MEMSPACE = {
    1: "hbm",
    2: "psum",
    3: "sbuf",
    4: "shared_hbm",
}

_INT_MEMSPACE_RE = re.compile(r", (\d+) : i32>")


def _rewrite_memspace_text(generic: str) -> str:
    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        name = _NKIPY_TO_NISA_MEMSPACE.get(n)
        if name is None:
            return m.group(0)
        return f", #nisa.mem<{name}>>"

    return _INT_MEMSPACE_RE.sub(repl, generic)


def _to_nki_module(src: str) -> tuple[nk_ir.Context, nk_ir.Module]:
    # Re-serialize through nkipy-opt with `--mlir-print-op-generic` so any
    # nkipy-dialect ops that survive into this phase (currently just
    # `nkipy.gather`, which we lower below) arrive in generic form
    # `"nkipy.gather"(...)`. The upstream MLIR Python bindings don't know
    # about the nkipy dialect; pretty-form `nkipy.gather(...)` would fail
    # to parse (`allow_unregistered_dialects` only covers generic form).
    from .nkipy_opt import run_nkipy_opt_passes  # avoid circular import
    src = run_nkipy_opt_passes(src, passes=[], print_generic=True)

    up_ctx = up_ir.Context()
    up_ctx.load_all_available_dialects()
    up_ctx.allow_unregistered_dialects = True
    with up_ctx:
        up_mod = up_ir.Module.parse(src)
        generic = up_mod.operation.get_asm(
            print_generic_op_form=True, assume_verified=True
        )
    generic = _rewrite_memspace_text(generic)

    nk_ctx = nk_ir.Context()
    _nki.register_all_dialects(nk_ctx)
    nk_ctx.allow_unregistered_dialects = True
    with nk_ctx:
        nk_mod = nk_ir.Module.parse(generic)
    return nk_ctx, nk_mod


# ---------------------------------------------------------------------------
# MemRef access trace
# ---------------------------------------------------------------------------


DYN_SENTINEL = -(1 << 63)


@dataclass
class _Access:
    """Mirror of C++ ``MemRefAccess``.

    ``indices`` contains one ``arith`` SSA value per base-rank dim (zero for
    dims with no subview offset). ``dropped_dims`` marks rank-reducing or
    unit-collapsed dims that should carry only a symbol offset in the NISA
    affine map (no iteration dim).

    The flat-affine map is built from this triple via ``_build_nisa_map``.
    """

    base: nk_ir.Value
    indices: list[nk_ir.Value]
    base_type: nk_ir.MemRefType
    dropped_dims: list[bool]  # len == base_rank

    @property
    def base_rank(self) -> int:
        return self.base_type.rank  # type: ignore[attr-defined]


def _reassoc_groups(attr: nk_ir.Attribute) -> list[list[int]]:
    outer = nk_ir.ArrayAttr(attr)
    groups: list[list[int]] = []
    for g in outer:
        inner = nk_ir.ArrayAttr(g)
        groups.append([int(nk_ir.IntegerAttr(x).value) for x in inner])
    return groups


def _const_int(v: nk_ir.Value) -> int | None:
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "arith.constant":
        return None
    try:
        attr = op.attributes["value"]
    except KeyError:
        return None
    try:
        return nk_ir.IntegerAttr(attr).value
    except Exception:
        return None


def _emit_const_index(ctx: nk_ir.Context, value: int, loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    attr = nk_ir.IntegerAttr.get(idx_ty, value)
    op = nk_ir.Operation.create(
        "arith.constant", results=[idx_ty], attributes={"value": attr}, loc=loc
    )
    return op.result


def _emit_addi(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
               loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.addi", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _emit_muli(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
               loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.muli", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _emit_divui(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
                loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.divui", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _get_base_and_offsets(ctx: nk_ir.Context, operand: nk_ir.Value,
                          loc: nk_ir.Location) -> _Access:
    """Port of C++ ``getBaseAndOffsets``. Walks subview/collapse/expand chains
    back to the base alloc or block arg, materialising arith ops as needed.

    The current insertion point must be positioned where new arith ops can be
    safely emitted (typically the op being rewritten).
    """
    base = operand
    base_type = operand.type
    indices: list[nk_ir.Value] = []
    dropped_dims: list[bool] = []

    changed = True
    while changed:
        changed = False
        owner = getattr(base, "owner", None)
        if owner is None:
            break
        op = owner.opview if hasattr(owner, "opview") else owner
        name = getattr(op, "name", None)

        if name == "memref.subview":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            src_rank = source_ty.rank
            try:
                static_offsets = [int(x) for x in op.attributes["static_offsets"]]
            except (KeyError, ValueError):
                break
            if len(static_offsets) != src_rank:
                break
            dyn_ops = list(op.operands)[1:]
            dyn_idx = 0
            subview_offsets: list[nk_ir.Value] = []
            for i in range(src_rank):
                if static_offsets[i] == DYN_SENTINEL:
                    subview_offsets.append(dyn_ops[dyn_idx])
                    dyn_idx += 1
                else:
                    subview_offsets.append(
                        _emit_const_index(ctx, static_offsets[i], loc)
                    )

            # Rank-reducing detection via static_sizes==1 vs result shape.
            try:
                static_sizes = [int(x) for x in op.attributes["static_sizes"]]
            except (KeyError, ValueError):
                break
            result_shape = list(getattr(op.results[0].type, "shape", ()))
            # Determine dropped dims: rank-reducing means result_rank < src_rank.
            # Heuristic: dims with static_sizes[i]==1 that are NOT present in
            # the result shape are dropped. We align by order: walk source dims,
            # match to result in order, preferring size equality.
            dropped = [False] * src_rank
            if len(result_shape) < src_rank:
                ri = 0
                for si in range(src_rank):
                    if ri < len(result_shape) and static_sizes[si] == result_shape[ri]:
                        ri += 1
                    else:
                        # If size == 1 and we haven't matched yet, drop it.
                        if static_sizes[si] == 1:
                            dropped[si] = True
                        elif ri < len(result_shape) and static_sizes[si] == 1:
                            dropped[si] = True
                        else:
                            # fall back: mark as dropped if no remaining result dim
                            # matches
                            dropped[si] = True
                # If we ended up dropping too many or too few, bail
                non_dropped = sum(1 for d in dropped if not d)
                if non_dropped != len(result_shape):
                    break

            # Accumulate offsets:
            if not indices:
                # No carried indices yet. Usually this is the first subview in
                # the chain, but it can also follow a collapse_shape that
                # reduced a multi-dim HBM operand to a lower-rank view — in
                # which case `dropped_dims` is already populated and must be
                # preserved. The subview is same-rank here (src_rank ==
                # result_rank) for a collapse→subview chain, so just seed
                # indices with the subview offsets and merge `dropped` with
                # any preserved `dropped_dims` element-wise.
                indices = subview_offsets
                if dropped_dims and len(dropped_dims) == src_rank:
                    dropped_dims = [a or b for a, b in zip(dropped_dims, dropped)]
                else:
                    dropped_dims = dropped
            else:
                # Nested subview. Expand current indices (in result-rank space
                # after prior ops) to source rank.
                if any(dropped):
                    # Rank-reducing: dropped dims get pure subview offset;
                    # kept dims get accumulated + subview offset.
                    expanded: list[nk_ir.Value] = []
                    kept_idx = 0
                    for si in range(src_rank):
                        if dropped[si]:
                            expanded.append(subview_offsets[si])
                        else:
                            assert kept_idx < len(indices)
                            expanded.append(
                                _emit_addi(indices[kept_idx], subview_offsets[si],
                                           ctx, loc)
                            )
                            kept_idx += 1
                    indices = expanded
                    merged_dropped = [False] * src_rank
                    kept_idx = 0
                    for si in range(src_rank):
                        if dropped[si]:
                            merged_dropped[si] = True
                        else:
                            if kept_idx < len(dropped_dims) and dropped_dims[kept_idx]:
                                merged_dropped[si] = True
                            kept_idx += 1
                    dropped_dims = merged_dropped
                else:
                    # Same-rank subview: add element-wise.
                    if len(indices) != src_rank:
                        break
                    indices = [
                        _emit_addi(indices[i], subview_offsets[i], ctx, loc)
                        for i in range(src_rank)
                    ]

            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.collapse_shape":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            try:
                groups = _reassoc_groups(op.attributes["reassociation"])
            except KeyError:
                break
            src_shape = list(getattr(source_ty, "shape", ()))
            if any(s < 0 for s in src_shape):
                break
            src_rank = len(src_shape)

            # Determine whether any group has multiple non-unit dims.
            has_multi_non_unit = any(
                sum(1 for d in grp if src_shape[d] != 1) > 1 for grp in groups
            )

            ms = getattr(source_ty, "memory_space", None)
            is_hbm = ms is not None and (
                "<hbm>" in str(ms) or "<shared_hbm>" in str(ms)
            )

            if has_multi_non_unit and is_hbm:
                # Stop tracing — HBM collapse is fine as-is; NCC handles it.
                break

            # Find primary dim (largest size) per group — for multi-non-unit.
            def primary_dim(group: list[int]) -> int:
                best_idx, best_size = -1, 0
                for i, d in enumerate(group):
                    if src_shape[d] > best_size:
                        best_size = src_shape[d]
                        best_idx = i
                return best_idx

            # Expand droppedDims from collapsed to source rank.
            expanded_dropped = [False] * src_rank
            if is_hbm:
                for gi, grp in enumerate(groups):
                    collapsed_dropped = (
                        gi < len(dropped_dims) and dropped_dims[gi] if dropped_dims
                        else False
                    )
                    if collapsed_dropped:
                        for d in grp:
                            expanded_dropped[d] = True
                    elif len(grp) > 1:
                        for d in grp:
                            if src_shape[d] == 1:
                                expanded_dropped[d] = True
            else:
                for gi, grp in enumerate(groups):
                    collapsed_dropped = (
                        gi < len(dropped_dims) and dropped_dims[gi] if dropped_dims
                        else False
                    )
                    if collapsed_dropped:
                        for d in grp:
                            expanded_dropped[d] = True
                    elif len(grp) > 1 and has_multi_non_unit:
                        p = primary_dim(grp)
                        for i, d in enumerate(grp):
                            if i != p:
                                expanded_dropped[d] = True
                    elif len(grp) > 1:
                        # Drop only the unit dims, but keep one per group so
                        # the group still has a home for its iteration dim.
                        # For single-non-unit groups that dim is obviously
                        # the keeper; for all-unit groups we keep the first
                        # position so downstream affine-map construction has
                        # somewhere to place d_gi.
                        non_unit = [d for d in grp if src_shape[d] != 1]
                        keeper = non_unit[0] if non_unit else grp[0]
                        for d in grp:
                            if d != keeper:
                                expanded_dropped[d] = True

            # Expand indices from collapsed rank to source rank.
            if indices:
                expanded_indices: list[nk_ir.Value] = []
                for gi, grp in enumerate(groups):
                    if len(grp) == 1:
                        expanded_indices.append(indices[gi])
                    else:
                        non_unit_count = sum(1 for d in grp if src_shape[d] != 1)
                        if non_unit_count <= 1:
                            zero = _emit_const_index(ctx, 0, loc)
                            for d in grp:
                                if src_shape[d] != 1:
                                    expanded_indices.append(indices[gi])
                                else:
                                    expanded_indices.append(zero)
                        else:
                            p = primary_dim(grp)
                            primary_size = src_shape[grp[p]]
                            zero = _emit_const_index(ctx, 0, loc)
                            size_val = _emit_const_index(ctx, primary_size, loc)
                            batch = _emit_divui(indices[gi], size_val, ctx, loc)
                            for i, d in enumerate(grp):
                                if i == p:
                                    expanded_indices.append(zero)
                                elif src_shape[d] == 1:
                                    expanded_indices.append(zero)
                                else:
                                    expanded_indices.append(batch)
                indices = expanded_indices

            dropped_dims = expanded_dropped
            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.expand_shape":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            try:
                groups = _reassoc_groups(op.attributes["reassociation"])
            except KeyError:
                break
            dst_shape = list(getattr(op.results[0].type, "shape", ()))
            src_rank = source_ty.rank

            if indices:
                src_indices: list[nk_ir.Value] = []
                for grp in groups:
                    combined = indices[grp[0]]
                    for k in range(1, len(grp)):
                        inner_size = dst_shape[grp[k]]
                        scale = _emit_const_index(ctx, inner_size, loc)
                        combined = _emit_muli(combined, scale, ctx, loc)
                        combined = _emit_addi(combined, indices[grp[k]], ctx, loc)
                    src_indices.append(combined)
                indices = src_indices

            if dropped_dims:
                new_dropped = [False] * src_rank
                for gi, grp in enumerate(groups):
                    all_dropped = all(
                        d < len(dropped_dims) and dropped_dims[d] for d in grp
                    )
                    if all_dropped:
                        new_dropped[gi] = True
                dropped_dims = new_dropped

            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.reinterpret_cast":
            # Pass-through when it preserves rank and has zero static offsets.
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            src_shape = list(getattr(source_ty, "shape", ()))
            dst_shape = list(getattr(op.results[0].type, "shape", ()))
            if src_shape != dst_shape:
                break
            try:
                st_off = [int(x) for x in op.attributes["static_offsets"]]
            except (KeyError, ValueError):
                st_off = []
            if any(x != 0 for x in st_off):
                break
            base = source
            base_type = source_ty
            changed = True
            continue

        break

    # Ensure we have rank-many indices (fresh allocs / block args need zeros).
    if not indices:
        rank = base_type.rank  # type: ignore[attr-defined]
        indices = [_emit_const_index(ctx, 0, loc) for _ in range(rank)]

    # Normalise dropped_dims length to base rank.
    rank = base_type.rank  # type: ignore[attr-defined]
    if len(dropped_dims) < rank:
        dropped_dims = dropped_dims + [False] * (rank - len(dropped_dims))
    elif len(dropped_dims) > rank:
        dropped_dims = dropped_dims[:rank]

    return _Access(
        base=base,
        indices=indices,
        base_type=base_type,  # type: ignore[arg-type]
        dropped_dims=dropped_dims,
    )


# ---------------------------------------------------------------------------
# Affine map construction
# ---------------------------------------------------------------------------


def _create_standard_nisa_map(
    ctx: nk_ir.Context,
    num_iter_dims: int,
    num_symbols: int,
    num_results: int,
    dropped_dims: list[bool],
) -> nk_ir.AffineMap:
    """Python mirror of C++ ``createStandardNisaMap``.

    d0 -> first kept position, d(N-1) -> last kept position, middle iter dims
    map to middle kept positions in order. Dropped dims get pure symbol or
    constant expressions; any remaining result has symbol or 0.
    """
    kept_positions = [
        i for i in range(num_results)
        if not (dropped_dims and i < len(dropped_dims) and dropped_dims[i])
    ]

    position_to_dim: dict[int, int] = {}
    if kept_positions and num_iter_dims > 0:
        position_to_dim[kept_positions[0]] = 0
        if num_iter_dims > 1 and len(kept_positions) > 1:
            position_to_dim[kept_positions[-1]] = num_iter_dims - 1
            mid_dim = 1
            for k in range(1, len(kept_positions) - 1):
                if mid_dim + 1 >= num_iter_dims:
                    break
                position_to_dim[kept_positions[k]] = mid_dim
                mid_dim += 1

    exprs: list[nk_ir.AffineExpr] = []
    for i in range(num_results):
        is_dropped = bool(dropped_dims) and i < len(dropped_dims) and dropped_dims[i]
        if is_dropped:
            if i < num_symbols:
                exprs.append(nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(nk_ir.AffineConstantExpr.get(0))
        elif i in position_to_dim:
            dim_expr = nk_ir.AffineDimExpr.get(position_to_dim[i])
            if i < num_symbols:
                exprs.append(dim_expr + nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(dim_expr)
        else:
            if i < num_symbols:
                exprs.append(nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(nk_ir.AffineConstantExpr.get(0))
    return nk_ir.AffineMap.get(num_iter_dims, num_symbols, exprs)


def _build_nisa_map(
    ctx: nk_ir.Context, num_iter_dims: int, access: _Access
) -> nk_ir.Attribute:
    amap = _create_standard_nisa_map(
        ctx,
        num_iter_dims,
        num_symbols=len(access.indices),
        num_results=access.base_rank,
        dropped_dims=access.dropped_dims,
    )
    return nisa.flatten_affine_map(amap, ctx)


def _operand_kwargs(
    prefix: str,
    access: _Access,
    flat_map: nk_ir.Attribute,
    tile_shape: list[int],
    tile_par_dims: int = 1,
) -> dict:
    return {
        f"{prefix}_memloc": access.base,
        f"{prefix}_indices": access.indices,
        f"{prefix}_ap": flat_map,
        f"{prefix}_static_tile_shape": list(tile_shape),
        f"{prefix}_tile_par_dims": tile_par_dims,
    }


def _empty_operand_kwargs(prefix: str) -> dict:
    """Kwargs for an 'omitted' optional operand (ap=None, memloc=None)."""
    return {
        f"{prefix}_memloc": None,
        f"{prefix}_indices": [],
        f"{prefix}_ap": None,
        f"{prefix}_static_tile_shape": [],
        f"{prefix}_tile_par_dims": 0,
    }


def _scalar_operand_kwargs(prefix: str, scalar: nk_ir.Value) -> dict:
    return {
        f"{prefix}_memloc": scalar,
        f"{prefix}_indices": [],
        f"{prefix}_ap": None,
        f"{prefix}_static_tile_shape": [],
        f"{prefix}_tile_par_dims": 0,
    }


# ---------------------------------------------------------------------------
# Rewrite context
# ---------------------------------------------------------------------------


_Pattern = Callable[["_RewriteContext", nk_ir.OpView], None]
_PATTERNS: dict[str, _Pattern] = {}


def pattern(*op_names: str) -> Callable[[_Pattern], _Pattern]:
    def register(fn: _Pattern) -> _Pattern:
        for name in op_names:
            _PATTERNS[name] = fn
        return fn

    return register


_LINALG_TO_ARITH_OP = {
    "linalg.add": nisa.ArithOp.Add,
    "linalg.sub": nisa.ArithOp.Subtract,
    "linalg.mul": nisa.ArithOp.Multiply,
    "linalg.max": nisa.ArithOp.Max,
    "linalg.min": nisa.ArithOp.Min,
}

_REDUCE_BODY_OP_TO_ARITH = {
    "arith.addf": nisa.ArithOp.Add,
    "arith.addi": nisa.ArithOp.Add,
    "arith.mulf": nisa.ArithOp.Multiply,
    "arith.muli": nisa.ArithOp.Multiply,
    "arith.maximumf": nisa.ArithOp.Max,
    "arith.minimumf": nisa.ArithOp.Min,
}

_ARITH_TO_CROSS_LANE = {
    nisa.ArithOp.Add: nisa.CrossLaneReduceArithOp.Add,
    nisa.ArithOp.Max: nisa.CrossLaneReduceArithOp.Max,
}


class _RewriteContext:
    def __init__(self, ctx: nk_ir.Context, module: nk_ir.Module):
        self.ctx = ctx
        self.module = module
        self.loc = nk_ir.Location.unknown(ctx)
        self._f32_const_cache: dict[tuple[int, float], nk_ir.Value] = {}

    def f32_const(self, block: nk_ir.Block, value: float) -> nk_ir.Value:
        key = (id(block), value)
        cached = self._f32_const_cache.get(key)
        if cached is not None:
            return cached
        with self.loc:
            f32 = nk_ir.F32Type.get(self.ctx)
            attr = nk_ir.FloatAttr.get(f32, value)
            with nk_ir.InsertionPoint.at_block_begin(block):
                const_op = nk_ir.Operation.create(
                    "arith.constant",
                    results=[f32],
                    attributes={"value": attr},
                    loc=self.loc,
                )
        self._f32_const_cache[key] = const_op.result
        return const_op.result


def _enclosing_block(op: nk_ir.OpView) -> nk_ir.Block:
    return op.operation.block  # type: ignore[attr-defined]


def _is_memspace(ty: nk_ir.Type, name: str) -> bool:
    ms = getattr(ty, "memory_space", None)
    if ms is None:
        return False
    return f"<{name}>" in str(ms)


def _is_hbm(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "hbm") or _is_memspace(ty, "shared_hbm")


def _is_sbuf(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "sbuf")


def _is_psum(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "psum")


def _static_shape(ty: nk_ir.Type) -> list[int] | None:
    shape = list(getattr(ty, "shape", ()))
    if any(s < 0 for s in shape):
        return None
    return shape


# ---------------------------------------------------------------------------
# Elementwise: linalg.add/sub/mul/max/min -> nisa.tensor_tensor_arith
# ---------------------------------------------------------------------------


@pattern("linalg.add", "linalg.sub", "linalg.mul", "linalg.max", "linalg.min")
def _rewrite_elementwise(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    arith_kind = _LINALG_TO_ARITH_OP[op.operation.name]
    operands = list(op.operation.operands)
    if len(operands) < 3:
        return
    lhs, rhs, dst = operands[0], operands[1], operands[2]

    shape = _static_shape(dst.type)
    if shape is None:
        return

    with nk_ir.InsertionPoint(op), rctx.loc:
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        lhs_acc = _get_base_and_offsets(rctx.ctx, lhs, rctx.loc)
        rhs_acc = _get_base_and_offsets(rctx.ctx, rhs, rctx.loc)
        rank = len(shape)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        lhs_map = _build_nisa_map(rctx.ctx, rank, lhs_acc)
        rhs_map = _build_nisa_map(rctx.ctx, rank, rhs_acc)

        kwargs: dict = {}
        kwargs.update(_operand_kwargs("dst", dst_acc, dst_map, shape))
        kwargs.update(_operand_kwargs("lhs", lhs_acc, lhs_map, shape))
        kwargs.update(_operand_kwargs("rhs", rhs_acc, rhs_map, shape))
        nisa.tensor_tensor_arith(
            op=arith_kind, engine=nisa.Engine.Vector, **kwargs
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# memref.copy + linalg.copy
# ---------------------------------------------------------------------------


def _pad_shape_to_2d(shape: list[int]) -> list[int]:
    if len(shape) < 2:
        return shape + [1] * (2 - len(shape))
    return shape


@pattern("memref.copy")
def _rewrite_memref_copy(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    src = op.operation.operands[0]
    dst = op.operation.operands[1]
    src_ty = src.type
    dst_ty = dst.type

    src_hbm, dst_hbm = _is_hbm(src_ty), _is_hbm(dst_ty)
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)

    needs_dma = src_hbm or dst_hbm
    on_tpb = (
        (src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)
    )
    if not (needs_dma or on_tpb):
        return

    shape = _static_shape(dst_ty)
    if shape is None:
        return
    shape = _pad_shape_to_2d(shape)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)

        num_iter = len(shape)
        src_map = _build_nisa_map(rctx.ctx, num_iter, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, num_iter, dst_acc)

        needs_psum_hop = needs_dma and (
            (src_hbm and dst_psum) or (src_psum and dst_hbm)
        )

        if needs_psum_hop:
            sbuf_attr = nk_ir.Attribute.parse("#nisa.mem<sbuf>")
            inter_ty = nk_ir.MemRefType.get(
                shape, dst_ty.element_type, memory_space=sbuf_attr  # type: ignore[attr-defined]
            )
            inter_val = nisa.alloc(memref_type=inter_ty, alignment=64)
            inter_acc = _get_base_and_offsets(rctx.ctx, inter_val, rctx.loc)
            inter_map = _build_nisa_map(rctx.ctx, num_iter, inter_acc)
            if src_hbm and dst_psum:
                nisa.dma_copy(
                    **_operand_kwargs("dst", inter_acc, inter_map, shape),
                    **_operand_kwargs("src", src_acc, src_map, shape),
                )
                nisa.tensor_copy(
                    **_operand_kwargs("dst", dst_acc, dst_map, shape),
                    **_operand_kwargs("src", inter_acc, inter_map, shape),
                    engine=nisa.Engine.Vector,
                )
            else:
                nisa.tensor_copy(
                    **_operand_kwargs("dst", inter_acc, inter_map, shape),
                    **_operand_kwargs("src", src_acc, src_map, shape),
                    engine=nisa.Engine.Vector,
                )
                nisa.dma_copy(
                    **_operand_kwargs("dst", dst_acc, dst_map, shape),
                    **_operand_kwargs("src", inter_acc, inter_map, shape),
                )
        elif needs_dma:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
            )
        else:
            nisa.tensor_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
                engine=nisa.Engine.Vector,
            )
    op.operation.erase()


@pattern("linalg.copy")
def _rewrite_linalg_copy(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    src_ty, dst_ty = src.type, dst.type
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)
    if not ((src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)):
        return
    shape = _static_shape(dst_ty)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.tensor_copy(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_operand_kwargs("src", src_acc, src_map, shape),
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# memref.alloc / memref.dealloc
# ---------------------------------------------------------------------------


@pattern("memref.alloc")
def _rewrite_memref_alloc(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    result_ty = op.operation.results[0].type
    if not (_is_sbuf(result_ty) or _is_psum(result_ty) or _is_hbm(result_ty)):
        return

    alignment = 0
    attrs = op.operation.attributes
    if "alignment" in attrs:
        alignment = nk_ir.IntegerAttr(attrs["alignment"]).value

    with nk_ir.InsertionPoint(op), rctx.loc:
        new_val = nisa.alloc(memref_type=result_ty, alignment=alignment)

    op.operation.results[0].replace_all_uses_with(new_val)
    op.operation.erase()


def _fold_reinterpret_casts(rctx: _RewriteContext) -> None:
    casts: list[nk_ir.OpView] = []

    def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
        if op_handle.name == "memref.reinterpret_cast":
            casts.append(op_handle.opview)
        return nk_ir.WalkResult.ADVANCE

    rctx.module.operation.walk(visit)

    for cast_op in casts:
        src = cast_op.operation.operands[0]
        src_owner = getattr(src, "owner", None)
        if src_owner is None:
            continue
        src_op = src_owner.opview if hasattr(src_owner, "opview") else src_owner
        if getattr(src_op, "name", None) != "nisa.alloc":
            continue
        try:
            st_off = [int(x) for x in cast_op.operation.attributes["static_offsets"]]
            if any(x != 0 for x in st_off):
                continue
        except (KeyError, ValueError):
            continue

        new_ty = cast_op.operation.results[0].type

        alignment = 0
        if "alignment" in src_op.attributes:
            alignment = nk_ir.IntegerAttr(src_op.attributes["alignment"]).value

        with nk_ir.InsertionPoint(src_op), rctx.loc:
            new_alloc = nisa.alloc(memref_type=new_ty, alignment=alignment)

        cast_op.operation.results[0].replace_all_uses_with(new_alloc)
        cast_op.operation.erase()
        if list(src.uses):
            src.replace_all_uses_with(new_alloc)
        src_op.erase()


@pattern("memref.dealloc")
def _rewrite_memref_dealloc(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    target = op.operation.operands[0]
    target_ty = target.type
    if not (_is_sbuf(target_ty) or _is_psum(target_ty)):
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        nisa.release(memref=target)
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.transpose
# ---------------------------------------------------------------------------


def _non_unit_dims(shape: list[int]) -> list[int]:
    return [i for i, s in enumerate(shape) if s != 1]


@pattern("linalg.transpose")
def _rewrite_linalg_transpose(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    src_ty, dst_ty = src.type, dst.type
    src_shape = _static_shape(src_ty)
    dst_shape = _static_shape(dst_ty)
    if src_shape is None or dst_shape is None:
        return

    attrs = op.operation.attributes
    if "permutation" not in attrs:
        return
    perm_str = str(attrs["permutation"])
    try:
        inside = perm_str.split(":", 1)[1].rstrip(">").strip()
        perm = [int(x.strip()) for x in inside.split(",") if x.strip()]
    except (IndexError, ValueError):
        return

    non_unit_src = _non_unit_dims(src_shape)
    if len(non_unit_src) > 2:
        return

    needs_transpose = False
    if len(non_unit_src) == 2:
        s0, s1 = non_unit_src[0], non_unit_src[1]
        d0 = perm.index(s0)
        d1 = perm.index(s1)
        needs_transpose = d0 > d1

    src_hbm = _is_hbm(src_ty)
    src_sbuf = _is_sbuf(src_ty)
    dst_sbuf = _is_sbuf(dst_ty)
    dst_hbm = _is_hbm(dst_ty)

    # 2D tile shapes derived from non-unit dims.
    src_tile = [src_shape[d] for d in non_unit_src]
    dst_tile = [dst_shape[d] for d in _non_unit_dims(dst_shape)]
    while len(src_tile) < 2:
        src_tile.append(1)
    while len(dst_tile) < 2:
        dst_tile.append(1)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        num_iter = 2
        src_map = _build_nisa_map(rctx.ctx, num_iter, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, num_iter, dst_acc)

        if needs_transpose:
            if not ((src_hbm or src_sbuf) and dst_sbuf):
                return
            nisa.dma_transpose(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_tile),
                **_operand_kwargs("src", src_acc, src_map, src_tile),
                permutation=[1, 0],
                dge_mode=nisa.DGEType.NoDGE,
                oob_is_err=True,
                engine=nisa.Engine.DMA,
            )
            op.operation.erase()
            return

        cross = (src_hbm and dst_sbuf) or (src_sbuf and dst_hbm)
        if cross:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_tile),
                **_operand_kwargs("src", src_acc, src_map, src_tile),
            )
            op.operation.erase()
            return


# ---------------------------------------------------------------------------
# linalg.matmul_transpose_a
# ---------------------------------------------------------------------------


def _index_const(rctx: _RewriteContext, value: int) -> nk_ir.Value:
    with rctx.loc:
        idx_ty = nk_ir.IndexType.get(rctx.ctx)
        attr = nk_ir.IntegerAttr.get(idx_ty, value)
        const_op = nk_ir.Operation.create(
            "arith.constant",
            results=[idx_ty],
            attributes={"value": attr},
            loc=rctx.loc,
        )
    return const_op.result


@pattern("linalg.matmul_transpose_a")
def _rewrite_matmul_transpose_a(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) != 3:
        return
    mat_a, mat_b, mat_c = operands
    a_ty, b_ty, c_ty = mat_a.type, mat_b.type, mat_c.type

    a_shape = _static_shape(a_ty)
    b_shape = _static_shape(b_ty)
    c_shape = _static_shape(c_ty)
    if a_shape is None or b_shape is None or c_shape is None:
        return
    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        return
    K, M = a_shape
    if b_shape[0] != K or c_shape[0] != M or c_shape[1] != b_shape[1]:
        return

    if not (_is_sbuf(a_ty) and _is_sbuf(b_ty) and _is_psum(c_ty)):
        return

    N = b_shape[1]

    with nk_ir.InsertionPoint(op), rctx.loc:
        a_acc = _get_base_and_offsets(rctx.ctx, mat_a, rctx.loc)
        b_acc = _get_base_and_offsets(rctx.ctx, mat_b, rctx.loc)
        c_acc = _get_base_and_offsets(rctx.ctx, mat_c, rctx.loc)
        a_map = _build_nisa_map(rctx.ctx, 2, a_acc)
        b_map = _build_nisa_map(rctx.ctx, 2, b_acc)
        c_map = _build_nisa_map(rctx.ctx, 2, c_acc)

        row_pos = _index_const(rctx, 0)
        col_pos = _index_const(rctx, 0)
        nisa.matmul(
            **_operand_kwargs("dst", c_acc, c_map, [M, N]),
            **_operand_kwargs("stationary", a_acc, a_map, [K, M]),
            **_operand_kwargs("moving", b_acc, b_map, [K, N]),
            row_pos=row_pos,
            col_pos=col_pos,
            psum_accumulate_flags=None,
            is_transpose=False,
            perf_opt=nisa.PerfOptMode.None_,
            psum_zero_region=nisa.MatmulZeroRegion.Size2048,
            engine=nisa.Engine.Tensor,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.reciprocal
# ---------------------------------------------------------------------------


@pattern("linalg.reciprocal")
def _rewrite_reciprocal(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    if not (
        (_is_sbuf(src.type) or _is_psum(src.type))
        and (_is_sbuf(dst.type) or _is_psum(dst.type))
    ):
        return
    shape = _static_shape(dst.type)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.reciprocal(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_operand_kwargs("src", src_acc, src_map, shape),
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.{exp, square, sqrt, abs, log, tanh} -> nisa.activation
# ---------------------------------------------------------------------------


_LINALG_TO_ACTIVATION = {
    "linalg.exp": nisa.ActivationFunction.exp,
    "linalg.square": nisa.ActivationFunction.square,
    "linalg.sqrt": nisa.ActivationFunction.sqrt,
    "linalg.abs": nisa.ActivationFunction.abs,
    "linalg.log": nisa.ActivationFunction.log,
    "linalg.tanh": nisa.ActivationFunction.tanh,
}


def _emit_activation(
    rctx: _RewriteContext,
    op: nk_ir.OpView,
    src: nk_ir.Value,
    dst: nk_ir.Value,
    act_kind,
) -> bool:
    if not (_is_sbuf(src.type) and _is_sbuf(dst.type)):
        return False
    shape = _static_shape(dst.type)
    if shape is None:
        return False

    block = _enclosing_block(op)
    bias = rctx.f32_const(block, 0.0)
    scale = rctx.f32_const(block, 1.0)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.activation(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_empty_operand_kwargs("reduce_res"),
            **_operand_kwargs("src", src_acc, src_map, shape),
            **_scalar_operand_kwargs("bias", bias),
            **_scalar_operand_kwargs("scale", scale),
            **_empty_operand_kwargs("alpha"),
            op=act_kind,
            engine=nisa.Engine.Scalar,
        )
    return True


@pattern(*_LINALG_TO_ACTIVATION.keys())
def _rewrite_linalg_activation(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    act_kind = _LINALG_TO_ACTIVATION[op.operation.name]
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    if _emit_activation(rctx, op, operands[0], operands[1], act_kind):
        op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.fill -> nisa.memset
# ---------------------------------------------------------------------------


@pattern("linalg.fill")
def _rewrite_linalg_fill(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    scalar, dst = operands[0], operands[1]
    dst_ty = dst.type
    if not (_is_sbuf(dst_ty) or _is_psum(dst_ty)):
        return
    shape = _static_shape(dst_ty)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.memset(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            value=scalar,
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.generic (scalar / broadcast / same-shape / type cast / reduction /
# unary math / powf)
# ---------------------------------------------------------------------------


_BODY_OP_TO_ARITH = {
    "arith.addf": nisa.ArithOp.Add,
    "arith.addi": nisa.ArithOp.Add,
    "arith.subf": nisa.ArithOp.Subtract,
    "arith.subi": nisa.ArithOp.Subtract,
    "arith.mulf": nisa.ArithOp.Multiply,
    "arith.muli": nisa.ArithOp.Multiply,
    "arith.divf": nisa.ArithOp.Divide,
    "arith.divsi": nisa.ArithOp.DivideInt,
    "arith.divui": nisa.ArithOp.DivideInt,
    "arith.remf": nisa.ArithOp.Mod,
    "arith.remsi": nisa.ArithOp.ModInt,
}

_CMPF_PRED_TO_ARITH = {
    1: nisa.ArithOp.IsEQ,
    2: nisa.ArithOp.IsGT,
    3: nisa.ArithOp.IsGE,
    4: nisa.ArithOp.IsLT,
    5: nisa.ArithOp.IsLE,
    6: nisa.ArithOp.IsNE,
}

_CMPI_PRED_TO_ARITH = {
    0: nisa.ArithOp.IsEQ,
    1: nisa.ArithOp.IsNE,
    2: nisa.ArithOp.IsLT,
    3: nisa.ArithOp.IsLE,
    4: nisa.ArithOp.IsGT,
    5: nisa.ArithOp.IsGE,
}

_BODY_MATH_TO_ACTIVATION = {
    "math.sin": nisa.ActivationFunction.sin,
    "math.copysign": nisa.ActivationFunction.sign,
}


def _defining_op(v: nk_ir.Value):
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    return owner.opview if hasattr(owner, "opview") else owner


def _predicate_int(op):
    attrs = op.attributes
    if "predicate" not in attrs:
        return None
    return nk_ir.IntegerAttr(attrs["predicate"]).value


def _is_constant_value(v: nk_ir.Value) -> bool:
    owner = getattr(v, "owner", None)
    if owner is None:
        return False
    op = owner.opview if hasattr(owner, "opview") else owner
    return getattr(op, "name", None) == "arith.constant"


def _shape_match_broadcast(in_shape, out_shape):
    if len(in_shape) != len(out_shape):
        return False
    had_broadcast = False
    for i, o in zip(in_shape, out_shape):
        if i == 1 and o > 1:
            had_broadcast = True
        elif i != o:
            return False
    return had_broadcast


def _analyze_generic_body(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if not ops:
        return None
    yield_op = ops[-1]
    if yield_op.name != "linalg.yield":
        return None
    yielded = list(yield_op.operands)
    if len(yielded) != 1:
        return None
    root = _defining_op(yielded[0])
    if root is None or not hasattr(root, "name"):
        return None
    name = root.name
    kind = _BODY_OP_TO_ARITH.get(name)
    if kind is not None:
        return kind, root.operands[0], root.operands[1]
    if name == "arith.uitofp":
        inner = _defining_op(root.operands[0])
        if inner is None:
            return None
        if inner.name == "arith.cmpf":
            pred = _predicate_int(inner)
            k = _CMPF_PRED_TO_ARITH.get(pred) if pred is not None else None
            if k is not None:
                return k, inner.operands[0], inner.operands[1]
        if inner.name in ("arith.andi", "arith.ori"):
            logical_kind = (
                nisa.ArithOp.LogicalAnd if inner.name == "arith.andi"
                else nisa.ArithOp.LogicalOr
            )
            lhs_cast = _defining_op(inner.operands[0])
            rhs_cast = _defining_op(inner.operands[1])
            if (lhs_cast and rhs_cast and
                lhs_cast.name == "arith.fptoui" and
                rhs_cast.name == "arith.fptoui"):
                return logical_kind, lhs_cast.operands[0], rhs_cast.operands[0]
        return None
    if name == "arith.extui":
        inner = _defining_op(root.operands[0])
        if inner is None or inner.name != "arith.cmpi":
            return None
        pred = _predicate_int(inner)
        k = _CMPI_PRED_TO_ARITH.get(pred) if pred is not None else None
        if k is not None:
            return k, inner.operands[0], inner.operands[1]
    return None


def _match_generic_powf(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return None
    inner, yield_op = ops[0], ops[1]
    if inner.name != "math.powf" or yield_op.name != "linalg.yield":
        return None
    if list(yield_op.operands) != [inner.results[0]]:
        return None
    return inner.operands[0], inner.operands[1]


def _match_reduction_body(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if not ops or ops[-1].name != "linalg.yield":
        return None
    body_ops = ops[:-1]
    found = None
    for bo in body_ops:
        k = _REDUCE_BODY_OP_TO_ARITH.get(bo.name)
        if k is not None:
            if found is not None:
                return None
            found = (k, bo)
    return found


def _classify_reduction(op: nk_ir.OpView):
    attrs = op.operation.attributes
    it_str = str(attrs["iterator_types"])
    kinds = []
    for token in it_str.split("#linalg.iterator_type<"):
        close = token.find(">")
        if close < 0:
            continue
        k = token[:close]
        if k in ("parallel", "reduction"):
            kinds.append(k)
    if not kinds:
        return None
    num_red = sum(1 for k in kinds if k == "reduction")
    if num_red == 0:
        return None
    is_left = all(kinds[i] == "reduction" for i in range(num_red))
    is_right = all(kinds[-1 - i] == "reduction" for i in range(num_red))
    if not (is_left or is_right):
        return None
    return num_red, is_left, is_right


def _rewrite_linalg_generic_reduction(
    rctx: _RewriteContext, op: nk_ir.OpView, num_ins: int
) -> bool:
    if num_ins != 1:
        return False
    operands = list(op.operation.operands)
    src = operands[0]
    dst = operands[1]

    match = _match_reduction_body(op)
    if match is None:
        return False
    arith_kind, inner = match

    block = op.regions[0].blocks[0]
    block_args = list(block.arguments)
    out_block_arg = block_args[-1]
    in0 = inner.operands[0]
    in1 = inner.operands[1]
    if not (str(in0) == str(out_block_arg) or str(in1) == str(out_block_arg)):
        return False

    classified = _classify_reduction(op)
    if classified is None:
        return False
    num_red_dims, is_left, is_right = classified

    src_ty = src.type
    dst_ty = dst.type
    if not _is_sbuf(src_ty):
        return False
    if not (_is_sbuf(dst_ty) or _is_psum(dst_ty)):
        return False
    dst_shape = _static_shape(dst_ty)
    src_shape = _static_shape(src_ty)
    if dst_shape is None or src_shape is None:
        return False

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)

        if is_left:
            cross_op = _ARITH_TO_CROSS_LANE.get(arith_kind)
            if cross_op is None:
                return False
            # cross_lane_reduce_arith uses each operand's data shape as its
            # iteration/tile domain. src spans the full input (parallel +
            # partition reduction dim), dst spans the output. Using dst_shape
            # for src tile (as before) made the hardware reduce a 1-wide
            # slice, returning 0 for axis=0 sums.
            src_map = _build_nisa_map(rctx.ctx, len(src_shape), src_acc)
            dst_map = _build_nisa_map(rctx.ctx, len(dst_shape), dst_acc)
            nisa.cross_lane_reduce_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_shape),
                **_operand_kwargs("src", src_acc, src_map, src_shape),
                reduce_op=cross_op,
                num_r_dim=0,
                engine=nisa.Engine.Gpsimd,
            )
            op.operation.erase()
            return True

        # Rightmost reduction: tensor_reduce_arith into temp, then
        # tensor_tensor_arith to accumulate into dst.
        #
        # tensor_reduce_arith's iteration domain spans the SOURCE shape
        # (parallel dims + reduction dims), so affine maps for both src and
        # temp_dst have `numSrcIterDims = len(src_shape)` dimensions. Tile
        # shapes reflect each operand's own shape: src_shape for src,
        # dst_shape for the dst/temp. Using dst_shape for src (as before)
        # made the hardware only reduce a 1-wide slice, giving wrong sums.
        num_src_iter_dims = len(src_shape)

        src_map = _build_nisa_map(rctx.ctx, num_src_iter_dims, src_acc)
        dst_reduce_map = _build_nisa_map(rctx.ctx, num_src_iter_dims, dst_acc)

        temp_ty = nk_ir.MemRefType.get(
            dst_shape,
            dst_ty.element_type,  # type: ignore[attr-defined]
            memory_space=dst_ty.memory_space,  # type: ignore[attr-defined]
        )
        temp_val = nisa.alloc(memref_type=temp_ty, alignment=0)
        temp_acc = _get_base_and_offsets(rctx.ctx, temp_val, rctx.loc)
        temp_reduce_map = _build_nisa_map(
            rctx.ctx, num_src_iter_dims, temp_acc,
        )

        nisa.tensor_reduce_arith(
            **_operand_kwargs("dst", temp_acc, temp_reduce_map, dst_shape),
            **_operand_kwargs("src", src_acc, src_map, src_shape),
            op=arith_kind,
            negated=False,
            num_r_dim=num_red_dims,
            engine=nisa.Engine.Vector,
        )

        # Accumulation uses the dst iteration domain only (parallel dims).
        num_dst_iter_dims = len(dst_shape)
        dst_accum_map = _build_nisa_map(rctx.ctx, num_dst_iter_dims, dst_acc)
        temp_accum_map = _build_nisa_map(
            rctx.ctx, num_dst_iter_dims, temp_acc,
        )
        nisa.tensor_tensor_arith(
            **_operand_kwargs("dst", dst_acc, dst_accum_map, dst_shape),
            **_operand_kwargs("lhs", dst_acc, dst_accum_map, dst_shape),
            **_operand_kwargs("rhs", temp_acc, temp_accum_map, dst_shape),
            op=arith_kind,
            engine=nisa.Engine.Vector,
        )
        nisa.release(memref=temp_val)

    op.operation.erase()
    return True


def _match_generic_unary_activation(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return None
    inner, yield_op = ops[0], ops[1]
    if yield_op.name != "linalg.yield":
        return None
    if list(yield_op.operands) != [inner.results[0]]:
        return None
    return _BODY_MATH_TO_ACTIVATION.get(inner.name)


def _match_generic_identity_body(op: nk_ir.OpView) -> bool:
    """True if the generic's body yields the first block argument directly.

    Mirrors the C++ LinalgGenericIdentityCopyPattern. Uses `walk()` to find
    the yield op — touching `block.operations` (by iteration or index)
    corrupts the NKI Python binding's iterator state and breaks later
    matchers on the same generic.
    """
    block = op.regions[0].blocks[0]
    yield_op: list[nk_ir.Operation] = []

    def visit(o: nk_ir.Operation) -> nk_ir.WalkResult:
        if o.name == "linalg.yield":
            yield_op.append(o)
            return nk_ir.WalkResult.INTERRUPT
        return nk_ir.WalkResult.ADVANCE

    op.operation.walk(visit)
    if not yield_op:
        return False
    terminator = yield_op[0]
    operands = list(terminator.operands)
    if len(operands) != 1:
        return False
    args = list(block.arguments)
    if not args:
        return False
    return operands[0] == args[0]


def _emit_copy_from_identity_generic(
    rctx: _RewriteContext,
    op: nk_ir.OpView,
    src: nk_ir.Value,
    dst: nk_ir.Value,
) -> None:
    """Lower an identity-body linalg.generic to nisa.tensor_copy / dma_copy.

    Same lowering strategy as _rewrite_memref_copy / _rewrite_linalg_copy.
    """
    src_ty, dst_ty = src.type, dst.type
    src_hbm, dst_hbm = _is_hbm(src_ty), _is_hbm(dst_ty)
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)

    needs_dma = src_hbm or dst_hbm
    on_tpb = (
        (src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)
    )
    if not (needs_dma or on_tpb):
        return

    shape = _static_shape(dst_ty)
    if shape is None:
        return

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        if needs_dma:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
            )
        else:
            nisa.tensor_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
                engine=nisa.Engine.Vector,
            )
    op.operation.erase()


def _match_generic_type_cast(op: nk_ir.OpView) -> bool:
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return False
    inner, yield_op = ops[0], ops[1]
    if inner.name not in ("arith.sitofp", "arith.fptosi"):
        return False
    if yield_op.name != "linalg.yield":
        return False
    if list(yield_op.operands) != [inner.results[0]]:
        return False
    return True


@pattern("linalg.generic")
def _rewrite_linalg_generic(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    attrs = op.operation.attributes
    if "operandSegmentSizes" not in attrs:
        return
    seg_str = str(attrs["operandSegmentSizes"])
    try:
        nums = [int(x.strip()) for x in seg_str.split(":")[1].strip(" >").split(",")]
        num_ins, num_outs = nums[0], nums[1]
    except (ValueError, IndexError):
        return

    if num_outs != 1:
        return

    if "iterator_types" not in attrs:
        return
    if "reduction" in str(attrs["iterator_types"]):
        _rewrite_linalg_generic_reduction(rctx, op, num_ins)
        return

    operands = list(op.operation.operands)
    inputs = operands[:num_ins]
    output = operands[num_ins]
    out_ty = output.type
    out_shape = _static_shape(out_ty)

    # Identity-body generic (body is just `linalg.yield %arg0`) — lower to
    # a copy. Ported from the pre-open-source LinalgGenericIdentityCopyPattern
    # in LinalgToNisa.cpp. Arises from broadcast_to in the tracer, and from
    # trivial transposes reconstructed by legalize-layout.
    if (num_ins == 1
            and "reduction" not in str(attrs["iterator_types"])
            and _match_generic_identity_body(op)):
        _emit_copy_from_identity_generic(rctx, op, inputs[0], output)
        return

    if num_ins == 1:
        act_kind = _match_generic_unary_activation(op)
        if act_kind is not None:
            if _emit_activation(rctx, op, inputs[0], output, act_kind):
                op.operation.erase()
                return

    if num_ins == 1 and _match_generic_type_cast(op):
        if (
            "parallel" in str(attrs["iterator_types"])
            and "reduction" not in str(attrs["iterator_types"])
            and out_shape is not None
            and _is_sbuf(inputs[0].type)
            and _is_sbuf(out_ty)
        ):
            zero = rctx.f32_const(_enclosing_block(op), 0.0)
            with nk_ir.InsertionPoint(op), rctx.loc:
                src_acc = _get_base_and_offsets(rctx.ctx, inputs[0], rctx.loc)
                dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
                rank = len(out_shape)
                src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
                dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
                nisa.tensor_scalar_arith(
                    **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                    **_operand_kwargs("src", src_acc, src_map, out_shape),
                    **_scalar_operand_kwargs("operand0", zero),
                    **_empty_operand_kwargs("operand1"),
                    op0=nisa.ArithOp.Add,
                    op1=None,
                    reverse_operands=nisa.TensScalarRevOps.None_,
                    engine=nisa.Engine.Vector,
                )
            op.operation.erase()
            return

    pow_match = _match_generic_powf(op) if num_ins == 2 else None
    if pow_match is not None and out_shape is not None:
        base, exp_v = pow_match
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        swap = str(base) != str(block_arg0)
        base_v = inputs[1] if swap else inputs[0]
        exp_v = inputs[0] if swap else inputs[1]
        if (_is_sbuf(base_v.type) and _is_sbuf(exp_v.type)
                and _is_sbuf(out_ty)):
            with nk_ir.InsertionPoint(op), rctx.loc:
                base_acc = _get_base_and_offsets(rctx.ctx, base_v, rctx.loc)
                exp_acc = _get_base_and_offsets(rctx.ctx, exp_v, rctx.loc)
                dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
                rank = len(out_shape)
                nisa.tensor_tensor_power(
                    **_operand_kwargs(
                        "dst", dst_acc, _build_nisa_map(rctx.ctx, rank, dst_acc),
                        out_shape,
                    ),
                    **_operand_kwargs(
                        "lhs", base_acc, _build_nisa_map(rctx.ctx, rank, base_acc),
                        out_shape,
                    ),
                    **_operand_kwargs(
                        "rhs", exp_acc, _build_nisa_map(rctx.ctx, rank, exp_acc),
                        out_shape,
                    ),
                    engine=nisa.Engine.Gpsimd,
                )
            op.operation.erase()
            return

    analysis = _analyze_generic_body(op)
    if analysis is None or out_shape is None:
        return
    arith_kind, body_lhs, body_rhs = analysis

    if not _is_sbuf(out_ty):
        return

    if num_ins == 1:
        input_v = inputs[0]
        if not _is_sbuf(input_v.type):
            return
        lhs_const = _is_constant_value(body_lhs)
        rhs_const = _is_constant_value(body_rhs)
        if lhs_const == rhs_const:
            return
        scalar_v = body_lhs if lhs_const else body_rhs
        reverse = (
            nisa.TensScalarRevOps.First if lhs_const else nisa.TensScalarRevOps.None_
        )
        with nk_ir.InsertionPoint(op), rctx.loc:
            src_acc = _get_base_and_offsets(rctx.ctx, input_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            nisa.tensor_scalar_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                **_operand_kwargs("src", src_acc, src_map, out_shape),
                **_scalar_operand_kwargs("operand0", scalar_v),
                **_empty_operand_kwargs("operand1"),
                op0=arith_kind,
                op1=None,
                reverse_operands=reverse,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return

    # num_ins == 2
    in0, in1 = inputs[0], inputs[1]
    in0_shape = _static_shape(in0.type)
    in1_shape = _static_shape(in1.type)
    if in0_shape is None or in1_shape is None:
        return
    if not (_is_sbuf(in0.type) and _is_sbuf(in1.type)):
        return
    if _is_constant_value(body_lhs) or _is_constant_value(body_rhs):
        return

    in0_bcast = _shape_match_broadcast(in0_shape, out_shape)
    in1_bcast = _shape_match_broadcast(in1_shape, out_shape)

    if in0_shape == out_shape and in1_shape == out_shape:
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        swap = str(body_lhs) != str(block_arg0)
        lhs_v = in1 if swap else in0
        rhs_v = in0 if swap else in1
        with nk_ir.InsertionPoint(op), rctx.loc:
            lhs_acc = _get_base_and_offsets(rctx.ctx, lhs_v, rctx.loc)
            rhs_acc = _get_base_and_offsets(rctx.ctx, rhs_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            lhs_map = _build_nisa_map(rctx.ctx, rank, lhs_acc)
            rhs_map = _build_nisa_map(rctx.ctx, rank, rhs_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            nisa.tensor_tensor_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                **_operand_kwargs("lhs", lhs_acc, lhs_map, out_shape),
                **_operand_kwargs("rhs", rhs_acc, rhs_map, out_shape),
                op=arith_kind,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return

    if in0_bcast != in1_bcast:
        tensor_v = in1 if in0_bcast else in0
        vec_v = in0 if in0_bcast else in1
        vec_shape = in0_shape if in0_bcast else in1_shape
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        block_arg1 = op.regions[0].blocks[0].arguments[1]
        vec_arg = block_arg0 if in0_bcast else block_arg1
        vec_is_lhs = str(body_lhs) == str(vec_arg)
        reverse = (
            nisa.TensScalarRevOps.First if vec_is_lhs else nisa.TensScalarRevOps.None_
        )
        with nk_ir.InsertionPoint(op), rctx.loc:
            src_acc = _get_base_and_offsets(rctx.ctx, tensor_v, rctx.loc)
            vec_acc = _get_base_and_offsets(rctx.ctx, vec_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
            vec_map = _build_nisa_map(rctx.ctx, rank, vec_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            # Match the deleted C++ pattern: all operands use
            # tile_par_dims = rank - 1 so the broadcast operand's
            # free-dim product = 1 (the broadcast dimension), which
            # NISA's tensor_scalar_arith verifier requires. The vec
            # operand carries its own shape, not the full out_shape.
            par_dims = rank - 1
            nisa.tensor_scalar_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape, par_dims),
                **_operand_kwargs("src", src_acc, src_map, out_shape, par_dims),
                **_operand_kwargs("operand0", vec_acc, vec_map, vec_shape, par_dims),
                **_empty_operand_kwargs("operand1"),
                op0=arith_kind,
                op1=None,
                reverse_operands=reverse,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return


def _build_dma_copy_indirect_op(
    *,
    ctx: nk_ir.Context,
    loc: nk_ir.Location,
    dst_memloc: nk_ir.Value,
    dst_indices: list[nk_ir.Value],
    dst_ap: nk_ir.Attribute,
    dst_tile_shape: list[int],
    src_memloc: nk_ir.Value,
    src_indices: list[nk_ir.Value],
    src_ap: nk_ir.Attribute,
    src_tile_shape: list[int],
    src_index_memloc: nk_ir.Value,
    src_index_ap: nk_ir.Attribute,
    src_index_tile_shape: list[int],
    src_indirect_max_index: int,
    tile_par_dims: int = 1,
) -> None:
    """Raw builder for nisa.dma_copy_indirect.

    We bypass the generated Python builder because it translates
    ``dst_indirect_max_index=None`` into ``[0 : i32]``, which the
    verifier then rejects since there's no matching ``dst_index``
    operand.  Here we set the attribute to an empty ArrayAttr so the
    verifier sees both ``dst_index`` and ``dst_indirect_max_index`` as
    absent.
    """
    i32_ty = nk_ir.IntegerType.get_signless(32, ctx)
    bool_true = nk_ir.BoolAttr.get(True, ctx)

    operands = [dst_memloc, *dst_indices, src_memloc, *src_indices,
                src_index_memloc]
    # operandSegmentSizes: 13 segments in declaration order.
    seg_sizes = [
        1, len(dst_indices), 0,        # dst
        1, len(src_indices), 0,        # src
        1, 0, 0,                       # src_index
        0, 0, 0,                       # dst_index (absent)
        0,                             # dma_qos (absent)
    ]
    attrs = {
        "dst_ap": dst_ap,
        "dst_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            dst_tile_shape, ctx
        ),
        "dst_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "src_ap": src_ap,
        "src_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            src_tile_shape, ctx
        ),
        "src_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "src_index_ap": src_index_ap,
        "src_index_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            src_index_tile_shape, ctx
        ),
        "src_index_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "dst_index_static_tile_shape": nk_ir.DenseI64ArrayAttr.get([], ctx),
        "dst_index_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, 0),
        "src_indirect_max_index": nk_ir.ArrayAttr.get(
            [nk_ir.IntegerAttr.get(i32_ty, src_indirect_max_index)], ctx,
        ),
        "dst_indirect_max_index": nk_ir.ArrayAttr.get([], ctx),
        "dst_rmw_op": nk_ir.ArrayAttr.get([], ctx),
        "oob_is_err": bool_true,
        "unique_indices": bool_true,
        "engine": nk_ir.IntegerAttr.get(i32_ty, nisa.Engine.DMA.value),
        "operandSegmentSizes": nk_ir.DenseI32ArrayAttr.get(seg_sizes, ctx),
    }
    nk_ir.Operation.create(
        "nisa.dma_copy_indirect",
        results=[],
        operands=operands,
        attributes=attrs,
        loc=loc,
    )


# ---------------------------------------------------------------------------
# nkipy.gather -> nisa.dma_copy_indirect
# ---------------------------------------------------------------------------
#
# Port of the deleted C++ NkipyGatherToNisaPattern.  The source table stays
# in HBM and each iteration gathers one row (or one partition's worth of
# rows) into SBUF via dma_copy_indirect, then DMAs the result back to the
# output HBM tensor.
#
# Layouts:
#   2D output [N, H]:  single gather of N rows (one index per partition).
#   3D output [N, I, H]: wrap in scf.for over I, gathering one row/partition
#                        per iteration; output[:, i, :] = source[indices[:, i]].
#
# The indirect DMA uses three specially-shaped affine maps:
#   dst     : standard [d0, d1]        — fill the SBUF output tile
#   src     : [s0, d1 + s1]            — look up one row; d0 is unused, the
#                                         row index comes from the index buffer
#   index   : standard [d0, d1]        — read N indices, one per partition


def _gather_src_indirect_map(ctx: nk_ir.Context) -> nk_ir.Attribute:
    # (d0, d1)[s0, s1] -> [s0, d1 + s1]
    s0 = nk_ir.AffineSymbolExpr.get(0)
    d1 = nk_ir.AffineDimExpr.get(1)
    s1 = nk_ir.AffineSymbolExpr.get(1)
    amap = nk_ir.AffineMap.get(2, 2, [s0, d1 + s1])
    return nisa.flatten_affine_map(amap, ctx)


def _gather_standard_2d_map(ctx: nk_ir.Context, num_symbols: int) -> nk_ir.Attribute:
    # Standard 2D map with d0/d1 on each dim, plus per-dim symbol offsets.
    d0 = nk_ir.AffineDimExpr.get(0)
    d1 = nk_ir.AffineDimExpr.get(1)
    exprs: list[nk_ir.AffineExpr] = []
    if num_symbols >= 1:
        exprs.append(d0 + nk_ir.AffineSymbolExpr.get(0))
    else:
        exprs.append(d0)
    if num_symbols >= 2:
        exprs.append(d1 + nk_ir.AffineSymbolExpr.get(1))
    else:
        exprs.append(d1)
    return nisa.flatten_affine_map(
        nk_ir.AffineMap.get(2, num_symbols, exprs), ctx
    )


def _emit_gather_iteration(
    rctx: _RewriteContext,
    indices_sbuf: nk_ir.Value,
    sbuf_output: nk_ir.Value,
    idx_access: _Access,
    src_access: _Access,
    out_access: _Access,
    zero_idx: nk_ir.Value,
    idx_offset: nk_ir.Value,
    out_indices: list[nk_ir.Value],
    N: int,
    H: int,
    base_v: int,
    tile_par_dims: int = 1,
) -> None:
    ctx = rctx.ctx
    loc = rctx.loc

    # --- 1. DMA one "column" of indices from HBM/SBUF into the indices SBUF ---
    dst_map_std = _gather_standard_2d_map(ctx, num_symbols=2)
    if not idx_access.indices:
        idx_src_indices = [zero_idx, idx_offset]
    else:
        idx_src_indices = list(idx_access.indices)
        # Add idx_offset to the innermost index
        idx_src_indices[-1] = _emit_addi(
            idx_src_indices[-1], idx_offset, ctx, loc
        )
    idx_src_map = _build_nisa_map(
        ctx, 2,
        _Access(
            base=idx_access.base,
            indices=idx_src_indices,
            base_type=idx_access.base_type,
            dropped_dims=idx_access.dropped_dims,
        ),
    )
    nisa.dma_copy(
        dst_memloc=indices_sbuf,
        dst_indices=[zero_idx, zero_idx],
        dst_ap=dst_map_std,
        dst_static_tile_shape=[N, 1],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=idx_access.base,
        src_indices=idx_src_indices,
        src_ap=idx_src_map,
        src_static_tile_shape=[N, 1],
        src_tile_par_dims=tile_par_dims,
        oob_is_err=True,
        engine=nisa.Engine.DMA,
    )

    # --- 2. dma_copy_indirect: gather H elements from HBM using SBUF indices ---
    # dst: standard [d0, d1] 2D map (no symbols — direct to SBUF alloc base).
    gather_dst_map = nisa.flatten_affine_map(
        nk_ir.AffineMap.get(
            2, 0,
            [nk_ir.AffineDimExpr.get(0), nk_ir.AffineDimExpr.get(1)],
        ),
        ctx,
    )
    # src: (d0, d1)[s0, s1] -> [s0, d1 + s1] — s0 is the indirect row index
    # supplied by the index buffer; d1+s1 covers the gather's free dim.
    gather_src_map = _gather_src_indirect_map(ctx)
    col_offset = (
        src_access.indices[1]
        if len(src_access.indices) > 1 else zero_idx
    )
    src_indirect_indices = [col_offset]
    # index: standard [d0, d1] 2D map, no symbol offsets
    index_map = nisa.flatten_affine_map(
        nk_ir.AffineMap.get(
            2, 0,
            [nk_ir.AffineDimExpr.get(0), nk_ir.AffineDimExpr.get(1)],
        ),
        ctx,
    )

    # Emit the op via builder and then repair its inherent properties.
    # The Python builder sets `dst_indirect_max_index = [0 : i32]` when
    # no dst_index is provided, which the verifier rejects.  MLIR's
    # `dma_op.attributes[...]` assignment adds a *discardable* attribute
    # but leaves the op's inherent `Properties` struct untouched, so we
    # rebuild the op by printing it, textually clearing the bogus attr,
    # and parsing back in place. Same trick the C++ side used implicitly
    # by constructing the op in one shot.
    dma_op = nisa.dma_copy_indirect(
        dst_memloc=sbuf_output,
        dst_indices=[],
        dst_ap=gather_dst_map,
        dst_static_tile_shape=[N, H],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=src_access.base,
        src_indices=src_indirect_indices,
        src_ap=gather_src_map,
        src_static_tile_shape=[N, H],
        src_tile_par_dims=tile_par_dims,
        src_index_memloc=indices_sbuf,
        src_index_indices=[],
        src_index_ap=index_map,
        src_index_static_tile_shape=[N, 1],
        src_index_tile_par_dims=tile_par_dims,
        dst_index_memloc=None,
        dst_index_indices=[],
        dst_index_ap=None,
        dst_index_static_tile_shape=[],
        dst_index_tile_par_dims=0,
        oob_is_err=True,
        src_indirect_max_index=base_v,
        unique_indices=True,
        engine=nisa.Engine.DMA,
    )

    # --- 3. DMA the gathered SBUF tile back out to the output HBM buffer ---
    src_copy_map = _gather_standard_2d_map(ctx, num_symbols=2)
    dst_copy_map = _build_nisa_map(
        ctx, 2,
        _Access(
            base=out_access.base,
            indices=out_indices,
            base_type=out_access.base_type,
            dropped_dims=out_access.dropped_dims,
        ),
    )
    nisa.dma_copy(
        dst_memloc=out_access.base,
        dst_indices=out_indices,
        dst_ap=dst_copy_map,
        dst_static_tile_shape=[N, H],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=sbuf_output,
        src_indices=[zero_idx, zero_idx],
        src_ap=src_copy_map,
        src_static_tile_shape=[N, H],
        src_tile_par_dims=tile_par_dims,
        oob_is_err=True,
        engine=nisa.Engine.DMA,
    )


@pattern("nkipy.gather")
def _rewrite_nkipy_gather(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 3:
        return
    source = operands[0]
    indices = operands[1]
    output = operands[2]

    src_ty = source.type
    idx_ty = indices.type
    out_ty = output.type
    if not (isinstance(src_ty, nk_ir.MemRefType)
            and isinstance(idx_ty, nk_ir.MemRefType)
            and isinstance(out_ty, nk_ir.MemRefType)):
        return

    src_shape = _static_shape(src_ty)
    out_shape = _static_shape(out_ty)
    if src_shape is None or out_shape is None:
        return
    if len(src_shape) != 2:
        return
    out_rank = len(out_shape)
    if out_rank < 2 or out_rank > 3:
        return

    H = src_shape[1]
    N = out_shape[0]
    base_v = src_shape[0]
    ctx = rctx.ctx

    sbuf_attr = nk_ir.Attribute.parse("#nisa.mem<sbuf>")
    i32_ty = nk_ir.IntegerType.get_signless(32, ctx)
    idx_elt_ty = idx_ty.element_type  # type: ignore[attr-defined]
    out_elt_ty = out_ty.element_type  # type: ignore[attr-defined]

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(ctx, source, rctx.loc)
        idx_acc = _get_base_and_offsets(ctx, indices, rctx.loc)
        out_acc = _get_base_and_offsets(ctx, output, rctx.loc)

        zero_idx = _emit_const_index(ctx, 0, rctx.loc)

        indices_sbuf_ty = nk_ir.MemRefType.get(
            [N, 1], idx_elt_ty, memory_space=sbuf_attr,
        )
        indices_sbuf = nisa.alloc(memref_type=indices_sbuf_ty, alignment=0)
        sbuf_output_ty = nk_ir.MemRefType.get(
            [N, H], out_elt_ty, memory_space=sbuf_attr,
        )
        sbuf_output = nisa.alloc(memref_type=sbuf_output_ty, alignment=0)

        if out_rank == 3:
            I_size = out_shape[1]
            upper = _emit_const_index(ctx, I_size, rctx.loc)
            one = _emit_const_index(ctx, 1, rctx.loc)

            # scf.for_ is a generator that yields the induction variable
            # inside its body's insertion point.  With no iter_args the
            # single yield is the IV; the body terminates with scf.yield.
            from nki.compiler._internal.dialects import scf as _scf
            for iv in _scf.for_(zero_idx, upper, one, iter_args=[]):
                out_indices = list(out_acc.indices) if out_acc.indices \
                    else [zero_idx, zero_idx, zero_idx]
                # Insert the loop IV at the second dim (I) of the output.
                if len(out_indices) == 3:
                    out_indices[1] = _emit_addi(
                        out_indices[1], iv, ctx, rctx.loc
                    )
                _emit_gather_iteration(
                    rctx, indices_sbuf, sbuf_output,
                    idx_acc, src_acc, out_acc,
                    zero_idx, iv, out_indices,
                    N, H, base_v,
                )
                _scf.yield_([])
        else:
            out_indices = list(out_acc.indices) if out_acc.indices \
                else [zero_idx, zero_idx]
            _emit_gather_iteration(
                rctx, indices_sbuf, sbuf_output,
                idx_acc, src_acc, out_acc,
                zero_idx, zero_idx, out_indices,
                N, H, base_v,
            )

        nisa.release(memref=indices_sbuf)
        nisa.release(memref=sbuf_output)

    op.operation.erase()


# ---------------------------------------------------------------------------
# Post-pass: fold HBM collapse_shape/expand_shape into the nisa.alloc.
# ---------------------------------------------------------------------------


def _alloc_defining_op(v: nk_ir.Value):
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "nisa.alloc":
        return None
    return op


def _is_block_arg(v: nk_ir.Value) -> bool:
    owner = getattr(v, "owner", None)
    return isinstance(owner, nk_ir.Block)


def _try_fold_hbm_reshape_alloc(rctx: _RewriteContext, op: nk_ir.OpView) -> bool:
    src = op.operation.operands[0]
    src_ty = src.type
    if not _is_hbm(src_ty):
        return False
    alloc_op = _alloc_defining_op(src)
    if alloc_op is None:
        return False
    dst_ty = op.operation.results[0].type
    if not isinstance(dst_ty, nk_ir.MemRefType):
        return False

    alignment = 0
    if "alignment" in alloc_op.attributes:
        alignment = nk_ir.IntegerAttr(alloc_op.attributes["alignment"]).value

    src_shape = list(getattr(src_ty, "shape", ()))
    src_elt = src_ty.element_type  # type: ignore[attr-defined]

    with nk_ir.InsertionPoint(alloc_op), rctx.loc:
        new_alloc = nisa.alloc(memref_type=dst_ty, alignment=alignment)

    for user in list(alloc_op.result.uses):
        user_op = user.owner
        if user_op.name != "nisa.dma_copy":
            continue
        if user_op.operands[0] != alloc_op.result:
            continue
        user_op.operands[0] = new_alloc
        existing = (
            user_op.attributes["dst_shape"]
            if "dst_shape" in user_op.attributes else None
        )
        if existing is None or str(existing) in ("", "array<i64>"):
            user_op.attributes["dst_shape"] = nk_ir.DenseI64ArrayAttr.get(src_shape)
        user_op.attributes["dst_elt_ty"] = nk_ir.TypeAttr.get(src_elt)

    op.operation.results[0].replace_all_uses_with(new_alloc)
    op.operation.erase()
    if not list(alloc_op.result.uses):
        alloc_op.erase()
    return True


def _try_fold_hbm_reshape_arg(rctx: _RewriteContext, op: nk_ir.OpView) -> bool:
    src = op.operation.operands[0]
    src_ty = src.type
    if not _is_hbm(src_ty):
        return False
    if not _is_block_arg(src):
        return False
    src_shape = list(getattr(src_ty, "shape", ()))
    src_elt = src_ty.element_type  # type: ignore[attr-defined]
    users = list(op.operation.results[0].uses)
    if not users:
        op.operation.erase()
        return True
    for user in users:
        user_op = user.owner
        if user_op.name != "nisa.dma_copy":
            return False
        if user_op.operands[1] != op.operation.results[0]:
            return False

    for user in users:
        user_op = user.owner
        user_op.operands[1] = src
        existing = (
            user_op.attributes["src_shape"]
            if "src_shape" in user_op.attributes else None
        )
        if existing is None or str(existing) in ("", "array<i64>"):
            user_op.attributes["src_shape"] = nk_ir.DenseI64ArrayAttr.get(src_shape)
        user_op.attributes["src_elt_ty"] = nk_ir.TypeAttr.get(src_elt)

    op.operation.erase()
    return True


def _fold_hbm_reshapes(rctx: _RewriteContext) -> None:
    while True:
        candidates: list[nk_ir.OpView] = []

        def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
            if op_handle.name in ("memref.collapse_shape", "memref.expand_shape"):
                candidates.append(op_handle.opview)
            return nk_ir.WalkResult.ADVANCE

        rctx.module.operation.walk(visit)
        progressed = False
        for op in candidates:
            if _try_fold_hbm_reshape_alloc(rctx, op):
                progressed = True
                continue
            if _try_fold_hbm_reshape_arg(rctx, op):
                progressed = True
        if not progressed:
            return


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


_DEAD_VIEW_OPS = (
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def _dce_dead_view_ops(rctx: _RewriteContext) -> None:
    while True:
        dead: list[nk_ir.OpView] = []

        def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
            if (
                op_handle.name in _DEAD_VIEW_OPS
                and not list(op_handle.results[0].uses)
            ):
                dead.append(op_handle.opview)
            return nk_ir.WalkResult.ADVANCE

        rctx.module.operation.walk(visit)
        if not dead:
            return
        for op in dead:
            op.operation.erase()


def _walk_and_rewrite(rctx: _RewriteContext) -> None:
    candidates: list[nk_ir.OpView] = []

    def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
        name = op_handle.name
        if name in _PATTERNS:
            candidates.append(op_handle.opview)
        return nk_ir.WalkResult.ADVANCE

    rctx.module.operation.walk(visit)

    for op in candidates:
        _PATTERNS[op.operation.name](rctx, op)

    _dce_dead_view_ops(rctx)
    _fold_reinterpret_casts(rctx)
    _dce_dead_view_ops(rctx)
    _fold_hbm_reshapes(rctx)


def _resolve_custom_ops(module: nk_ir.Module, ctx: nk_ir.Context) -> None:
    """Python port of the deleted C++ ResolveCustomOpsPass.

    Replaces calls to custom-op declarations with the inlined NISA body
    stashed in the module attribute ``nkipy.custom_op_bodies`` (a dict of
    funcname → MLIR-text NISA body).  Matches both conventions:

    - *Output-as-argument* (`func @f(%in, %out) { return }`): allocate
      buffers for the trailing arguments and replace call results with
      the allocated buffers.
    - *Return-value* (kernel_builder, `func @f(%in) -> %out`): splice the
      body and rewire `func.return` operands to call results.

    Must run before ``_finalize_for_nki`` which strips ``nkipy.*`` attrs.
    """
    module_op = module.operation
    if "nkipy.custom_op_bodies" not in module_op.attributes:
        return
    bodies_attr = nk_ir.DictAttr(module_op.attributes["nkipy.custom_op_bodies"])

    # Collect custom-op declarations (body-less func.func with
    # `nkipy.custom_op` attr).
    decls: list[tuple[str, nk_ir.OpView]] = []
    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        if "nkipy.custom_op" not in op.attributes:
            continue
        # A declaration has an empty region body.
        regions = list(op.operation.regions)
        if regions and list(regions[0].blocks):
            # Not a pure declaration; skip (body already present).
            continue
        sym = nk_ir.StringAttr(op.attributes["sym_name"]).value
        decls.append((sym, op))

    if not decls:
        del module_op.attributes["nkipy.custom_op_bodies"]
        return

    for func_name, decl_op in decls:
        # Look up the stashed NISA body string.
        if func_name not in bodies_attr:
            raise RuntimeError(
                f"no stashed NISA body for custom op '{func_name}'"
            )
        body_text = nk_ir.StringAttr(bodies_attr[func_name]).value
        body_module = nk_ir.Module.parse(body_text, ctx)

        # Find the (single) non-declaration func in the parsed body.
        nisa_func = None
        for bop in body_module.body.operations:
            if bop.operation.name != "func.func":
                continue
            regions = list(bop.operation.regions)
            if regions and list(regions[0].blocks):
                nisa_func = bop
                break
        if nisa_func is None:
            raise RuntimeError(
                f"no function body in stashed NISA module for '{func_name}'"
            )

        func_ty = nk_ir.TypeAttr(nisa_func.attributes["function_type"]).value
        num_results = len(list(func_ty.results))  # type: ignore[attr-defined]
        num_args = len(list(func_ty.inputs))  # type: ignore[attr-defined]
        # Output-names drives the output-as-argument split.
        if "nki.output_names" in nisa_func.attributes:
            num_outputs = len(list(
                nk_ir.ArrayAttr(nisa_func.attributes["nki.output_names"])
            ))
        else:
            num_outputs = 0

        is_return_value_style = num_results > 0
        if is_return_value_style:
            num_inputs = num_args
            num_outputs = num_results
        else:
            num_inputs = num_args - num_outputs

        # Collect all call sites for this custom op across the module.
        call_sites: list[nk_ir.OpView] = []

        def collect_calls(op: nk_ir.Operation) -> nk_ir.WalkResult:
            if op.name == "func.call":
                callee = nk_ir.FlatSymbolRefAttr(
                    op.attributes["callee"]
                ).value
                if callee == func_name:
                    call_sites.append(op.opview)
            return nk_ir.WalkResult.ADVANCE

        module.operation.walk(collect_calls)

        nisa_block = list(nisa_func.operation.regions[0].blocks)[0]
        nisa_args = list(nisa_block.arguments)

        for call_op in call_sites:
            call_operation = call_op.operation
            call_operands = list(call_operation.operands)

            with nk_ir.InsertionPoint(call_operation), call_operation.location:

                def maybe_cast(arg: nk_ir.Value, expected: nk_ir.Type) -> nk_ir.Value:
                    if str(arg.type) == str(expected):
                        return arg
                    v = arg
                    while True:
                        owner = getattr(v, "owner", None)
                        if owner is None:
                            break
                        owner_op = owner.opview if hasattr(owner, "opview") else owner
                        if getattr(owner_op, "name", None) != "memref.cast":
                            break
                        src = owner_op.operands[0]
                        if str(src.type) == str(expected):
                            return src
                        v = src
                    cast_op = nk_ir.Operation.create(
                        "memref.cast",
                        results=[expected],
                        operands=[arg],
                        loc=call_operation.location,
                    )
                    return cast_op.result

                # `pairs` maps every NISA-body Value that must be
                # rewritten (block args + cloned-op results) to its
                # replacement in the caller. Lookups are O(len(pairs))
                # which is fine for the short bodies we inline.
                pairs: list[tuple[nk_ir.Value, nk_ir.Value]] = []

                if is_return_value_style:
                    for nisa_arg, call_arg in zip(nisa_args, call_operands):
                        pairs.append(
                            (nisa_arg, maybe_cast(call_arg, nisa_arg.type))
                        )
                    return_operands: list[nk_ir.Value] = []
                    for body_op in list(nisa_block.operations):
                        if body_op.operation.name == "func.return":
                            for op_v in body_op.operation.operands:
                                replacement = op_v
                                for old, new in pairs:
                                    if op_v == old:
                                        replacement = new
                                        break
                                return_operands.append(replacement)
                            continue
                        _clone_op_with_map(body_op, pairs, pairs, ctx)
                    for i, retv in enumerate(return_operands):
                        call_op.results[i].replace_all_uses_with(retv)
                else:
                    # Input args come first.
                    for i in range(num_inputs):
                        pairs.append(
                            (nisa_args[i],
                             maybe_cast(call_operands[i], nisa_args[i].type))
                        )
                    # Allocate outputs for trailing NISA args.
                    out_allocs: list[nk_ir.Value] = []
                    for i in range(num_outputs):
                        nisa_out_ty = nisa_args[num_inputs + i].type
                        alloc_op = nk_ir.Operation.create(
                            "memref.alloc",
                            results=[nisa_out_ty],
                            operands=[],
                            attributes={
                                "operandSegmentSizes":
                                    nk_ir.DenseI32ArrayAttr.get([0, 0], ctx),
                            },
                            loc=call_operation.location,
                        )
                        pairs.append(
                            (nisa_args[num_inputs + i], alloc_op.result)
                        )
                        out_allocs.append(alloc_op.result)
                    for body_op in list(nisa_block.operations):
                        if body_op.operation.name == "func.return":
                            continue
                        _clone_op_with_map(body_op, pairs, pairs, ctx)
                    for i in range(num_outputs):
                        call_op.results[i].replace_all_uses_with(out_allocs[i])

            call_operation.erase()

        # Fix enclosing function return types after inlining — NISA-body
        # result types may differ from the caller's declared return type
        # (e.g. non-strided vs strided memrefs).
        for op in module.body.operations:
            if op.operation.name != "func.func":
                continue
            if "nkipy.custom_op" in op.attributes:
                continue
            regions = list(op.operation.regions)
            if not regions or not list(regions[0].blocks):
                continue
            block = list(regions[0].blocks)[0]
            term = block.operations[len(list(block.operations)) - 1]
            if term.operation.name != "func.return":
                continue
            ret_types = [v.type for v in term.operation.operands]
            func_ty_attr = op.attributes["function_type"]
            cur_ty = nk_ir.TypeAttr(func_ty_attr).value
            cur_results = list(cur_ty.results)  # type: ignore[attr-defined]
            if len(ret_types) == len(cur_results) and all(
                str(a) == str(b) for a, b in zip(ret_types, cur_results)
            ):
                continue
            new_inputs = list(cur_ty.inputs)  # type: ignore[attr-defined]
            new_ty = nk_ir.FunctionType.get(new_inputs, ret_types)
            op.attributes["function_type"] = nk_ir.TypeAttr.get(new_ty)

        decl_op.operation.erase()

    del module_op.attributes["nkipy.custom_op_bodies"]


def _clone_op_with_map(
    src_op: nk_ir.OpView,
    pairs: list[tuple[nk_ir.Value, nk_ir.Value]],
    results_pairs: list[tuple[nk_ir.Value, nk_ir.Value]],
    ctx: nk_ir.Context,
) -> None:
    """Clone ``src_op`` at the current insertion point, rebinding any
    operand appearing in ``pairs`` to its partner.  Each entry in
    ``pairs`` is (original_value, replacement_value); we do a linear
    scan on the clone's operands comparing by MLIR Value equality.
    Newly-produced results are appended to ``results_pairs`` so later
    operations in the same body can find them.
    """
    operation = src_op.operation
    # Clone inserts at the currently-active InsertionPoint.
    cloned = operation.clone()

    def find_replacement(v: nk_ir.Value) -> nk_ir.Value | None:
        for old, new in pairs:
            if v == old:
                return new
        return None

    def patch(op: nk_ir.Operation) -> nk_ir.WalkResult:
        for i in range(len(op.operands)):
            repl = find_replacement(op.operands[i])
            if repl is not None:
                op.operands[i] = repl
        return nk_ir.WalkResult.ADVANCE

    cloned.walk(patch)
    for i, r in enumerate(operation.results):
        results_pairs.append((r, cloned.results[i]))


def _finalize_for_nki(module: nk_ir.Module, ctx: nk_ir.Context, target: str) -> None:
    def strip_nkipy_attrs(op: nk_ir.Operation) -> nk_ir.WalkResult:
        to_remove = [
            named.name for named in op.attributes
            if named.name.startswith("nkipy.")
        ]
        for name in to_remove:
            del op.attributes[name]
        return nk_ir.WalkResult.ADVANCE

    module.operation.walk(strip_nkipy_attrs)

    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        if "nki.output_names" in op.attributes:
            continue
        func_ty_attr = op.attributes["function_type"]
        func_ty = nk_ir.TypeAttr(func_ty_attr).value
        num_results = len(list(func_ty.results))  # type: ignore[attr-defined]
        if num_results == 0:
            continue
        names = [
            f"output_{i}" if num_results > 1 else "output"
            for i in range(num_results)
        ]
        op.attributes["nki.output_names"] = nk_ir.ArrayAttr.get(
            [nk_ir.StringAttr.get(n) for n in names]
        )

    target_attr = nk_ir.Attribute.parse(f"#nisa.target<{target}>")
    module.operation.attributes["nisa.target"] = target_attr


def linalg_to_nisa(
    mlir_text: str, target: str = "trn2", print_generic: bool = True,
) -> str:
    """Translate post-Phase-4 MLIR to NISA MLIR (text -> text).

    Defaults to generic form because the NISA pretty-printer omits the element
    type for ``nisa.dma_copy``'s ``view(...)`` syntax, which its own parser
    then rejects. Generic form roundtrips cleanly through the downstream NKI
    parser. Callers that consume the output as text (STRING_CHECK/FILECHECK)
    should pass ``print_generic=False``.
    """
    ctx, module = _to_nki_module(mlir_text)
    with ctx:
        rctx = _RewriteContext(ctx, module)
        _walk_and_rewrite(rctx)
        _resolve_custom_ops(module, ctx)
        _finalize_for_nki(module, ctx, target)
        out = module.operation.get_asm(
            print_generic_op_form=print_generic, assume_verified=True
        )
    # Strip the bogus `dst_indirect_max_index = [0 : i32]` the Python
    # builder for nisa.dma_copy_indirect injects on gather-only ops.
    # The verifier requires it to be absent when dst_index is absent,
    # but we can't clear the inherent property from Python — just edit
    # the text on the way out.
    out = re.sub(
        r",\s*dst_indirect_max_index\s*=\s*\[0\s*:\s*i32\]",
        "",
        out,
    )
    return out


__all__ = [
    "linalg_to_nisa",
    "pattern",
]
