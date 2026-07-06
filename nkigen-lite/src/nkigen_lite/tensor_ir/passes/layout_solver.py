"""
Layout Solver: assigns each value a three-way I/P/F dim classification.

  I (iteration): loop indices, bare-int DMA, not in SBUF tile
  P (partition): SBUF dim-0, computes in parallel
  F (free):      SBUF dim-1, contiguous per partition

Whole dims are assigned to groups; dims are never split or reordered within
a group (each group keeps ascending dim order). A P-extent larger than the
hardware partition count (128) is legal here — downstream tiling
(compute_tile_sizes) chunks the P group to fit; the solver's default-layout
scoring merely stops rewarding P-extent beyond 128.

Solving is seeded by matmul hard constraints (tensor engine fixes P/F for
all three operands) and propagated outward through layout-preserving ops;
values demanded in conflicting layouts keep the first assignment (no
conversion is planned — every op currently round-trips through
layout-agnostic HBM, so a conflict costs correctness nothing today).

Uses nkigen_lite.core.Graph as the graph representation (SSA-based IR with
object references and use-lists).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nkigen_lite.core import Graph, Op, Value
from nkigen_lite.tensor_ir.passes.hardware import TRN2

PARTITION_MAX = TRN2.partition_max


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Layout assignment for a tensor: three-way I/P/F classification.

    i_dims — iteration dims (bare-int loop indices, not in SBUF tile)
    p_dims — partition dims (SBUF dim-0; extent may exceed 128, tiling
             chunks it to the hardware partition count downstream)
    f_dims — free dims (SBUF dim-1, contiguous per partition)

    Invariant: dims within each group are always sorted ascending (row-major
    canonical form). Two Layouts with the same I/P/F group membership are equal
    regardless of the order in which dims were originally specified.
    """

    i_dims: tuple[int, ...]
    p_dims: tuple[int, ...]
    f_dims: tuple[int, ...]

    def __post_init__(self):
        # Normalize: sort dims within each group to enforce canonical form.
        object.__setattr__(self, "i_dims", tuple(sorted(self.i_dims)))
        object.__setattr__(self, "p_dims", tuple(sorted(self.p_dims)))
        object.__setattr__(self, "f_dims", tuple(sorted(self.f_dims)))
        # Validate: no dim appears in multiple groups.
        all_dims = self.i_dims + self.p_dims + self.f_dims
        if len(all_dims) != len(set(all_dims)):
            raise ValueError(
                f"Layout has overlapping groups: I={self.i_dims}, P={self.p_dims}, F={self.f_dims}"
            )

    def p_extent(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[d] for d in self.p_dims) if self.p_dims else 1

    def f_extent(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[d] for d in self.f_dims) if self.f_dims else 1

    def is_valid(self, shape: tuple[int, ...]) -> bool:
        all_dims = self.i_dims + self.p_dims + self.f_dims
        if len(all_dims) != len(shape):
            return False
        if set(all_dims) != set(range(len(shape))):
            return False
        return True


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _value_shape(v: Value) -> tuple[int, ...]:
    """Extract shape from a core.Value (whose type is TensorType)."""
    return v.type.shape


def _all_values(graph: Graph) -> dict[str, Value]:
    """Build a name→Value map for a graph (inputs + all op results)."""
    vals: dict[str, Value] = {}
    for v in graph.inputs:
        vals[v.name] = v
    for op in graph.ops:
        for r in op.results:
            vals[r.name] = r
    return vals


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def get_matmul_layouts(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
) -> tuple[Layout, Layout, Layout]:
    """Determine fixed layouts for matmul operands.

    A[..., M, K] @ B[..., K, N] → C[..., M, N]
    Stationary A: I=batch, P={K}, F={M}
    Moving B:     I=batch, P={K}, F={N}
    Output C:     I=batch, P={M}, F={N}
    """
    a_rank = len(a_shape)
    b_rank = len(b_shape)
    c_rank = len(c_shape)

    a_m_idx = a_rank - 2
    a_k_idx = a_rank - 1
    b_k_idx = b_rank - 2
    b_n_idx = b_rank - 1
    c_m_idx = c_rank - 2
    c_n_idx = c_rank - 1

    a_batch = tuple(range(a_rank - 2))
    b_batch = tuple(range(b_rank - 2))
    c_batch = tuple(range(c_rank - 2))

    a_layout = Layout(i_dims=a_batch, p_dims=(a_k_idx,), f_dims=(a_m_idx,))
    b_layout = Layout(i_dims=b_batch, p_dims=(b_k_idx,), f_dims=(b_n_idx,))
    c_layout = Layout(i_dims=c_batch, p_dims=(c_m_idx,), f_dims=(c_n_idx,))

    return a_layout, b_layout, c_layout


def _is_last_two_swap(perm: tuple[int, ...]) -> bool:
    """Check if permutation only swaps the last two dimensions."""
    n = len(perm)
    if n < 2:
        return False
    return (
        perm[n - 1] == n - 2
        and perm[n - 2] == n - 1
        and all(perm[i] == i for i in range(n - 2))
    )


def solve_graph(graph: Graph) -> dict[str, Layout]:
    """Assign layouts to all values in the graph."""
    layouts: dict[str, Layout] = {}
    values = _all_values(graph)

    # --- Phase 1: Seed matmul hard constraints ---
    frozen: set[str] = set()
    for op in graph.ops:
        if op.opcode == "matmul":
            a_val, b_val = op.inputs[0], op.inputs[1]
            c_val = op.results[0]
            a_layout, b_layout, c_layout = get_matmul_layouts(
                _value_shape(a_val), _value_shape(b_val), _value_shape(c_val)
            )
            for val, layout in [(a_val, a_layout), (b_val, b_layout), (c_val, c_layout)]:
                if val.name not in layouts:
                    layouts[val.name] = layout
                frozen.add(val.name)

    # --- Phase 2: Classify transposes and propagate through trivial ones ---
    graph_output_names = {v.name for v in graph.outputs.values()}
    opaque: set[str] = set()
    for op in graph.ops:
        if op.opcode != "transpose":
            continue
        perm = op.attrs.get("perm", ())
        if not _is_last_two_swap(perm):
            out_name = op.results[0].name
            if out_name not in frozen and out_name not in graph_output_names:
                opaque.add(out_name)
        else:
            out_val = op.results[0]
            if out_val.name not in frozen:
                continue
            inp_val = op.inputs[0]
            if inp_val.name in frozen:
                continue
            out_layout = layouts[out_val.name]
            inp_shape = _value_shape(inp_val)
            new_i = tuple(perm[d] for d in out_layout.i_dims)
            new_p = tuple(perm[d] for d in out_layout.p_dims)
            new_f = tuple(perm[d] for d in out_layout.f_dims)
            candidate = Layout(i_dims=new_i, p_dims=new_p, f_dims=new_f)
            if candidate.is_valid(inp_shape):
                layouts[inp_val.name] = candidate

    # --- Phase 3: Seed graph inputs with defaults ---
    for v in graph.inputs:
        if v.name not in layouts:
            layouts[v.name] = _default_layout(_value_shape(v))

    # --- Phase 4: Forward propagation ---
    unresolved: list[Op] = []
    for op in graph.ops:
        if op.opcode == "matmul":
            continue
        out_val = op.results[0]
        if out_val.name in layouts or out_val.name in opaque:
            continue
        out_shape = _value_shape(out_val)

        candidate = None
        for inp_val in op.inputs:
            if inp_val.name not in layouts:
                continue
            c = _adapt_layout(layouts[inp_val.name], _value_shape(inp_val), out_shape, op)
            if not c or not c.is_valid(out_shape):
                continue
            if candidate is None:
                candidate = c
            elif any(f < p for f in candidate.f_dims for p in candidate.p_dims):
                if not any(f < p for f in c.f_dims for p in c.p_dims):
                    candidate = c

        if candidate:
            layouts[out_val.name] = candidate
        else:
            unresolved.append(op)

    # --- Phase 5: Backward propagation for unresolved values ---
    for op in reversed(unresolved):
        out_val = op.results[0]
        if out_val.name in layouts or out_val.name in opaque:
            continue
        out_shape = _value_shape(out_val)
        for consumer_op in out_val.uses:
            consumer_out = consumer_op.results[0]
            if consumer_out.name in layouts:
                candidate = _adapt_layout(
                    layouts[consumer_out.name], _value_shape(consumer_out), out_shape, op
                )
                if candidate and candidate.is_valid(out_shape):
                    layouts[out_val.name] = candidate
                    break

    # --- Phase 6: Backward propagation through reshape/broadcast/transpose chains ---
    op_producing: dict[str, Op] = {}
    for op in graph.ops:
        for r in op.results:
            op_producing[r.name] = op

    _CHAIN_OPS = {"reshape", "broadcast_to", "transpose"}

    def _chain_back(val_name: str, layout: Layout):
        if val_name not in op_producing:
            return
        op = op_producing[val_name]
        if op.opcode not in _CHAIN_OPS:
            return
        out_shape = _value_shape(op.results[0])
        for inp_val in op.inputs:
            if inp_val.name in frozen or inp_val.name in opaque:
                continue
            if len(inp_val.uses) > 1:
                continue
            inp_shape = _value_shape(inp_val)
            candidate = _adapt_layout(layout, out_shape, inp_shape, op)
            if not candidate or not candidate.is_valid(inp_shape):
                continue
            layouts[inp_val.name] = candidate
            _chain_back(inp_val.name, candidate)

    for name in frozen:
        if name in layouts:
            _chain_back(name, layouts[name])

    # --- Phase 7: Fill remaining with defaults ---
    for name, val in values.items():
        if name not in layouts and name not in opaque:
            layouts[name] = _default_layout(_value_shape(val))

    return layouts


def _propagate_reshape_layout(
    src_layout: Layout, src_shape: tuple[int, ...], dst_shape: tuple[int, ...]
) -> Layout | None:
    """Propagate layout through a cross-rank reshape using cumulative product matching.

    Maps each source dim block to destination dim block(s), preserving the
    I/P/F group assignment. Returns None if reshape crosses group boundaries.
    """
    src_rank = len(src_shape)
    dst_rank = len(dst_shape)

    def _group_of(d: int) -> str:
        if d in src_layout.i_dims:
            return "i"
        if d in src_layout.p_dims:
            return "p"
        return "f"

    new_i: list[int] = []
    new_p: list[int] = []
    new_f: list[int] = []
    si, di = 0, 0

    while si < src_rank and di < dst_rank:
        s_prod = src_shape[si]
        d_prod = dst_shape[di]
        s_start, d_start = si, di

        while s_prod != d_prod:
            if s_prod < d_prod:
                si += 1
                if si >= src_rank:
                    return None
                s_prod *= src_shape[si]
            else:
                di += 1
                if di >= dst_rank:
                    return None
                d_prod *= dst_shape[di]

        # Determine the effective group: ignore size-1 I-dims
        group = None
        for s in range(s_start, si + 1):
            g = _group_of(s)
            if g == "i" and src_shape[s] == 1:
                continue
            if group is None:
                group = g
            elif g != group:
                return None
        if group is None:
            group = "i"

        dst_dims = list(range(d_start, di + 1))
        if group == "i":
            new_i.extend(dst_dims)
        elif group == "p":
            new_p.extend(dst_dims)
        else:
            new_f.extend(dst_dims)
        si += 1
        di += 1

    if si != src_rank or di != dst_rank:
        return None

    candidate = Layout(i_dims=tuple(new_i), p_dims=tuple(new_p), f_dims=tuple(new_f))

    # Safety: reject if P-extent changes
    src_p_ext = src_layout.p_extent(src_shape)
    dst_p_ext = candidate.p_extent(dst_shape)
    if dst_p_ext != src_p_ext:
        return None

    return candidate


def _adapt_layout(
    src_layout: Layout,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    op: Op,
) -> Layout | None:
    """Adapt a layout from source to destination through the given op.

    Returns the adapted layout or None if the op is opaque/incompatible.
    """
    src_rank = len(src_shape)
    dst_rank = len(dst_shape)

    if op.opcode == "transpose":
        perm_attr = op.attrs.get("perm", tuple(range(dst_rank)))
        if not _is_last_two_swap(perm_attr):
            return None
        new_i = tuple(perm_attr[d] for d in src_layout.i_dims)
        new_p = tuple(perm_attr[d] for d in src_layout.p_dims)
        new_f = tuple(perm_attr[d] for d in src_layout.f_dims)
        return Layout(i_dims=new_i, p_dims=new_p, f_dims=new_f)

    if op.opcode == "reduce":
        assert op.attrs.get("keepdims", False), (
            f"reduce without keepdims=True not supported: {op}"
        )
        return src_layout

    if op.opcode in ("broadcast_to", "slice", "concat"):
        return src_layout

    if op.opcode == "reshape":
        if src_rank == dst_rank:
            return src_layout
        return _propagate_reshape_layout(src_layout, src_shape, dst_shape)

    # For elementwise ops (mul, add, sub, exp, etc.): same rank -> same layout
    if dst_rank == src_rank:
        return src_layout

    return None


def _default_layout(shape: tuple[int, ...]) -> Layout:
    """Assign a default layout using contiguous I|P|F splits.

    Scoring: (1 + K / f_extent) / utilization.
    Lower is better. Prefers layouts that maximize both partition
    utilization and f-extent (amortizing the per-iteration overhead K).
    """
    rank = len(shape)
    if rank == 0:
        return Layout(i_dims=(), p_dims=(), f_dims=())
    if rank == 1:
        return Layout(i_dims=(), p_dims=(), f_dims=(0,))
    if rank == 2:
        return Layout(i_dims=(), p_dims=(0,), f_dims=(1,))

    K = 1024  # per-iteration overhead in element-equivalents

    def _score(layout: Layout) -> float:
        p_ext = layout.p_extent(shape) if layout.p_dims else 1
        f_ext = layout.f_extent(shape) if layout.f_dims else 1
        util = min(p_ext, PARTITION_MAX) / PARTITION_MAX
        return (1.0 + K / f_ext) / util

    best_layout = Layout(i_dims=(), p_dims=(0,), f_dims=tuple(range(1, rank)))
    best_score = _score(best_layout)

    # Enumerate contiguous splits: dims [0:i_end) = I, [i_end:f_start) = P, [f_start:rank) = F
    for i_end in range(rank):
        for f_start in range(i_end + 1, rank):
            layout = Layout(
                i_dims=tuple(range(i_end)),
                p_dims=tuple(range(i_end, f_start)),
                f_dims=tuple(range(f_start, rank)),
            )
            s = _score(layout)
            if s < best_score:
                best_score = s
                best_layout = layout

    return best_layout


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_solution(graph: Graph, layouts: dict[str, Layout]):
    print(f"\n{'='*80}")
    print(f"  Layout Solution: {graph.name}")
    print(f"{'='*80}\n")

    print(f"{'Value':<25} {'Shape':<20} {'I-dims':<12} {'P-dims':<12} {'F-dims':<12}")
    print("-" * 82)

    values = _all_values(graph)
    all_names = [v.name for v in graph.inputs] + [op.results[0].name for op in graph.ops]
    for name in all_names:
        if name not in layouts:
            continue
        val = values[name]
        shape = _value_shape(val)
        layout = layouts[name]

        i_str = str(tuple(shape[d] for d in layout.i_dims)) if layout.i_dims else "()"
        p_str = str(tuple(shape[d] for d in layout.p_dims)) if layout.p_dims else "()"
        f_str = str(tuple(shape[d] for d in layout.f_dims)) if layout.f_dims else "()"

        short_name = name[:24]
        print(f"{short_name:<25} {str(shape):<20} {i_str:<12} {p_str:<12} {f_str:<12}")

    print()
