"""Insert dealloc ops after last use of on-chip allocations.

Operates on nki_ir.Graph (output of tile_and_lower). Computes liveness
for every alloc'd SBUF/PSUM value and inserts a dealloc immediately
after its last use, freeing on-chip memory for reuse.

Values that are already explicitly deallocated (e.g. PSUM after matmul)
are skipped. Graph outputs and values captured by sub-graphs (loop
bodies) have their lifetime extended appropriately.
"""

from __future__ import annotations

from nkigen_lite.core import Graph, Op, Value
from nkigen_lite.nki_ir.ir import MemorySpace, TileType

_NO_DST_ALIAS_OPS = frozenset({
    "alloc", "scalar_const", "affine", "reg_compare", "load_register",
    "dealloc", "fori_loop", "if_else", "while_loop", "constant",
})


def insert_deallocs(graph: Graph) -> int:
    """Insert dealloc ops for on-chip allocations at their last use point.

    Operates recursively on sub-graphs (fori_loop bodies, if_else branches)
    so that buffers allocated inside loops are freed within the same scope.

    Returns the number of dealloc ops inserted.
    """
    count = _insert_deallocs_in_graph(graph)

    for op in graph.ops:
        for attr_key in ("body", "then_body", "else_body", "cond_body", "body_body"):
            sub = op.attrs.get(attr_key) if op.attrs else None
            if sub is not None and isinstance(sub, Graph):
                count += insert_deallocs(sub)

    return count


def _insert_deallocs_in_graph(graph: Graph) -> int:
    """Insert deallocs in a single graph (non-recursive)."""
    alloc_values = _find_alloc_values(graph)
    if not alloc_values:
        return 0

    already_deallocd = _find_already_deallocd(graph, alloc_values)
    output_names = _output_alloc_roots(graph, alloc_values)

    last_use = _compute_last_use(graph, alloc_values)

    count = 0
    for alloc_name, op_idx in sorted(last_use.items(), key=lambda x: x[1], reverse=True):
        if alloc_name in already_deallocd:
            continue
        if alloc_name in output_names:
            continue
        alloc_val = alloc_values[alloc_name]
        _insert_dealloc_after(graph, alloc_val, op_idx)
        count += 1

    return count


def _find_alloc_values(graph: Graph) -> dict[str, Value]:
    """Find all on-chip alloc results (SBUF and PSUM)."""
    allocs: dict[str, Value] = {}
    for op in graph.ops:
        if op.opcode == "alloc" and op.results:
            val = op.result
            if isinstance(val.type, TileType) and val.type.memory in (
                MemorySpace.SBUF,
                MemorySpace.PSUM,
            ):
                allocs[val.name] = val
    return allocs


def _find_already_deallocd(graph: Graph, alloc_values: dict[str, Value]) -> set[str]:
    """Find alloc roots that already have an explicit dealloc (possibly via alias)."""
    alias_to_alloc = _build_alias_map(graph, alloc_values)
    deallocd: set[str] = set()
    for op in graph.ops:
        if op.opcode == "dealloc" and op.inputs:
            name = op.inputs[0].name
            root = alias_to_alloc.get(name, name)
            deallocd.add(root)
    return deallocd


def _output_alloc_roots(graph: Graph, alloc_values: dict[str, Value]) -> set[str]:
    """Find alloc roots that back graph output values (must not be deallocated)."""
    alias_to_alloc = _build_alias_map(graph, alloc_values)
    roots: set[str] = set()
    for v in graph.outputs.values():
        root = alias_to_alloc.get(v.name)
        if root is not None:
            roots.add(root)
    return roots


def _build_alias_map(graph: Graph, alloc_values: dict[str, Value]) -> dict[str, str]:
    """Build a map from every value name to its underlying alloc root.

    In nki_ir, all compute ops follow the dst-passing convention:
    input[0] is the pre-allocated destination buffer, and the result
    occupies that same buffer. So the result aliases input[0]'s alloc.
    """
    alias_to_alloc: dict[str, str] = {}
    for name in alloc_values:
        alias_to_alloc[name] = name

    for op in graph.ops:
        if op.opcode in _NO_DST_ALIAS_OPS:
            continue
        if not op.results:
            continue
        if not op.inputs:
            continue

        dst_input = op.inputs[0]
        if not isinstance(dst_input.type, TileType):
            continue
        if dst_input.type.memory not in (MemorySpace.SBUF, MemorySpace.PSUM):
            continue

        dst_root = alias_to_alloc.get(dst_input.name)
        if dst_root is not None:
            for r in op.results:
                alias_to_alloc[r.name] = dst_root

    return alias_to_alloc


def _collect_sub_graph_captures(op: Op) -> set[str]:
    """Collect value names used inside sub-graphs of a control-flow op."""
    captured: set[str] = set()
    for attr_key in ("body", "then_body", "else_body", "cond_body", "body_body"):
        sub = op.attrs.get(attr_key) if op.attrs else None
        if sub is None or not isinstance(sub, Graph):
            continue
        sub_defined = {v.name for v in sub.inputs}
        for sub_op in sub.ops:
            for r in sub_op.results:
                sub_defined.add(r.name)
        for sub_op in sub.ops:
            for inp in sub_op.inputs:
                if inp.name not in sub_defined:
                    captured.add(inp.name)
            for nested_key in ("body", "then_body", "else_body", "cond_body", "body_body"):
                nested = sub_op.attrs.get(nested_key) if sub_op.attrs else None
                if nested is not None and isinstance(nested, Graph):
                    nested_caps = _collect_nested_captures(nested, sub_defined)
                    captured.update(nested_caps)
    return captured


def _collect_nested_captures(graph: Graph, outer_defined: set[str]) -> set[str]:
    """Recursively collect captures from nested sub-graphs."""
    captured: set[str] = set()
    local_defined = {v.name for v in graph.inputs}
    for op in graph.ops:
        for r in op.results:
            local_defined.add(r.name)
    for op in graph.ops:
        for inp in op.inputs:
            if inp.name not in local_defined and inp.name not in outer_defined:
                captured.add(inp.name)
        for attr_key in ("body", "then_body", "else_body", "cond_body", "body_body"):
            nested = op.attrs.get(attr_key) if op.attrs else None
            if nested is not None and isinstance(nested, Graph):
                all_defined = local_defined | outer_defined
                nested_caps = _collect_nested_captures(nested, all_defined)
                captured.update(nested_caps)
    return captured


def _compute_last_use(
    graph: Graph,
    alloc_values: dict[str, Value],
) -> dict[str, int]:
    """Compute last-use op index for each alloc'd value.

    A value is "used" at op index i if:
    - It appears directly in op.inputs at index i (or any alias of it does)
    - It is captured by a sub-graph (fori_loop body, if_else body) at index i

    The last use is the latest such index across all aliases of the alloc.
    """
    alias_to_alloc = _build_alias_map(graph, alloc_values)
    last_use: dict[str, int] = {}
    ops = graph.ops

    for i, op in enumerate(ops):
        if op.opcode in ("alloc", "dealloc"):
            continue

        for inp in op.inputs:
            root = alias_to_alloc.get(inp.name)
            if root is not None and root in alloc_values:
                last_use[root] = i

        if op.opcode in ("fori_loop", "if_else", "while_loop"):
            captures = _collect_sub_graph_captures(op)
            for cap_name in captures:
                root = alias_to_alloc.get(cap_name)
                if root is not None and root in alloc_values:
                    last_use[root] = i

    return last_use


def _insert_dealloc_after(graph: Graph, alloc_val: Value, op_idx: int) -> None:
    """Insert a dealloc op after the op at op_idx."""
    dealloc_op = Op("dealloc", [alloc_val], [], counter=graph.counter)
    graph.ops.insert(op_idx + 1, dealloc_op)
