"""Fold chained ``transpose`` ops into a single transpose (tensor IR peephole).

``transpose(transpose(x, p1), p2)`` is one transpose ``transpose(x, compose)``
with ``compose[i] = p1[p2[i]]`` — permuting twice is permuting once by the
composed permutation. Emitting the chain materializes the intermediate through
HBM (load → remap → store → reload); the composed form skips that round-trip.

Composing is never costlier than the chain: a single transpose (batch-only DMA
remap or an on-chip P↔F swap) replaces two. When the inner transpose has no
other consumers it becomes dead and DCE drops its whole materialization — the
qwen3 attention path chains ``(0,2,1,3)`` then ``(0,1,3,2)`` on a
(1,8,4096,128) tensor purely to feed the QK^T matmul, a ~16 MB intermediate
that folds away entirely.

Runs before ``decompose`` (which never rewrites transpose) alongside the other
data-movement folds in the lowering pipeline.
"""

from __future__ import annotations

from nkigen_lite.core import Graph, Op


def _compose(p1: tuple[int, ...], p2: tuple[int, ...]) -> tuple[int, ...]:
    """Perm of ``transpose(transpose(x, p1), p2)``: ``compose[i] = p1[p2[i]]``.

    ``p1`` maps output-i to source dim ``p1[i]``; applying ``p2`` on top selects
    ``p1[p2[i]]`` as the source dim for the final output position ``i``.
    """
    return tuple(p1[p2[i]] for i in range(len(p2)))


def fold_transpose(graph: Graph) -> int:
    """Compose every ``transpose`` whose input is another ``transpose`` into a
    single transpose reading the inner transpose's source. Returns the fold
    count. Dead inner transposes are removed by the trailing DCE.
    """
    folds = 0
    for op in list(graph.ops):
        if op.opcode != "transpose":
            continue
        inner_val = op.inputs[0]
        inner = inner_val.producer
        if inner is None or inner.opcode != "transpose":
            continue
        src = inner.inputs[0]
        compose = _compose(inner.attrs["perm"], op.attrs["perm"])
        new_op = Op(
            "transpose", [src], [op.result.type],
            {"perm": compose}, counter=graph.counter,
        )
        graph.insert_before(op, new_op)
        graph.replace_value(op.result, new_op.result)
        folds += 1
    if folds:
        graph.dce()
    return folds
