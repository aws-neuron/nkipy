"""Fold ``broadcast_to`` into elementwise consumers (tensor IR peephole).

A ``broadcast_to`` materialized through HBM is pure waste when its only
consumers are elementwise binary ops: the vector/scalar engine broadcasts a
size-1 *free* dim natively (``tensor_scalar_arith``), and a partition-1
operand is fanned across lanes by the load itself. So instead of emitting a
separate broadcast segment (load source → replicate → store the full tensor →
reload in the consumer), we rewire the consumers to read the un-broadcast
source and let the elementwise tile machinery broadcast on-chip.

This is only sound when the broadcast is *collapse-preserving* for every
consumer — i.e. the source, right-aligned to the consumer's output shape and
collapsed to 2D ``(P, F)``, has ``P in {1, rep_P}`` and ``F in {1, rep_F}``.
That is exactly the case the elementwise emitter's ``emit_binary_op`` handles
without an HBM round-trip (native F broadcast, or partition broadcast). A
*middle* broadcast (e.g. GQA head expansion ``(1,8,1,64) -> (1,8,8,64)``) is
NOT collapse-preserving: folding it would put a mismatched partition extent in
the consumer's tile loop and break the leading-dims-onto-partition collapse,
turning one packed 2D loop into an unrolled per-row one. Those stay
materialized.

Running before ``decompose`` also removes the extra HBM round-trip that
``decompose``'s docstring flags as the source of a residual ~1/65536
floor-divide precision error: ``div(a, broadcast_to(b))`` folds to
``div(a, b)`` → ``mul(a, reciprocal(b))``, so the reciprocal is computed on the
small operand and broadcast natively rather than after a scratch replicate.
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import Graph, Value


# Binary elementwise opcodes whose lowering (``emit_binary_op``) broadcasts a
# size-1 partition/free operand on-chip. Unary ops are excluded: rewiring
# ``neg(broadcast_to(x))`` to ``neg(x)`` would shrink the result shape its own
# consumers observe. ``where`` is excluded: its copy_predicated lowering needs
# exactly-matching operand shapes.
_FOLDABLE_BINARY = frozenset({
    "add", "sub", "mul", "div", "maximum", "minimum",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
})


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...] | None:
    """Right-aligned numpy broadcast of *shapes*, or None if incompatible."""
    rank = max((len(s) for s in shapes), default=0)
    out = [1] * rank
    for s in shapes:
        al = (1,) * (rank - len(s)) + tuple(s)
        for i, d in enumerate(al):
            if d == 1:
                continue
            if out[i] == 1:
                out[i] = d
            elif out[i] != d:
                return None
    return tuple(out)


def _collapses_cleanly(src_shape: tuple[int, ...], rep_shape: tuple[int, ...]) -> bool:
    """True if ``src_shape`` collapses to the elementwise rep's ``(P, F)``.

    For a rank>=3 rep the whole segment loops ``(prod(rep[:-1]), rep[-1])``,
    and an operand can join that loop only if its collapsed partition is
    ``rep_P`` or 1 and its collapsed free is ``rep_F`` or 1. For rank<3 reps the
    generic 2D path handles any broadcast, so folding is always collapse-safe.
    """
    if len(rep_shape) < 3:
        return True
    if len(src_shape) > len(rep_shape):
        return False
    rep_P = prod(rep_shape[:-1])
    rep_F = rep_shape[-1]
    al = (1,) * (len(rep_shape) - len(src_shape)) + tuple(src_shape)
    P = prod(al[:-1])
    F = al[-1]
    return (F == rep_F or F == 1) and (P == rep_P or P == 1)


def _can_fold(bcast_op, src: Value) -> bool:
    """True if every consumer of the broadcast result can instead read *src*
    with its result shape and elementwise-collapse both preserved."""
    result = bcast_op.result
    consumers = result.uses
    if not consumers:
        return False
    for op in consumers:
        if op.opcode not in _FOLDABLE_BINARY:
            return False
        # Substitute src for the broadcast result and require the consumer's
        # output shape is unchanged (the other operand must still carry the
        # full target shape).
        sub_shapes = [
            src.type.shape if inp is result else inp.type.shape
            for inp in op.inputs
        ]
        new_out = _broadcast_shapes(*sub_shapes)
        if new_out != op.results[0].type.shape:
            return False
        if not _collapses_cleanly(src.type.shape, op.results[0].type.shape):
            return False
    return True


def fold_broadcast(graph: Graph) -> int:
    """Rewire elementwise consumers of collapse-safe ``broadcast_to`` ops to
    their source and drop the now-dead broadcasts. Returns the fold count."""
    output_values = set(graph.outputs.values())
    folds = 0
    for op in list(graph.ops):
        if op.opcode != "broadcast_to":
            continue
        src = op.inputs[0]
        # A broadcast that is itself a graph output must stay materialized.
        if op.result in output_values:
            continue
        if not _can_fold(op, src):
            continue
        op.result.replace_all_uses_with(src)
        folds += 1
    if folds:
        graph.dce()
    return folds
