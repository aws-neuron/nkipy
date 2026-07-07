"""Top-level lowering pipeline: tensor_ir → nki_ir.

Pipeline: canonicalize → decompose → direct_lower. Produces legal NKI IR
directly. Layouts are decided per segment inside direct_lower (see
passes/layout.py), not as a separate global pass.
"""

from __future__ import annotations

from nkigen_lite.core import Graph
from nkigen_lite import nki_ir
from nkigen_lite.tensor_ir.passes.canonicalize import canonicalize
from nkigen_lite.tensor_ir.passes.decompose import decompose
from nkigen_lite.tensor_ir.passes.hardware import HardwareProfile, TRN2


def lower_to_nki(
    graph: Graph,
    target: HardwareProfile = TRN2,
    skip_canonicalize: bool = False,
    skip_decompose: bool = False,
    verify_each_phase: bool = False,
) -> nki_ir.Graph:
    """Lower a tensor_ir graph to nki_ir through the full pass pipeline.

    Args:
        graph: tensor_ir Graph to lower.
        target: Hardware target parameters. Currently unused — no pass in
            this pipeline reads it yet (tile-legality constants are instead
            imported directly from nki_ir.ir). Accepted for forward
            compatibility with a future cost model / multi-target lowering.
        skip_canonicalize: Skip the canonicalize pass.
        skip_decompose: Skip the decompose pass.
        verify_each_phase: Run Graph.verify after every nki_ir phase.

    Returns:
        nki_ir Graph ready for interpretation or code generation.
    """
    # Phase 1-2: simplify tensor_ir
    if not skip_canonicalize:
        canonicalize(graph)
    if not skip_decompose:
        decompose(graph)

    # Phase 3: direct lower to nki_ir (per-segment layout decisions)
    from nkigen_lite.tensor_ir.passes.basic.direct_lower import lower_graph
    return lower_graph(graph)
