"""Top-level lowering pipeline: tensor_ir → nki_ir.

Pipeline: canonicalize → decompose → layout_solver → direct_lower
Produces legal NKI IR directly.
"""

from __future__ import annotations

from nkigen_lite.core import Graph
from nkigen_lite import nki_ir
from nkigen_lite.tensor_ir.passes.canonicalize import canonicalize
from nkigen_lite.tensor_ir.passes.decompose import decompose
from nkigen_lite.tensor_ir.passes.layout_solver import Layout, solve_graph
from nkigen_lite.tensor_ir.passes.hardware import HardwareProfile, TRN2


def lower_to_nki(
    graph: Graph,
    target: HardwareProfile = TRN2,
    layouts: dict[str, Layout] | None = None,
    skip_canonicalize: bool = False,
    skip_decompose: bool = False,
    verify_each_phase: bool = False,
) -> nki_ir.Graph:
    """Lower a tensor_ir graph to nki_ir through the full pass pipeline.

    Args:
        graph: tensor_ir Graph to lower.
        target: Hardware target parameters.
        layouts: Pre-assigned layouts (skips layout solver if given).
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

    # Phase 3: layout solving
    if layouts is None:
        layouts = solve_graph(graph)

    # Phase 4: direct lower to nki_ir
    from nkigen_lite.tensor_ir.passes.basic.direct_lower import lower_graph
    return lower_graph(graph, layouts)
