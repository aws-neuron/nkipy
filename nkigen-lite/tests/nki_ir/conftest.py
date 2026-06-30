"""Fixtures for nki_ir tests.

Provides ``compile_and_run`` — compiles an nki_ir graph via Kernel
Builder and executes on Trainium hardware.

HW tests are marked with ``@pytest.mark.hw`` so they can be run
separately or ordered after interpreter tests::

    pytest nkigen_lite/tests/nki_ir/ -m "not hw"   # interpreter only
    pytest nkigen_lite/tests/nki_ir/ -m hw          # HW only
    pytest nkigen_lite/tests/nki_ir/                # all (interpreter first)
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.nki_ir import Graph
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

import nki.compiler.kernel_builder as nb


@pytest.fixture
def compile_and_run():
    """Compile an nki_ir graph and execute on Trainium.

    Returns a callable: ``(graph, inputs, outputs) -> outputs_dict``.
    """
    opts = nb.CompileOptions(target="trn2")

    def _run(
        graph: Graph,
        inputs: dict[str, np.ndarray],
        outputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        kernel_fn = build_kb_kernel(graph)
        nb.compile_and_execute(
            kernel_fn, inputs=inputs, outputs=outputs, compile_opts=opts,
        )
        return outputs

    return _run


def pytest_collection_modifyitems(items):
    """Auto-mark HW tests and order them after interpreter tests."""
    interp_tests = []
    hw_tests = []
    for item in items:
        if "compile_and_run" in item.fixturenames or "_hw" in item.name:
            item.add_marker(pytest.mark.hw)
            hw_tests.append(item)
        else:
            interp_tests.append(item)
    items[:] = interp_tests + hw_tests
