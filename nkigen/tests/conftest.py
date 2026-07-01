"""
Root conftest.py for NKIPyKernelGen test suite.

Provides:
  - Centralized sys.path setup (replaces per-file path hacks)
  - ``--dump-ir`` CLI flag: dumps intermediate MLIR after each compiler pass
  - Per-worker Neuron core partitioning for pytest-xdist

Test modes (LLVM, HW, STRING_CHECK, FILECHECK) are declared
per-test via the @nkigen_test decorator or run_kernel_test().
Hardware tests auto-skip when no Trainium device is detected.
Use standard pytest selection to run subsets: pytest tests/passes/, pytest -k "not e2e", etc.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Centralized sys.path setup
# ---------------------------------------------------------------------------
# This replaces all the scattered sys.path.insert() hacks in individual test
# files. By adding these paths once here (loaded automatically by pytest),
# utility modules become importable without path manipulation in each file.

_tests_dir = Path(__file__).parent

# Allow 'from harness import ...' in all tests
sys.path.insert(0, str(_tests_dir))

# Allow 'from pass_utils import ...' in pass tests
sys.path.insert(0, str(_tests_dir / "passes"))


# ---------------------------------------------------------------------------
# --dump-ir CLI option
# ---------------------------------------------------------------------------
# When passed, run_kernel_test() automatically saves intermediate MLIR after
# every compiler pass so you can inspect the IR without modifying test code.
#
# Usage:
#   pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s
#
# IR files are written to  tests/<category>/outputs/<test_name>/
# e.g.  00_input.mlir, 01_prepare_arithmetic.mlir, ...


def pytest_addoption(parser):
    parser.addoption(
        "--dump-ir",
        action="store_true",
        default=False,
        help="Dump intermediate MLIR after each compiler pass to tests/*/outputs/<test>/",
    )


def pytest_configure(config):
    """Store the --dump-ir flag where harness.py can read it."""
    # Using a module-level global avoids passing config through every call site.
    import harness as _harness
    _harness._DUMP_IR_ENABLED = config.getoption("--dump-ir", default=False)

    # -----------------------------------------------------------------------
    # Per-worker Neuron core partitioning for pytest-xdist
    # -----------------------------------------------------------------------
    # Each xdist worker gets a unique logical Neuron core via
    # NEURON_RT_VISIBLE_CORES. This prevents workers from racing for the same
    # cores (NRT allocates all visible cores on init).
    #
    # We set NEURON_LOGICAL_NC_CONFIG=1 so each logical NC = 1 physical core,
    # giving 128 logical cores on trn2.48xlarge. Tests compile with --lnc 1,
    # so each worker needs only 1 logical core.
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        worker_num = int(worker_id.replace("gw", ""))
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(worker_num)
