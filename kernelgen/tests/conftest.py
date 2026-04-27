"""
Root conftest.py for NKIPyKernelGen test suite.

Provides:
  - Centralized sys.path setup (replaces per-file path hacks)
  - ``--dump-ir`` CLI flag: dumps intermediate MLIR after each compiler pass

Test modes (LLVM, BIR_SIM, HW, STRING_CHECK, FILECHECK) are declared
per-test via the @nkipy_kernelgen_test decorator or run_kernel_test().
Hardware tests auto-skip when no Trainium device is detected.
Use standard pytest selection to run subsets: pytest tests/passes/, pytest -k "not e2e", etc.
"""

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
