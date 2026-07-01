"""
Passes test conftest.py.

Provides:
  - Auto-marks all pass tests with 'passes' marker
  - Shared pass utilities exposed as fixtures
"""

import pytest

from pass_utils import (
    run_passes,
    trace_to_mlir_with_preprocessing,
    compile_through_passes,
    compile_knob_pipeline,
    save_mlir_to_file,
    get_test_output_dir,
    run_filecheck,
    assert_ir_unchanged,
    get_filecheck_path,
    verify_tiled_mlir_with_numpy,
)

# Auto-mark all tests in this directory tree
pytestmark = pytest.mark.passes


@pytest.fixture
def filecheck():
    """Fixture providing the run_filecheck function."""
    return run_filecheck


@pytest.fixture
def pass_runner():
    """Fixture providing the run_passes function."""
    return run_passes


@pytest.fixture
def pass_compiler():
    """Fixture providing the compile_through_passes function."""
    return compile_through_passes


@pytest.fixture
def knob_pipeline():
    """Fixture providing the compile_knob_pipeline function."""
    return compile_knob_pipeline


@pytest.fixture
def mlir_preprocessor():
    """Fixture providing trace_to_mlir_with_preprocessing."""
    return trace_to_mlir_with_preprocessing


@pytest.fixture
def mlir_verifier():
    """Fixture providing the verify_tiled_mlir_with_numpy function."""
    return verify_tiled_mlir_with_numpy
