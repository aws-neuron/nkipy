"""Test that both MLIR IR modules can be imported without conflicts."""

import pytest


def test_dual_mlir_import():
    """Test importing both nkipy_kernelgen._mlir.ir and nki.compiler._internal.ir."""
    from nkipy_kernelgen._mlir import ir as nkipy_ir

    from nki.compiler._internal import ir as nki_ir
    from nki.compiler._internal import register_all_dialects
