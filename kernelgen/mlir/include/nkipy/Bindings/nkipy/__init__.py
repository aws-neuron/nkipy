# Import from system MLIR package (not bundled)
# This avoids nanobind type conflicts with nki.compiler._internal
from mlir import ir
from mlir import dialects

# Import NKIPy-specific extensions
from ._mlir_libs._nkipy import nkipy

# Re-export for convenience
__all__ = ['ir', 'dialects', 'nkipy']