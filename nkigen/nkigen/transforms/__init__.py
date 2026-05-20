"""
MLIR Transformation Passes for NKIPyKernelGen

This package provides Python implementations of MLIR transformation passes
for lowering traced NumPy functions to Neuron hardware.
"""

__all__ = []

# Import pass management
from ..pass_manager import apply_passes

__all__.extend([
    "apply_passes",
])

# Import nkipy-opt wrapper for passes that can't run from Python
try:
    from .nkipy_opt import (
        get_nkipy_opt_path,
        run_nkipy_opt_passes,
        apply_complete_knob_pipeline,
    )

    __all__.extend([
        "get_nkipy_opt_path",
        "run_nkipy_opt_passes",
        "apply_complete_knob_pipeline",
    ])
except ImportError as e:
    # nkipy-opt wrapper not available
    import warnings
    warnings.warn(
        f"nkipy-opt wrapper not available: {e}",
        ImportWarning
    )
