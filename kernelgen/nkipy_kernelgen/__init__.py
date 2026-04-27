"""
NKIPyKernelGen - Lowering from NumPy to NKI compiler

This package provides tools to trace Python functions with NumPy operations
and convert them to MLIR for compilation with neuronxcc.
"""

from .trace import trace
from .traced_array import TracedArray
from .custom_op import CustomOp
from .execution import verify_against_numpy
from .pass_manager import apply_passes
from . import apis
from . import transforms

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "trace",
    "TracedArray",
    "CustomOp",
    "verify_against_numpy",
    "apply_passes",
    "apis",
    "transforms",
]
