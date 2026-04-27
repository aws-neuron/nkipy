"""
Execution engine for running MLIR on CPU and verifying against NumPy.
"""

import numpy as np
from typing import Callable, List, Tuple, Any
from nkipy_kernelgen.llvm import LLVMModule


def verify_against_numpy(
    traced_func: Callable, numpy_func: Callable, input_arrays: List[np.ndarray],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> Tuple[bool, Any, Any]:

    # Get NumPy result
    numpy_result = numpy_func(*input_arrays)

    # Get MLIR module
    module = traced_func.to_mlir()

    runner = LLVMModule(module, traced_func.__name__)
    mlir_result = runner(*input_arrays)

    matches = np.allclose(mlir_result, numpy_result, rtol=rtol, atol=atol)
    # Return results without raising an error - let the test decide what to do
    return matches, mlir_result, numpy_result