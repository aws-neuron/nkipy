# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy operation discovery and unit testing."""

from typing import Callable, Dict, List, Optional

import numpy as np

from nkipy.tools.kernel_agent.executor import ExecutionResult, run_kernel

# Target operations to test
TARGET_OPS = {
    "binary": [
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "maximum",
        "minimum",
        "floor_divide",
        "mod",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "logical_and",
        "logical_or",
        "logical_xor",
    ],
    "comparison": [
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    ],
    "unary": [
        "abs",
        "exp",
        "log",
        "sqrt",
        "square",
        "negative",
        "sign",
        "sin",
        "cos",
        "tan",
        "arctan",
        "tanh",
        "ceil",
        "floor",
        "rint",
        "trunc",
    ],
    "reduction": [
        "sum",
        "max",
        "min",
        "mean",
        "any",
    ],
    "transform": [
        "reshape",
        "transpose",
        "expand_dims",
        "copy",
        "repeat",
        "broadcast_to",
    ],
    "creation": [
        "zeros_like",
        "empty_like",
        "full_like",
    ],
    "indexing": [
        "where",
        "take",
        "take_along_axis",
    ],
    "linalg": [
        "matmul",
    ],
}

# Ops requiring special handling
BINARY_OPS = set(
    TARGET_OPS["binary"]
    + TARGET_OPS["comparison"]
    + ["logical_and", "logical_or", "logical_xor"]
)
INTEGER_OPS = {"bitwise_and", "bitwise_or", "bitwise_xor", "floor_divide", "mod"}
POSITIVE_INPUT_OPS = {"log", "sqrt"}


def get_all_ops() -> List[str]:
    """Get flat list of all target operations."""
    return [op for ops in TARGET_OPS.values() for op in ops]


def get_op_category(op_name: str) -> Optional[str]:
    """Get category for an operation."""
    for cat, ops in TARGET_OPS.items():
        if op_name in ops:
            return cat
    return None


def make_kernel(op_name: str) -> Optional[Callable]:
    """Create a simple test kernel for an operation."""
    numpy_func = getattr(np, op_name, None)
    if numpy_func is None:
        return None

    if op_name in BINARY_OPS:

        def kernel(a, b):
            return numpy_func(a, b)
    elif op_name == "reshape":

        def kernel(x):
            return numpy_func(x, (-1,))
    elif op_name == "transpose":

        def kernel(x):
            return numpy_func(x)
    elif op_name == "expand_dims":

        def kernel(x):
            return numpy_func(x, axis=0)
    elif op_name == "broadcast_to":

        def kernel(x):
            return numpy_func(x, (2,) + x.shape)
    elif op_name == "repeat":

        def kernel(x):
            return numpy_func(x, 2, axis=0)
    elif op_name == "where":

        def kernel(cond, a, b):
            return numpy_func(cond, a, b)
    elif op_name == "take":

        def kernel(x, indices):
            return numpy_func(x, indices, axis=0)
    elif op_name == "take_along_axis":

        def kernel(x, indices):
            return numpy_func(x, indices, axis=0)
    elif op_name in ("zeros_like", "empty_like"):

        def kernel(x):
            return numpy_func(x)
    elif op_name == "full_like":

        def kernel(x):
            return numpy_func(x, 1.0)
    elif op_name == "matmul":

        def kernel(a, b):
            return numpy_func(a, b)
    else:
        # Default: unary
        def kernel(x):
            return numpy_func(x)

    return kernel


def make_inputs(
    op_name: str, dtype: str = "float32", shape: tuple = (32, 32)
) -> Dict[str, np.ndarray]:
    """Create test inputs for an operation."""
    np_dtype = getattr(np, dtype)

    # Use deterministic values
    size = int(np.prod(shape))
    base = np.linspace(0.1, 1.0, size).reshape(shape).astype(np_dtype)

    if op_name in POSITIVE_INPUT_OPS:
        base = np.abs(base) + 0.1

    if op_name in BINARY_OPS:
        return {"a": base, "b": base * 0.5 + 0.1}
    elif op_name == "where":
        cond = (base > 0.5).astype(np.bool_)
        return {"cond": cond, "a": base, "b": base * 2}
    elif op_name in ("take", "take_along_axis"):
        indices = np.zeros((shape[0],), dtype=np.int32)
        return {
            "x": base,
            "indices": indices.reshape(-1, 1)
            if op_name == "take_along_axis"
            else indices,
        }
    elif op_name == "matmul":
        return {"a": base, "b": base.T}
    else:
        return {"x": base}


def test_op(
    op_name: str,
    dtype: str = "float32",
    run_hardware: bool = True,
) -> ExecutionResult:
    """Test a single operation."""
    kernel = make_kernel(op_name)
    if kernel is None:
        from nkipy.tools.kernel_agent.executor import StageResult

        return ExecutionResult(
            numpy=StageResult(success=False, error=f"Unknown op: {op_name}")
        )

    # Skip unsupported dtype combinations
    if op_name in INTEGER_OPS and dtype not in ("int32", "int16", "int8"):
        dtype = "int32"

    inputs = make_inputs(op_name, dtype)
    return run_kernel(kernel, inputs, run_hardware=run_hardware)


def discover_ops(
    ops: Optional[List[str]] = None,
    dtypes: Optional[List[str]] = None,
    run_hardware: bool = True,
) -> Dict:
    """Discover support status for operations."""
    ops = ops or get_all_ops()
    dtypes = dtypes or ["float32"]

    results = {"ops": {}, "summary": {"pass": 0, "fail": 0, "total": 0}}

    for op in ops:
        results["ops"][op] = {}
        for dtype in dtypes:
            result = test_op(op, dtype, run_hardware=run_hardware)
            status = result.summary
            results["ops"][op][dtype] = status
            results["summary"]["total"] += 1
            if status == "pass":
                results["summary"]["pass"] += 1
            else:
                results["summary"]["fail"] += 1

    return results


def print_results(results: Dict) -> None:
    """Print results in concise format."""
    s = results["summary"]
    print(f"\n=== Op Discovery: {s['pass']}/{s['total']} passed ===\n")

    for op, dtypes in results["ops"].items():
        statuses = [f"{d}:{s[:4]}" for d, s in dtypes.items()]
        print(f"  {op:20} {' '.join(statuses)}")
